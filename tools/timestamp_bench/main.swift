// tsbench — dissect MTL4 writeTimestamp overhead (precise vs relaxed).
//
// Context: putting precise GPU timestamps around every kernel costs ~1.3x in uzu.
//
// Experiments:
//   overhead  — per-kernel stamps in a uzu-style barrier-separated chain (repro)
//   stampcost — fixed per-stamp cost with trivial kernels
//   encsplit  — relaxed stamps with one encoder per kernel: accuracy vs precise,
//               and the cost of splitting encoders itself
//   cbstamp   — MTL4CommandBuffer.writeTimestampIntoHeap between encoders
//   serial    — does a precise stamp imply ordering? (correctness probe)
//   clock     — what clock/units the heap timestamps use
//
// Build: ./build.sh   Run: ./tsbench [all|overhead|stampcost|encsplit|cbstamp|serial|clock]

import Foundation
import Metal

let kernelSource = """
#include <metal_stdlib>
using namespace metal;
struct Params { uint iters; float scale; };

// ALU spin: reads+writes one float4 per thread (L2-resident at small grids)
kernel void work(device float4 *buf [[buffer(0)]],
                 constant Params &p [[buffer(1)]],
                 uint gid [[thread_position_in_grid]])
{
    float4 v0 = buf[gid];
    float4 v1 = v0 + float4(1.0f);
    float4 v2 = v0 + float4(2.0f);
    float4 v3 = v0 + float4(3.0f);
    const float4 s = float4(p.scale);
    for (uint i = 0; i < p.iters; ++i) {
        v0 = fma(v0, s, float4(0.25f));
        v1 = fma(v1, s, float4(0.5f));
        v2 = fma(v2, s, float4(0.75f));
        v3 = fma(v3, s, float4(1.0f));
    }
    buf[gid] = v0 + v1 + v2 + v3;
}

// DRAM streaming: read a big slice, write almost nothing (matmul weight-load stand-in)
kernel void stream(const device float4 *src [[buffer(0)]],
                   device float4 *dst [[buffer(1)]],
                   constant Params &p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]])
{
    float4 v = src[gid];
    float4 acc = float4(0.0f);
    for (uint i = 0; i < p.iters; ++i) { acc = fma(v, float4(p.scale), acc); }
    if (acc.x == 12345.678f) dst[0] = acc; // data-dependent: never true, keeps the read alive
}

// serial probe: out[0] = in[0] + 1
kernel void bump(device uint *buf [[buffer(0)]]) { buf[0] += 1; }
"""

func nowNs() -> UInt64 { clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) }
func fmtUs(_ ns: Double) -> String { String(format: "%9.1f us", ns / 1e3) }
func fmtMs(_ ns: Double) -> String { String(format: "%8.3f ms", ns / 1e6) }

@available(macOS 26.0, *)
enum StampMode: String { case none, relaxed, precise }

@available(macOS 26.0, *)
struct Trial {
    var gpuNs: Double = 0
    var encodeNs: Double = 0
    var stamps: [UInt64] = []
}

@available(macOS 26.0, *)
final class TsBench {
    let device: MTLDevice
    let queue: any MTL4CommandQueue
    let residencySet: any MTLResidencySet
    let lib: any MTLLibrary
    var psos: [String: any MTLComputePipelineState] = [:]

    struct Params { var iters: UInt32; var scale: Float }
    let paramsBuffer: any MTLBuffer

    init() {
        device = MTLCreateSystemDefaultDevice()!
        queue = device.makeMTL4CommandQueue()!
        let rsd = MTLResidencySetDescriptor()
        rsd.initialCapacity = 4096
        residencySet = try! device.makeResidencySet(descriptor: rsd)
        queue.addResidencySet(residencySet)
        lib = try! device.makeLibrary(source: kernelSource, options: nil)
        paramsBuffer = device.makeBuffer(length: 16, options: .storageModeShared)!
        register(paramsBuffer)
    }

    func register(_ alloc: any MTLAllocation) {
        residencySet.addAllocation(alloc)
        residencySet.commit()
        residencySet.requestResidency()
    }

    func pso(_ name: String) -> any MTLComputePipelineState {
        if let p = psos[name] { return p }
        let p = try! device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
        psos[name] = p
        return p
    }

    func setParams(iters: UInt32, scale: Float = 0.9999) {
        var p = Params(iters: iters, scale: scale)
        memcpy(paramsBuffer.contents(), &p, MemoryLayout<Params>.size)
    }

    func makeBufs(count: Int, bytes: Int, shared: Bool = true) -> [any MTLBuffer] {
        (0..<count).map { _ in
            let b = device.makeBuffer(length: bytes, options: shared ? .storageModeShared : .storageModePrivate)!
            if shared { memset(b.contents(), 0x3c, bytes) }
            register(b)
            return b
        }
    }

    func makeHeap(count: Int) -> any MTL4CounterHeap {
        let d = MTL4CounterHeapDescriptor()
        d.type = .timestamp
        d.count = count
        return try! device.makeCounterHeap(descriptor: d)
    }

    /// Argument table with `bufs` at slots 0..., and params buffer at `paramsIndex` (if non-nil).
    func table(_ bufs: [any MTLBuffer], paramsIndex: Int?) -> any MTL4ArgumentTable {
        let d = MTL4ArgumentTableDescriptor()
        d.maxBufferBindCount = bufs.count + (paramsIndex != nil ? 1 : 0)
        let t = try! device.makeArgumentTable(descriptor: d)
        for (i, b) in bufs.enumerated() {
            t.setAddress(b.gpuAddress, index: i)
        }
        if let pi = paramsIndex { t.setAddress(paramsBuffer.gpuAddress, index: pi) }
        return t
    }

    func commitAndWait(_ cb: any MTL4CommandBuffer) -> Double {
        let sema = DispatchSemaphore(value: 0)
        var g = 0.0
        let opts = MTL4CommitOptions()
        opts.addFeedbackHandler { fb in g = (fb.gpuEndTime - fb.gpuStartTime) * 1e9; sema.signal() }
        queue.commit([cb], options: opts)
        sema.wait()
        return g
    }

    func resolve(_ heap: any MTL4CounterHeap, _ n: Int) -> [UInt64] {
        let data = try! heap.resolveCounterRange(0..<n)!
        return data.withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
    }

    func stamp(_ enc: any MTL4ComputeCommandEncoder, _ heap: any MTL4CounterHeap, _ idx: Int, _ mode: StampMode) {
        switch mode {
        case .relaxed: enc.writeTimestamp(granularity: .relaxed, counterHeap: heap, index: idx)
        case .precise: enc.writeTimestamp(granularity: .precise, counterHeap: heap, index: idx)
        case .none: break
        }
    }

    /// params slot per kernel kind: work -> 1, stream -> 2, bump -> none
    func paramsIndex(for kernel: String) -> Int? {
        switch kernel {
        case "work": return 1
        case "stream": return 2
        default: return nil
        }
    }

    /// One encoder, N dispatches of the same pso. Slot 0 is rebound per dispatch:
    /// if `offsets` has >1 entry, slot0 = bufs[0] + offsets[i] (streaming over one big buffer);
    /// else slot0 = bufs[i % count] (rotate buffers). Other slots bound once.
    /// uzu-style: optional dispatch barrier between kernels. Stamps bracket each kernel.
    func runChain(nKernels n: Int, grid: Int, kernel: String, bufs: [any MTLBuffer], offsets: [Int],
                  mode: StampMode, barriers: Bool) -> Trial {
        let nStamps = mode == .none ? 0 : 2 * n
        let heap = nStamps > 0 ? makeHeap(count: nStamps) : nil
        let alloc = device.makeCommandAllocator()!
        let cb = device.makeCommandBuffer()!
        cb.beginCommandBuffer(allocator: alloc)
        let enc = cb.makeComputeCommandEncoder()!
        let t = table(bufs, paramsIndex: paramsIndex(for: kernel))
        enc.setComputePipelineState(pso(kernel))
        enc.setArgumentTable(t)
        let gridSize = MTLSize(width: grid, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(256, pso(kernel).maxTotalThreadsPerThreadgroup), height: 1, depth: 1)

        let t0 = nowNs()
        var si = 0
        for i in 0..<n {
            if offsets.count > 1 {
                t.setAddress(bufs[0].gpuAddress + UInt64(offsets[i % offsets.count]), index: 0)
            } else if bufs.count > 1 {
                t.setAddress(bufs[i % bufs.count].gpuAddress, index: 0)
            }
            if heap != nil { stamp(enc, heap!, si, mode); si += 1 }
            enc.dispatchThreads(threadsPerGrid: gridSize, threadsPerThreadgroup: tgSize)
            if heap != nil { stamp(enc, heap!, si, mode); si += 1 }
            if barriers {
                enc.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: .device)
            }
        }
        let encodeNs = Double(nowNs() - t0)
        enc.endEncoding()
        cb.endCommandBuffer()
        let gpuNs = commitAndWait(cb)
        let stamps = heap != nil ? resolve(heap!, nStamps) : []
        return Trial(gpuNs: gpuNs, encodeNs: encodeNs, stamps: stamps)
    }

    /// N kernels, each in its OWN encoder (relaxed stamps become per-encoder-boundary).
    func runSplitEncoders(nKernels n: Int, grid: Int, kernel: String, bufs: [any MTLBuffer], offsets: [Int],
                          mode: StampMode, barriers: Bool, cbLevelStamps: Bool = false) -> Trial {
        let nStamps = mode == .none ? 0 : 2 * n
        let heap = nStamps > 0 ? makeHeap(count: nStamps) : nil
        let alloc = device.makeCommandAllocator()!
        let cb = device.makeCommandBuffer()!
        cb.beginCommandBuffer(allocator: alloc)
        let p = pso(kernel)
        let gridSize = MTLSize(width: grid, height: 1, depth: 1)
        let tgSize = MTLSize(width: min(256, p.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)

        let t0 = nowNs()
        for i in 0..<n {
            let enc = cb.makeComputeCommandEncoder()!
            let t = table(bufs, paramsIndex: paramsIndex(for: kernel))
            if offsets.count > 1 {
                t.setAddress(bufs[0].gpuAddress + UInt64(offsets[i % offsets.count]), index: 0)
            } else if bufs.count > 1 {
                t.setAddress(bufs[i % bufs.count].gpuAddress, index: 0)
            }
            enc.setComputePipelineState(p)
            enc.setArgumentTable(t)
            if let heap, !cbLevelStamps { stamp(enc, heap, 2 * i, mode) }
            enc.dispatchThreads(threadsPerGrid: gridSize, threadsPerThreadgroup: tgSize)
            if let heap, !cbLevelStamps { stamp(enc, heap, 2 * i + 1, mode) }
            if barriers {
                enc.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: .device)
            }
            enc.endEncoding()
            if let heap, cbLevelStamps {
                cb.writeTimestamp(counterHeap: heap, index: 2 * i)     // after encoder i's work
                // second stamp lands at top of next iteration's encoder; emulate with a
                // post-work stamp only: use 2*i for post, leave odd entries for analysis
            }
        }
        if let heap, cbLevelStamps { cb.writeTimestamp(counterHeap: heap, index: 2 * n - 1) }
        let encodeNs = Double(nowNs() - t0)
        cb.endCommandBuffer()
        let gpuNs = commitAndWait(cb)
        let stamps = heap != nil ? resolve(heap!, nStamps) : []
        return Trial(gpuNs: gpuNs, encodeNs: encodeNs, stamps: stamps)
    }

    func repeated(_ trials: Int = 5, warmup: Int = 2, _ body: () -> Trial) -> Trial {
        for _ in 0..<warmup { _ = body() }
        let rs = (0..<trials).map { _ in body() }.sorted { $0.gpuNs < $1.gpuNs }
        return rs[rs.count / 2]
    }

    func calibrate(grid: Int, kernel: String, bufs: [any MTLBuffer], offsets: [Int], targetUs: Double) -> UInt32 {
        var iters: UInt32 = 100
        for round in 0..<12 {
            setParams(iters: iters)
            let r = repeated(3, warmup: 3) {
                runChain(nKernels: 8, grid: grid, kernel: kernel, bufs: bufs, offsets: offsets, mode: .none, barriers: true)
            }
            let perKernelUs = r.gpuNs / 8 / 1e3
            var next = max(1, Int((Double(iters) * targetUs / perKernelUs).rounded()))
            if round < 2 { next = max(next, Int(iters) / 4) }
            if abs(next - Int(iters)) <= max(1, Int(iters) / 10) { iters = UInt32(next); break }
            iters = UInt32(next)
        }
        setParams(iters: iters)
        return iters
    }

    // MARK: overhead repro — ALU kernels ~15us (L2) and DRAM stream kernels ~20us

    func expOverhead(n: Int = 128, targetUs: Double = 15) {
        print("== exp overhead: per-kernel stamps, barrier-separated chain ==")
        let grid = 16384
        let bufs = makeBufs(count: 1, bytes: grid * 16)
        let iters = calibrate(grid: grid, kernel: "work", bufs: bufs, offsets: [0], targetUs: targetUs)
        print(String(format: "work kernel: grid=%d iters=%d (target %.0f us)", grid, iters, targetUs))
        print(String(format: "%-36@ %10@ %10@ %10@ %8@", "variant", "GPU total", "per kernel", "encode CPU", "x base"))
        var base = 0.0
        let extraBufs = makeBufs(count: 8, bytes: grid * 16)
        for (name, mode, barriers, nb) in [
            ("plain (barriers)", StampMode.none, true, 1),
            ("relaxed 2/kernel", .relaxed, true, 1),
            ("precise 2/kernel", .precise, true, 1),
            ("plain, no barriers, 8 bufs", .none, false, 8),
            ("precise 2/kernel, no barriers", .precise, false, 8),
        ] as [(String, StampMode, Bool, Int)] {
            let offs = [Int](repeating: 0, count: 8)
            let r = repeated(3, warmup: 1) { runChain(nKernels: n, grid: grid, kernel: "work", bufs: Array(extraBufs.prefix(nb)), offsets: offs, mode: mode, barriers: barriers) }
            if mode == .none && barriers { base = r.gpuNs }
            print(String(format: "%-36@ %10@ %10@ %10@ %8.3f", name, fmtMs(r.gpuNs), fmtUs(r.gpuNs / Double(n)), fmtMs(r.encodeNs), base > 0 ? r.gpuNs / base : 1))
            fflush(stdout)
        }

        // DRAM streaming variant: 32 kernels x 4MB each = 128MB weight traffic
        print("\n-- DRAM streaming kernels (4MB read each, ~matmul-weight profile) --")
        let sbytes = 4 * 1024 * 1024
        let sgrid = sbytes / 16
        let wbuf = makeBufs(count: 1, bytes: 32 * sbytes, shared: false)
        let dbuf = makeBufs(count: 1, bytes: 4096)
        setParams(iters: 4)
        let offsets = (0..<32).map { $0 * sbytes }
        var sbase = 0.0
        for (name, mode) in [("plain", StampMode.none), ("relaxed 2/kernel", .relaxed), ("precise 2/kernel", .precise)] as [(String, StampMode)] {
            let r = repeated(3, warmup: 1) {
                runChain(nKernels: 32, grid: sgrid, kernel: "stream", bufs: wbuf + dbuf, offsets: offsets, mode: mode, barriers: true)
            }
            if mode == .none { sbase = r.gpuNs }
            let perK = r.gpuNs / 32 / 1e3
            print(String(format: "%-36@ %10@ %10.1f us (%5.1f GB/s) %10@ %8.3f", name, fmtMs(r.gpuNs), perK, 4.0 / perK * 1e3, fmtMs(r.encodeNs), r.gpuNs / sbase))
            fflush(stdout)
        }
        print()
    }

    // MARK: stampcost — fixed per-stamp cost with trivial kernels

    func expStampCost(n: Int = 1000) {
        print("== exp stampcost: trivial kernels, barrier-separated ==")
        let grid = 256
        setParams(iters: 1)
        let bufs = makeBufs(count: 1, bytes: grid * 16)
        var base = 0.0
        for mode in [StampMode.none, .relaxed, .precise] {
            let r = repeated { runChain(nKernels: n, grid: grid, kernel: "work", bufs: bufs, offsets: [0], mode: mode, barriers: true) }
            if mode == .none { base = r.gpuNs }
            let perStamp = (r.gpuNs - base) / Double(mode == .none ? 1 : 2 * n)
            print(String(format: "%-8@ GPU %10@  overhead %10@ total, %7.0f ns/stamp | encode CPU %10@",
                         mode.rawValue, fmtMs(r.gpuNs), fmtMs(r.gpuNs - base), mode == .none ? 0 : perStamp, fmtMs(r.encodeNs)))
        }
        print()
    }

    // MARK: encsplit — relaxed stamps with one encoder per kernel

    func expEncSplit(n: Int = 96, targetUs: Double = 15) {
        print("== exp encsplit: one encoder per kernel ==")
        let grid = 16384
        let bufs = makeBufs(count: 1, bytes: grid * 16)
        let iters = calibrate(grid: grid, kernel: "work", bufs: bufs, offsets: [0], targetUs: targetUs)
        print(String(format: "work kernel: grid=%d iters=%d", grid, iters))

        let plainChain = repeated { runChain(nKernels: n, grid: grid, kernel: "work", bufs: bufs, offsets: [0], mode: .none, barriers: true) }
        let splitPlain = repeated { runSplitEncoders(nKernels: n, grid: grid, kernel: "work", bufs: bufs, offsets: [0], mode: .none, barriers: true) }
        let splitRelaxed = repeated { runSplitEncoders(nKernels: n, grid: grid, kernel: "work", bufs: bufs, offsets: [0], mode: .relaxed, barriers: true) }
        let chainPrecise = repeated { runChain(nKernels: n, grid: grid, kernel: "work", bufs: bufs, offsets: [0], mode: .precise, barriers: true) }

        print(String(format: "chain, plain            : %10@  encode %@", fmtMs(plainChain.gpuNs), fmtMs(plainChain.encodeNs)))
        print(String(format: "split encoders, plain   : %10@  encode %@", fmtMs(splitPlain.gpuNs), fmtMs(splitPlain.encodeNs)))
        print(String(format: "split encoders, relaxed : %10@  encode %@", fmtMs(splitRelaxed.gpuNs), fmtMs(splitRelaxed.encodeNs)))
        print(String(format: "chain, precise          : %10@  encode %@", fmtMs(chainPrecise.gpuNs), fmtMs(chainPrecise.encodeNs)))

        // accuracy: per-kernel duration, precise (ground truth) vs relaxed-split
        func durations(_ s: [UInt64], _ n: Int, gpuNs: Double) -> [Double] {
            guard let f = s.first(where: { $0 != 0 }), let l = s.last(where: { $0 != 0 }), l > f else { return [] }
            let scale = gpuNs / Double(l - f)
            return (0..<n).map { Double(s[2 * $0 + 1] &- s[2 * $0]) * scale }
        }
        let dp = durations(chainPrecise.stamps, n, gpuNs: chainPrecise.gpuNs)
        let dr = durations(splitRelaxed.stamps, n, gpuNs: splitRelaxed.gpuNs)
        func stats(_ name: String, _ d: [Double]) {
            guard !d.isEmpty else { print("\(name): no data"); return }
            let s = d.sorted()
            print(String(format: "%-28@ per-kernel us: min %6.1f p50 %6.1f p90 %6.1f max %6.1f",
                         name, s.first! / 1e3, s[s.count / 2] / 1e3, s[Int(Double(s.count) * 0.9)] / 1e3, s.last! / 1e3))
        }
        stats("precise (chain)", dp)
        stats("relaxed (split encoders)", dr)
        let zeros = splitRelaxed.stamps.filter { $0 == 0 }.count
        print("relaxed split stamps zero: \(zeros)/\(splitRelaxed.stamps.count)")
        print()
    }

    // MARK: cbstamp — MTL4CommandBuffer.writeTimestampIntoHeap between encoders

    func expCBStamp(n: Int = 96, targetUs: Double = 15) {
        print("== exp cbstamp: command-buffer-level writeTimestampIntoHeap ==")
        let grid = 16384
        let bufs = makeBufs(count: 1, bytes: grid * 16)
        let iters = calibrate(grid: grid, kernel: "work", bufs: bufs, offsets: [0], targetUs: targetUs)
        print(String(format: "work kernel: grid=%d iters=%d", grid, iters))
        let plain = repeated { runSplitEncoders(nKernels: n, grid: grid, kernel: "work", bufs: bufs, offsets: [0], mode: .none, barriers: true) }
        let cbst = repeated { runSplitEncoders(nKernels: n, grid: grid, kernel: "work", bufs: bufs, offsets: [0], mode: .precise, barriers: true, cbLevelStamps: true) }
        print(String(format: "split encoders plain    : %10@", fmtMs(plain.gpuNs)))
        print(String(format: "split + cb-level stamps : %10@", fmtMs(cbst.gpuNs)))
        let s = cbst.stamps
        let nz = s.filter { $0 != 0 }
        print("cb-level stamps written: \(nz.count)/\(s.count), distinct: \(Set(nz).count)")
        if nz.count >= 2 {
            let scale = cbst.gpuNs / Double(nz.last! - nz.first!)
            let gaps = (0..<nz.count - 1).map { Double(nz[$0 + 1] - nz[$0]) * scale / 1e3 }
            print(String(format: "inter-stamp gaps us: min %.1f p50 %.1f max %.1f",
                         gaps.min()!, gaps.sorted()[gaps.count / 2], gaps.max()!))
        }
        print()
    }

    // MARK: serial — does precise stamping order adjacent dispatches?

    func expSerial(n: Int = 256) {
        print("== exp serial: bump kernel chain, NO barriers; is precise stamping a fence? ==")
        let bufs = makeBufs(count: 1, bytes: 16)
        let zero = makeBufs(count: 1, bytes: 16) // unused spare slot
        _ = zero
        func runChainNoBarrier(mode: StampMode) -> UInt32 {
            let b = bufs[0]
            memset(b.contents(), 0, 16)
            let nStamps = mode == .none ? 0 : 2 * n
            let heap = nStamps > 0 ? makeHeap(count: nStamps) : nil
            let alloc = device.makeCommandAllocator()!
            let cb = device.makeCommandBuffer()!
            cb.beginCommandBuffer(allocator: alloc)
            let enc = cb.makeComputeCommandEncoder()!
            let d = MTL4ArgumentTableDescriptor(); d.maxBufferBindCount = 1
            let t = try! device.makeArgumentTable(descriptor: d)
            t.setAddress(b.gpuAddress, index: 0)
            enc.setComputePipelineState(pso("bump"))
            enc.setArgumentTable(t)
            var si = 0
            for _ in 0..<n {
                if let heap { stamp(enc, heap, si, mode); si += 1 }
                enc.dispatchThreads(threadsPerGrid: MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                if let heap { stamp(enc, heap, si, mode); si += 1 }
            }
            enc.endEncoding()
            cb.endCommandBuffer()
            _ = commitAndWait(cb)
            return b.contents().load(as: UInt32.self)
        }
        let vNone = runChainNoBarrier(mode: .none)
        let vRelaxed = runChainNoBarrier(mode: .relaxed)
        let vPrecise = runChainNoBarrier(mode: .precise)
        print("expected final value if serialized: \(n)")
        print("no stamps      : \(vNone)  \(vNone == n ? "(ordered — dispatches self-serialize here)" : "(RACES — no ordering)")")
        print("relaxed stamps : \(vRelaxed)")
        print("precise stamps : \(vPrecise)")
        print()
    }

    // MARK: clock

    func expClock() {
        print("== exp clock: timestamp clock domain ==")
        let (cpuTs, gpuTs) = device.sampleTimestamps()
        print("device.sampleTimestamps: cpu=\(cpuTs) gpu=\(gpuTs)  (mach_absolute_time=\(mach_absolute_time()))")
        let grid = 65536
        setParams(iters: 500)
        let bufs = makeBufs(count: 1, bytes: grid * 16)
        let r = runChain(nKernels: 4, grid: grid, kernel: "work", bufs: bufs, offsets: [0], mode: .precise, barriers: true)
        if let first = r.stamps.first, let last = r.stamps.last {
            let span = Double(last &- first)
            print(String(format: "heap stamps: first=%u last=%u span=%.0f raw units", first, last, span))
            print(String(format: "feedback GPU time: %.0f ns -> heap clock = %.3f MHz", r.gpuNs, span / r.gpuNs * 1e3))
            print(String(format: "heap stamp - sampleTimestamps.gpuTimestamp = %lld (different clock domain)",
                         Int64(bitPattern: first) - Int64(bitPattern: gpuTs)))
        }
        print()
    }
}

if #available(macOS 26.0, *) {
    let which = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "all"
    let bench = TsBench()
    switch which {
    case "overhead": bench.expOverhead()
    case "stampcost": bench.expStampCost()
    case "encsplit": bench.expEncSplit()
    case "cbstamp": bench.expCBStamp()
    case "serial": bench.expSerial()
    case "clock": bench.expClock()
    default:
        bench.expClock()
        bench.expOverhead()
        bench.expStampCost()
        bench.expEncSplit()
        bench.expCBStamp()
        bench.expSerial()
    }
} else {
    FileHandle.standardError.write("tsbench requires macOS 26+\n".data(using: .utf8)!)
    exit(1)
}
