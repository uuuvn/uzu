import Metal
if #available(macOS 26.0, *) {
    let device = MTLCreateSystemDefaultDevice()!
    for n in [4096, 8192, 16384, 32768, 65536, 131072] {
        let d = MTL4CounterHeapDescriptor(); d.type = .timestamp; d.count = n
        if let _ = try? device.makeCounterHeap(descriptor: d) {
            print("\(n): ok")
        } else {
            print("\(n): FAIL")
        }
    }
}
