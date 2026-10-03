# tsbench — MTL4 GPU timestamp overhead & semantics (Apple Silicon, Metal 4)

Repro and reverse-engineering of the "~1.3x slowdown from precise per-kernel GPU timestamps"
(Discord, eugenebokhan/uuuvn). Machine: Apple M1 Max, macOS 27, Metal 4.

Build & run:

```sh
./build.sh
./tsbench all        # or: clock | overhead | stampcost | encsplit | cbstamp | serial
```

**WARNING**: all buffers must be added to an `MTLResidencySet` registered on the
MTL4 command queue (`queue.addResidencySet`, as uzu's `MetalContext` does). Without
it, argument-table-bound buffers are accessed through an uncached ~2.2 GB/s path and
every measurement is garbage (~170x slowdown on DRAM streams).

## TL;DR

| timestamp mode | GPU cost per stamp | CPU encode per stamp | what you get |
|---|---|---|---|
| `relaxed` (same encoder) | ~0.3 µs | ~0 | **garbage**: collapses to encoder begin/end, ≤5 stamps written per encoder, rest resolve to 0 |
| `relaxed` (encoder per kernel) | ~0.5 µs + encoder split | +12 µs (split) | per-encoder-boundary stamps; p50 accurate, tail noisy |
| `precise` | **~8–11 µs** | **~5 µs** | exact per-dispatch, but a full pipeline fence per stamp |
| `MTL4CommandBuffer.writeTimestampIntoHeap` (between encoders) | ~24 µs | — | precise, but even pricier |

Overhead on a barrier-separated kernel chain (uzu's shape), stamps around each kernel:

- ~16 µs ALU kernels: precise **2.07x**, relaxed 1.0x
- ~18 µs DRAM-streaming kernels (4 MB weight read each): precise **1.7–1.9x**, relaxed 1.0x
- trivial kernels: precise **+10.5 µs/stamp** flat

So the reported 1.3x is exactly what you get when blocks are ~40–60 µs: two precise
stamps add ~+17 µs per block.

## What `precise` actually does (driver RE)

`-[AGXG13XFamilyComputeContext_mtlnext writeTimestampWithGranularity:intoHeap:atIndex:]`
(AGXMetalG13X, dyld-cache carve via `carve_dylib.py`, disasm via radare2):

1. Both granularities just append a 16-byte `{counterHeap, index}` record to a deferred
   list on the encoder context (bump allocator; the GPU-visible stamp site is patched in
   later when the list is flushed).
2. `precise` additionally, per stamp:
   - closes the current GPU command segment: emits a sync command (`0x6000xxxx` family —
     the same command family `encodeSyncComputeWithBackFacingBarrierSrcMask:…` emits for
     `barrierAfterEncoderStages:…`),
   - emits an end-of-segment word `0xA0000000`,
   - starts a new segment with `0x40000000`.
   I.e. the header-doc "may cause splitting of command encoders" is literal: **every
   precise stamp splits the GPU command stream and fences it**.
3. Semantics proof (`./tsbench serial`): 256 chained `buf[0] += 1` dispatches with no
   barriers produce final value **7** (dispatches race), with relaxed stamps still **7**,
   with precise stamps **256** — a precise stamp fully serializes execution around it.
4. The timestamp written is a **24 MHz counter** (41.7 ns resolution) in a GPU-local
   clock domain — epoch is unrelated to `mach_absolute_time`/`device.sampleTimestamps`
   (which return mach-time nanoseconds) and appears to reset when the GPU power-gates.
   Only deltas are meaningful.

Why ~10 µs per stamp: each stamp costs a full compute-pipeline drain plus a
command-processor timestamp packet (segment teardown + restart). It is *not* "one u64
from a register" — there is no shader-visible clock; the stamp is executed by the CP at
a hard segment boundary.

## How imprecise is `relaxed`?

Inside one encoder, relaxed stamps are sampled **at encoder boundaries only**
(`probe12`-style experiment, N kernels × 2 stamps in a single encoder):

| kernels in one encoder | stamps written / requested | distinct values |
|---|---|---|
| 1 | 2/2 | 2 |
| 2 | 4/4 | 2 |
| 3 | 5/6 | 2 |
| ≥4 | 5/2N | 2 |

First stamp = encoder start, next four = encoder end, everything else = **0**
(never written). Per-kernel durations computed from them are 0 or nonsense.

With **one encoder per kernel**, relaxed stamps are all written and distinct, and
per-kernel durations track precise ones (p50 within ~10%; p90 tail inflated because
encoder teardown lands inside the measurement). But the split itself costs
~7 µs GPU + ~12 µs CPU per encoder, so "relaxed + split" ≈ 1.3x — roughly half the
cost of precise, with worse fidelity.

## Practical options for blockwise timing in uzu

1. Sample per **N kernels** (amortize): precise stamp cost is flat ~8–11 µs, so per-block
   sampling at ~500 µs granularity costs ~2%.
2. If blocks already align with encoder boundaries, relaxed stamps are ~free and correct
   at that granularity.
3. Do not put relaxed stamps inside a shared encoder expecting per-kernel numbers.
4. `MTL4CommandBuffer.writeTimestampIntoHeap` between encoders is precise but the
   most expensive option (~24 µs/stamp) — skip.

## In-engine verification (uzu-engine, Qwen3.5-4B-M, 21-token prefill + decode)

`crates/uzu-engine/examples/kernel_timestamps.rs` instruments the engine itself: the
codegen (`build/metal/bindgen/dispatch.rs`) brackets every kernel dispatch with precise
stamps when `UZU_KERNEL_TIMESTAMPS=1` or `kernel_timestamps::set_enabled(true)`; results
are resolved per command buffer in the MTL4 commit feedback handler.

Measured on M1 Max (wall clock, warm):

| pass | baseline | precise-stamped | overhead |
|---|---|---|---|
| prefill 21 tokens + 1st token (550 kernels, 1 command buffer) | 75.4 ms | 91.7 ms | **1.22x** |
| decode step (350 kernels, 1 command buffer) | 10.8 ms | 18.5 ms | **1.72x** |

Sum of stamped per-kernel durations vs commit-feedback GPU time: prefill 79.3 / 90.6 ms,
decode 12.5 / 18.6 ms — i.e. ~11 ms / ~6 ms of pure fence+stamp cost, ≈ 17 µs per kernel
(2 stamps × ~8.5 µs), matching the microbenchmarks. Decode suffers more because its
kernels are small. Per-kernel attribution is otherwise sane: decode is 55% `Gemv`
(129 calls, 79 µs mean; the 1056 µs outlier is the vocab projection), prefill is 80%
`Gemm` (128 calls, 567 µs mean). Heap tick rate cross-checks exactly to 24 MHz against
feedback GPU time, and `device.queryTimestampFrequency` returns 24 MHz.

Notes:
- Counter heaps cap at **4096 entries** (2048 kernels per command buffer); extra
  dispatches go unstamped.
- Heap creation failing used to print `AGX: MTLCounterHeap: Invalid heap size requested`
  when asking for more.

## Files

- `main.swift` — the bench (experiments: clock, overhead, stampcost, encsplit, cbstamp, serial)
- `build.sh` — `swiftc -O`
- `carve_dylib.py` — carve a dylib out of the split macOS 27 dyld cache (segments at vmaddrs) for r2
- `wrap_macho.py` — wrap raw code bytes in a Mach-O for `llvm-objdump -d`
- `introspect.m`, `introspect2.m` — runtime ObjC introspection: driver class names, method IMPs, code dumps
