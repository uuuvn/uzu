//! Optional per-kernel GPU timestamp instrumentation (Metal 4 counter heaps).
//!
//! When enabled, every generated kernel dispatch is bracketed with
//! `MTL4TimestampGranularity::PRECISE` timestamp writes into a counter heap
//! owned by the command buffer being encoded. On command buffer completion the
//! heap is resolved and per-kernel durations are recorded into a global sink
//! that can be drained with [`take`].
//!
//! Precise stamps split the GPU command stream and fence execution around each
//! kernel (see tools/timestamp_bench), so per-kernel timings are accurate but
//! the total pass gets significantly slower (~1.3-2x). Disabled by default;
//! enable via [`set_enabled`] or the `UZU_KERNEL_TIMESTAMPS` env var.

use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};

fn enabled_flag() -> &'static AtomicBool {
    static ENABLED: OnceLock<AtomicBool> = OnceLock::new();
    ENABLED.get_or_init(|| AtomicBool::new(std::env::var_os("UZU_KERNEL_TIMESTAMPS").is_some()))
}

pub fn set_enabled(enabled: bool) {
    enabled_flag().store(enabled, Ordering::Relaxed);
}

pub fn is_enabled() -> bool {
    enabled_flag().load(Ordering::Relaxed)
}

/// GPU duration of one kernel dispatch, in raw timestamp-counter ticks.
#[derive(Debug, Clone)]
pub struct KernelTiming {
    /// DSL kernel name (from the codegen template).
    pub name: &'static str,
    pub duration_ticks: u64,
}

/// Kernel timings of one completed command buffer.
#[derive(Debug)]
pub struct KernelTimingBatch {
    /// Command buffer GPU time from commit feedback, for cross-checking.
    pub feedback_gpu_time_ns: u64,
    /// `queryTimestampFrequency` of the device, i.e. nominal ticks per second.
    pub timestamp_frequency_hz: u64,
    /// First and last stamp written in the command buffer, in ticks.
    pub span_ticks: (u64, u64),
    pub kernels: Vec<KernelTiming>,
}

static RECORDED: Mutex<Vec<KernelTimingBatch>> = Mutex::new(Vec::new());

pub(super) fn record(batch: KernelTimingBatch) {
    RECORDED.lock().expect("kernel timings lock poisoned").push(batch);
}

/// Drain all batches recorded so far, in completion order.
pub fn take() -> Vec<KernelTimingBatch> {
    std::mem::take(&mut *RECORDED.lock().expect("kernel timings lock poisoned"))
}
