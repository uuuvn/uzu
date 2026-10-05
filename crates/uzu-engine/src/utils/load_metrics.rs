//! Lightweight, opt-in instrumentation for measuring where model load time
//! goes: shader compilation, weight disk reads, and everything else.
//!
//! Disabled by default and costs a single relaxed atomic load per instrumented
//! operation. Enable with [`enable`] before loading a model, then read the
//! accumulated timings with [`snapshot`].

use std::{
    collections::BTreeSet,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{Duration, Instant},
};

use parking_lot::Mutex;

/// A named load-time bucket that durations are accumulated into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMetric {
    /// Reading and parsing `tokenizer.json`.
    Tokenizer,
    /// Constructing the decoder: weight validation, GPU buffer allocation,
    /// kernel instantiation. Is a superset of [`LoadMetric::MetalLibrary`],
    /// [`LoadMetric::MetalPipelineState`] and [`LoadMetric::WeightsDiskRead`].
    Decoder,
    /// Decompressing and parsing embedded Metal libraries
    /// (`new_library_with_data`).
    MetalLibrary,
    /// Specializing Metal functions and creating compute pipeline states,
    /// i.e. AIR -> native shader compilation.
    MetalPipelineState,
    /// Reading weight bytes from `model.safetensors` (the `read_exact_at`
    /// calls only, buffer allocation excluded).
    WeightsDiskRead,
}

impl LoadMetric {
    pub const ALL: [LoadMetric; 5] = [
        LoadMetric::Tokenizer,
        LoadMetric::Decoder,
        LoadMetric::MetalLibrary,
        LoadMetric::MetalPipelineState,
        LoadMetric::WeightsDiskRead,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            LoadMetric::Tokenizer => "tokenizer.json load",
            LoadMetric::Decoder => "decoder construction (superset)",
            LoadMetric::MetalLibrary => "metal library load",
            LoadMetric::MetalPipelineState => "metal pipeline state creation (shader compilation)",
            LoadMetric::WeightsDiskRead => "weights disk read",
        }
    }
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static NANOS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static CREATIONS: Mutex<BTreeSet<Box<str>>> = Mutex::new(BTreeSet::new());

pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

pub fn disable() {
    ENABLED.store(false, Ordering::Relaxed);
}

pub fn reset() {
    for nanos in &NANOS {
        nanos.store(0, Ordering::Relaxed);
    }
    CREATIONS.lock().clear();
}

pub fn snapshot() -> Vec<(LoadMetric, Duration)> {
    LoadMetric::ALL
        .into_iter()
        .map(|metric| (metric, Duration::from_nanos(NANOS[metric as usize].load(Ordering::Relaxed))))
        .collect()
}

/// Records the creation of one pipeline state, identified by `cache_key`, if
/// metrics are enabled. Use to enumerate which pipeline variants a workload
/// actually triggers.
pub fn record_creation(cache_key: &str) {
    if ENABLED.load(Ordering::Relaxed) {
        CREATIONS.lock().insert(cache_key.into());
    }
}

/// All pipeline state cache keys recorded since the last [`reset`], sorted.
pub fn creations() -> Vec<Box<str>> {
    CREATIONS.lock().iter().cloned().collect()
}

/// Accumulates the time until it is dropped into `metric`, if metrics are
/// enabled. Returns `None` (and measures nothing) otherwise.
pub fn record(metric: LoadMetric) -> Option<LoadMetricGuard> {
    if ENABLED.load(Ordering::Relaxed) {
        Some(LoadMetricGuard::new(metric))
    } else {
        None
    }
}

pub struct LoadMetricGuard {
    metric: LoadMetric,
    start: Instant,
}

impl LoadMetricGuard {
    fn new(metric: LoadMetric) -> Self {
        Self {
            metric,
            start: Instant::now(),
        }
    }
}

impl Drop for LoadMetricGuard {
    fn drop(&mut self) {
        NANOS[self.metric as usize].fetch_add(self.start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}
