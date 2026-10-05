//! Prints a breakdown of where model load time goes: shader compilation
//! (Metal library loading + pipeline state creation), weight disk reads,
//! tokenizer loading, and everything else.
//!
//! Run it twice: the second run benefits from Metal's warm on-disk shader
//! cache. Run with `UZU_CLEAN_COMPILE=1` to bypass that cache and measure the
//! first-run-ever experience. Note that weight reads still hit the OS page
//! cache unless the file was never read before.
//!
//! ```sh
//! cargo run --release --example model_load_time -- <model-path>
//! UZU_CLEAN_COMPILE=1 cargo run --release --example model_load_time -- <model-path>
//! ```

#[cfg(all(feature = "metal", target_vendor = "apple"))]
mod measurement {
    use std::{
        path::PathBuf,
        time::{Duration, Instant},
    };

    use uzu_engine::{
        backends::metal::Metal,
        engine::Engine,
        load_metrics::{self, LoadMetric},
    };

    fn print_metric(
        metrics: &[(LoadMetric, Duration)],
        metric: LoadMetric,
        total: Duration,
    ) {
        let duration = metrics.iter().find(|(m, _)| *m == metric).map(|(_, d)| *d).unwrap_or_default();
        let share = if total.is_zero() {
            0.0
        } else {
            duration.as_secs_f64() / total.as_secs_f64() * 100.0
        };
        println!("  {:<48} {:>10.2?} {:>6.1}%", metric.label(), duration, share);
    }

    pub fn run() {
        let model_path = std::env::args().nth(1).map(PathBuf::from).expect("usage: model_load_time <model-path>");
        println!("model: {}", model_path.display());
        if std::env::var("UZU_CLEAN_COMPILE").is_ok() {
            println!("UZU_CLEAN_COMPILE is set: Metal's on-disk shader cache is bypassed (first-run shader timings)");
        } else {
            println!("UZU_CLEAN_COMPILE is not set: Metal's on-disk shader cache is active");
        }

        load_metrics::enable();

        let start = Instant::now();
        let engine = Engine::<Metal>::new().expect("failed to create engine");
        let engine_time = start.elapsed();

        load_metrics::reset();
        let start = Instant::now();
        let _model = engine.load_language_model(&model_path).expect("failed to load model");
        let load_time = start.elapsed();

        let metrics = load_metrics::snapshot();
        let accounted: Duration = [
            LoadMetric::MetalLibrary,
            LoadMetric::MetalPipelineState,
            LoadMetric::WeightsDiskRead,
            LoadMetric::Tokenizer,
        ]
        .iter()
        .map(|metric| metrics.iter().find(|(m, _)| m == metric).map(|(_, d)| *d).unwrap_or_default())
        .sum();

        println!();
        println!("engine creation (Metal context): {:>10.2?}", engine_time);
        println!("model load total:                {:>10.2?}", load_time);
        for metric in [
            LoadMetric::MetalPipelineState,
            LoadMetric::MetalLibrary,
            LoadMetric::WeightsDiskRead,
            LoadMetric::Tokenizer,
            LoadMetric::Decoder,
        ] {
            print_metric(&metrics, metric, load_time);
        }
        let other = load_time.saturating_sub(accounted);
        let share = if load_time.is_zero() {
            0.0
        } else {
            other.as_secs_f64() / load_time.as_secs_f64() * 100.0
        };
        println!("  {:<48} {:>10.2?} {:>6.1}%", "other (everything else)", other, share);
    }
}

fn main() {
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    measurement::run();
    #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
    eprintln!("the model_load_time example requires the metal backend on an Apple platform");
}
