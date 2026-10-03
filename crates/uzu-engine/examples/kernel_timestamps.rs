//! Per-kernel GPU timings for one forward pass, using precise MTL4 counter-heap
//! timestamps (see `src/backends/metal/kernel_timestamps.rs`).
//!
//! Downloads the model through the normal registry/download path (cached after
//! first run), loads it with the Metal backend, then runs the same
//! prefill+decode step twice: once as a baseline and once with precise
//! per-kernel timestamps, printing both the overhead and per-kernel stats.
//!
//! ```sh
//! cargo run --release -p uzu-engine --example kernel_timestamps
//! UZU_MODEL=meta-llama/Llama-3.2-1B-Instruct cargo run --release -p uzu-engine --example kernel_timestamps
//! ```

use std::{collections::BTreeMap, time::Instant};

use uzu_engine::{
    backends::metal::{Metal, kernel_timestamps},
    engine::language_model::LanguageModel,
};

const DEFAULT_MODEL: &str = "trymirai/Qwen3.5-4B-M";
const PROMPT: &str = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. ";

fn download_model(repo_id: &str) -> String {
    use uzu::{
        engine::{Engine, EngineConfig},
        storage::DownloadPhase,
    };
    let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    runtime.block_on(async {
        let config = EngineConfig::default().with_allow_ollama_usage(false).with_allow_lmstudio_usage(false);
        let engine = Engine::new(config).await.expect("failed to create engine");
        let model = engine
            .model(repo_id.to_string())
            .await
            .unwrap_or_else(|error| panic!("failed to look up {repo_id} in the registry: {error}"))
            .unwrap_or_else(|| panic!("model {repo_id} not found in the registry"));

        let stream = engine.download(&model).await.expect("failed to start model download");
        while stream.next().await.is_some() {}

        let state = engine.download_state(&model).await.expect("model has no download state");
        assert!(
            matches!(state.phase, DownloadPhase::Downloaded {}),
            "model {repo_id} download did not complete: {:?}",
            state.phase,
        );

        engine.model_path(&model).await.expect("model path unavailable after download")
    })
}

/// One prefill + two decode steps. Returns (prefill wall, decode wall).
fn forward_once(
    model: &LanguageModel<Metal>,
    tokens: &[u64],
) -> (std::time::Duration, std::time::Duration) {
    let mut state = model.create_empty_state(None, 42).expect("failed to create state");
    let options = model.default_stream_options();
    let mut stream = model.stream(tokens, &mut state, options).expect("failed to create stream");
    let start = Instant::now();
    let token = stream.next().expect("stream ended before first token").expect("forward pass failed");
    let prefill_wall = start.elapsed();
    let start = Instant::now();
    stream.next().expect("stream ended before second token").expect("decode failed");
    let decode_wall = start.elapsed();
    let text = model.tokenizer().decode(&[token as u32], false).unwrap_or_default();
    println!("    first token: {token} {text:?} | prefill {:.3} ms, decode {:.3} ms",
             prefill_wall.as_secs_f64() * 1e3, decode_wall.as_secs_f64() * 1e3);
    drop(stream); // waits for all in-flight GPU work
    (prefill_wall, decode_wall)
}

/// Prefill + two decode steps with timestamps enabled. Returns
/// (wall times, prefill batches, decode batches).
#[allow(clippy::type_complexity)]
fn forward_instrumented(
    model: &LanguageModel<Metal>,
    tokens: &[u64],
) -> (std::time::Duration, std::time::Duration, Vec<kernel_timestamps::KernelTimingBatch>, Vec<kernel_timestamps::KernelTimingBatch>) {
    let mut state = model.create_empty_state(None, 42).expect("failed to create state");
    let options = model.default_stream_options();
    kernel_timestamps::set_enabled(true);
    kernel_timestamps::take(); // drain anything from model load
    let mut stream = model.stream(tokens, &mut state, options).expect("failed to create stream");

    let start = Instant::now();
    let token = stream.next().expect("stream ended before first token").expect("prefill failed");
    let text = model.tokenizer().decode(&[token as u32], false).unwrap_or_default();
    let prefill_wall = start.elapsed();
    let prefill_batches = kernel_timestamps::take();

    let start = Instant::now();
    let token2 = stream.next().expect("stream ended before second token").expect("decode failed");
    let decode_wall = start.elapsed();
    let decode_batches = kernel_timestamps::take();
    println!("    tokens: {token} {text:?}, then {token2}");

    drop(stream); // waits for all in-flight GPU work
    kernel_timestamps::set_enabled(false);
    (prefill_wall, decode_wall, prefill_batches, decode_batches)
}

#[derive(Default)]
struct Agg {
    count: usize,
    total_ticks: u64,
    min_ticks: u64,
    max_ticks: u64,
}

fn report(
    label: &str,
    batches: &[kernel_timestamps::KernelTimingBatch],
) {
    println!("\n== {label}: {} command buffer(s) with stamps ==", batches.len());
    let mut all: BTreeMap<&'static str, Agg> = BTreeMap::new();
    let mut total_kernel_ticks = 0u64;
    for (i, batch) in batches.iter().enumerate() {
        let span_ticks = batch.span_ticks.1.saturating_sub(batch.span_ticks.0);
        let effective_mhz = if batch.feedback_gpu_time_ns > 0 {
            span_ticks as f64 / batch.feedback_gpu_time_ns as f64 * 1e3
        } else {
            0.0
        };
        println!(
            "  batch {i}: {} kernels | feedback GPU {:.3} ms | stamp span {:.3} ms @ {} Hz | effective {:.1} MHz",
            batch.kernels.len(),
            batch.feedback_gpu_time_ns as f64 / 1e6,
            span_ticks as f64 / batch.timestamp_frequency_hz as f64 * 1e3,
            batch.timestamp_frequency_hz,
            effective_mhz,
        );
        for kernel in &batch.kernels {
            let agg = all.entry(kernel.name).or_insert_with(|| Agg {
                min_ticks: u64::MAX,
                ..Default::default()
            });
            agg.count += 1;
            agg.total_ticks += kernel.duration_ticks;
            agg.min_ticks = agg.min_ticks.min(kernel.duration_ticks);
            agg.max_ticks = agg.max_ticks.max(kernel.duration_ticks);
            total_kernel_ticks += kernel.duration_ticks;
        }
    }

    let freq = batches.first().map(|b| b.timestamp_frequency_hz).unwrap_or(24_000_000) as f64;
    let us = |ticks: u64| ticks as f64 / freq * 1e6;
    let total_feedback_ns: u64 = batches.iter().map(|b| b.feedback_gpu_time_ns).sum();
    println!("  sum of per-kernel durations : {:.3} ms", us(total_kernel_ticks) / 1e3);
    println!("  sum of feedback GPU times   : {:.3} ms", total_feedback_ns as f64 / 1e6);
    println!("  (difference = fence/stamp cost, ~2 precise stamps per kernel)");

    println!("  {:<40} {:>6} {:>10} {:>10} {:>10} {:>10}", "kernel", "count", "total us", "mean us", "min us", "max us");
    let mut rows: Vec<_> = all.iter().collect();
    rows.sort_by_key(|(_, agg)| std::cmp::Reverse(agg.total_ticks));
    for (name, agg) in rows {
        println!(
            "  {:<40} {:>6} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
            name,
            agg.count,
            us(agg.total_ticks),
            us(agg.total_ticks / agg.count as u64),
            us(agg.min_ticks),
            us(agg.max_ticks),
        );
    }
}

fn main() {
    let repo_id = std::env::var("UZU_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    println!("== model: {repo_id} (downloading if needed) ==");
    let model_path = download_model(&repo_id);
    println!("model at: {model_path}");

    let engine = uzu_engine::engine::Engine::<Metal>::new().expect("failed to create engine");
    let model = engine.load_language_model(std::path::Path::new(&model_path)).expect("failed to load model");

    let tokens: Vec<u64> = model
        .tokenizer()
        .encode(PROMPT, true)
        .expect("tokenization failed")
        .get_ids()
        .iter()
        .map(|&t| t as u64)
        .collect();
    println!("prompt: {} tokens", tokens.len());

    println!("\n== baseline (timestamps off) ==");
    for _ in 0..2 {
        forward_once(&model, &tokens);
    }

    println!("\n== instrumented (precise per-kernel timestamps) ==");
    let (prefill_wall, decode_wall, prefill_batches, decode_batches) = forward_instrumented(&model, &tokens);
    println!("    prefill+1st token wall: {:.3} ms", prefill_wall.as_secs_f64() * 1e3);
    println!("    decode step wall:       {:.3} ms", decode_wall.as_secs_f64() * 1e3);

    report("prefill + 1st token", &prefill_batches);
    report("decode step", &decode_batches);
}
