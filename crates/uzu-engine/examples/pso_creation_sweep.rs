//! Enumerates the pipeline state variants a model actually compiles at run
//! time (lazily, outside of `LanguageModel` construction) by running prefill
//! and decode over a matrix of prefix and suffix sizes, and captures each
//! created pipeline's cache key.
//!
//! The sizes are model-independent: they are chosen from the static dispatch
//! thresholds of the lazily-specialized kernels, so the matrix exercises every
//! reachable variant bucket on any model:
//!
//! - GEMV handles only m <= 8 (`GEMV_MAX_BATCH`), so decode (m=1), 8, and 9.
//! - GEMM simdgroup tiling splits at m in {8, 32, 64}
//!   (`simdgroup_quant_tile`), and MXU tiling buckets m at {16, 63, 255, 511}
//!   (M5+ only), so 31, 32, 63, 64, 65, and a >1024-chunk prefill.
//! - Attention uses the single/two-pass kernels for suffix <= 8 (two-pass once
//!   kv > 1024) and the GEMM kernel for suffix > 8, keyed by
//!   align_q = suffix % 32 == 0 and align_k = kv % simd_bk == 0, so suffix
//!   sizes 9 (unaligned) and 32 (aligned) with aligned and unaligned prefixes.
//! - Sampling kernels specialize on the sampling method, so greedy, fully
//!   stochastic, and repetition-penalty runs are included.
//!
//! ```sh
//! cargo run --release --example pso_creation_sweep -- <model-path>
//! UZU_CLEAN_COMPILE=1 cargo run --release --example pso_creation_sweep -- <model-path>
//! ```

#[cfg(all(feature = "metal", target_vendor = "apple"))]
mod measurement {
    use std::{
        collections::BTreeSet,
        path::PathBuf,
        time::{Duration, Instant},
    };

    use uzu_engine::{
        backends::metal::Metal,
        engine::{
            Engine,
            language_model::stream::{LanguageModelStreamOptions, SamplingMethod},
        },
        load_metrics::{self, LoadMetric},
    };

    #[derive(Clone)]
    enum SamplingSpec {
        Default,
        Greedy,
        Stochastic,
        RepetitionPenalty,
    }

    impl SamplingSpec {
        fn label(&self) -> &'static str {
            match self {
                SamplingSpec::Default => "default",
                SamplingSpec::Greedy => "greedy",
                SamplingSpec::Stochastic => "stochastic(t,k,p,minp)",
                SamplingSpec::RepetitionPenalty => "repetition_penalty",
            }
        }
    }

    struct RunSpec {
        prefix: usize,
        suffix: usize,
        sampling: SamplingSpec,
    }

    fn pipeline_time() -> Duration {
        load_metrics::snapshot()
            .into_iter()
            .find(|(metric, _)| *metric == LoadMetric::MetalPipelineState)
            .map(|(_, duration)| duration)
            .unwrap_or_default()
    }

    fn apply_sampling(
        options: &mut LanguageModelStreamOptions,
        sampling: &SamplingSpec,
    ) {
        options.sampling_method = match sampling {
            SamplingSpec::Default => return,
            SamplingSpec::Greedy => SamplingMethod::Greedy,
            SamplingSpec::Stochastic => SamplingMethod::Stochastic {
                temperature: Some(0.7),
                top_k: Some(32),
                top_p: Some(0.9),
                min_p: Some(0.05),
                repetition_penalty: None,
                suffix_repetition_length: None,
            },
            SamplingSpec::RepetitionPenalty => SamplingMethod::Stochastic {
                temperature: Some(0.7),
                top_k: None,
                top_p: None,
                min_p: None,
                repetition_penalty: Some(1.1),
                suffix_repetition_length: Some(64),
            },
        };
    }

    pub fn run() {
        let model_path = std::env::args().nth(1).map(PathBuf::from).expect("usage: pso_creation_sweep <model-path>");
        println!("model: {}", model_path.display());

        load_metrics::enable();

        let engine = Engine::<Metal>::new().expect("failed to create engine");

        load_metrics::reset();
        let load_start = Instant::now();
        let model = engine.load_language_model(&model_path).expect("failed to load model");
        let load_time = load_start.elapsed();
        let load_creations = load_metrics::creations();
        println!(
            "load: {:?}, {} pipeline states created (eager), {:?} in pipeline creation",
            load_time,
            load_creations.len(),
            pipeline_time(),
        );

        let specs = [
            RunSpec {
                prefix: 1,
                suffix: 2,
                sampling: SamplingSpec::Default,
            },
            RunSpec {
                prefix: 8,
                suffix: 2,
                sampling: SamplingSpec::Default,
            },
            RunSpec {
                prefix: 9,
                suffix: 2,
                sampling: SamplingSpec::Default,
            },
            RunSpec {
                prefix: 31,
                suffix: 1,
                sampling: SamplingSpec::Default,
            },
            RunSpec {
                prefix: 32,
                suffix: 1,
                sampling: SamplingSpec::Default,
            },
            RunSpec {
                prefix: 63,
                suffix: 1,
                sampling: SamplingSpec::Default,
            },
            RunSpec {
                prefix: 64,
                suffix: 1,
                sampling: SamplingSpec::Default,
            },
            RunSpec {
                prefix: 65,
                suffix: 1,
                sampling: SamplingSpec::Default,
            },
            RunSpec {
                prefix: 1040,
                suffix: 3,
                sampling: SamplingSpec::Default,
            },
            RunSpec {
                prefix: 4,
                suffix: 2,
                sampling: SamplingSpec::Greedy,
            },
            RunSpec {
                prefix: 4,
                suffix: 2,
                sampling: SamplingSpec::Stochastic,
            },
            RunSpec {
                prefix: 4,
                suffix: 2,
                sampling: SamplingSpec::RepetitionPenalty,
            },
        ];

        let mut cumulative: BTreeSet<Box<str>> = load_creations.iter().cloned().collect();
        for spec in &specs {
            let mut state = model.create_empty_state(None, 42).expect("failed to create state");
            let input: Vec<u64> = (0..spec.prefix).map(|i| 100 + (i % 1000) as u64).collect();
            let mut options = model.default_stream_options();
            apply_sampling(&mut options, &spec.sampling);

            load_metrics::reset();
            let start = Instant::now();
            let mut produced = 0;
            {
                let stream = model.stream(&input, &mut state, options).expect("failed to create stream");
                for token in stream.take(spec.suffix) {
                    token.expect("stream error");
                    produced += 1;
                }
            }
            let elapsed = start.elapsed();

            let creations = load_metrics::creations();
            let new: Vec<Box<str>> = creations.iter().filter(|key| !cumulative.contains(*key)).cloned().collect();
            cumulative.extend(creations.iter().cloned());

            println!();
            println!(
                "prefix={:<5} suffix={:<2} sampling={:<24} produced={:<2} wall={:>8.2?} pipeline={:>8.2?} new_variants={}",
                spec.prefix,
                spec.suffix,
                spec.sampling.label(),
                produced,
                elapsed,
                pipeline_time(),
                new.len(),
            );
            for key in &new {
                println!("    + {key}");
            }
        }

        println!();
        println!(
            "total unique pipeline state variants (load + runs): {} ({} eager at load, {} lazy)",
            cumulative.len(),
            load_creations.len(),
            cumulative.len() - load_creations.len(),
        );
    }
}

fn main() {
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    measurement::run();
    #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
    eprintln!("the pso_creation_sweep example requires the metal backend on an Apple platform");
}
