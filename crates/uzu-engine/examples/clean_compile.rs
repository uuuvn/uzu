//! Measures the cost of Metal pipeline state creation the way the engine
//! actually performs it, so the effect of `UZU_CLEAN_COMPILE` (bypass of
//! Metal's on-disk shader cache) can be observed directly.
//!
//! Run this example twice: the second run is much faster because Metal serves
//! pipeline states from its on-disk shader cache. Then run it with
//! `UZU_CLEAN_COMPILE=1`: every run behaves like the first-ever run, without
//! the user's real shader cache being touched.
//!
//! ```sh
//! cargo run --release --example clean_compile
//! cargo run --release --example clean_compile
//! UZU_CLEAN_COMPILE=1 cargo run --release --example clean_compile
//! ```

#[cfg(all(feature = "metal", target_vendor = "apple"))]
mod measurement {
    use std::time::{Duration, Instant};

    use uzu_engine::{
        backends::{
            common::{Backend, Context, Kernels, kernel::TensorAddScaleKernel},
            metal::{Metal, MetalContext},
        },
        data_type::DataType,
    };

    pub fn run() {
        if std::env::var("UZU_CLEAN_COMPILE").is_ok() {
            println!("UZU_CLEAN_COMPILE is set: Metal's on-disk shader cache is bypassed, expect first-run timings");
        } else {
            println!("UZU_CLEAN_COMPILE is not set: Metal's on-disk shader cache is active");
        }

        let start = Instant::now();
        let context = MetalContext::new().expect("failed to create Metal context");
        println!("context creation: {:?}", start.elapsed());

        let mut total = Duration::ZERO;
        let mut count = 0u32;
        for data_type in [DataType::F32, DataType::F16, DataType::BF16] {
            for in_place in [false, true] {
                let start = Instant::now();
                // Each kernel instantiation specializes a function and creates a
                // pipeline state for it, going through the same code path the
                // engine uses for all of its kernels.
                let _kernel =
                    <<<Metal as Backend>::Kernels as Kernels>::TensorAddScaleKernel as TensorAddScaleKernel>::new(
                        &context, data_type, in_place,
                    )
                    .expect("failed to create TensorAddScaleKernel");
                let elapsed = start.elapsed();
                total += elapsed;
                count += 1;
                println!("TensorAddScaleKernel {data_type:?} in_place={in_place}: {elapsed:?}");
            }
        }

        println!("created {count} pipeline states in {total:?} (avg {:?})", total / count);
    }
}

fn main() {
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    measurement::run();
    #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
    eprintln!("the clean_compile example requires the metal backend on an Apple platform");
}
