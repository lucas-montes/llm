use crate::benchmarks::config;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use llm::{
    gemma3::{Gemma3, InitParams},
    modules::Module,
};

fn bench(c: &mut Criterion) {
    let mut benchmark = c.benchmark_group("gemma");
    config::set_full_model_benchmark_configs(&mut benchmark);

    let mut gemma: Gemma3 = InitParams::gemma3_270m().into();

    for indeces in [
        vec![1, 2, 3],                               // 3 tokens - very short
        vec![1, 2, 640, 8],                          // 4 tokens - short phrase
        vec![15, 42, 100, 200, 500, 1000],           // 6 tokens - sentence
        vec![1, 15, 42, 100, 200, 500, 1000, 2000],  // 8 tokens - longer sentence
        vec![5, 10, 15, 20, 25, 30, 35, 40, 45, 50], // 10 tokens - paragraph start
        (1..=64).collect(),                          // 64 tokens - common batch size
        (1..=128).collect(),                         // 128 tokens - medium context
        (1..=256).collect(),                         // 256 tokens - long context
    ] {
        benchmark.bench_with_input(
            BenchmarkId::new("rust", indeces.len()),
            &indeces,
            |b, value| {
                b.iter(|| {
                    let _ = gemma.forward(black_box(value)).unwrap();
                });
            },
        );
    }

    benchmark.finish()
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = config::get_default_profiling_configs();
    targets = bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, bench);

criterion_main!(benches);
