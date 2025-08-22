use crate::benchmarks::config;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use llm::erf::erf;

fn bench(c: &mut Criterion) {
    let mut benchmark = c.benchmark_group("erf-comparaison");
    config::set_default_benchmark_configs(&mut benchmark);

    for value in [
        f32::MIN,
        f32::MAX,
        0.0,
        -1.0,
        1.0,
        24.23,
        89.69,
        420.420,
        0.000000009,
        -3333333.423423,
    ] {
        benchmark.bench_with_input(BenchmarkId::new("rust", value), &value, |b, value| {
            b.iter(|| {
                erf(black_box(*value));
            });
        });
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
