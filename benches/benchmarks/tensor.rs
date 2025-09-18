use crate::benchmarks::config;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench(c: &mut Criterion) {
    let mut benchmark = c.benchmark_group("tensors");
    config::set_default_benchmark_configs(&mut benchmark);

    let seed = Some(23);

    for (rows, cols) in [
        (1, 1),
        (10, 10),
        (100, 100),
        (1000, 1000),
        (2048, 2048),
        (4096, 4096),
        (10000, 10000),
    ] {
        let tensor = llm::tensor::Tensor::rand(&[rows, cols], seed);
        let size = format!("rows({rows})xcols({cols})");

        benchmark.bench_with_input(
            BenchmarkId::new("matmul", &size),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let _ = tensor.matmul(black_box(tensor)).unwrap();
                });
            },
        );


        // benchmark.bench_with_input(
        //     BenchmarkId::new("regular_rqsrt", &size),
        //     &tensor,
        //     |b, tensor| {
        //         b.iter(|| {
        //             tensor.rsqrt_slow();
        //         });
        //     },
        // );

        // benchmark.bench_with_input(
        //     BenchmarkId::new("qwake_rqsrt", &size),
        //     &tensor,
        //     |b, tensor| {
        //         b.iter(|| {
        //             tensor.rsqrt();
        //         });
        //     },
        // );

        // benchmark.bench_with_input(BenchmarkId::new("smid_rqsrt", &size), &tensor, |b, tensor| {
        //     b.iter(|| {
        //         tensor.rsqrt_simd();
        //     });
        // });
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
