use criterion::criterion_main;

mod benchmarks;

criterion_main! {
    benchmarks::modules::benches,
    benchmarks::tensor::benches,
    benchmarks::gemma::benches,
}
