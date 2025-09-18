use llm::{
    gemma3::{Gemma3, InitParams},
    modules::Module,
};
use std::time::Instant;
use sysinfo::{System, SystemExt, ProcessExt};

fn benchmark_rust(
    model: &mut Gemma3,
    tokens: &[usize],
    warmups: usize,
    iterations: usize
) -> std::collections::HashMap<String, String> {
    // Clean up and get initial memory
    let mut sys = System::new_all();
    sys.refresh_all();
    let pid = sysinfo::get_current_pid().expect("Failed to get current PID");
    let initial_mem = sys.process(pid)
        .map(|p| p.memory() as f64 / 1024.0) // KB to MB
        .unwrap_or(0.0);

    // Warm-up runs
    println!("Running {} warm-up inferences...", warmups);
    for _ in 0..warmups {
        let _ = model.forward(tokens);
    }

    // Benchmark time
    println!("Running {} benchmark inferences...", iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.forward(tokens);
    }
    let time_taken = start.elapsed();
    let mean_time_sec = time_taken.as_secs_f64() / iterations as f64;

    // Measure peak memory during benchmark
    let mut peak_mem = initial_mem;
    for _ in 0..iterations {
        let _ = model.forward(tokens);
        sys.refresh_process(pid);
        if let Some(process) = sys.process(pid) {
            let current_mem = process.memory() as f64 / 1024.0;
            peak_mem = peak_mem.max(current_mem);
        }
    }
    let peak_mem_delta = peak_mem - initial_mem;

    let mut results = std::collections::HashMap::new();
    results.insert("warmups".to_string(), warmups.to_string());
    results.insert("iterations".to_string(), iterations.to_string());
    results.insert("mean_time_sec".to_string(), format!("{:.6}", mean_time_sec));
    results.insert("peak_memory_mb".to_string(), format!("{:.2}", peak_mem_delta));
    results.insert("input_tokens".to_string(), format!("{:?}", tokens));

    results
}

fn main() {
    let mut gemma: Gemma3 = InitParams::gemma3_270m().into();
    let tokens = [1, 2, 3];

    let results = benchmark_rust(&mut gemma, &tokens, 10, 100);

    println!("\nBenchmark Results (Rust):");
    for (key, value) in results {
        println!("{}: {}", key, value);
    }
}
