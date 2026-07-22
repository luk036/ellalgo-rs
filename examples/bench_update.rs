use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::SingleCut;
use ellalgo_rs::ell::Ell;
use ellalgo_rs::SearchSpace;
use std::hint::black_box;
use std::time::Instant;

fn bench(c: &mut Ell, dim: usize, n_iter: u64) -> f64 {
    let grad = Arr::from_fn(dim, |i| (i + 1) as f64 / dim as f64);
    let cut = (grad, SingleCut(0.0));
    let start = Instant::now();
    for _ in 0..n_iter {
        black_box(c.update_central_cut(&cut));
    }
    start.elapsed().as_nanos() as f64 / n_iter as f64
}

fn main() {
    println!("=== Rust (ellalgo-rs) - Single Update Benchmark ===\n");
    println!("  {:<10} {:>12}", "Dim", "ns/update");

    for &dim in &[2, 16, 32, 64, 128] {
        let n_iter = if dim <= 16 { 100000 } else { 50000 };
        let mut ellip = Ell::new_with_scalar(100.0, Arr::new(dim));
        let ns = bench(&mut ellip, dim, n_iter);
        println!("  {:<10} {:>10.1}", dim, ns);
    }
}
