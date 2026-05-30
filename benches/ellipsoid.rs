use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim, SingleCut};
use ellalgo_rs::ell::Ell;

#[derive(Debug, Default)]
pub struct BenchOracle;

impl OracleOptim<Arr> for BenchOracle {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
        let gradient = 2.0 * xc;
        let f = xc.dot(xc);
        if f < *gamma {
            *gamma = f;
            ((gradient, SingleCut(f)), true)
        } else {
            ((gradient, SingleCut(f)), false)
        }
    }
}

fn bench_ellipsoid_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("ellipsoid");

    for dimension in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("optimization", dimension),
            dimension,
            |b, &dim| {
                b.iter(|| {
                    let mut ellip = Ell::new_with_scalar(10.0, Arr::new(dim));
                    let mut oracle = BenchOracle;
                    let mut gamma = f64::NEG_INFINITY;
                    let options = Options::default();

                    cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_ellipsoid_update);
criterion_main!(benches);
