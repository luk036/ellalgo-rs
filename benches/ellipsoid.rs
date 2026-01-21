use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim};
use ellalgo_rs::ell::Ell;
use ndarray::prelude::*;

type Arr = Array1<f64>;

#[derive(Debug, Default)]
pub struct BenchOracle;

impl OracleOptim<Arr> for BenchOracle {
    type CutChoice = f64;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
        let x_val = xc[0];
        let y_val = xc[1];
        let f0 = x_val + y_val;
        let f1 = f0 - 3.0;
        if f1 > 0.0 {
            return ((array![1.0, 1.0], f1), false);
        }
        let f2 = -x_val + y_val + 1.0;
        if f2 > 0.0 {
            return ((array![-1.0, 1.0], f2), false);
        }
        let f3 = *gamma - f0;
        if f3 > 0.0 {
            return ((array![-1.0, -1.0], f3), false);
        }
        *gamma = f0;
        ((array![-1.0, -1.0], 0.0), true)
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
                    let mut ellip = Ell::new_with_scalar(10.0, Array1::zeros(dim));
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
