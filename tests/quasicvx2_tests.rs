//! Port of Python test_quasicvx2.py — Quasiconvex optimization without round robin.
//!
//! Constraints:
//!   1. exp(x) <= y   → constraint violation: exp(x) - y > 0
//!   2. y > 0          → constraint violation: -y > 0 (when y <= 0)
//!   3. x > 0          → constraint violation: -x > 0 (when x <= 0)
//!
//! Objective: maximize sqrt(x) / y (minimize -sqrt(x) + gamma * y)

use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim, SingleCut};
use ellalgo_rs::ell::Ell;

#[derive(Debug)]
struct MyQuasicvxOracle {
    idx: i32,
}

impl Default for MyQuasicvxOracle {
    fn default() -> Self {
        MyQuasicvxOracle { idx: -1 }
    }
}

impl OracleOptim<Arr> for MyQuasicvxOracle {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
        let x = xc[0];
        let y = xc[1];

        let num_constraints = 3;
        for _ in 0..num_constraints {
            self.idx += 1;
            if self.idx == num_constraints {
                self.idx = 0;
            }

            let (grad, fj) = match self.idx {
                0 => {
                    // constraint 1: exp(x) <= y
                    let tmp = x.exp();
                    (Arr::from(vec![tmp, -1.0]), tmp - y)
                }
                1 => {
                    // constraint 2: y > 0
                    (Arr::from(vec![0.0, -1.0]), -y)
                }
                2 => {
                    // constraint 3: x > 0
                    (Arr::from(vec![-1.0, 0.0]), -x)
                }
                _ => unreachable!(),
            };

            if fj > 0.0 {
                return ((grad, SingleCut(fj)), false);
            }
        }

        // objective: maximize sqrt(x) / y
        let tmp2 = x.sqrt();
        let fj = -tmp2 + *gamma * y;
        if fj > 0.0 {
            // infeasible
            return ((Arr::from(vec![-0.5 / tmp2, *gamma]), SingleCut(fj)), false);
        }

        *gamma = tmp2 / y;
        ((Arr::from(vec![-0.5 / tmp2, *gamma]), SingleCut(0.0)), true)
    }
}

#[test]
fn test_case_feasible() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![1.0, 1.0]));
    let mut oracle = MyQuasicvxOracle::default();
    let mut gamma = 0.0;
    let options = Options::default();
    let (x_opt, _) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);
    assert!(x_opt.is_some());
}

#[test]
fn test_case_infeasible1() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![100.0, 100.0]));
    let mut oracle = MyQuasicvxOracle::default();
    let mut gamma = 0.0;
    let options = Options::default();
    let (x_opt, _) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);
    assert!(x_opt.is_none());
}

#[test]
fn test_case_infeasible2() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![1.0, 1.0]));
    let mut oracle = MyQuasicvxOracle::default();
    let options = Options::default();
    let (x_opt, _) = cutting_plane_optim(&mut oracle, &mut ellip, &mut 100.0, &options);
    assert!(x_opt.is_none());
}
