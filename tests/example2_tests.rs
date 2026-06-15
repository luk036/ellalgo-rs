//! Port of Python test_example2.py — Feasibility example with round robin.
//!
//! Constraints:
//!   1. x + y <= 3.0
//!   2. -x + y + 1.0 <= 0

use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_feas, Options, OracleFeas, SingleCut};
use ellalgo_rs::ell::Ell;

#[derive(Debug)]
struct MyOracle2 {
    idx: i32,
}

impl Default for MyOracle2 {
    fn default() -> Self {
        MyOracle2 { idx: -1 }
    }
}

impl OracleFeas<Arr> for MyOracle2 {
    type CutChoice = SingleCut;

    fn assess_feas(&mut self, xc: &Arr) -> Option<(Arr, SingleCut)> {
        let x = xc[0];
        let y = xc[1];

        let num_constraints = 2;
        for _ in 0..num_constraints {
            self.idx += 1;
            if self.idx == num_constraints {
                self.idx = 0;
            }

            let (grad, fj) = match self.idx {
                0 => (Arr::from(vec![1.0, 1.0]), x + y - 3.0),
                1 => (Arr::from(vec![-1.0, 1.0]), -x + y + 1.0),
                _ => unreachable!(),
            };

            if fj > 0.0 {
                return Some((grad, SingleCut(fj)));
            }
        }
        None
    }
}

#[test]
fn test_case_feasible() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
    let mut omega = MyOracle2::default();
    let options = Options::default();
    let (x_feas, num_iters) = cutting_plane_feas(&mut omega, &mut ellip, &options);
    assert!(x_feas.is_some());
    assert_eq!(num_iters, 1);
}

#[test]
fn test_case_infeasible() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![100.0, 100.0]));
    let mut omega = MyOracle2::default();
    let options = Options::default();
    let (x_feas, num_iters) = cutting_plane_feas(&mut omega, &mut ellip, &options);
    assert!(x_feas.is_none());
    assert_eq!(num_iters, 0);
}
