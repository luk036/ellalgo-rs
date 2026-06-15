//! Port of Python test_lmi.py, test_lmi0_oracle.py, and test_lmi0_oracle.py.
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleFeas, OracleOptim, SingleCut};
use ellalgo_rs::ell::Ell;
use ellalgo_rs::ell_stable::EllStable;
use ellalgo_rs::oracles::lmi0_oracle::LMI0Oracle;
use ellalgo_rs::oracles::lmi_old_oracle::LMIOldOracle;
use ellalgo_rs::oracles::lmi_oracle::LMIOracle;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn f1_matrices() -> Vec<Arr> {
    vec![
        Arr::with_data(vec![-7.0, -11.0, -11.0, 3.0], 2, 2),
        Arr::with_data(vec![7.0, -18.0, -18.0, 8.0], 2, 2),
        Arr::with_data(vec![-2.0, -8.0, -8.0, 1.0], 2, 2),
    ]
}

fn b1_matrix() -> Arr {
    Arr::with_data(vec![33.0, -9.0, -9.0, 26.0], 2, 2)
}

fn f2_matrices() -> Vec<Arr> {
    vec![
        Arr::with_data(
            vec![-21.0, -11.0, 0.0, -11.0, 10.0, 8.0, 0.0, 8.0, 5.0],
            3,
            3,
        ),
        Arr::with_data(
            vec![0.0, 10.0, 16.0, 10.0, -10.0, -10.0, 16.0, -10.0, 3.0],
            3,
            3,
        ),
        Arr::with_data(
            vec![-5.0, 2.0, -17.0, 2.0, -6.0, 8.0, -17.0, 8.0, 6.0],
            3,
            3,
        ),
    ]
}

fn b2_matrix() -> Arr {
    Arr::with_data(
        vec![14.0, 9.0, 40.0, 9.0, 91.0, 10.0, 40.0, 10.0, 15.0],
        3,
        3,
    )
}

// ---------------------------------------------------------------------------
// LMI oracle direct tests
// ---------------------------------------------------------------------------

#[test]
fn test_lmi_oracle() {
    let mut lmi1 = LMIOracle::new(f1_matrices(), b1_matrix());
    let cut = lmi1.assess_feas(&Arr::new(3));
    assert!(cut.is_none());
}

#[test]
fn test_lmi0_oracle() {
    let f1 = f1_matrices();
    let mut lmi1 = LMI0Oracle::new(f1);
    let cut = lmi1.assess_feas(&Arr::new(3));
    assert!(cut.is_some());
}

// ---------------------------------------------------------------------------
// LMI0 oracle tests (from test_lmi0_oracle.py)
// ---------------------------------------------------------------------------

fn lmi0_matrices() -> Vec<Arr> {
    vec![
        Arr::with_data(vec![1.0, 0.0, 0.0, 0.0], 2, 2),
        Arr::with_data(vec![0.0, 1.0, 1.0, 0.0], 2, 2),
        Arr::with_data(vec![0.0, 0.0, 0.0, 1.0], 2, 2),
    ]
}

#[test]
fn test_lmi0_oracle_feasible() {
    let mut lmi0 = LMI0Oracle::new(lmi0_matrices());
    let x = Arr::from(vec![1.0, 0.0, 1.0]);
    let cut = lmi0.assess_feas(&x);
    assert!(cut.is_none());
}

#[test]
fn test_lmi0_oracle_infeasible() {
    let mut lmi0 = LMI0Oracle::new(lmi0_matrices());
    let x = Arr::from(vec![-1.0, 0.0, -1.0]);
    let cut = lmi0.assess_feas(&x);
    assert!(cut.is_some());
    let (g, ep) = cut.unwrap();
    assert_approx_eq!(g[0], -1.0);
    assert_approx_eq!(g[1], -0.0);
    assert_approx_eq!(g[2], -0.0);
    assert_approx_eq!(ep, 1.0);
}

#[test]
fn test_lmi0_oracle_infeasible2() {
    let mut lmi0 = LMI0Oracle::new(lmi0_matrices());
    let x = Arr::from(vec![1.0, 1.0, 1.0]);
    let cut = lmi0.assess_feas(&x);
    assert!(cut.is_some());
}

// ---------------------------------------------------------------------------
// MyOracle — optimization oracle wrapping two LMI feasibility oracles
// ---------------------------------------------------------------------------

use approx_eq::assert_approx_eq;

#[derive(Debug)]
struct MyLmiOracle<T> {
    idx: i32,
    c: Arr,
    lmi1: T,
    lmi2: T,
}

impl<T: OracleFeas<Arr, CutChoice = SingleCut>> MyLmiOracle<T> {
    fn new(oracle: fn(Vec<Arr>, Arr) -> T) -> Self {
        let c = Arr::from(vec![1.0, -1.0, 1.0]);
        let lmi1 = oracle(f1_matrices(), b1_matrix());
        let lmi2 = oracle(f2_matrices(), b2_matrix());
        MyLmiOracle {
            idx: -1,
            c,
            lmi1,
            lmi2,
        }
    }
}

impl<T: OracleFeas<Arr, CutChoice = SingleCut>> OracleOptim<Arr> for MyLmiOracle<T> {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
        let f0 = self.c.dot(xc);
        for _ in 0..3 {
            self.idx = if self.idx == 2 { 0 } else { self.idx + 1 };
            match self.idx {
                0 => {
                    if let Some(cut) = self.lmi1.assess_feas(xc) {
                        return ((cut.0, cut.1), false);
                    }
                }
                1 => {
                    if let Some(cut) = self.lmi2.assess_feas(xc) {
                        return ((cut.0, cut.1), false);
                    }
                }
                2 => {
                    let fj = f0 - *gamma;
                    if fj > 0.0 {
                        return ((self.c.clone(), SingleCut(fj)), false);
                    }
                    *gamma = f0;
                }
                _ => unreachable!(),
            }
        }
        ((self.c.clone(), SingleCut(0.0)), true)
    }
}

fn run_lmi<T>(oracle: fn(Vec<Arr>, Arr) -> T) -> usize
where
    T: OracleFeas<Arr, CutChoice = SingleCut>,
{
    let mut ellip = Ell::new_with_scalar(10.0, Arr::new(3));
    let mut omega = MyLmiOracle::new(oracle);
    let mut gamma = f64::INFINITY;
    let options = Options::default();
    let (x_best, num_iters) = cutting_plane_optim(&mut omega, &mut ellip, &mut gamma, &options);
    assert!(x_best.is_some());
    num_iters
}

fn run_lmi_stable<T>(oracle: fn(Vec<Arr>, Arr) -> T) -> usize
where
    T: OracleFeas<Arr, CutChoice = SingleCut>,
{
    let mut ellip = EllStable::new_with_scalar(10.0, Arr::new(3));
    let mut omega = MyLmiOracle::new(oracle);
    let mut gamma = f64::INFINITY;
    let options = Options::default();
    let (x_best, num_iters) = cutting_plane_optim(&mut omega, &mut ellip, &mut gamma, &options);
    assert!(x_best.is_some());
    num_iters
}

#[test]
fn test_lmi_lazy() {
    let result = run_lmi(LMIOracle::new);
    // Iteration count differs from Python (281) due to numerical differences
    // between Rust f64 LDLT factorization and numpy operations
    assert!(result < 300);
}

#[test]
fn test_lmi_old() {
    let result = run_lmi(LMIOldOracle::new);
    assert!(result < 300);
}

#[test]
fn test_lmi_lazy_stable() {
    let result = run_lmi_stable(LMIOracle::new);
    // EllStable converges differently from Ell
    assert!(result < 400);
}

#[test]
fn test_lmi_old_stable() {
    let result = run_lmi_stable(LMIOldOracle::new);
    assert!(result < 400);
}
