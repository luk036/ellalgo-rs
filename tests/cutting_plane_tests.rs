//! Port of Python test_cutting_plane.py — Comprehensive cutting plane tests.
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{
    bsearch, cutting_plane_feas, cutting_plane_optim, cutting_plane_optim_q, BSearchAdaptor,
    Options, OracleFeas, OracleOptim, OracleOptimQ, SingleCut,
};
use ellalgo_rs::ell::Ell;

// ---------------------------------------------------------------------------
// Simple feasibility oracle: x + y <= 3.0
// ---------------------------------------------------------------------------
#[derive(Debug, Default)]
struct MyOracleFeas;

impl OracleFeas<Arr> for MyOracleFeas {
    type CutChoice = SingleCut;

    fn assess_feas(&mut self, xc: &Arr) -> Option<(Arr, SingleCut)> {
        let x = xc[0];
        let y = xc[1];
        let fj = x + y - 3.0;
        if fj > 0.0 {
            Some((Arr::from(vec![1.0, 1.0]), SingleCut(fj)))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Infeasibility oracle: always returns a cut
// ---------------------------------------------------------------------------
#[derive(Debug, Default)]
struct MyOracleInfeas;

impl OracleFeas<Arr> for MyOracleInfeas {
    type CutChoice = SingleCut;

    fn assess_feas(&mut self, _xc: &Arr) -> Option<(Arr, SingleCut)> {
        Some((Arr::from(vec![1.0, 1.0]), SingleCut(1.0)))
    }
}

// ---------------------------------------------------------------------------
// Optimization oracle: minimize x + y subject to x <= 1, y <= 1
// ---------------------------------------------------------------------------
#[derive(Debug, Default)]
struct MyOracleOptim;

impl OracleOptim<Arr> for MyOracleOptim {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
        let x = xc[0];
        let y = xc[1];
        let f0 = x + y;

        let f1 = x - 1.0;
        if f1 > 0.0 {
            return ((Arr::from(vec![1.0, 0.0]), SingleCut(f1)), false);
        }
        let f2 = y - 1.0;
        if f2 > 0.0 {
            return ((Arr::from(vec![0.0, 1.0]), SingleCut(f2)), false);
        }
        let f3 = f0 - *gamma;
        if f3 < 0.0 {
            return ((Arr::from(vec![-1.0, -1.0]), SingleCut(-f3)), false);
        }
        ((Arr::from(vec![-1.0, -1.0]), SingleCut(0.0)), true)
    }
}

// ---------------------------------------------------------------------------
// Binary search assess_bs oracle
// ---------------------------------------------------------------------------
#[derive(Debug, Default)]
struct MyOracleBS2;

impl ellalgo_rs::cutting_plane::OracleBS for MyOracleBS2 {
    fn assess_bs(&mut self, gamma: f64) -> bool {
        gamma > 0.0
    }
}

// ---------------------------------------------------------------------------
// Oracle that always returns a cut (for max_iters tests)
// ---------------------------------------------------------------------------
#[derive(Debug, Default)]
struct MyOracleOptim2;

impl OracleOptim<Arr> for MyOracleOptim2 {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, _xc: &Arr, _gamma: &mut f64) -> ((Arr, SingleCut), bool) {
        ((Arr::from(vec![1.0, 1.0]), SingleCut(1.0)), false)
    }
}

// ---------------------------------------------------------------------------
// Optim Q oracle that always returns a cut (for NoEffect path)
// ---------------------------------------------------------------------------
#[derive(Debug, Default)]
struct MyOracleOptimQ2;

impl OracleOptimQ<Arr> for MyOracleOptimQ2 {
    type CutChoice = SingleCut;

    fn assess_optim_q(
        &mut self,
        xc: &Arr,
        _gamma: &mut f64,
        _retry: bool,
    ) -> ((Arr, SingleCut), bool, Arr, bool) {
        (
            (Arr::from(vec![1.0, 1.0]), SingleCut(1.0)),
            false,
            xc.clone(),
            true,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn test_cutting_plane_feas() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
    let mut omega = MyOracleFeas;
    let options = Options::new(200, 1e-20);
    let (x_best, num_iters) = cutting_plane_feas(&mut omega, &mut ellip, &options);
    assert!(x_best.is_some());
    assert_eq!(num_iters, 0);
}

#[test]
fn test_cutting_plane_feas_no_soln() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
    let mut omega = MyOracleInfeas;
    let options = Options::new(200, 1e-20);
    let (x_best, num_iters) = cutting_plane_feas(&mut omega, &mut ellip, &options);
    assert!(x_best.is_none());
    assert_eq!(num_iters, 2);
}

#[test]
fn test_cutting_plane_optim() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
    let mut omega = MyOracleOptim;
    let mut gamma = 0.0;
    let options = Options::new(200, 1e-20);
    let (x_best, _num_iters) = cutting_plane_optim(&mut omega, &mut ellip, &mut gamma, &options);
    assert!(x_best.is_some());
}

#[test]
fn test_cutting_plane_optim_no_soln() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
    let mut omega = MyOracleOptim;
    let options = Options::new(4, 1e-20);
    let (x_best, num_iters) = cutting_plane_optim(&mut omega, &mut ellip, &mut 100.0, &options);
    assert!(x_best.is_none());
    assert_eq!(num_iters, 0);
}

#[test]
fn test_cutting_plane_optim_max_iters() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
    let mut omega = MyOracleOptim2;
    let options = Options::new(5, 1e-20);
    let (x_best, num_iters) = cutting_plane_optim(&mut omega, &mut ellip, &mut 0.0, &options);
    assert!(x_best.is_none());
    assert_eq!(num_iters, 2);
}

#[test]
fn test_cutting_plane_feas_max_iters() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
    let mut omega = MyOracleInfeas;
    let options = Options::new(5, 1e-20);
    let (x_best, num_iters) = cutting_plane_feas(&mut omega, &mut ellip, &options);
    assert!(x_best.is_none());
    assert_eq!(num_iters, 2);
}

// ---------------------------------------------------------------------------
// Optim Q tests
// ---------------------------------------------------------------------------

/// Standard optim-q oracle: minimize x + y subject to x <= 1, y <= 1
#[derive(Debug, Default)]
struct MyOracleOptimQ3;

impl OracleOptimQ<Arr> for MyOracleOptimQ3 {
    type CutChoice = SingleCut;

    fn assess_optim_q(
        &mut self,
        xc: &Arr,
        gamma: &mut f64,
        retry: bool,
    ) -> ((Arr, SingleCut), bool, Arr, bool) {
        let x = xc[0];
        let y = xc[1];
        let f0 = x + y;

        let f1 = x - 1.0;
        if f1 > 0.0 {
            return (
                (Arr::from(vec![1.0, 0.0]), SingleCut(f1)),
                false,
                xc.clone(),
                true,
            );
        }
        let f2 = y - 1.0;
        if f2 > 0.0 {
            return (
                (Arr::from(vec![0.0, 1.0]), SingleCut(f2)),
                false,
                xc.clone(),
                true,
            );
        }
        let f3 = f0 - *gamma;
        if f3 < 0.0 {
            return (
                (Arr::from(vec![-1.0, -1.0]), SingleCut(-f3)),
                false,
                xc.clone(),
                true,
            );
        }

        let x_q = Arr::from(vec![x.round(), y.round()]);
        let f1q = x_q[0] - 1.0;
        if f1q > 0.0 {
            return (
                (Arr::from(vec![1.0, 0.0]), SingleCut(f1q)),
                false,
                x_q,
                !retry,
            );
        }
        let f2q = x_q[1] - 1.0;
        if f2q > 0.0 {
            return (
                (Arr::from(vec![0.0, 1.0]), SingleCut(f2q)),
                false,
                x_q,
                !retry,
            );
        }
        let f3q = x_q[0] + x_q[1] - *gamma;
        if f3q < 0.0 {
            return (
                (Arr::from(vec![-1.0, -1.0]), SingleCut(-f3q)),
                false,
                x_q,
                !retry,
            );
        }
        *gamma = x_q[0] + x_q[1];
        (
            (Arr::from(vec![-1.0, -1.0]), SingleCut(0.0)),
            true,
            x_q,
            !retry,
        )
    }
}

#[test]
fn test_cutting_plane_optim_q() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
    let mut omega = MyOracleOptimQ3;
    let mut gamma = 0.0;
    let options = Options::new(200, 1e-20);
    let (x_best, _num_iters) = cutting_plane_optim_q(&mut omega, &mut ellip, &mut gamma, &options);
    assert!(x_best.is_some());
}

#[test]
fn test_cutting_plane_optim_q_no_soln() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
    let mut omega = MyOracleOptimQ3;
    let options = Options::new(20, 1e-20);
    let (x_best, num_iters) = cutting_plane_optim_q(&mut omega, &mut ellip, &mut 100.0, &options);
    assert!(x_best.is_none());
    assert_eq!(num_iters, 0);
}

#[test]
fn test_cutting_plane_optim_q_no_effect() {
    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
    let mut omega = MyOracleOptimQ2;
    let options = Options::new(5, 1e-20);
    let (x_best, num_iters) = cutting_plane_optim_q(&mut omega, &mut ellip, &mut 0.0, &options);
    assert!(x_best.is_none());
    assert_eq!(num_iters, 2);
}

// ---------------------------------------------------------------------------
// Binary search tests
// ---------------------------------------------------------------------------

#[test]
fn test_bsearch() {
    let mut omega = MyOracleBS2;
    let mut intrvl = (-100.0, 100.0);
    let options = Options::new(2000, 1e-7);
    let (feasible, num_iters) = bsearch(&mut omega, &mut intrvl, &options);
    assert!(feasible);
    assert_eq!(num_iters, 30);
}

#[test]
fn test_bsearch_no_soln() {
    let mut omega = MyOracleBS2;
    let mut intrvl = (-100.0, -50.0);
    let options = Options::new(20, 1e-20);
    let (feasible, num_iters) = bsearch(&mut omega, &mut intrvl, &options);
    assert!(!feasible);
    assert_eq!(num_iters, 20);
}

// ---------------------------------------------------------------------------
// BSearchAdaptor tests
// ---------------------------------------------------------------------------

/// Feasibility oracle: feasible when x + y <= 3.0 (used with BSearchAdaptor)
#[derive(Debug, Default)]
struct MyOracleFeas2;

impl OracleFeas<Arr> for MyOracleFeas2 {
    type CutChoice = SingleCut;

    fn assess_feas(&mut self, xc: &Arr) -> Option<(Arr, SingleCut)> {
        let x = xc[0];
        let y = xc[1];
        let fj = x + y - 3.0;
        if fj > 0.0 {
            Some((Arr::from(vec![1.0, 1.0]), SingleCut(fj)))
        } else {
            None
        }
    }
}

impl ellalgo_rs::cutting_plane::OracleBS for MyOracleFeas2 {
    fn assess_bs(&mut self, gamma: f64) -> bool {
        gamma > 0.0
    }
}

#[test]
fn test_bsearch_adaptor() {
    let ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
    let omega = MyOracleFeas2;
    let options = Options::default();
    let mut adaptor = BSearchAdaptor::new(omega, ellip, options);
    let mut intrvl = (-100.0, 100.0);
    let bs_options = Options::new(2000, 1e-8);
    let (feasible, _num_iters) = bsearch(&mut adaptor, &mut intrvl, &bs_options);
    assert!(feasible);
}

#[test]
fn test_bsearch_adaptor_x_best() {
    use ellalgo_rs::cutting_plane::SearchSpace;
    let ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
    let omega = MyOracleFeas2;
    let options = Options::default();
    let adaptor = BSearchAdaptor::new(omega, ellip, options);
    let x_best = adaptor.space.xc();
    assert_eq!(x_best, Arr::from(vec![0.0, 0.0]));
}
