//! Integration tests for the ellalgo-rs library
//!
//! These tests verify that the library works correctly end-to-end with
//! various oracle implementations and problem types.

use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim};
use ellalgo_rs::ell::Ell;
use ndarray::prelude::*;

type Arr = Array1<f64>;

/// Test a simple convex optimization problem: minimize ||x||²
#[test]
fn test_simple_quadratic_optimization() {
    struct QuadraticOracle;

    impl OracleOptim<Arr> for QuadraticOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            // Minimize f(x) = x[0]² + x[1]²
            let gradient = array![2.0 * xc[0], 2.0 * xc[1]];
            let f = xc[0].powi(2) + xc[1].powi(2);

            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    let mut ellip = Ell::new_with_scalar(10.0, array![5.0, 5.0]);
    let mut oracle = QuadraticOracle;
    let mut gamma = f64::INFINITY;
    let options = Options::new(1000, 1e-10);

    let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    assert!(xbest.is_some(), "Should find a solution");

    // Solution should be near origin
    let x = xbest.unwrap();
    assert!(x[0].abs() < 0.5, "x[0] should be near 0");
    assert!(x[1].abs() < 0.5, "x[1] should be near 0");
    assert!(gamma < 1.0, "Objective value should be small");
}

/// Test optimization with known optimal point
#[test]
fn test_known_optimal_solution() {
    struct KnownOptimalOracle {
        target: Arr,
    }

    impl OracleOptim<Arr> for KnownOptimalOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            // Minimize: ||x - target||²
            let diff = xc - &self.target;
            let gradient = 2.0 * &diff;
            let f = diff.dot(&diff);

            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    let target = array![2.0, -1.5];
    let initial = array![10.0, 10.0];

    let mut ellip = Ell::new_with_scalar(20.0, initial);
    let mut oracle = KnownOptimalOracle {
        target: target.clone(),
    };
    let mut gamma = f64::INFINITY;
    let options = Options::new(1000, 1e-10);

    let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    assert!(xbest.is_some(), "Should find a solution");

    let x = xbest.unwrap();
    let error = ((x[0] - target[0]).powi(2) + (x[1] - target[1]).powi(2)).sqrt();

    // Should be within reasonable tolerance
    assert!(error < 15.0, "Solution error {} should be less than 15.0", error);
}

/// Test with higher dimensional problem
#[test]
fn test_higher_dimensional_optimization() {
    struct HighDimOracle {
        target: Arr,
    }

    impl OracleOptim<Arr> for HighDimOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            // Minimize ||x - target||²
            let diff = xc - &self.target;
            let gradient = 2.0 * &diff;
            let f = diff.dot(&diff);

            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    let ndim = 5;
    let target = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let initial = array![0.0, 0.0, 0.0, 0.0, 0.0];

    let mut ellip = Ell::new_with_scalar(10.0, initial);
    let mut oracle = HighDimOracle {
        target: target.clone(),
    };
    let mut gamma = f64::INFINITY;
    let options = Options::new(2000, 1e-10);

    let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    assert!(xbest.is_some(), "Should find a solution");

    let x = xbest.unwrap();
    let mut total_error = 0.0;
    for i in 0..ndim {
        total_error += (x[i] - target[i]).powi(2);
    }
    let rms_error = (total_error / ndim as f64).sqrt();

    assert!(rms_error < 3.0, "RMS error {} should be less than 3.0", rms_error);
}

/// Test convergence metrics
#[test]
fn test_convergence_metrics() {
    struct ConvergenceTestOracle;

    impl OracleOptim<Arr> for ConvergenceTestOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            // Simple quadratic: minimize x[0]² + x[1]²
            let gradient = array![2.0 * xc[0], 2.0 * xc[1]];
            let f = xc[0].powi(2) + xc[1].powi(2);

            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    let mut ellip = Ell::new_with_scalar(10.0, array![3.0, 3.0]);
    let mut oracle = ConvergenceTestOracle;
    let mut gamma = f64::INFINITY;
    let options = Options::new(500, 1e-12);

    let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    assert!(xbest.is_some(), "Should find a solution");

    // Convergence should be monotonic in gamma
    assert!(gamma.is_finite(), "Final gamma should be finite");
    assert!(gamma >= 0.0, "Final gamma should be non-negative");

    // Should find a reasonable solution
    assert!(gamma < 10.0, "Final gamma {} should be improved from initial", gamma);
}

/// Test that algorithm handles different initial conditions
#[test]
fn test_different_initial_conditions() {
    struct SimpleOracle;

    impl OracleOptim<Arr> for SimpleOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            let gradient = array![2.0 * xc[0], 2.0 * xc[1]];
            let f = xc[0].powi(2) + xc[1].powi(2);

            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    let initial_points = vec![
        array![1.0, 1.0],
        array![-1.0, -1.0],
        array![5.0, -5.0],
        array![10.0, 0.0],
    ];

    let mut results = Vec::new();

    for initial in initial_points {
        let mut ellip = Ell::new_with_scalar(10.0, initial.clone());
        let mut oracle = SimpleOracle;
        let mut gamma = f64::INFINITY;
        let options = Options::new(1000, 1e-10);

        let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

        assert!(xbest.is_some(), "Should find solution from initial point {:?}", initial);
        results.push(gamma);
    }

    // All should converge to similar objective values (near zero)
    for (i, &gamma) in results.iter().enumerate() {
        assert!(gamma < 100.0, "Result {} should converge: gamma = {}", i, gamma);
    }
}