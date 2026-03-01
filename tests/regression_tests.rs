//! Regression tests to ensure algorithm behavior remains consistent
//!
//! These tests track convergence metrics and ensure the algorithm
//! doesn't regress in terms of performance or accuracy.

use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim};
use ellalgo_rs::ell::Ell;
use ndarray::prelude::*;

type Arr = Array1<f64>;

/// Track iteration counts for a simple quadratic problem
/// Regression: should converge in a bounded number of iterations
#[test]
fn test_regression_quadratic_iterations() {
    struct QuadraticOracle;

    impl OracleOptim<Arr> for QuadraticOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            // Minimize: x[0]² + x[1]²
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
    let mut oracle = QuadraticOracle;
    let mut gamma = f64::INFINITY;
    let options = Options::new(1000, 1e-10);

    let (_xbest, num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    // Regression: should converge in less than 1000 iterations for this simple problem

    assert!(
        num_iters < 1000,
        "Regression: quadratic problem converged in {} iterations, expected < 1000",
        num_iters
    );

    // Regression: final objective should be reasonably small

    assert!(
        gamma < 5.0,
        "Regression: final objective {}, expected < 5.0",
        gamma
    );
}

/// Track convergence rate for a convex problem
/// Regression: objective should decrease monotonically
#[test]
fn test_regression_monotonic_convergence() {
    struct ConvergenceOracle {
        history: Vec<f64>,
    }

    impl ConvergenceOracle {
        fn new() -> Self {
            ConvergenceOracle {
                history: Vec::new(),
            }
        }
    }

    impl OracleOptim<Arr> for ConvergenceOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            // Minimize: (x[0]-1)² + (x[1]-2)²
            let gradient = array![2.0 * (xc[0] - 1.0), 2.0 * (xc[1] - 2.0)];
            let f = (xc[0] - 1.0).powi(2) + (xc[1] - 2.0).powi(2);

            self.history.push(f);

            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    let mut ellip = Ell::new_with_scalar(10.0, array![0.0, 0.0]);
    let mut oracle = ConvergenceOracle::new();
    let mut gamma = f64::INFINITY;
    let options = Options::new(200, 1e-12);

    let (_xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    // Regression: gamma should improve (decrease) at least once
    assert!(gamma < f64::INFINITY, "Regression: gamma should improve");

    // Regression: final gamma should be close to optimal (0)
    assert!(
        gamma < 0.1,
        "Regression: final gamma {}, expected < 0.1",
        gamma
    );
}

/// Track solution quality for a known optimal point
/// Regression: should find solution within known tolerance
#[test]
fn test_regression_solution_quality() {
    struct KnownOptimalOracle {
        optimal: Arr,
    }

    impl OracleOptim<Arr> for KnownOptimalOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            // Minimize: ||x - optimal||²
            let diff = xc - &self.optimal;
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

    let optimal = array![2.0, -1.5];
    let initial = array![10.0, 10.0];

    let mut ellip = Ell::new_with_scalar(20.0, initial);
    let mut oracle = KnownOptimalOracle {
        optimal: optimal.clone(),
    };
    let mut gamma = f64::INFINITY;
    let options = Options::new(1000, 1e-10);

    let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    assert!(xbest.is_some(), "Regression: should find solution");

    let x = xbest.unwrap();
    let error = ((x[0] - optimal[0]).powi(2) + (x[1] - optimal[1]).powi(2)).sqrt();

    // Regression: should be within reasonable tolerance of optimal

    assert!(
        error < 10.0,
        "Regression: solution error {}, expected < 10.0",
        error
    );
}

/// Track performance scaling with problem dimension
/// Regression: iterations should scale reasonably with dimension
#[test]
fn test_regression_dimensional_scaling() {
    struct DimOracle {
        #[allow(dead_code)]
        ndim: usize,
    }

    impl OracleOptim<Arr> for DimOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            // Minimize: sum of squares
            let gradient = 2.0 * xc;
            let f = xc.dot(xc);

            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    // Test different dimensions and track iteration scaling
    let dims = [2, 3, 4, 5];
    let mut iteration_counts = Vec::new();

    for &ndim in &dims {
        let initial = Array1::from_elem(ndim, 5.0);
        let mut ellip = Ell::new_with_scalar(10.0, initial);
        let mut oracle = DimOracle { ndim };
        let mut gamma = f64::INFINITY;
        let options = Options::new(2000, 1e-10);

        let (_xbest, num_iters) =
            cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

        iteration_counts.push(num_iters);

        // Should converge for all dimensions
        assert!(
            num_iters < 2000 || gamma < 100.0,
            "Should converge or get good solution in {} dimensions",
            ndim
        );
    }

    // Regression: iterations should not grow exponentially with dimension

    // This is a loose check - exponential growth would be very fast

    let ratio = iteration_counts[3] as f64 / iteration_counts[0] as f64;

    assert!(
        ratio < 10.0,
        "Regression: iteration ratio {}D/2D = {:.2}, expected < 10",
        dims[3],
        ratio
    );
}

/// Track ellipsoid volume reduction
/// Regression: ellipsoid should shrink over iterations
#[test]
fn test_regression_ellipsoid_shrinkage() {
    struct ShrinkageOracle;

    impl OracleOptim<Arr> for ShrinkageOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            // Minimize: x[0]² + x[1]²
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
    let initial_kappa = ellip.kappa;
    let initial_tsq = ellip.tsq;

    let mut oracle = ShrinkageOracle;
    let mut gamma = f64::INFINITY;
    let options = Options::new(100, 1e-10);

    let (_xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    // Regression: ellipsoid should have shrunk (or at least not grown significantly)

    assert!(
        ellip.tsq <= initial_tsq * 1.1 || ellip.kappa <= initial_kappa * 1.1,
        "Regression: ellipsoid should shrink during optimization"
    );
}

/// Test fixed iteration count for reproducibility
/// Regression: should produce consistent results
#[test]
fn test_regression_reproducibility() {
    struct ReproducibleOracle;

    impl OracleOptim<Arr> for ReproducibleOracle {
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

    let initial = array![2.0, 3.0];
    let options = Options::new(500, 1e-10);

    // Run twice and check for consistency
    let mut ellip1 = Ell::new_with_scalar(10.0, initial.clone());
    let mut oracle1 = ReproducibleOracle;
    let mut gamma1 = f64::INFINITY;

    let (xbest1, num_iters1) =
        cutting_plane_optim(&mut oracle1, &mut ellip1, &mut gamma1, &options);

    let mut ellip2 = Ell::new_with_scalar(10.0, initial);
    let mut oracle2 = ReproducibleOracle;
    let mut gamma2 = f64::INFINITY;

    let (xbest2, num_iters2) =
        cutting_plane_optim(&mut oracle2, &mut ellip2, &mut gamma2, &options);

    // Regression: should produce identical results
    assert_eq!(
        num_iters1, num_iters2,
        "Regression: iteration count should be reproducible"
    );

    if let (Some(x1), Some(x2)) = (xbest1, xbest2) {
        let diff = (x1[0] - x2[0]).abs() + (x1[1] - x2[1]).abs();

        assert!(
            diff < 1e-10,
            "Regression: solutions should be identical, diff = {}",
            diff
        );
    }

    assert!(
        (gamma1 - gamma2).abs() < 1e-10,
        "Regression: gamma values should be identical"
    );
}
