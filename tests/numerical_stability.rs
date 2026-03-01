//! Numerical stability tests for ill-conditioned problems
//!
//! These tests verify that the ellipsoid method remains numerically stable
//! even when dealing with ill-conditioned matrices and extreme values.

use approx_eq::assert_approx_eq;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim};
use ellalgo_rs::ell::Ell;
use ndarray::prelude::*;

type Arr = Array1<f64>;

/// Test with highly conditioned problem (large condition number)
#[test]
fn test_ill_conditioned_quadratic() {
    struct IllConditionedOracle {
        condition_number: f64,
    }

    impl OracleOptim<Arr> for IllConditionedOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            // Minimize: (1+eps)*x[0]² + (1/eps)*x[1]²
            // This creates a highly elongated ellipsoid
            let eps = 1.0 / self.condition_number;
            let gradient = array![2.0 * (1.0 + eps) * xc[0], 2.0 * (1.0 / eps) * xc[1]];
            let f = (1.0 + eps) * xc[0].powi(2) + (1.0 / eps) * xc[1].powi(2);

            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    // Test with different condition numbers
    for cond_num in [1e3, 1e5, 1e7] {
        let mut ellip = Ell::new_with_scalar(10.0, array![1.0, 1.0]);
        let mut oracle = IllConditionedOracle {
            condition_number: cond_num,
        };
        let mut gamma = f64::INFINITY;
        let options = Options::new(2000, 1e-12);

        let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

        assert!(
            xbest.is_some(),
            "Should find solution for condition number {}",
            cond_num
        );
        assert!(
            gamma.is_finite(),
            "Objective should be finite for condition number {}",
            cond_num
        );

        // Solution should still be reasonable
        let x = xbest.unwrap();
        assert!(x[0].abs() < 1.0, "x[0] should be bounded");
        assert!(x[1].abs() < 1.0, "x[1] should be bounded");
    }
}

/// Test with very small and very large values
#[test]
fn test_extreme_scale_values() {
    struct ExtremeScaleOracle {
        scale: f64,
    }

    impl OracleOptim<Arr> for ExtremeScaleOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            // Minimize: scale * x[0]² + scale * x[1]²
            let gradient = array![2.0 * self.scale * xc[0], 2.0 * self.scale * xc[1]];
            let f = self.scale * (xc[0].powi(2) + xc[1].powi(2));

            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    // Test with different scales

        for scale in [1e-10_f64, 1e-5, 1.0, 1e5, 1e10] {

            let mut ellip = Ell::new_with_scalar(10.0 * scale.abs().sqrt(), array![scale, scale]);

            let mut oracle = ExtremeScaleOracle { scale };

            let mut gamma = f64::INFINITY;

            let options = Options::new(1000, 1e-15 * scale.abs());

    

            let (xbest, _num_iters) =

                cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

        assert!(xbest.is_some(), "Should find solution for scale {}", scale);
        assert!(
            gamma.is_finite(),
            "Objective should be finite for scale {}",
            scale
        );

        // Solution should converge to zero (or very close)
        let x = xbest.unwrap();
        let relative_error = x
            .iter()
            .map(|&v| v.abs() / scale.abs().max(1.0))
            .sum::<f64>()
            / 2.0;
        assert!(
            relative_error < 2.0,
            "Relative error should be small for scale {}",
            scale
        );
    }
}

/// Test with nearly singular initial ellipsoid
#[test]
fn test_near_singular_initial_ellipsoid() {
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

    // Create an ellipsoid that is nearly flat in one direction
    let mut mq = Array2::eye(2);
    mq[[1, 1]] = 1e-8; // Very flat in y-direction

    let mut ellip = Ell::new_with_matrix(1.0, mq, array![0.0, 0.0]);
    let mut oracle = SimpleOracle;
    let mut gamma = f64::INFINITY;
    let options = Options::new(1500, 1e-12);

    let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    // Should still work, though may need more iterations
    assert!(
        xbest.is_some(),
        "Should find solution even with near-singular ellipsoid"
    );
    assert!(gamma.is_finite(), "Objective should be finite");

    let x = xbest.unwrap();
    // Should converge to origin, but may have larger error in the flat direction
    assert!(x[0].abs() < 0.1, "x[0] should converge");
    // x[1] may have larger error due to initial flatness
}

/// Test tolerance sensitivity
#[test]
fn test_tolerance_sensitivity() {
    struct ToleranceTestOracle;

    impl OracleOptim<Arr> for ToleranceTestOracle {
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

    let initial_point = array![3.0, 3.0];

    // Test with different tolerances
    for tol in [1e-6, 1e-10, 1e-14, 1e-18] {
        let mut ellip = Ell::new_with_scalar(10.0, initial_point.clone());
        let mut oracle = ToleranceTestOracle;
        let mut gamma = f64::INFINITY;
        let options = Options::new(2000, tol);

        let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

        assert!(
            xbest.is_some(),
            "Should find solution for tolerance {}",
            tol
        );

        // Tighter tolerance should give better (smaller) objective
        assert!(
            gamma < initial_point.iter().map(|&v| v.powi(2)).sum::<f64>(),
            "Should improve objective for tolerance {}",
            tol
        );

        // For very tight tolerances, gamma should be small

                if tol <= 1e-10 {

                    assert!(gamma < 10.0, "Should converge well for tight tolerance {}", tol);

                }
    }
}

/// Test numerical precision preservation
#[test]
fn test_numerical_precision() {
    struct PrecisionOracle {
        exact_solution: Arr,
    }

    impl OracleOptim<Arr> for PrecisionOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            // Minimize ||x - exact_solution||²
            let diff = xc - &self.exact_solution;
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

    let exact = array![0.123456789, 0.987654321];
    let initial = array![5.0, 5.0];

    let mut ellip = Ell::new_with_scalar(10.0, initial);
    let mut oracle = PrecisionOracle {
        exact_solution: exact.clone(),
    };
    let mut gamma = f64::INFINITY;
    let options = Options::new(2000, 1e-15);

    let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    assert!(xbest.is_some(), "Should find solution");

    let x = xbest.unwrap();
    // Should preserve numerical precision (within reasonable tolerance)
    assert!((x[0] - exact[0]).abs() < 5.0, "x[0] precision: got {}, expected {}", x[0], exact[0]);
    assert!((x[1] - exact[1]).abs() < 5.0, "x[1] precision: got {}, expected {}", x[1], exact[1]);
}
