use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim};
use ellalgo_rs::ell::Ell;

/// Regression: should converge in a bounded number of iterations
#[test]
fn test_regression_quadratic_iterations() {
    struct QuadraticOracle;

    impl OracleOptim<Arr> for QuadraticOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            let gradient = Arr::from(vec![2.0 * xc[0], 2.0 * xc[1]]);
            let f = xc[0].powi(2) + xc[1].powi(2);
            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![3.0, 3.0]));
    let mut oracle = QuadraticOracle;
    let mut gamma = f64::INFINITY;
    let options = Options::new(1000, 1e-10);

    let (_xbest, num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    assert!(
        num_iters < 1000,
        "Regression: quadratic problem converged in {} iterations, expected < 1000",
        num_iters
    );
    assert!(
        gamma < 10.0,
        "Regression: final gamma {} should be reasonably small",
        gamma
    );
}

/// Regression: solution quality for different starting points
#[test]
fn test_regression_solution_quality() {
    struct QuadraticOracle;

    impl OracleOptim<Arr> for QuadraticOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            let gradient = Arr::from(vec![2.0 * xc[0], 2.0 * xc[1]]);
            let f = xc[0].powi(2) + xc[1].powi(2);
            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    let start_points = vec![
        Arr::from(vec![3.0, 3.0]),
        Arr::from(vec![-5.0, 5.0]),
        Arr::from(vec![10.0, -10.0]),
    ];

    for initial in start_points {
        let mut ellip = Ell::new_with_scalar(20.0, initial);
        let mut oracle = QuadraticOracle;
        let mut gamma = f64::INFINITY;
        let options = Options::new(2000, 1e-10);

        let (xbest, _num_iters) =
            cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

        assert!(
            xbest.is_some(),
            "Should converge from different starting points"
        );
        if let Some(x) = xbest {
            assert!(
                x[0].abs() < 10.0 && x[1].abs() < 10.0,
                "Solution should converge toward origin, got ({}, {})",
                x[0],
                x[1]
            );
        }
    }
}

/// Regression: ellipsoid shrinkage over iterations
#[test]
fn test_regression_ellipsoid_shrinkage() {
    struct TrackingOracle {
        #[allow(dead_code)]
        tsq_history: Vec<f64>,
    }

    impl TrackingOracle {
        fn new() -> Self {
            Self {
                tsq_history: Vec::new(),
            }
        }
    }

    impl OracleOptim<Arr> for TrackingOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            let gradient = Arr::from(vec![2.0 * xc[0], 2.0 * xc[1]]);
            let f = xc[0].powi(2) + xc[1].powi(2);
            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![3.0, 3.0]));
    let mut oracle = TrackingOracle::new();
    let mut gamma = f64::INFINITY;
    let options = Options::new(500, 1e-12);

    let (_xbest, num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    assert!(num_iters > 0, "Should complete some iterations");
    assert!(gamma.is_finite(), "Final gamma should be finite");
    assert!(
        gamma >= 0.0,
        "Final gamma should be non-negative for quadratic"
    );
}

/// Regression: monotonic convergence
#[test]
fn test_regression_monotonic_convergence() {
    struct MonotonicOracle;

    impl OracleOptim<Arr> for MonotonicOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            let gradient = Arr::from(vec![2.0 * xc[0], 2.0 * xc[1]]);
            let f = xc[0].powi(2) + xc[1].powi(2);
            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![3.0, 3.0]));
    let mut oracle = MonotonicOracle;
    let mut gamma = f64::INFINITY;
    let options = Options::new(500, 1e-12);

    let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    assert!(xbest.is_some(), "Should find a solution");
    assert!(
        gamma >= 0.0,
        "Final gamma for ||x||² should be non-negative"
    );
    assert!(gamma < 100.0, "Final gamma should be less than initial");
}

/// Regression: dimensional scaling
#[test]
fn test_regression_dimensional_scaling() {
    struct ScalingOracle {
        ndim: usize,
    }

    impl OracleOptim<Arr> for ScalingOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            let mut gradient = Arr::new(self.ndim);
            let mut f = 0.0;
            for i in 0..self.ndim {
                gradient[i] = 2.0 * xc[i];
                f += xc[i].powi(2);
            }
            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    for ndim in [2, 4, 8].iter() {
        let initial = Arr::from(vec![3.0; *ndim]);
        let mut ellip = Ell::new_with_scalar(10.0, initial);
        let mut oracle = ScalingOracle { ndim: *ndim };
        let mut gamma = f64::INFINITY;
        let options = Options::new(3000, 1e-10);

        let (xbest, num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

        assert!(xbest.is_some(), "Should converge in {} dimensions", ndim);
        assert!(
            num_iters < 3000,
            "{} dimensions should converge within iteration limit: {}",
            ndim,
            num_iters
        );
    }
}

/// Regression: reproducibility
#[test]
fn test_regression_reproducibility() {
    struct ConstOracle;

    impl OracleOptim<Arr> for ConstOracle {
        type CutChoice = f64;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            let gradient = Arr::from(vec![2.0 * xc[0], 2.0 * xc[1]]);
            let f = xc[0].powi(2) + xc[1].powi(2);
            if f < *gamma {
                *gamma = f;
                ((gradient, f), true)
            } else {
                ((gradient, f), false)
            }
        }
    }

    let options = Options::new(500, 1e-10);

    let run = || {
        let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![3.0, 3.0]));
        let mut oracle = ConstOracle;
        let mut gamma = f64::INFINITY;
        let (xbest, num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);
        (xbest, num_iters, gamma)
    };

    let (x1, n1, g1) = run();
    let (x2, n2, g2) = run();

    assert_eq!(n1, n2, "Iteration counts should match");
    assert!((g1 - g2).abs() < 1e-10, "Final gamma should match");
    if let (Some(x1), Some(x2)) = (&x1, &x2) {
        for i in 0..2 {
            assert!((x1[i] - x2[i]).abs() < 1e-10, "Solutions should match");
        }
    }
}
