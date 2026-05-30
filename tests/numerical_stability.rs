use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim, SingleCut};
use ellalgo_rs::ell::Ell;

#[test]
fn test_ill_conditioned_quadratic() {
    struct IllConditionedOracle {
        condition_number: f64,
    }

    impl OracleOptim<Arr> for IllConditionedOracle {
        type CutChoice = SingleCut;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
            let eps = 1.0 / self.condition_number;
            let gradient = Arr::from(vec![2.0 * (1.0 + eps) * xc[0], 2.0 * (1.0 / eps) * xc[1]]);
            let f = (1.0 + eps) * xc[0].powi(2) + (1.0 / eps) * xc[1].powi(2);
            if f < *gamma {
                *gamma = f;
                ((gradient, SingleCut(f)), true)
            } else {
                ((gradient, SingleCut(f)), false)
            }
        }
    }

    for cond_num in [1e3, 1e5, 1e7] {
        let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![1.0, 1.0]));
        let mut oracle = IllConditionedOracle {
            condition_number: cond_num,
        };
        let mut gamma = f64::INFINITY;
        let options = Options::new(2000, 1e-12);

        let (xbest, _num_iters) =
            cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

        assert!(
            xbest.is_some(),
            "Should converge with cond_num = {}",
            cond_num
        );
        assert!(
            gamma.is_finite(),
            "Final gamma should be finite for cond_num = {}",
            cond_num
        );
    }
}

#[test]
fn test_near_singular_initial_ellipsoid() {
    struct NearSingularOracle;

    impl OracleOptim<Arr> for NearSingularOracle {
        type CutChoice = SingleCut;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
            let gradient = Arr::from(vec![2.0 * xc[0], 2.0 * xc[1]]);
            let f = xc[0].powi(2) + xc[1].powi(2);
            if f < *gamma {
                *gamma = f;
                ((gradient, SingleCut(f)), true)
            } else {
                ((gradient, SingleCut(f)), false)
            }
        }
    }

    let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![3.0, 3.0]));
    let mut oracle = NearSingularOracle;
    let mut gamma = f64::INFINITY;
    let options = Options::new(1000, 1e-10);

    let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    assert!(
        xbest.is_some(),
        "Should handle near-singular initial ellipsoid"
    );
    assert!(
        gamma < 10.0,
        "Should converge to reasonable value: {}",
        gamma
    );
}

#[test]
fn test_extreme_scale_values() {
    struct ExtremeScaleOracle {
        scale: f64,
    }

    impl OracleOptim<Arr> for ExtremeScaleOracle {
        type CutChoice = SingleCut;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
            let s = self.scale;
            let gradient = Arr::from(vec![2.0 * s * xc[0], 2.0 * xc[1]]);
            let f = s * xc[0].powi(2) + xc[1].powi(2);
            if f < *gamma {
                *gamma = f;
                ((gradient, SingleCut(f)), true)
            } else {
                ((gradient, SingleCut(f)), false)
            }
        }
    }

    for scale in [1e-6f64, 1e6f64] {
        let mut ellip =
            Ell::new_with_scalar(10.0 * scale.abs().sqrt(), Arr::from(vec![scale, scale]));
        let mut oracle = ExtremeScaleOracle { scale };
        let mut gamma = f64::INFINITY;
        let options = Options::new(2000, 1e-10);

        let (xbest, _num_iters) =
            cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

        assert!(xbest.is_some(), "Should handle scale = {}", scale);
    }
}

#[test]
fn test_tolerance_sensitivity() {
    struct ToleranceOracle;

    impl OracleOptim<Arr> for ToleranceOracle {
        type CutChoice = SingleCut;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
            let gradient = Arr::from(vec![2.0 * xc[0], 2.0 * xc[1]]);
            let f = xc[0].powi(2) + xc[1].powi(2);
            if f < *gamma {
                *gamma = f;
                ((gradient, SingleCut(f)), true)
            } else {
                ((gradient, SingleCut(f)), false)
            }
        }
    }

    for tolerance in [1e-6, 1e-10, 1e-14] {
        let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![3.0, 3.0]));
        let mut oracle = ToleranceOracle;
        let mut gamma = f64::INFINITY;
        let options = Options::new(2000, tolerance);

        let (xbest, _num_iters) =
            cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

        assert!(
            xbest.is_some(),
            "Should converge with tolerance = {}",
            tolerance
        );
    }
}

#[test]
fn test_numerical_precision() {
    struct PrecisionOracle;

    impl OracleOptim<Arr> for PrecisionOracle {
        type CutChoice = SingleCut;

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
            let gradient = Arr::from(vec![2.0 * xc[0], 2.0 * xc[1]]);
            let f = xc[0].powi(2) + xc[1].powi(2);
            if f < *gamma {
                *gamma = f;
                ((gradient, SingleCut(f)), true)
            } else {
                ((gradient, SingleCut(f)), false)
            }
        }
    }

    let initial_point = Arr::from(vec![1000.0, -1000.0]);
    let mut ellip = Ell::new_with_scalar(10.0, initial_point.clone());
    let mut oracle = PrecisionOracle;
    let mut gamma = f64::INFINITY;
    let options = Options::new(3000, 1e-12);

    let (xbest, _num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    assert!(xbest.is_some(), "Should converge from far starting point");
    let gamma_orig = initial_point[0].powi(2) + initial_point[1].powi(2);
    assert!(
        gamma < gamma_orig,
        "gamma should improve from initial value"
    );
}
