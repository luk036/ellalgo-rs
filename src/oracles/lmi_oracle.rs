use super::ldlt_mgr::LDLTMgr;
use crate::cutting_plane::OracleFeas;
use ndarray::{Array1, Array2};

pub type Arr = Array1<f64>;
pub type Cut = (Arr, f64);

/// The `LMIOracle` struct represents an oracle for a Linear Matrix Inequality (LMI) constraint.
/// It contains the necessary data to evaluate the LMI constraint, including the matrix `mat_f`,
/// the matrix `mat_f0`, and an `LDLTMgr` instance for managing the Cholesky decomposition.
/// This oracle can be used to check the feasibility of a given point with respect to the LMI constraint.
pub struct LMIOracle {
    mat_f: Vec<Array2<f64>>,
    mat_f0: Array2<f64>,
    ldlt_mgr: LDLTMgr,
}

impl LMIOracle {
    pub fn new(mat_f: Vec<Array2<f64>>, mat_b: Array2<f64>) -> Self {
        let ldlt_mgr = LDLTMgr::new(mat_b.nrows());
        LMIOracle {
            mat_f,
            mat_f0: mat_b,
            ldlt_mgr,
        }
    }
}

impl OracleFeas<Arr> for LMIOracle {
    type CutChoices = f64; // single cut

    fn assess_feas(&mut self, x: &Array1<f64>) -> Option<Cut> {
        fn get_elem(
            mat_f0: &Array2<f64>,
            mat_f: &[Array2<f64>],
            x: &Array1<f64>,
            i: usize,
            j: usize,
        ) -> f64 {
            mat_f0[(i, j)]
                - mat_f
                    .iter()
                    .zip(x.iter())
                    .map(|(mat_fk, xk)| mat_fk[(i, j)] * xk)
                    .sum::<f64>()
        }

        let get_elem = |i: usize, j: usize| get_elem(&self.mat_f0, &self.mat_f, x, i, j);

        if self.ldlt_mgr.factor(get_elem) {
            None
        } else {
            let ep = self.ldlt_mgr.witness();
            let g = self
                .mat_f
                .iter()
                .map(|mat_fk| self.ldlt_mgr.sym_quad(mat_fk))
                .collect();
            Some((g, ep))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use super::{ProfitOracle, ProfitOracleQ, ProfitRbOracle};
    use crate::cutting_plane::{cutting_plane_optim, Options, OracleOptim};
    use crate::ell::Ell;
    use ndarray::{array, Array2, ShapeError};

    struct MyOracle {
        c: Array1<f64>,
        lmi1: LMIOracle,
        lmi2: LMIOracle,
    }

    impl OracleOptim<Arr> for MyOracle {
        type CutChoices = f64; // single cut

        fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
            if let Some(cut) = self.lmi1.assess_feas(xc) {
                return (cut, false);
            }

            if let Some(cut) = self.lmi2.assess_feas(xc) {
                return (cut, false);
            }

            let f0 = self.c.dot(xc);
            let fj = f0 - *gamma;
            if fj > 0.0 {
                return ((self.c.clone(), fj), false);
            }

            *gamma = f0;
            ((self.c.clone(), 0.0), true)
        }
    }

    fn run_lmi(oracle1: LMIOracle, oracle2: LMIOracle) -> usize {
        let xinit = Arr::zeros(3);
        let mut ellip = Ell::new_with_scalar(10.0, xinit);
        let mut omega = MyOracle {
            c: array![1.0, -1.0, 1.0],
            lmi1: oracle1,
            lmi2: oracle2,
        };
        let mut gamma = f64::INFINITY;
        let (xbest, num_iters) =
            cutting_plane_optim(&mut omega, &mut ellip, &mut gamma, &Options::default());
        assert!(xbest.is_some());
        num_iters
    }

    #[test]
    fn test_lmi() -> Result<(), ShapeError> {
        let f1 = vec![
            Array2::from_shape_vec((2, 2), vec![-7.0, -11.0, -11.0, 3.0])?,
            Array2::from_shape_vec((2, 2), vec![7.0, -18.0, -18.0, 8.0])?,
            Array2::from_shape_vec((2, 2), vec![-2.0, -8.0, -8.0, 1.0])?,
        ];
        let b1 = Array2::from_shape_vec((2, 2), vec![33.0, -9.0, -9.0, 26.0])?;
        let f2 = vec![
            Array2::from_shape_vec(
                (3, 3),
                vec![-21.0, -11.0, 0.0, -11.0, 10.0, 8.0, 0.0, 8.0, 5.0],
            )?,
            Array2::from_shape_vec(
                (3, 3),
                vec![0.0, 10.0, 16.0, 10.0, -10.0, -10.0, 16.0, -10.0, 3.0],
            )?,
            Array2::from_shape_vec(
                (3, 3),
                vec![-5.0, 2.0, -17.0, 2.0, -6.0, 8.0, -17.0, 8.0, 6.0],
            )?,
        ];
        let b2 = Array2::from_shape_vec(
            (3, 3),
            vec![14.0, 9.0, 40.0, 9.0, 91.0, 10.0, 40.0, 10.0, 15.0],
        )?;

        let oracle1 = LMIOracle::new(f1, b1);
        let oracle2 = LMIOracle::new(f2, b2);
        let result = run_lmi(oracle1, oracle2);
        assert_eq!(result, 281);
        Ok(())
    }
}
