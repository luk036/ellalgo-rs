use super::cutting_plane::OracleOptim;
use ndarray::prelude::*;

type Arr = Array1<f64>;

#[derive(Debug)]
pub struct MyOracle {}

impl OracleOptim for MyOracle {
    type ArrayType = Arr;
    type CutChoices = f64; // single cut

    /**
     * @brief
     *
     * @param[in] z
     * @param[in,out] t
     * @return std::tuple<Cut, double>
     */
    fn assess_optim(&mut self, z: &Arr, t: &mut f64) -> ((Arr, f64), bool) {
        let x = z[0];
        let y = z[1];

        // constraint 1: x + y <= 3
        let fj = x + y - 3.0;
        if fj > 0.0 {
            return ((array![1.0, 1.0], fj), false);
        }
        // constraint 2: x - y >= 1
        let fj = -x + y + 1.0;
        if fj > 0.0 {
            return ((array![-1.0, 1.0], fj), false);
        }
        // objective: maximize x + y
        let f0 = x + y;
        let fj = *t - f0;
        if fj < 0.0 {
            *t = f0;
            return ((array![-1.0, -1.0], 0.0), true);
        }
        ((array![-1.0, -1.0], fj), false)
    }
}

mod tests {
    use super::*;
    use crate::cutting_plane::{cutting_plane_optim, CutStatus, Options};
    use crate::ell::Ell;
    use ndarray::array;
    // use super::ell_stable::EllStable;

    #[test]
    pub fn test_feasible() {
        let mut ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracle {};
        let mut t = -1.0e100; // std::numeric_limits<double>::min()
        let options = Options {
            max_iter: 2000,
            tol: 1e-10,
        };
        let (x_opt, _niter, _status) = cutting_plane_optim(&mut oracle, &mut ell, &mut t, &options);
        if let Some(x) = x_opt {
            assert!(x[0] >= 0.0);
        } else {
            assert!(false); // not feasible
        }
    }

    #[test]
    pub fn test_infeasible1() {
        let mut ell = Ell::new(array![10.0, 10.0], array![100.0, 100.0]); // wrong initial guess
                                                                          // or ellipsoid is too small
        let mut oracle = MyOracle {};
        let mut t = -1.0e100; // std::numeric_limits<double>::min()
        let options = Options {
            max_iter: 2000,
            tol: 1e-12,
        };
        let (x_opt, _niter, status) = cutting_plane_optim(&mut oracle, &mut ell, &mut t, &options);
        if let Some(_x) = x_opt {
            assert!(false);
        } else {
            assert_eq!(status, CutStatus::NoSoln); // no sol'n
        }
    }

    #[test]
    pub fn test_infeasible2() {
        let mut ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracle {};
        // wrong initial guess
        let options = Options {
            max_iter: 2000,
            tol: 1e-12,
        };
        let (x_opt, _niter, status) =
            cutting_plane_optim(&mut oracle, &mut ell, &mut 100.0, &options);
        if let Some(_x) = x_opt {
            assert!(false);
        } else {
            assert_eq!(status, CutStatus::NoSoln); // no sol'n
        }
    }
}
