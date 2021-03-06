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
        let sqrtx = z[0];
        let ly = z[1];

        // constraint 1: exp(x) <= y, or sqrtx**2 <= ly
        let fj = sqrtx * sqrtx - ly;
        if fj > 0.0 {
            return ((array![2.0 * sqrtx, -1.0], fj), false);
        }

        // objective: minimize -sqrt(x) / y
        let tmp2 = ly.exp();
        let tmp3 = *t * tmp2;
        let fj = -sqrtx + tmp3;
        if fj < 0.0 {
            // feasible
            *t = sqrtx / tmp2;
            return ((array![-1.0, sqrtx], 0.0), true);
        }
        ((array![-1.0, tmp3], fj), false)
    }
}

mod tests {
    use super::*;
    use crate::cutting_plane::{cutting_plane_optim, CutStatus, Options};
    use crate::ell::Ell;
    // use crate::ell_stable::EllStable;
    use ndarray::array;
    // use super::ell_stable::EllStable;

    #[test]
    pub fn test_feasible() {
        let mut ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracle {};
        let mut t = 0.0;
        let options = Options {
            max_iter: 2000,
            tol: 1e-10,
        };
        let (x_opt, _niter, _status) = cutting_plane_optim(&mut oracle, &mut ell, &mut t, &options);
        if let Some(x) = x_opt {
            assert!(x[0] * x[0] >= 0.49 && x[0] * x[0] <= 0.51);
            assert!(x[1].exp() >= 1.6 && x[1].exp() <= 1.7);
        } else {
            assert!(false); // not feasible
        }
    }

    #[test]
    pub fn test_feasible_stable() {
        let mut ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracle {};
        let mut t = 0.0;
        let options = Options {
            max_iter: 2000,
            tol: 1e-10,
        };
        let (x_opt, _niter, _status) = cutting_plane_optim(&mut oracle, &mut ell, &mut t, &options);
        if let Some(x) = x_opt {
            assert!(x[0] * x[0] >= 0.49 && x[0] * x[0] <= 0.51);
            assert!(x[1].exp() >= 1.6 && x[1].exp() <= 1.7);
        } else {
            assert!(false); // not feasible
        }
    }
}
