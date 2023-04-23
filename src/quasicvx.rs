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
     * @param[in,out] target
     * @return std::tuple<Cut, double>
     */
    fn assess_optim(&mut self, z: &Arr, target: &mut f64) -> ((Arr, f64), bool) {
        let sqrtx = z[0];
        let ly = z[1];

        // constraint 1: exp(x) <= y, or sqrtx**2 <= ly
        let fj = sqrtx * sqrtx - ly;
        if fj > 0.0 {
            return ((array![2.0 * sqrtx, -1.0], fj), false);
        }

        // objective: minimize -sqrt(x) / y
        let tmp2 = ly.exp();
        let tmp3 = *target * tmp2;
        let fj = -sqrtx + tmp3;
        if fj < 0.0 {
            // feasible
            *target = sqrtx / tmp2;
            return ((array![-1.0, sqrtx], 0.0), true);
        }
        ((array![-1.0, tmp3], fj), false)
    }
}

#[cfg(test)]
mod tests {
    use super::MyOracle;
    use crate::cutting_plane::{cutting_plane_optim, Options};
    use crate::ell::Ell;
    use ndarray::array;

    #[test]
    pub fn test_feasible() {
        let mut ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracle {};
        let mut target = 0.0;
        let options = Options {
            max_iter: 2000,
            tol: 1e-10,
        };
        let (x_opt, _niter) = cutting_plane_optim(&mut oracle, &mut ell, &mut target, &options);
        assert!(!x_opt.is_none());
        if let Some(x) = x_opt {
            assert!(x[0] * x[0] >= 0.49 && x[0] * x[0] <= 0.51);
            assert!(x[1].exp() >= 1.6 && x[1].exp() <= 1.7);
        }
    }

    #[test]
    pub fn test_feasible_stable() {
        let mut ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracle {};
        let mut target = 0.0;
        let options = Options {
            max_iter: 2000,
            tol: 1e-10,
        };
        let (x_opt, _niter) = cutting_plane_optim(&mut oracle, &mut ell, &mut target, &options);
        assert!(!x_opt.is_none());
        if let Some(x) = x_opt {
            assert!(x[0] * x[0] >= 0.49 && x[0] * x[0] <= 0.51);
            assert!(x[1].exp() >= 1.6 && x[1].exp() <= 1.7);
        }
    }
}
