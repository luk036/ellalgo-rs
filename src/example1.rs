use super::cutting_plane::OracleOptim;
use ndarray::prelude::*;

type Arr = Array1<f64>;

#[derive(Debug)]
pub struct MyOracle {}

impl OracleOptim<Arr> for MyOracle {
    type CutChoices = f64; // single cut

    /// The function assess_optim takes in two parameters, z and gamma, and returns a tuple containing an
    /// array and a double, along with a boolean value.
    ///
    /// Arguments:
    ///
    /// * `z`: The parameter `z` is an array of length 2, representing the values of `x` and `y`
    /// respectively.
    /// * `gamma`: The parameter `gamma` is a mutable reference to a `f64` variable. It is used to store the
    /// current best solution for the optimization problem.
    fn assess_optim(&mut self, z: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
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
        let fj = *gamma - f0;
        if fj < 0.0 {
            *gamma = f0;
            return ((array![-1.0, -1.0], 0.0), true);
        }
        ((array![-1.0, -1.0], fj), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cutting_plane::{cutting_plane_optim, Options};
    use crate::ell::Ell;
    // use ndarray::array;
    // use super::ell_stable::EllStable;

    #[test]
    pub fn test_feasible() {
        let mut ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracle {};
        let mut gamma = -1.0e100; // std::numeric_limits<double>::min()
        let options = Options {
            max_iters: 2000,
            tol: 1e-10,
        };
        let (x_opt, _niter) = cutting_plane_optim(&mut oracle, &mut ell, &mut gamma, &options);
        assert!(x_opt.is_some());
        if let Some(x) = x_opt {
            assert!(x[0] >= 0.0);
        }
    }

    #[test]
    pub fn test_infeasible1() {
        let mut ell = Ell::new(array![10.0, 10.0], array![100.0, 100.0]); // wrong initial guess
                                                                          // or ellipsoid is too small
        let mut oracle = MyOracle {};
        let mut gamma = -1.0e100; // std::numeric_limits<double>::min()
        let options = Options {
            max_iters: 2000,
            tol: 1e-12,
        };
        let (x_opt, _niter) = cutting_plane_optim(&mut oracle, &mut ell, &mut gamma, &options);
        assert_eq!(x_opt, None);
    }

    #[test]
    pub fn test_infeasible2() {
        let mut ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracle {};
        // wrong initial guess
        let options = Options {
            max_iters: 2000,
            tol: 1e-12,
        };
        let (x_opt, _niter) = cutting_plane_optim(&mut oracle, &mut ell, &mut 100.0, &options);
        assert_eq!(x_opt, None);
    }
}
