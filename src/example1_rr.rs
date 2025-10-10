use super::cutting_plane::OracleOptim;
use ndarray::prelude::*;

type Arr = Array1<f64>;

#[derive(Debug)]
pub struct MyOracle {
    /// for round robin
    idx: i32,
}

impl Default for MyOracle {
    fn default() -> Self {
        MyOracle { idx: -1 }
    }
}

impl OracleOptim<Arr> for MyOracle {
    type CutChoice = f64; // single cut

    /// The function assess_optim takes in two parameters, xc and gamma, and returns a tuple containing an
    /// array and a double, along with a boolean value.
    ///
    /// Arguments:
    ///
    /// * `xc`: The parameter `xc` is an array of length 2, representing the values of `x` and `y`
    /// respectively.
    /// * `gamma`: The parameter `gamma` is a mutable reference to a `f64` variable. It is used to store the
    /// current best solution for the optimization problem.
    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
        let x = xc[0];
        let y = xc[1];
        let f0 = x + y;

        let num_constraints = 3;
        for _ in 0..num_constraints {
            self.idx += 1;
            if self.idx == num_constraints {
                self.idx = 0; // round robin
            }
            let fj = match self.idx {
                0 => f0 - 3.0,
                1 => -x + y + 1.0,
                2 => *gamma - f0,
                _ => unreachable!(),
            };
            if fj > 0.0 {
                return (
                    (
                        match self.idx {
                            0 => array![1.0, 1.0],
                            1 => array![-1.0, 1.0],
                            2 => array![-1.0, -1.0],
                            _ => unreachable!(),
                        },
                        fj,
                    ),
                    false,
                );
            }
        }
        *gamma = f0;
        ((array![-1.0, -1.0], 0.0), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cutting_plane::{cutting_plane_optim, Options};
    use crate::ell::Ell;
    // use ndarray::array;
    // use super::ell_stable::EllStable;

    /// Tests the feasibility of the optimization problem using the cutting plane method.
    ///
    /// This test creates a new `Ell` instance with a scalar radius of 10.0 and a center at `[0.0, 0.0]`.
    /// It then creates a new `MyOracle` instance as the optimization oracle.
    /// The `gamma` variable is initialized to negative infinity.
    /// The `Options` struct is configured with a tolerance of 1e-10.
    /// The `cutting_plane_optim` function is called with the oracle, ellipsoid, gamma, and options.
    /// The test asserts that the best solution `xbest` is not `None`, and that the number of iterations is 25.
    #[test]
    pub fn test_feasible() {
        let mut ellip = Ell::new_with_scalar(10.0, array![0.0, 0.0]);
        let mut oracle = MyOracle::default();
        let mut gamma = f64::NEG_INFINITY;
        let options = Options {
            tolerance: 1e-10,
            ..Default::default()
        };
        let (xbest, num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);
        assert!(xbest.is_some());
        assert_eq!(num_iters, 25);
    }

    #[test]
    pub fn test_infeasible1() {
        let mut ellip = Ell::new(array![10.0, 10.0], array![100.0, 100.0]); // wrong initial guess
                                                                            // or ellipsoid is too small
        let mut oracle = MyOracle::default();
        let mut gamma = f64::NEG_INFINITY;
        let options = Options::default();
        let (xbest, _num_iters) =
            cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);
        assert!(xbest.is_none());
    }

    #[test]
    pub fn test_infeasible2() {
        let mut ellip = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracle::default();
        // wrong initial guess
        let options = Options::default();
        let (xbest, _niter) = cutting_plane_optim(&mut oracle, &mut ellip, &mut 100.0, &options);
        assert!(xbest.is_none());
    }
}
