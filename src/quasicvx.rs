use super::cutting_plane::OracleOptim;
use ndarray::prelude::*;

type Arr = Array1<f64>;

#[derive(Debug, Default)]
pub struct MyOracle {
    idx: usize,
}

impl OracleOptim<Arr> for MyOracle {
    type CutChoices = f64; // single cut

    /// The function assesses optimization based on input values and returns a tuple along with a
    /// boolean flag.
    ///
    /// Arguments:
    ///
    /// * `xc`: The `xc` parameter in the `assess_optim` function represents an array containing two
    /// elements. The first element, `xc[0]`, is assigned to the variable `sqrtx`, and the second
    /// element, `xc[1]`, is assigned to the variable `logy`. These
    /// * `gamma`: The `gamma` parameter is a mutable reference to a `f64` value. It is being updated
    /// within the `assess_optim` function based on the calculations performed on the input values `xc`
    /// and the internal state of the function.
    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
        let sqrtx = xc[0];
        let logy = xc[1];

        let num_constraints = 2;
        for _ in 0..num_constraints {
            self.idx += 1;
            if self.idx == num_constraints {
                self.idx = 0; // round robin
            }
            let fj = match self.idx {
                0 => sqrtx * sqrtx - logy,
                1 => -sqrtx + *gamma * logy.exp(),
                _ => unreachable!(),
            };
            if fj > 0.0 {
                return (
                    (
                        match self.idx {
                            0 => array![2.0 * sqrtx, -1.0],
                            1 => array![-1.0, *gamma * logy.exp()],
                            _ => unreachable!(),
                        },
                        fj,
                    ),
                    false,
                );
            }
        }
        *gamma = sqrtx / logy.exp();
        ((array![-1.0, sqrtx], 0.0), true)
    }
}

#[cfg(test)]
mod tests {
    use super::MyOracle;
    use crate::cutting_plane::{cutting_plane_optim, Options};
    use crate::ell::Ell;
    // use crate::ell_stable::EllStable;
    use ndarray::array;

    #[test]
    pub fn test_feasible() {
        let mut ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracle::default();
        let mut gamma = 0.0;
        let options = Options {
            max_iters: 2000,
            tolerance: 1e-8,
        };
        let (x_opt, num_iters) = cutting_plane_optim(&mut oracle, &mut ell, &mut gamma, &options);
        assert!(x_opt.is_some());
        if let Some(x) = x_opt {
            assert!(x[0] * x[0] >= 0.49 && x[0] * x[0] <= 0.51);
            assert!(x[1].exp() >= 1.6 && x[1].exp() <= 1.7);
        }
        assert_eq!(num_iters, 35);
    }

    // #[test]
    // pub fn test_feasible_stable() {
    //     let mut ell = EllStable::new(array![10.0, 10.0], array![0.0, 0.0]);
    //     let mut oracle = MyOracle::default();
    //     let mut gamma = 0.0;
    //     let options = Options {
    //         max_iters: 2000,
    //         tolerance: 1e-8,
    //     };
    //     let (x_opt, num_iters) = cutting_plane_optim(&mut oracle, &mut ell, &mut gamma, &options);
    //     assert!(x_opt.is_some());
    //     if let Some(x) = x_opt {
    //         assert!(x[0] * x[0] >= 0.49 && x[0] * x[0] <= 0.51);
    //         assert!(x[1].exp() >= 1.6 && x[1].exp() <= 1.7);
    //     }
    //     assert_eq!(num_iters, 35);
    // }
}
