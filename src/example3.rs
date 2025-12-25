use super::cutting_plane::{OracleFeas, OracleFeas2};
use ndarray::prelude::*;

type Arr = Array1<f64>;

/// A struct representing a custom oracle for some optimization problem.
///
/// This oracle is used to evaluate the feasibility of a given solution.
/// It keeps track of an index `idx` and a target value `target`.
#[derive(Debug)]
pub struct MyOracle3 {
    /// The index of the current solution being evaluated.
    pub idx: i32,
    /// The target value for the optimization problem.
    pub target: f64,
}

// impl MyOracle3 {
//     /// Creates a new `MyOracle3` instance with the index set to 0 and the target value set to a very small negative number.
//     #[inline]
//     pub fn new() -> Self {
//         MyOracle3 {
//             idx: 0,
//             target: -1e100,
//         }
//     }
// }

impl Default for MyOracle3 {
    /// Creates a new `MyOracle3` instance with the index set to 0 and the target value set to a very small negative number.
    ///
    /// This is the default implementation for the `MyOracle3` struct, which is used to represent a custom oracle for some optimization problem.
    /// The oracle is used to evaluate the feasibility of a given solution, and this default implementation initializes the index to 0 and the target value to a very small negative number.
    #[inline]
    fn default() -> Self {
        MyOracle3 {
            idx: -1,
            target: -1e100,
        }
    }
}

impl OracleFeas<Arr> for MyOracle3 {
    type CutChoice = f64; // single cut

    /// The function assess_feas takes in an array xc and checks if it satisfies two constraints,
    /// returning an optional tuple of an array and a float if any constraint is violated.
    ///
    /// Arguments:
    ///
    /// * `xc`: The parameter `xc` is an array of size 2, representing the coordinates of a point in a
    ///   2-dimensional space. The first element `xc[0]` represents the x-coordinate, and the second
    ///   element `xc[1]` represents the y-coordinate.
    ///
    /// Returns:
    ///
    /// The function `assess_feas` returns an `Option` containing a tuple `(Arr, f64)`.
    fn assess_feas(&mut self, xc: &Arr) -> Option<(Arr, f64)> {
        let x_val = xc[0];
        let y_val = xc[1];

        let num_constraints = 4;
        for _ in 0..num_constraints {
            self.idx += 1;
            if self.idx == num_constraints {
                self.idx = 0; // round robin
            }
            let fj = match self.idx {
                0 => -x_val - 1.0,
                1 => -y_val - 2.0,
                2 => x_val + y_val - 1.0,
                3 => 2.0 * x_val - 3.0 * y_val - self.target,
                _ => unreachable!(),
            };
            if fj > 0.0 {
                return Some((
                    match self.idx {
                        0 => array![-1.0, 0.0],
                        1 => array![0.0, -1.0],
                        2 => array![1.0, 1.0],
                        3 => array![2.0, -3.0],
                        _ => unreachable!(),
                    },
                    fj,
                ));
            }
        }
        None
    }
}

impl OracleFeas2<Arr> for MyOracle3 {
    fn update(&mut self, gamma: f64) {
        self.target = gamma;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cutting_plane::{bsearch, BSearchAdaptor, Options};
    use crate::ell::Ell;
    // use ndarray::array;
    // use super::ell_stable::EllStable;

    #[test]
    pub fn test_feasible() {
        let ellip = Ell::new_with_scalar(100.0, array![0.0, 0.0]);
        let omega = MyOracle3::default();
        let options = Options {
            tolerance: 1e-8,
            ..Default::default()
        };
        let mut adaptor = BSearchAdaptor::new(omega, ellip, options);
        let mut intrvl = (-100.0, 100.0);
        let options2 = Options {
            tolerance: 1e-8,
            ..Default::default()
        };
        let (feasible, num_iters) = bsearch(&mut adaptor, &mut intrvl, &options2);
        assert!(feasible);
        assert_eq!(num_iters, 34);
    }
}
