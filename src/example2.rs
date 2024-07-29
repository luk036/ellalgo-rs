use super::cutting_plane::OracleFeas;
use ndarray::prelude::*;

type Arr = Array1<f64>;

#[derive(Debug, Default)]
struct MyOracleFeas {
    idx: usize,
}

impl MyOracleFeas {
    /// Calculates the value of the constraint `x + y <= 3`.
    ///
    /// This function takes two `f64` values `x` and `y` and returns the value of the constraint `x + y - 3.0`.
    /// This can be used to check if the constraint is satisfied (the returned value is less than or equal to 0) or violated (the returned value is greater than 0).
    fn fn1(&self, x: f64, y: f64) -> f64 {
        x + y - 3.0
    }

    /// Calculates the value of the constraint `x - y >= 1`.
    ///
    /// This function takes two `f64` values `x` and `y` and returns the value of the constraint `x - y + 1.0`.
    /// This can be used to check if the constraint is satisfied (the returned value is greater than or equal to 0) or violated (the returned value is less than 0).
    fn fn2(&self, x: f64, y: f64) -> f64 {
        -x + y + 1.0
    }

    /// Calculates the gradient of the first constraint function `fn1`.
    ///
    /// This function returns the gradient of the constraint function `fn1` as a 1-dimensional array. The gradient is a constant vector `[1.0, 1.0]`, as the constraint function is linear.
    fn grad1(&self) -> Arr {
        array![1.0, 1.0]
    }

    /// Calculates the gradient of the second constraint function `fn2`.
    ///
    /// This function returns the gradient of the constraint function `fn2` as a 1-dimensional array. The gradient is a constant vector `[-1.0, 1.0]`, as the constraint function is linear.
    fn grad2(&self) -> Arr {
        array![-1.0, 1.0]
    }
}

impl OracleFeas<Arr> for MyOracleFeas {
    type CutChoices = f64;

    /// The function assess_feas takes in an array xc and checks if it satisfies two constraints,
    /// returning an optional tuple of an array and a float if any constraint is violated.
    ///
    /// Arguments:
    ///
    /// * `xc`: The parameter `xc` is an array of size 2, representing the coordinates of a point in a
    /// 2-dimensional space. The first element `xc[0]` represents the x-coordinate, and the second
    /// element `xc[1]` represents the y-coordinate.
    ///
    /// Returns:
    ///
    /// The function `assess_feas` returns an `Option` containing a tuple `(Arr, f64)`.
    fn assess_feas(&mut self, xc: &Arr) -> Option<(Arr, f64)> {
        let x = xc[0];
        let y = xc[1];

        for _ in 0..2 {
            self.idx += 1;
            if self.idx == 2 {
                self.idx = 0; // round robin
            }
            let fj = match self.idx {
                0 => self.fn1(x, y),
                1 => self.fn2(x, y),
                _ => unreachable!(),
            };
            if fj > 0.0 {
                return Some((
                    match self.idx {
                        0 => self.grad1(),
                        1 => self.grad2(),
                        _ => unreachable!(),
                    },
                    fj,
                ));
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cutting_plane::{cutting_plane_feas, Options};
    use crate::ell::Ell;
    use ndarray::array;

    // use super::ell_stable::EllStable;

    /// Tests the feasibility of an ellipsoid using the cutting plane method.
    ///
    /// This test creates a new `Ell` instance with a center at `[10.0, 10.0]` and a radius of `[0.0, 0.0]`.
    /// It then creates a new `MyOracleFeas` instance and calls the `cutting_plane_feas` function with the
    /// `Ell` and `Options` instances. The test asserts that the returned `x_opt` is `Some` and that the
    /// number of iterations is 1.
    #[test]
    pub fn test_feasible() {
        let mut ellip = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracleFeas::default();
        let options = Options::default();
        let (x_opt, num_iters) = cutting_plane_feas(&mut oracle, &mut ellip, &options);
        assert!(x_opt.is_some());
        assert_eq!(num_iters, 1)
    }

    #[test]
    pub fn test_infeasible() {
        let mut ellip = Ell::new(array![10.0, 10.0], array![100.0, 100.0]);
        let mut oracle = MyOracleFeas::default();
        let options = Options::default();
        let (x_opt, num_iters) = cutting_plane_feas(&mut oracle, &mut ellip, &options);
        assert!(x_opt.is_none());
        assert_eq!(num_iters, 1)
    }
}
