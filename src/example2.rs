use super::cutting_plane::OracleFeas;
use ndarray::prelude::*;

type Arr = Array1<f64>;

#[derive(Debug)]
struct MyOracleFeas {
    idx: usize,
}

impl MyOracleFeas {
    // constraint 1: x + y <= 3
    fn fn1(&self, x: f64, y: f64) -> f64 {
        x + y - 3.0
    }

    // constraint 2: x - y >= 1
    fn fn2(&self, x: f64, y: f64) -> f64 {
        -x + y + 1.0
    }

    fn grad1(&self) -> Arr {
        array![1.0, 1.0]
    }

    fn grad2(&self) -> Arr {
        array![-1.0, 1.0]
    }

    #[allow(dead_code)]
    pub fn new() -> Self {
        MyOracleFeas { idx: 0 }
    }
}

impl OracleFeas<Arr> for MyOracleFeas {
    type CutChoices = f64;

    /// The function assess_feas takes in an array z and checks if it satisfies two constraints,
    /// returning an optional tuple of an array and a float if any constraint is violated.
    ///
    /// Arguments:
    ///
    /// * `z`: The parameter `z` is an array of size 2, representing the coordinates of a point in a
    /// 2-dimensional space. The first element `z[0]` represents the x-coordinate, and the second
    /// element `z[1]` represents the y-coordinate.
    ///
    /// Returns:
    ///
    /// The function `assess_feas` returns an `Option` containing a tuple `(Arr, f64)`.
    fn assess_feas(&mut self, z: &Arr) -> Option<(Arr, f64)> {
        let x = z[0];
        let y = z[1];

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

    #[test]
    pub fn test_example2() {
        let mut ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracleFeas::new();
        let options = Options {
            max_iters: 2000,
            tol: 1e-12,
        };
        let (feasible, _niter) = cutting_plane_feas(&mut oracle, &mut ell, &options);
        assert!(feasible);
    }
}
