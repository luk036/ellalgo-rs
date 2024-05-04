use super::cutting_plane::OracleFeas2;
use ndarray::prelude::*;

type Arr = Array1<f64>;

#[derive(Debug)]
pub struct MyOracle3 {
    idx: usize,
    target: f64,
}

impl MyOracle3 {
    #[inline]
    pub fn new() -> Self {
        MyOracle3 {
            idx: 0,
            target: -1e100,
        }
    }
}


impl Default for MyOracle3 {
    #[inline]
    fn default() -> Self {
        MyOracle3 {
            idx: 0,
            target: -1e100,
        }
    }
}

impl OracleFeas2<Arr> for MyOracle3 {
    type CutChoices = f64; // single cut

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

        for _ in 0..4 {
            self.idx += 1;
            if self.idx == 4 {
                self.idx = 0; // round robin
            }
            let fj = match self.idx {
                0 => -x - 1.0,
                1 => -y - 2.0,
                2 => x + y - 1.0,
                3 => 2.0 * x - 3.0 * y - self.target,
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

    fn update(&mut self, gamma: f64) {
        self.target = gamma;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cutting_plane::{BSearchAdaptor, bsearch, Options};
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
