use super::cutting_plane::OracleOptim;
use ndarray::prelude::*;

type Arr = Array1<f64>;

#[derive(Debug)]
pub struct MyOracle {
    idx: usize,
}

impl Default for MyOracle {
    #[inline]
    fn default() -> Self {
        MyOracle {
            idx: 0,
        }
    }
}

impl OracleOptim<Arr> for MyOracle {
    type CutChoices = f64; // single cut

    /// * @brief
    ///  *
    ///  * @param[in] z
    ///  * @param[in,out] gamma
    ///  * @return std::tuple<Cut, double>
    fn assess_optim(&mut self, z: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
        let sqrtx = z[0];
        let ly = z[1];

        for _ in 0..2 {
            self.idx += 1;
            if self.idx == 2 {
                self.idx = 0; // round robin
            }
            let fj = match self.idx {
                0 => sqrtx * sqrtx - ly,
                1 => -sqrtx + *gamma * ly.exp(),
                _ => unreachable!(),
            };
            if fj > 0.0 {
                return (
                    (
                        match self.idx {
                            0 => array![2.0 * sqrtx, -1.0],
                            1 => array![-1.0, *gamma * ly.exp()],
                            _ => unreachable!(),
                        },
                        fj,
                    ),
                    false,
                );
            }
        }
        return ((array![-1.0, sqrtx], 0.0), true);
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
        assert_eq!(num_iters, 47); // why not 35?
    }

    #[test]
    pub fn test_feasible_stable() {
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
        assert_eq!(num_iters, 47); // why not 35?
    }
}
