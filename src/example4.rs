use super::cutting_plane::OracleOptim;
use crate::arr::Arr;
use crate::cutting_plane::SingleCut;

#[derive(Debug)]
pub struct MyOracle {
    idx: i32,
}

impl Default for MyOracle {
    #[inline]
    fn default() -> Self {
        MyOracle { idx: -1 }
    }
}

impl OracleOptim<Arr> for MyOracle {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
        let x_val = xc[0];
        let y_val = xc[1];
        let f0 = 2.0 * x_val - 3.0 * y_val;

        let num_constraints = 4;
        for _ in 0..num_constraints {
            self.idx += 1;
            if self.idx == num_constraints {
                self.idx = 0;
            }
            let fj = match self.idx {
                0 => -x_val - 1.0,
                1 => -y_val - 2.0,
                2 => x_val + y_val - 1.0,
                3 => *gamma - f0,
                _ => unreachable!(),
            };
            if fj > 0.0 {
                return (
                    (
                        match self.idx {
                            0 => Arr::from(vec![-1.0, 0.0]),
                            1 => Arr::from(vec![0.0, -1.0]),
                            2 => Arr::from(vec![1.0, 1.0]),
                            3 => Arr::from(vec![-2.0, 3.0]),
                            _ => unreachable!(),
                        },
                        SingleCut(fj),
                    ),
                    false,
                );
            }
        }
        *gamma = f0;
        ((Arr::from(vec![-2.0, 3.0]), SingleCut(0.0)), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cutting_plane::{cutting_plane_optim, Options};
    use crate::ell::Ell;

    #[test]
    pub fn test_feasible() {
        let mut ellip = Ell::new_with_scalar(10.0, Arr::from(vec![0.0, 0.0]));
        let mut oracle = MyOracle::default();
        let mut gamma = f64::NEG_INFINITY;
        let options = Options {
            tolerance: 1e-10,
            ..Default::default()
        };
        let (xbest, num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);
        assert!(xbest.is_some());
        assert_eq!(num_iters, 82);
    }
}
