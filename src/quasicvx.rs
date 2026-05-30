use super::cutting_plane::OracleOptim;
use crate::arr::Arr;
use crate::cutting_plane::SingleCut;

#[derive(Debug)]
pub struct MyOracle {
    idx: i32,
}

impl Default for MyOracle {
    fn default() -> Self {
        MyOracle { idx: -1 }
    }
}

impl OracleOptim<Arr> for MyOracle {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
        let sqrtx = xc[0];
        let logy = xc[1];

        let num_constraints = 2;
        for _ in 0..num_constraints {
            self.idx += 1;
            if self.idx == num_constraints {
                self.idx = 0;
            }
            let func_val = match self.idx {
                0 => sqrtx * sqrtx - logy,
                1 => -sqrtx + *gamma * logy.exp(),
                _ => unreachable!(),
            };
            if func_val > 0.0 {
                return (
                    (
                        match self.idx {
                            0 => Arr::from(vec![2.0 * sqrtx, -1.0]),
                            1 => Arr::from(vec![-1.0, *gamma * logy.exp()]),
                            _ => unreachable!(),
                        },
                        SingleCut(func_val),
                    ),
                    false,
                );
            }
        }
        *gamma = sqrtx / logy.exp();
        ((Arr::from(vec![-1.0, sqrtx]), SingleCut(0.0)), true)
    }
}

#[cfg(test)]
mod tests {
    use super::MyOracle;
    use crate::arr::Arr;
    use crate::cutting_plane::{cutting_plane_optim, Options};
    use crate::ell::Ell;

    #[test]
    pub fn test_feasible() {
        let mut ell = Ell::new(Arr::from(vec![10.0, 10.0]), Arr::from(vec![0.0, 0.0]));
        let mut oracle = MyOracle::default();
        let mut gamma = 0.0;
        let options = Options {
            max_iters: 2000,
            tolerance: 1e-8,
            verbose: false,
        };
        let (x_opt, num_iters) = cutting_plane_optim(&mut oracle, &mut ell, &mut gamma, &options);
        assert!(x_opt.is_some());
        if let Some(x) = x_opt {
            assert!(x[0] * x[0] >= 0.49 && x[0] * x[0] <= 0.51);
            assert!(x[1].exp() >= 1.6 && x[1].exp() <= 1.7);
        }
        assert_eq!(num_iters, 35);
    }
}
