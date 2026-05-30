use super::cutting_plane::OracleFeas;
use crate::arr::Arr;
use crate::cutting_plane::SingleCut;

#[derive(Debug)]
pub struct MyOracle3 {
    pub idx: i32,
    pub target: f64,
}

impl Default for MyOracle3 {
    #[inline]
    fn default() -> Self {
        MyOracle3 {
            idx: -1,
            target: -1e100,
        }
    }
}

impl OracleFeas<Arr> for MyOracle3 {
    type CutChoice = SingleCut;

    fn assess_feas(&mut self, xc: &Arr) -> Option<(Arr, SingleCut)> {
        let x_val = xc[0];
        let y_val = xc[1];

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
                3 => 2.0 * x_val - 3.0 * y_val - self.target,
                _ => unreachable!(),
            };
            if fj > 0.0 {
                return Some((
                    match self.idx {
                        0 => Arr::from(vec![-1.0, 0.0]),
                        1 => Arr::from(vec![0.0, -1.0]),
                        2 => Arr::from(vec![1.0, 1.0]),
                        3 => Arr::from(vec![2.0, -3.0]),
                        _ => unreachable!(),
                    },
                    SingleCut(fj),
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
    use crate::cutting_plane::{bsearch, BSearchAdaptor, Options};
    use crate::ell::Ell;

    #[test]
    pub fn test_feasible() {
        let ellip = Ell::new_with_scalar(100.0, Arr::from(vec![0.0, 0.0]));
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
