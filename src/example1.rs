use super::cutting_plane::OracleOptim;
use crate::arr::Arr;
use crate::cutting_plane::SingleCut;

#[derive(Debug, Default)]
pub struct MyOracle;

impl OracleOptim<Arr> for MyOracle {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
        let x_val = xc[0];
        let y_val = xc[1];
        let f0 = x_val + y_val;
        let f1 = f0 - 3.0;
        if f1 > 0.0 {
            return ((Arr::from(vec![1.0, 1.0]), SingleCut(f1)), false);
        }
        let f2 = -x_val + y_val + 1.0;
        if f2 > 0.0 {
            return ((Arr::from(vec![-1.0, 1.0]), SingleCut(f2)), false);
        }
        let f3 = *gamma - f0;
        if f3 > 0.0 {
            return ((Arr::from(vec![-1.0, -1.0]), SingleCut(f3)), false);
        }
        *gamma = f0;
        ((Arr::from(vec![-1.0, -1.0]), SingleCut(0.0)), true)
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
        let mut oracle = MyOracle;
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
        let mut ellip = Ell::new(Arr::from(vec![10.0, 10.0]), Arr::from(vec![100.0, 100.0]));
        let mut oracle = MyOracle;
        let mut gamma = f64::NEG_INFINITY;
        let options = Options::default();
        let (xbest, _num_iters) =
            cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);
        assert!(xbest.is_none());
    }

    #[test]
    pub fn test_infeasible2() {
        let mut ellip = Ell::new(Arr::from(vec![10.0, 10.0]), Arr::from(vec![0.0, 0.0]));
        let mut oracle = MyOracle;
        let options = Options::default();
        let (xbest, _niter) = cutting_plane_optim(&mut oracle, &mut ellip, &mut 100.0, &options);
        assert!(xbest.is_none());
    }
}
