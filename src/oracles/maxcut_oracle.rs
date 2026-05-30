use crate::arr::Arr;
use crate::cutting_plane::{OracleOptim, SingleCut};

pub struct MaxcutOracle {
    weights: Arr,
    n: usize,
}

impl MaxcutOracle {
    pub fn new(weights: Arr) -> Self {
        let n = weights.rows();
        assert_eq!(n, weights.cols(), "weight matrix must be square");
        assert!(weights.is_2d(), "weights must be a 2D matrix");
        Self { weights, n }
    }
}

impl OracleOptim<Arr> for MaxcutOracle {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
        let x = xc.map(|v| if v >= 0.0 { 1.0 } else { -1.0 });

        let mut cut_value = 0.0;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if x[i] != x[j] {
                    cut_value += self.weights[(i, j)];
                }
            }
        }

        let mut grad = Arr::new(self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                if x[i] != x[j] {
                    grad[i] += self.weights[(i, j)];
                }
            }
        }
        grad *= 2.0;

        if cut_value > *gamma {
            *gamma = cut_value;
            ((-grad, SingleCut(-cut_value)), true)
        } else {
            ((-grad, SingleCut(*gamma)), false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arr::Arr;

    #[test]
    fn test_maxcut_oracle() {
        let w = Arr::from_shape_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
        let mut oracle = MaxcutOracle::new(w);

        let mut gamma = f64::NEG_INFINITY;
        let xc = Arr::from(vec![1.0, 1.0]);
        let ((_grad, _beta), improved) = oracle.assess_optim(&xc, &mut gamma);

        assert!(improved);
    }

    #[test]
    fn test_maxcut_oracle_not_optimal() {
        let w = Arr::from_shape_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
        let mut oracle = MaxcutOracle::new(w);

        let mut gamma = f64::NEG_INFINITY;
        let xc = Arr::from(vec![1.0, 1.0]);
        let ((_grad, _beta), improved) = oracle.assess_optim(&xc, &mut gamma);
        assert!(improved);
        assert!(gamma > f64::NEG_INFINITY);

        let xc2 = Arr::from(vec![1.0, 1.0]);
        let ((_grad2, _beta2), improved2) = oracle.assess_optim(&xc2, &mut gamma);
        assert!(!improved2);
    }
}
