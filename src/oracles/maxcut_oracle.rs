use crate::cutting_plane::OracleOptim;
use ndarray::prelude::*;

type Arr = Array1<f64>;

/// Max-cut oracle
pub struct MaxcutOracle {
    weights: Array2<f64>,
    n: usize,
}

impl MaxcutOracle {
    pub fn new(weights: Array2<f64>) -> Self {
        let n = weights.nrows();
        assert_eq!(n, weights.ncols(), "weight matrix must be square");
        Self { weights, n }
    }
}

impl OracleOptim<Arr> for MaxcutOracle {
    type CutChoice = f64;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
        let x = xc.mapv(|v| if v >= 0.0 { 1.0 } else { -1.0 });

        // cut weight = sum w_ij where x_i != x_j
        let mut cut_value = 0.0;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if x[i] != x[j] {
                    cut_value += self.weights[[i, j]];
                }
            }
        }

        // grad_i = sum w_ij where x_i != x_j
        let mut grad = Arr::zeros(self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                if x[i] != x[j] {
                    grad[i] += self.weights[[i, j]];
                }
            }
        }
        grad *= 2.0;

        if cut_value > *gamma {
            *gamma = cut_value;
            ((-grad, -cut_value), true)
        } else {
            ((-grad, *gamma), false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxcut_oracle() {
        let w = array![[0.0, 1.0], [1.0, 0.0]];
        let mut oracle = MaxcutOracle::new(w);

        let mut gamma = f64::NEG_INFINITY;
        let xc = array![1.0, 1.0];
        let ((_grad, _beta), improved) = oracle.assess_optim(&xc, &mut gamma);

        assert!(improved);
    }

    #[test]
    fn test_maxcut_oracle_not_optimal() {
        let w = array![[0.0, 1.0], [1.0, 0.0]];
        let mut oracle = MaxcutOracle::new(w);

        // First call sets gamma to some value
        let mut gamma = f64::NEG_INFINITY;
        let xc = array![1.0, 1.0];
        let ((_grad, _beta), improved) = oracle.assess_optim(&xc, &mut gamma);
        assert!(improved);
        assert!(gamma > f64::NEG_INFINITY);

        // Second call with same x - should not improve since same cut value
        let xc2 = array![1.0, 1.0];
        let ((_grad2, _beta2), improved2) = oracle.assess_optim(&xc2, &mut gamma);
        // Same value should not improve
        assert!(!improved2);
    }
}
