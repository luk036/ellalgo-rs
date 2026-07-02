use crate::arr::Arr;
use crate::cutting_plane::{OracleOptim, SingleCut};

pub struct SvmOracle {
    data: Arr,
    labels: Vec<i32>,
    nfeat: usize,
}

impl SvmOracle {
    pub fn new(data: Arr, labels: Vec<i32>) -> Self {
        let nfeat = data.cols();
        Self {
            data,
            labels,
            nfeat,
        }
    }
}

impl OracleOptim<Arr> for SvmOracle {
    type CutChoice = SingleCut;

    /// Assess SVM margin: $$ y_i (w^T x_i + b) \ge 1 $$
    ///
    /// Maximizes the margin by minimizing $$ \|w\| $$ subject to
    /// correct classification. Returns the gradient of the most violated constraint.
    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
        let n = self.nfeat;
        let w = Arr::from_fn(n, |i| xc[i]);
        let b = xc[n];

        let mut min_val = f64::INFINITY;
        let mut min_idx = 0;

        for i in 0..self.data.rows() {
            let y_i = self.labels[i] as f64;
            let xi_arr = self.data.row(i);
            let margin = y_i * (w.dot(&xi_arr) + b);
            if margin < min_val {
                min_val = margin;
                min_idx = i;
            }
        }

        if min_val >= 1.0 {
            *gamma = 0.0;
            return ((Arr::new(n + 1), SingleCut(0.0)), true);
        }

        let y_i = self.labels[min_idx] as f64;
        let x_i = self.data.row(min_idx);
        let grad_vec: Vec<f64> = x_i
            .iter()
            .map(|&x| -y_i * x)
            .chain(std::iter::once(-y_i))
            .collect();
        let grad_with_b = Arr::from(grad_vec);

        *gamma = min_val;
        ((grad_with_b, SingleCut(min_val)), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arr::Arr;

    #[test]
    fn test_svm_oracle() {
        let data = Arr::from_shape_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let labels = vec![1, 1, -1, -1];
        let mut oracle = SvmOracle::new(data, labels);

        let mut gamma = f64::NEG_INFINITY;
        let xc = Arr::from(vec![0.0, 0.0, 0.0]);
        let ((_grad, _beta), improved) = oracle.assess_optim(&xc, &mut gamma);

        assert!(improved);
    }

    #[test]
    fn test_svm_oracle_optimal() {
        let data = Arr::from_shape_vec(2, 2, vec![1.0, 0.0, -1.0, 0.0]);
        let labels = vec![1, -1];
        let mut oracle = SvmOracle::new(data, labels);

        let mut gamma = f64::NEG_INFINITY;
        let xc = Arr::from(vec![1.0, 0.0, 0.0]);
        let ((_grad, _beta), improved) = oracle.assess_optim(&xc, &mut gamma);
        assert!(improved);
        assert_eq!(gamma, 0.0);
    }
}
