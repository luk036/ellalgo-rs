use crate::arr::Arr;
use crate::cutting_plane::OracleOptim;
use ndarray::{Array1, Array2};

/// Hard-margin SVM oracle
pub struct SvmOracle {
    data: Array2<f64>,
    labels: Array1<i32>,
    nfeat: usize,
}

impl SvmOracle {
    pub fn new(data: Array2<f64>, labels: Array1<i32>) -> Self {
        let nfeat = data.ncols();
        Self {
            data,
            labels,
            nfeat,
        }
    }
}

impl OracleOptim<Arr> for SvmOracle {
    type CutChoice = f64;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
        let n = self.nfeat;
        let w = Arr::from_fn(n, |i| xc[i]);
        let b = xc[n];

        let mut min_val = f64::INFINITY;
        let mut min_idx = 0;

        for i in 0..self.data.nrows() {
            let y_i = self.labels[i] as f64;
            let x_i = self.data.row(i);
            let xi_arr = Arr::from(x_i.to_vec());
            let margin = y_i * (w.dot(&xi_arr) + b);
            if margin < min_val {
                min_val = margin;
                min_idx = i;
            }
        }

        if min_val >= 1.0 {
            *gamma = 0.0;
            return ((Arr::new(n + 1), 0.0), true);
        }

        // SVM subgradient: -y_i * x_i for w, -y_i for b
        let y_i = self.labels[min_idx] as f64;
        let x_i = self.data.row(min_idx);
        let grad_vec: Vec<f64> = x_i
            .iter()
            .map(|&x| -y_i * x)
            .chain(std::iter::once(-y_i))
            .collect();
        let grad_with_b = Arr::from(grad_vec);

        *gamma = min_val;
        ((grad_with_b, min_val), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arr::Arr;
    use ndarray::array;

    #[test]
    fn test_svm_oracle() {
        let data = array![[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]];
        let labels = array![1, 1, -1, -1];
        let mut oracle = SvmOracle::new(data, labels);

        let mut gamma = f64::NEG_INFINITY;
        let xc = Arr::from(vec![0.0, 0.0, 0.0]);
        let ((_grad, _beta), improved) = oracle.assess_optim(&xc, &mut gamma);

        assert!(improved);
    }

    #[test]
    fn test_svm_oracle_optimal() {
        let data = array![[1.0, 0.0], [-1.0, 0.0]];
        let labels = array![1, -1];
        let mut oracle = SvmOracle::new(data, labels);

        let mut gamma = f64::NEG_INFINITY;
        let xc = Arr::from(vec![1.0, 0.0, 0.0]);
        let ((_grad, _beta), improved) = oracle.assess_optim(&xc, &mut gamma);
        assert!(improved);
        assert_eq!(gamma, 0.0);
    }
}
