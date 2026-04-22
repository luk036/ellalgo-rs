use crate::cutting_plane::OracleOptim;
use ndarray::prelude::*;

type Arr = Array1<f64>;

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
        let w = xc.slice(s![0..n]);
        let b = xc[n];

        let mut min_val = f64::INFINITY;
        let mut min_idx = 0;

        for i in 0..self.data.nrows() {
            let y_i = self.labels[i] as f64;
            let x_i = self.data.row(i);
            let margin = y_i * (w.dot(&x_i) + b);
            if margin < min_val {
                min_val = margin;
                min_idx = i;
            }
        }

        if min_val >= 1.0 {
            *gamma = 0.0;
            return ((Arr::zeros(n + 1), 0.0), true);
        }

        // SVM subgradient: -y_i * x_i for w, -y_i for b
        let y_i = self.labels[min_idx] as f64;
        let x_i = self.data.row(min_idx);
        let grad_vec: Vec<f64> = x_i
            .iter()
            .map(|&x| -y_i * x)
            .chain(std::iter::once(-y_i))
            .collect();
        let grad_with_b = Array1::from_vec(grad_vec);

        *gamma = min_val;
        ((grad_with_b, min_val), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svm_oracle() {
        let data = array![[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]];
        let labels = array![1, 1, -1, -1];
        let mut oracle = SvmOracle::new(data, labels);

        let mut gamma = f64::NEG_INFINITY;
        let xc = array![0.0, 0.0, 0.0];
        let ((_grad, _beta), improved) = oracle.assess_optim(&xc, &mut gamma);

        assert!(improved);
    }
}
