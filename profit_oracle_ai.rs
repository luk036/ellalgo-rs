use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut1;
use ndarray::ArrayViewMut2;
use ndarray::s;
use ndarray::stack;
use ndarray::Axis;
use ndarray::Zip;
use ndarray::concatenate;
use ndarray::arr1;
use ndarray::arr2;
use ndarray::ScalarOperand;
use ndarray::ZipExt;
use ndarray::IntoDimension;
use ndarray::Dimension;
use ndarray::RemoveAxis;
use ndarray::AsArray;
use ndarray::AsArrayError;
use ndarray::Linalg;
use ndarray::linalg::Dot;
use ndarray::linalg::general_mat_vec_mul;
use ndarray::linalg::general_mat_mul;
use ndarray::linalg::general_dot;
use ndarray::linalg::general_mat_mat_mul;
use ndarray::linalg::general_mat_vec_mul_to;
use ndarray::linalg::general_mat_mul_to;
use ndarray::linalg::general_dot_to;
use ndarray::linalg::general_mat_mat_mul_to;
use ndarray::linalg::general_mat_vec_mul_into;
use ndarray::linalg::general_mat_mul_into;
use ndarray::linalg::general_dot_into;
use ndarray::linalg::general_mat_mat_mul_into;
use ndarray::linalg::general_mat_vec_mul_tr;
use ndarray::linalg::general_mat_mul_tr;
use ndarray::linalg::general_dot_tr;
use ndarray::linalg::general_mat_mat_mul_tr;
use ndarray::linalg::general_mat_vec_mul_to_tr;
use ndarray::linalg::general_mat_mul_to_tr;
use ndarray::linalg::general_dot_to_tr;
use ndarray::linalg::general_mat_mat_mul_to_tr;
use ndarray::linalg::general_mat_vec_mul_into_tr;
use ndarray::linalg::general_mat_mul_into_tr;
use ndarray::linalg::general_dot_into_tr;
use ndarray::linalg::general_mat_mat_mul_into_tr;

type Arr = Array2<f64>;
type Cut = (Arr, f64);

struct ProfitOracle {
    log_pA: f64,
    log_k: f64,
    price_out: Arr,
    elasticities: Arr,
}

impl OracleOptim for ProfitOracle {
    fn assess_optim(&self, y: ArrayView1<f64>, target: f64) -> (Cut, Option<f64>) {
        let fj: f64;
        let mut g: Arr;
        let log_Cobb = self.log_pA + self.elasticities.dot(&y);
        let q = &self.price_out * &y.mapv(f64::exp);
        let vx = q.slice(s![0]) + q.slice(s![1]);
        if (fj = y[0] - self.log_k) > 0.0 {
            g = arr2(&[[1.0, 0.0]]);
            return ((g, fj), None);
        }
        if (fj = f64::ln(target + vx) - log_Cobb) >= 0.0 {
            g = q / (target + vx) - &self.elasticities;
            return ((g, fj), None);
        }
        let target = f64::exp(log_Cobb) - vx;
        g = q / (target + vx) - &self.elasticities;
        ((g, 0.0), Some(target))
    }
}

fn main() {
    let params = (1.0, 2.0, 3.0);
    let elasticities = arr1(&[1.0, 2.0]);
    let price_out = arr1(&[3.0, 4.0]);
    let oracle = ProfitOracle {
        log_pA: f64::ln(params.0 * params.1),
        log_k: f64::ln(params.2),
        price_out: stack(Axis(0), &[&price_out, &price_out]).unwrap(),
        elasticities: elasticities.into_shape((1, 2)).unwrap(),
    };
    let y = arr1(&[1.0, 2.0]);
    let target = 3.0;
    let (cut, opt) = oracle.assess_optim(y.view(), target);
}

