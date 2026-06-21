use crate::arr::Arr;
use crate::cutting_plane::{CutStatus, ParallelCut, SearchSpace, SingleCut, UpdateByCutChoice};
use crate::ell_calc::EllCalc;

/// Numerically stable ellipsoid search space using LDL^T factorization.
///
/// `EllStable` = {x | (x - xc)^T mq^-1 (x - xc) ≤ κ}
#[derive(Debug, Clone)]
pub struct EllStable {
    mq: Arr,
    xc: Arr,
    kappa: f64,
    helper: EllCalc,
    tsq: f64,
}

impl EllStable {
    pub fn new_with_matrix(kappa: f64, mq: Arr, xc: Arr) -> EllStable {
        let helper = EllCalc::new(xc.len());
        EllStable {
            kappa,
            mq,
            xc,
            helper,
            tsq: 0.0,
        }
    }

    pub fn new(val: Arr, xc: Arr) -> EllStable {
        EllStable::new_with_matrix(1.0, Arr::from_diag(&val), xc)
    }

    pub fn new_with_scalar(val: f64, xc: Arr) -> EllStable {
        EllStable::new_with_matrix(val, Arr::eye(xc.len()), xc)
    }

    /// Update the ellipsoid using $$ LDL^T $$ factorization.
    ///
    /// The shape matrix is stored as $$ M = \kappa LDL^T $$.
    ///
    /// $$
    /// \begin{aligned}
    /// w &= L^{-1}g \\\\
    /// z &= D^{-1}w \\\\
    /// \omega &= w^T z = \sum_i w_i z_i \\\\
    /// q &= L^{-T}z \\\\
    /// x_c &\leftarrow x_c - \frac{\rho}{\omega}\,q
    /// \end{aligned}
    /// $$
    ///
    /// The rank-one update modifies the $$ LDL^T $$ factors directly.
    fn update_core<T, F>(&mut self, grad: &Arr, beta: &T, f_core: F) -> CutStatus
    where
        T: UpdateByCutChoice<Self, ArrayType = Arr>,
        F: FnOnce(&T, f64) -> (CutStatus, (f64, f64, f64)),
    {
        let ndim = self.xc.len();
        let n = ndim;

        // calculate inv(L)*grad
        let mut inv_ml_g = grad.clone();
        for i in 1..ndim {
            let row_start_i = i * n;
            for j in 0..i {
                let val = self.mq.at(j, i) * inv_ml_g[j];
                self.mq.data_mut()[row_start_i + j] = val;
                inv_ml_g[i] -= val;
            }
        }

        // calculate inv(D)*inv(L)*grad
        let mut inv_md_inv_ml_g = inv_ml_g.clone();
        for i in 0..ndim {
            inv_md_inv_ml_g[i] *= self.mq.at(i, i);
        }

        // calculate omega
        let mut gg_t = inv_md_inv_ml_g.clone();
        let mut omega = 0.0;
        for i in 0..ndim {
            gg_t[i] *= inv_ml_g[i];
            omega += gg_t[i];
        }

        self.tsq = self.kappa * omega;
        let (status, (rho, sigma, delta)) = f_core(beta, self.tsq);

        if status != CutStatus::Success {
            return status;
        }

        // calculate mq*grad = inv(L')*inv(D)*inv(L)*grad
        let mut g_t = inv_md_inv_ml_g.clone();
        for i in (1..ndim).rev() {
            for j in i..ndim {
                g_t[i - 1] -= self.mq.at(j, i - 1) * g_t[j];
            }
        }

        // calculate xc
        let rho_over_omega = rho / omega;
        for i in 0..ndim {
            self.xc[i] -= rho_over_omega * g_t[i];
        }

        // Rank-one update
        let mu_val = sigma / (1.0 - sigma);
        let mut oldt = omega / mu_val;
        let last_idx = ndim - 1;
        for j in 0..last_idx {
            let temp = oldt + gg_t[j];
            let beta2 = inv_md_inv_ml_g[j] / temp;
            self.mq.data_mut()[j * n + j] *= oldt / temp;
            let row_start_j = j * n;
            for l in (j + 1)..ndim {
                self.mq.data_mut()[row_start_j + l] += beta2 * self.mq.at(l, j);
            }
            oldt = temp;
        }
        let temp = oldt + gg_t[last_idx];
        self.mq.data_mut()[last_idx * n + last_idx] *= oldt / temp;
        self.kappa *= delta;

        status
    }
}

impl SearchSpace for EllStable {
    type ArrayType = Arr;

    fn xc(&self) -> Self::ArrayType {
        self.xc.clone()
    }

    fn tsq(&self) -> f64 {
        self.tsq
    }

    fn update_bias_cut<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoice<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_bias_cut_by(self, grad)
    }

    fn update_central_cut<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoice<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_central_cut_by(self, grad)
    }

    fn update_q<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoice<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_q_by(self, grad)
    }

    fn set_xc(&mut self, x: Self::ArrayType) {
        self.xc = x;
    }
}

impl UpdateByCutChoice<EllStable> for SingleCut {
    type ArrayType = Arr;

    fn update_bias_cut_by(&self, ellip: &mut EllStable, grad: &Self::ArrayType) -> CutStatus {
        let helper = ellip.helper.clone();
        ellip.update_core(grad, self, |beta, tsq| helper.calc_bias_cut(beta.0, tsq))
    }

    fn update_central_cut_by(&self, ellip: &mut EllStable, grad: &Self::ArrayType) -> CutStatus {
        let helper = ellip.helper.clone();
        ellip.update_core(grad, self, |_beta, tsq| helper.calc_central_cut(tsq))
    }

    fn update_q_by(&self, ellip: &mut EllStable, grad: &Self::ArrayType) -> CutStatus {
        let helper = ellip.helper.clone();
        ellip.update_core(grad, self, |beta, tsq| helper.calc_bias_cut_q(beta.0, tsq))
    }
}

impl UpdateByCutChoice<EllStable> for ParallelCut {
    type ArrayType = Arr;

    fn update_bias_cut_by(&self, ellip: &mut EllStable, grad: &Self::ArrayType) -> CutStatus {
        let helper = ellip.helper.clone();
        ellip.update_core(grad, self, |beta, tsq| {
            helper.calc_single_or_parallel_bias_cut(beta, tsq)
        })
    }

    fn update_central_cut_by(&self, ellip: &mut EllStable, grad: &Self::ArrayType) -> CutStatus {
        let helper = ellip.helper.clone();
        ellip.update_core(grad, self, |beta, tsq| {
            helper.calc_single_or_parallel_central_cut(beta, tsq)
        })
    }

    fn update_q_by(&self, ellip: &mut EllStable, grad: &Self::ArrayType) -> CutStatus {
        let helper = ellip.helper.clone();
        ellip.update_core(grad, self, |beta, tsq| {
            helper.calc_single_or_parallel_q(beta, tsq)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_construct() {
        let ellip = EllStable::new_with_scalar(0.01, Arr::new(4));
        assert_approx_eq!(ellip.kappa, 0.01);
        assert_eq!(ellip.xc, Arr::new(4));
        assert_approx_eq!(ellip.tsq, 0.0);
    }

    #[test]
    fn test_update_central_cut() {
        let mut ellip = EllStable::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), SingleCut(0.0));
        let status = ellip.update_central_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_eq!(ellip.xc, -0.01 * Arr::ones(4));
        assert_approx_eq!(ellip.kappa, 0.16 / 15.0);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_bias_cut() {
        let mut ellip = EllStable::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), SingleCut(0.05));
        let status = ellip.update_bias_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(ellip.xc[0], -0.03);
        assert_approx_eq!(ellip.kappa, 0.008);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_parallel_central_cut() {
        let mut ellip = EllStable::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), ParallelCut(0.0, Some(0.05)));
        let status = ellip.update_central_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_eq!(ellip.xc, -0.01 * Arr::ones(4));
        assert_approx_eq!(ellip.kappa, 0.012);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_parallel() {
        let mut ellip = EllStable::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), ParallelCut(0.01, Some(0.04)));
        let status = ellip.update_bias_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(ellip.xc[0], -0.0116);
        assert_approx_eq!(ellip.kappa, 0.01232);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_parallel_no_effect() {
        let mut ellip = EllStable::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), ParallelCut(-0.04, Some(0.0625)));
        let status = ellip.update_bias_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_eq!(ellip.xc, Arr::new(4));
        assert_approx_eq!(ellip.kappa, 0.01);
    }

    #[test]
    fn test_update_q_no_effect() {
        let mut ellip = EllStable::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), ParallelCut(-0.04, Some(0.0625)));
        let status = ellip.update_q(&cut);
        assert_eq!(status, CutStatus::NoEffect);
        assert_eq!(ellip.xc, Arr::new(4));
        assert_approx_eq!(ellip.kappa, 0.01);
    }

    #[test]
    fn test_update_q() {
        let mut ellip = EllStable::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), ParallelCut(0.01, Some(0.04)));
        let status = ellip.update_q(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(ellip.xc[0], -0.0116);
        assert_approx_eq!(ellip.kappa, 0.01232);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_new() {
        let val = Arr::from(vec![1.0, 1.0]);
        let xc = Arr::from(vec![0.0, 0.0]);
        let ellip = EllStable::new(val, xc);
        assert_eq!(ellip.kappa, 1.0);
        assert_eq!(ellip.mq, Arr::eye(2));
    }
}
