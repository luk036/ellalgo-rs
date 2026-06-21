use crate::arr::Arr;
use crate::cutting_plane::{CutStatus, ParallelCut, SearchSpace, SingleCut, UpdateByCutChoice};
use crate::ell_calc::EllCalc;

/// Ellipsoid Search Space
///
/// $$ \text{Ell} = \\{x \mid (x - x_c)^T M_Q^{-1} (x - x_c) \le \kappa \\} $$
#[derive(Debug, Clone)]
pub struct Ell {
    pub no_defer_trick: bool,
    pub mq: Arr,
    pub xc: Arr,
    pub kappa: f64,
    helper: EllCalc,
    pub tsq: f64,
}

impl Ell {
    /// Construct a new `Ell` from a matrix and center.
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::arr::Arr;
    /// use ellalgo_rs::ell::Ell;
    /// let mq = Arr::eye(2);
    /// let xc = Arr::from(vec![0.0, 0.0]);
    /// let ellip = Ell::new_with_matrix(1.0, mq, xc);
    /// assert_eq!(ellip.kappa, 1.0);
    /// ```
    pub fn new_with_matrix(kappa: f64, mq: Arr, xc: Arr) -> Ell {
        let helper = EllCalc::new(xc.len());
        Ell {
            kappa,
            mq,
            xc,
            helper,
            no_defer_trick: false,
            tsq: 0.0,
        }
    }

    /// Create a new `Ell` with diagonal matrix from `val`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::arr::Arr;
    /// use ellalgo_rs::ell::Ell;
    /// let val = Arr::from(vec![1.0, 1.0]);
    /// let xc = Arr::from(vec![0.0, 0.0]);
    /// let ellip = Ell::new(val, xc);
    /// assert_eq!(ellip.kappa, 1.0);
    /// ```
    pub fn new(val: Arr, xc: Arr) -> Ell {
        Ell::new_with_matrix(1.0, Arr::from_diag(&val), xc)
    }

    /// Create a new `Ell` with scalar value and center.
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::arr::Arr;
    /// use ellalgo_rs::ell::Ell;
    /// let val = 1.0;
    /// let xc = Arr::from(vec![0.0, 0.0]);
    /// let ellip = Ell::new_with_scalar(val, xc);
    /// assert_eq!(ellip.kappa, 1.0);
    /// ```
    pub fn new_with_scalar(val: f64, xc: Arr) -> Ell {
        Ell::new_with_matrix(val, Arr::eye(xc.len()), xc)
    }

    /// Construct from a covariance matrix.
    pub fn from_covariance(cov: Arr, xc: Arr) -> Ell {
        Ell::new_with_matrix(1.0, cov, xc)
    }

    /// Update the ellipsoid using a gradient cut.
    ///
    /// Given a gradient $$g$$ and offset $$ \beta $$, the ellipsoid
    /// $$ \{ x : (x - x_c)^T M^{-1} (x - x_c) \le \kappa \} $$ is updated:
    ///
    /// $$
    /// \begin{aligned}
    /// \tilde{g} &= M\,g \\\\
    /// \omega &= g^T \tilde{g} \\\\
    /// \tau^2 &= \kappa\,\omega \\\\
    /// x_c &\leftarrow x_c - \frac{\rho}{\omega}\,\tilde{g} \\\\
    /// M &\leftarrow M - \frac{\sigma}{\omega}\,\tilde{g}\,\tilde{g}^T \\\\
    /// \kappa &\leftarrow \delta\,\kappa
    /// \end{aligned}
    /// $$
    ///
    /// where $$ \rho, \sigma, \delta $$ are returned by the cut strategy.
    fn update_core<T, F>(&mut self, grad: &Arr, beta: &T, cut_strategy: F) -> CutStatus
    where
        T: UpdateByCutChoice<Self, ArrayType = Arr>,
        F: FnOnce(&T, f64) -> (CutStatus, (f64, f64, f64)),
    {
        let grad_t = self.mq.dot_mv(grad);
        let omega = grad.dot(&grad_t);

        self.tsq = self.kappa * omega;
        let (status, (rho, sigma, delta)) = cut_strategy(beta, self.tsq);
        if status != CutStatus::Success {
            return status;
        }

        let n = self.xc.len();
        let rho_over_omega = rho / omega;
        for i in 0..n {
            self.xc[i] -= rho_over_omega * grad_t[i];
        }

        let ratio = sigma / omega;
        for i in 0..n {
            let r_qg = ratio * grad_t[i];
            for j in 0..=i {
                let update = r_qg * grad_t[j];
                let idx = i * n + j;
                self.mq.data_mut()[idx] -= update;
                if i != j {
                    self.mq.data_mut()[j * n + i] = self.mq.data()[idx];
                }
            }
        }

        self.kappa *= delta;

        if self.no_defer_trick {
            self.mq *= self.kappa;
            self.kappa = 1.0;
        }
        status
    }
}

impl SearchSpace for Ell {
    type ArrayType = Arr;

    #[inline]
    fn xc(&self) -> Self::ArrayType {
        self.xc.clone()
    }

    #[inline]
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

trait CutType {
    fn call_bias_cut(&self, helper: &EllCalc, tsq: f64) -> (CutStatus, (f64, f64, f64));
    fn call_central_cut(&self, helper: &EllCalc, tsq: f64) -> (CutStatus, (f64, f64, f64));
    fn call_q_cut(&self, helper: &EllCalc, tsq: f64) -> (CutStatus, (f64, f64, f64));
}

impl CutType for SingleCut {
    fn call_bias_cut(&self, helper: &EllCalc, tsq: f64) -> (CutStatus, (f64, f64, f64)) {
        helper.calc_bias_cut(self.0, tsq)
    }
    fn call_central_cut(&self, helper: &EllCalc, tsq: f64) -> (CutStatus, (f64, f64, f64)) {
        helper.calc_central_cut(tsq)
    }
    fn call_q_cut(&self, helper: &EllCalc, tsq: f64) -> (CutStatus, (f64, f64, f64)) {
        helper.calc_bias_cut_q(self.0, tsq)
    }
}

impl CutType for ParallelCut {
    fn call_bias_cut(&self, helper: &EllCalc, tsq: f64) -> (CutStatus, (f64, f64, f64)) {
        helper.calc_single_or_parallel_bias_cut(self, tsq)
    }
    fn call_central_cut(&self, helper: &EllCalc, tsq: f64) -> (CutStatus, (f64, f64, f64)) {
        helper.calc_single_or_parallel_central_cut(self, tsq)
    }
    fn call_q_cut(&self, helper: &EllCalc, tsq: f64) -> (CutStatus, (f64, f64, f64)) {
        helper.calc_single_or_parallel_q(self, tsq)
    }
}

impl<T: CutType> UpdateByCutChoice<Ell> for T {
    type ArrayType = Arr;

    fn update_bias_cut_by(&self, ellip: &mut Ell, grad: &Self::ArrayType) -> CutStatus {
        let helper = ellip.helper.clone();
        ellip.update_core(grad, self, |beta, tsq| beta.call_bias_cut(&helper, tsq))
    }

    fn update_central_cut_by(&self, ellip: &mut Ell, grad: &Self::ArrayType) -> CutStatus {
        let helper = ellip.helper.clone();
        ellip.update_core(grad, self, |beta, tsq| beta.call_central_cut(&helper, tsq))
    }

    fn update_q_by(&self, ellip: &mut Ell, grad: &Self::ArrayType) -> CutStatus {
        let helper = ellip.helper.clone();
        ellip.update_core(grad, self, |beta, tsq| beta.call_q_cut(&helper, tsq))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_construct() {
        let ellip = Ell::new_with_scalar(0.01, Arr::new(4));
        assert!(!ellip.no_defer_trick);
        assert_approx_eq!(ellip.kappa, 0.01);
        assert_eq!(ellip.mq, Arr::eye(4));
        assert_eq!(ellip.xc, Arr::new(4));
        assert_approx_eq!(ellip.tsq, 0.0);
    }

    #[test]
    fn test_update_central_cut() {
        let mut ellip = Ell::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), SingleCut(0.0));
        let status = ellip.update_central_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_eq!(ellip.xc, -0.01 * Arr::ones(4));
        assert_eq!(ellip.mq, &Arr::eye(4) - &(0.1 * Arr::full(4, 4, 1.0)));
        assert_approx_eq!(ellip.kappa, 0.16 / 15.0);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_bias_cut() {
        let mut ellip = Ell::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), SingleCut(0.05));
        let status = ellip.update_bias_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(ellip.xc[0], -0.03);
        assert_approx_eq!(ellip.mq.at(0, 0), 0.8);
        assert_approx_eq!(ellip.kappa, 0.008);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_parallel_central_cut() {
        let mut ellip = Ell::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), ParallelCut(0.0, Some(0.05)));
        let status = ellip.update_central_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_eq!(ellip.xc, -0.01 * Arr::ones(4));
        assert_eq!(ellip.mq, &Arr::eye(4) - &(0.2 * Arr::full(4, 4, 1.0)));
        assert_approx_eq!(ellip.kappa, 0.012);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_parallel() {
        let mut ellip = Ell::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), ParallelCut(0.01, Some(0.04)));
        let status = ellip.update_bias_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(ellip.xc[0], -0.0116);
        assert_approx_eq!(ellip.mq.at(0, 0), 1.0 - 0.232);
        assert_approx_eq!(ellip.kappa, 0.01232);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_parallel_no_effect() {
        let mut ellip = Ell::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), ParallelCut(-0.04, Some(0.0625)));
        let status = ellip.update_bias_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_eq!(ellip.xc, Arr::new(4));
        assert_eq!(ellip.mq, Arr::eye(4));
        assert_approx_eq!(ellip.kappa, 0.01);
    }

    #[test]
    fn test_update_q_no_effect() {
        let mut ellip = Ell::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), ParallelCut(-0.04, Some(0.0625)));
        let status = ellip.update_q(&cut);
        assert_eq!(status, CutStatus::NoEffect);
        assert_eq!(ellip.xc, Arr::new(4));
        assert_eq!(ellip.mq, Arr::eye(4));
        assert_approx_eq!(ellip.kappa, 0.01);
    }

    #[test]
    fn test_update_q() {
        let mut ellip = Ell::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), ParallelCut(0.01, Some(0.04)));
        let status = ellip.update_q(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(ellip.xc[0], -0.0116);
        assert_approx_eq!(ellip.mq.at(0, 0), 1.0 - 0.232);
        assert_approx_eq!(ellip.kappa, 0.01232);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_central_cut_mq() {
        let mut ellip = Ell::new_with_scalar(0.01, Arr::new(4));
        let cut = (0.5 * Arr::ones(4), SingleCut(0.0));
        let _ = ellip.update_central_cut(&cut);
        let mq_expected = &Arr::eye(4) - &(0.1 * Arr::full(4, 4, 1.0));
        for i in 0..4 {
            for j in 0..4 {
                assert_approx_eq!(ellip.mq.at(i, j), mq_expected.at(i, j));
            }
        }
    }

    #[test]
    fn test_no_defer_trick() {
        let mut ellip = Ell::new_with_scalar(0.01, Arr::new(4));
        ellip.no_defer_trick = true;
        let cut = (0.5 * Arr::ones(4), SingleCut(0.0));
        let _ = ellip.update_central_cut(&cut);
        assert_approx_eq!(ellip.kappa, 1.0);
        let mq_expected = &(&Arr::eye(4) - &(0.1 * Arr::full(4, 4, 1.0))) * (0.16 / 15.0);
        for i in 0..4 {
            for j in 0..4 {
                assert_approx_eq!(ellip.mq.at(i, j), mq_expected.at(i, j));
            }
        }
    }

    #[test]
    fn test_from_covariance() {
        let cov = Arr::from_diag(&Arr::from(vec![2.0, 3.0, 4.0, 5.0]));
        let xc = Arr::from(vec![1.0, 2.0, 3.0, 4.0]);
        let ellip = Ell::from_covariance(cov.clone(), xc.clone());
        assert_eq!(ellip.kappa, 1.0);
        assert_eq!(ellip.mq, cov);
        assert_eq!(ellip.xc, xc);
    }
}
