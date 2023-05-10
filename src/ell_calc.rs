// mod lib;
use crate::cutting_plane::CutStatus;

/**
 * @brief Ellipsoid Search Space
 *
 *  EllCalcCore = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
#[derive(Debug, Clone)]
pub struct EllCalcCore {
    n_f: f64,
    n_plus_1: f64,
    half_n: f64,
    cst1: f64,
    cst2: f64,
    cst3: f64,
}

impl EllCalcCore {
    /**
     * @brief Construct a new EllCalc object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    pub fn new(n_f: f64) -> EllCalcCore {
        let n_plus_1 = n_f + 1.0;
        let half_n = n_f / 2.0;
        let n_sq = n_f * n_f;
        let cst0 = 1.0 / (n_f + 1.0);
        let cst1 = n_sq / (n_sq - 1.0);
        let cst2 = 2.0 * cst0;
        let cst3 = n_f * cst0;

        EllCalcCore {
            n_f,
            n_plus_1,
            half_n,
            cst1,
            cst2,
            cst3,
        }
    }

    /**
     * @brief
     *
     * @param[in] b0
     * @param[in] b1
     * @return i32
     */
    pub fn calc_ll_dc_core(&self, b0: f64, b1: f64, tsq: &f64) -> (f64, f64, f64) {
        let b1sqn = b1 * (b1 / tsq);
        let t1n = 1.0 - b1sqn;
        let b0b1n = b0 * (b1 / tsq);
        // let t0 = tsq - b0 * b0;
        let t0n = 1.0 - b0 * (b0 / tsq);
        // let t1 = tsq - b1sq;
        let bsum = b0 + b1;
        let bsumn = bsum / tsq;
        let bav = bsum / 2.0;
        let tempn = self.half_n * bsumn * (b1 - b0);
        let xi = (t0n * t1n + tempn * tempn).sqrt();
        let sigma = self.cst3 + (1.0 + b0b1n - xi) / (bsumn * bav) / self.n_plus_1;
        let rho = sigma * bav;
        let delta = self.cst1 * ((t0n + t1n) / 2.0 + xi / self.n_f);
        (rho, sigma, delta)
    }

    /**
     * @brief
     *
     * @param[in] b1
     * @param[in] b1sq
     * @return void
     */
    pub fn calc_ll_cc_core(&self, b1: f64, tsq: &f64) -> (f64, f64, f64) {
        let b1sqn = b1 * (b1 / tsq);
        let temp = self.half_n * b1sqn;
        let xi = (1.0 - b1sqn + temp * temp).sqrt();
        let sigma = self.cst3 + self.cst2 * (1.0 - xi) / b1sqn;
        let rho = sigma * b1 / 2.0;
        let delta = self.cst1 * (1.0 - b1sqn / 2.0 + xi / self.n_f);
        (rho, sigma, delta)
    }

    /**
     * @brief Deep Cut
     *
     * @param[in] beta
     * @return i32
     */
    pub fn calc_dc_core(&self, beta: &f64, tau: &f64, gamma: &f64) -> (f64, f64, f64) {
        let rho = gamma / self.n_plus_1;
        let sigma = 2.0 * rho / (tau + beta);
        let alpha = beta / tau;
        let delta = self.cst1 * (1.0 - alpha * alpha);
        (rho, sigma, delta)
    }

    /**
     * @brief Central Cut
     *
     * @param[in] tau
     * @return i32
     */
    pub fn calc_cc_core(&self, tsq: &f64) -> (f64, f64, f64) {
        // self.mu = self.half_n_minus_1;
        let tau = tsq.sqrt();
        let sigma = self.cst2;
        let rho = tau / self.n_plus_1;
        let delta = self.cst1;
        (rho, sigma, delta)
    }
}

/**
 * @brief Ellipsoid Search Space
 *
 *  EllCalc = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
#[derive(Debug, Clone)]
pub struct EllCalc {
    n_f: f64,
    pub calculator: EllCalcCore,
    pub use_parallel_cut: bool,
}

impl EllCalc {
    /**
     * @brief Construct a new EllCalc object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    pub fn new(n_f: f64) -> EllCalc {
        let calculator = EllCalcCore::new(n_f);

        EllCalc {
            n_f,
            calculator,
            use_parallel_cut: true,
        }
    }

    // pub fn update_cut(&mut self, beta: f64) -> CutStatus { self.calc_dc(beta) }

    pub fn calc_single_or_ll_dc(
        &self,
        beta: &(f64, Option<f64>),
        tsq: &f64,
    ) -> (CutStatus, (f64, f64, f64)) {
        let (b0, b1_opt) = *beta;
        if let Some(b1) = b1_opt {
            self.calc_ll_dc(b0, b1, tsq)
        } else {
            self.calc_dc(&b0, tsq)
        }
    }

    pub fn calc_single_or_ll_cc(
        &self,
        beta: &(f64, Option<f64>),
        tsq: &f64,
    ) -> (CutStatus, (f64, f64, f64)) {
        let (_b0, b1_opt) = *beta;
        if let Some(b1) = b1_opt {
            self.calc_ll_cc(b1, tsq)
        } else {
            self.calc_cc(tsq)
        }
    }

    pub fn calc_single_or_ll_q(
        &self,
        beta: &(f64, Option<f64>),
        tsq: &f64,
    ) -> (CutStatus, (f64, f64, f64)) {
        let (b0, b1_opt) = *beta;
        if let Some(b1) = b1_opt {
            self.calc_ll_q(b0, b1, tsq)
        } else {
            self.calc_q(&b0, tsq)
        }
    }

    /**
     * @brief
     *
     * @param[in] b0
     * @param[in] b1
     * @return i32
     */
    pub fn calc_ll_dc(&self, b0: f64, b1: f64, tsq: &f64) -> (CutStatus, (f64, f64, f64)) {
        if b1 < b0 {
            return (CutStatus::NoSoln, (0.0, 0.0, 0.0)); // no sol'n
        }

        let b1sqn = b1 * (b1 / tsq);
        let t1n = 1.0 - b1sqn;
        if t1n < 0.0 || !self.use_parallel_cut {
            return self.calc_dc(&b0, tsq);
        }

        // let b0b1n = b0 * (b1 / tsq);
        // if self.n_f * b0b1n < -1.0 {
        //     return (CutStatus::NoEffect, (0.0, 0.0, 1.0)); // no effect
        // }

        (
            CutStatus::Success,
            self.calculator.calc_ll_dc_core(b0, b1, tsq),
        )
    }

    /**
     * @brief
     *
     * @param[in] b0
     * @param[in] b1
     * @return i32
     */
    pub fn calc_ll_q(&self, b0: f64, b1: f64, tsq: &f64) -> (CutStatus, (f64, f64, f64)) {
        if b1 < b0 {
            return (CutStatus::NoSoln, (0.0, 0.0, 0.0)); // no sol'n
        }

        let b1sqn = b1 * (b1 / tsq);
        let t1n = 1.0 - b1sqn;
        if t1n < 0.0 || !self.use_parallel_cut {
            return self.calc_q(&b0, tsq);
        }

        let b0b1n = b0 * (b1 / tsq);
        if self.n_f * b0b1n < -1.0 {
            return (CutStatus::NoEffect, (0.0, 0.0, 1.0)); // no effect
        }

        (
            CutStatus::Success,
            self.calculator.calc_ll_dc_core(b0, b1, tsq),
        )
    }

    /**
     * @brief
     *
     * @param[in] b1
     * @param[in] b1sq
     * @return void
     */
    pub fn calc_ll_cc(&self, b1: f64, tsq: &f64) -> (CutStatus, (f64, f64, f64)) {
        if b1 < 0.0 {
            return (CutStatus::NoSoln, (0.0, 0.0, 0.0)); // no effect
        }
        let b1sqn = b1 * (b1 / tsq);
        let t1n = 1.0 - b1sqn;
        if t1n < 0.0 || !self.use_parallel_cut {
            return self.calc_cc(tsq);
        }
        (CutStatus::Success, self.calculator.calc_ll_cc_core(b1, tsq))
    }

    /**
     * @brief Deep Cut
     *
     * @param[in] beta
     * @return i32
     */
    pub fn calc_dc(&self, beta: &f64, tsq: &f64) -> (CutStatus, (f64, f64, f64)) {
        if *tsq < beta * beta {
            return (CutStatus::NoSoln, (0.0, 0.0, 0.0)); // no sol'n
        }

        let tau = tsq.sqrt();
        let gamma = tau + self.n_f * beta;
        // if gamma < 0.0 {
        //     return (CutStatus::NoEffect, (0.0, 0.0, 1.0)); // no effect
        // }

        (
            CutStatus::Success,
            self.calculator.calc_dc_core(beta, &tau, &gamma),
        )
    }

    /**
     * @brief Deep Cut
     *
     * @param[in] beta
     * @return i32
     */
    pub fn calc_q(&self, beta: &f64, tsq: &f64) -> (CutStatus, (f64, f64, f64)) {
        let tau = tsq.sqrt();

        if tau < *beta {
            return (CutStatus::NoSoln, (0.0, 0.0, 0.0)); // no sol'n
        }

        let gamma = tau + self.n_f * beta;
        if gamma < 0.0 {
            return (CutStatus::NoEffect, (0.0, 0.0, 1.0)); // no effect
        }

        (
            CutStatus::Success,
            self.calculator.calc_dc_core(beta, &tau, &gamma),
        )
    }

    /**
     * @brief Central Cut
     *
     * @param[in] tau
     * @return i32
     */
    #[inline]
    pub fn calc_cc(&self, tsq: &f64) -> (CutStatus, (f64, f64, f64)) {
        // self.mu = self.half_n_minus_1;
        (CutStatus::Success, self.calculator.calc_cc_core(tsq))
    }
}

// pub trait UpdateByCutChoices {
//     fn update_by(self, ell: &mut EllCalc) -> CutStatus;
// }
#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    pub fn test_construct() {
        let calculator = EllCalcCore::new(4.0);
        assert_eq!(calculator.n_f, 4.0);
        assert_eq!(calculator.half_n, 2.0);
        assert_approx_eq!(calculator.cst1, 16.0 / 15.0);
        assert_approx_eq!(calculator.cst2, 0.4);
        assert_approx_eq!(calculator.cst3, 0.8);
    }

    #[test]
    pub fn test_calc_cc() {
        let ell_calc = EllCalc::new(4.0);
        // ell_calc.tsq = 0.01;
        let (status, (rho, sigma, delta)) = ell_calc.calc_cc(&0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.4);
        assert_approx_eq!(rho, 0.02);
        assert_approx_eq!(delta, 16.0 / 15.0);
    }

    #[test]
    pub fn test_calc_dc() {
        let ell_calc = EllCalc::new(4.0);
        // ell_calc.tsq = 0.01;
        let (status, (_rho, _sigma, _delta)) = ell_calc.calc_dc(&0.11, &0.01);
        assert_eq!(status, CutStatus::NoSoln);
        let (status, (_rho, _sigma, _delta)) = ell_calc.calc_dc(&0.0, &0.01);
        assert_eq!(status, CutStatus::Success);
        let (status, (_rho, _sigma, _delta)) = ell_calc.calc_q(&-0.05, &0.01);
        assert_eq!(status, CutStatus::NoEffect);

        // ell_calc.tsq = 0.01;
        let (status, (rho, sigma, delta)) = ell_calc.calc_dc(&0.05, &0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(rho, 0.06);
        assert_approx_eq!(delta, 0.8);
    }

    #[test]
    pub fn test_calc_ll_cc() {
        let ell_calc = EllCalc::new(4.0);
        // ell_calc.tsq = 0.01;
        let (status, (rho, sigma, delta)) = ell_calc.calc_ll_cc(0.11, &0.01);
        assert_eq!(status, CutStatus::Success);
        // Central cut
        assert_approx_eq!(sigma, 0.4);
        assert_approx_eq!(rho, 0.02);
        assert_approx_eq!(delta, 16.0 / 15.0);

        let (status, (rho, sigma, delta)) = ell_calc.calc_ll_cc(0.05, &0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(rho, 0.02);
        assert_approx_eq!(delta, 1.2);
    }

    #[test]
    pub fn test_calc_ll() {
        let ell_calc = EllCalc::new(4.0);
        // ell_calc.tsq = 0.01;
        let (status, (_rho, _sigma, _delta)) = ell_calc.calc_ll_dc(0.07, 0.03, &0.01);
        assert_eq!(status, CutStatus::NoSoln);

        let (status, (rho, sigma, delta)) = ell_calc.calc_ll_dc(0.0, 0.05, &0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(rho, 0.02);
        assert_approx_eq!(delta, 1.2);

        let (status, (rho, sigma, delta)) = ell_calc.calc_ll_dc(0.05, 0.11, &0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(rho, 0.06);
        assert_approx_eq!(delta, 0.8);

        let (status, (_rho, _sigma, _delta)) = ell_calc.calc_ll_q(-0.07, 0.07, &0.01);
        assert_eq!(status, CutStatus::NoEffect);

        let (status, (rho, sigma, delta)) = ell_calc.calc_ll_dc(0.01, 0.04, &0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.928);
        assert_approx_eq!(rho, 0.0232);
        assert_approx_eq!(delta, 1.232);

        let (status, (rho, sigma, delta)) = ell_calc.calc_ll_q(-0.04, 0.0625, &0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.0);
        assert_approx_eq!(rho, 0.0);
        assert_approx_eq!(delta, 1.0);
    }
}
