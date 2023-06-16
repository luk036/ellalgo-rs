// mod lib;
use crate::cutting_plane::{CutStatus, SearchSpace, SearchSpaceQ, UpdateByCutChoices};
use crate::ell_calc::EllCalc;
// #[macro_use]
// extern crate ndarray;
// use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Array2;

/**
 * @brief Ellipsoid Search Space
 *
 *  Ell = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
 *
 */
#[derive(Debug, Clone)]
pub struct Ell {
    pub no_defer_trick: bool,
    mq: Array2<f64>,
    xc: Array1<f64>,
    kappa: f64,
    ndim: usize,
    helper: EllCalc,
    tsq: f64,
}

impl Ell {
    /**
     * @brief Construct a new Ell object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    pub fn new_with_matrix(kappa: f64, mq: Array2<f64>, xc: Array1<f64>) -> Ell {
        let ndim = xc.len();
        let helper = EllCalc::new(ndim as f64);

        Ell {
            kappa,
            mq,
            xc,
            ndim,
            helper,
            no_defer_trick: false,
            tsq: 0.0,
        }
    }

    /**
     * @brief Construct a new Ell object
     *
     * @param[in] val
     * @param[in] x
     */
    pub fn new(val: Array1<f64>, xc: Array1<f64>) -> Ell {
        Ell::new_with_matrix(1.0, Array2::from_diag(&val), xc)
    }

    /**
     * @brief Construct a new Ell object
     *
     * @param[in] val
     * @param[in] xc
     */
    pub fn new_with_scalar(val: f64, xc: Array1<f64>) -> Ell {
        Ell::new_with_matrix(val, Array2::eye(xc.len()), xc)
    }

    /**
     * @brief Update ellipsoid core function using the cut
     *
     *  $grad^T * (x - xc) + beta <= 0$
     *
     * @tparam T
     * @param[in] cut
     * @return (i32, f64)
     */
    fn update_core<T, F>(&mut self, grad: &Array1<f64>, beta: &T, cut_strategy: F) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Array1<f64>>,
        F: FnOnce(&T, &f64) -> (CutStatus, (f64, f64, f64)),
    {
        let grad_t = self.mq.dot(grad);
        let omega = grad.dot(&grad_t);

        self.tsq = self.kappa * omega;
        // let status = self.helper.calc_dc(*beta);
        let (status, (rho, sigma, delta)) = cut_strategy(beta, &self.tsq);
        if status != CutStatus::Success {
            return status;
        }

        self.xc -= &((rho / omega) * &grad_t); // n

        // n*(n+1)/2 + n
        // self.mq -= (self.sigma / omega) * xt::linalg::outer(grad_t, grad_t);
        let r = sigma / omega;
        for i in 0..self.ndim {
            let r_grad_t = r * grad_t[i];
            for j in 0..i {
                self.mq[[i, j]] -= r_grad_t * grad_t[j];
                self.mq[[j, i]] = self.mq[[i, j]];
            }
            self.mq[[i, i]] -= r_grad_t * grad_t[i];
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
    type ArrayType = Array1<f64>;

    /**
     * @brief copy the whole array anyway
     *
     * @return Array1<f64>
     */
    fn xc(&self) -> Self::ArrayType {
        self.xc.clone()
    }

    fn tsq(&self) -> f64 {
        self.tsq
    }

    fn update_dc<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_dc_by(self, grad)
    }

    fn update_cc<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_cc_by(self, grad)
    }
}

impl SearchSpaceQ for Ell {
    type ArrayType = Array1<f64>;

    /**
     * @brief copy the whole array anyway
     *
     * @return Array1<f64>
     */
    fn xc(&self) -> Self::ArrayType {
        self.xc.clone()
    }

    fn tsq(&self) -> f64 {
        self.tsq
    }

    fn update_q<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_q_by(self, grad)
    }
}

impl UpdateByCutChoices<Ell> for f64 {
    type ArrayType = Array1<f64>;

    fn update_dc_by(&self, ellip: &mut Ell, grad: &Self::ArrayType) -> CutStatus {
        let beta = self;
        let helper = ellip.helper.clone();
        ellip.update_core(grad, beta, |beta, tsq| helper.calc_dc(beta, tsq))
    }

    fn update_cc_by(&self, ellip: &mut Ell, grad: &Self::ArrayType) -> CutStatus {
        let beta = self;
        let helper = ellip.helper.clone();
        ellip.update_core(grad, beta, |_beta, tsq| helper.calc_cc(tsq))
    }

    fn update_q_by(&self, ellip: &mut Ell, grad: &Self::ArrayType) -> CutStatus {
        let beta = self;
        let helper = ellip.helper.clone();
        ellip.update_core(grad, beta, |beta, tsq| helper.calc_q(beta, tsq))
    }
}

impl UpdateByCutChoices<Ell> for (f64, Option<f64>) {
    type ArrayType = Array1<f64>;

    fn update_dc_by(&self, ellip: &mut Ell, grad: &Self::ArrayType) -> CutStatus {
        let beta = self;
        let helper = ellip.helper.clone();
        ellip.update_core(grad, beta, |beta, tsq| {
            helper.calc_single_or_ll_dc(beta, tsq)
        })
    }

    fn update_cc_by(&self, ellip: &mut Ell, grad: &Self::ArrayType) -> CutStatus {
        let beta = self;
        let helper = ellip.helper.clone();
        ellip.update_core(grad, beta, |beta, tsq| {
            helper.calc_single_or_ll_cc(beta, tsq)
        })
    }

    fn update_q_by(&self, ellip: &mut Ell, grad: &Self::ArrayType) -> CutStatus {
        let beta = self;
        let helper = ellip.helper.clone();
        ellip.update_core(grad, beta, |beta, tsq| {
            helper.calc_single_or_ll_q(beta, tsq)
        })
    }
}
