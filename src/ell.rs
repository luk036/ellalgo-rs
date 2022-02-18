// mod cutting_plane;
use crate::cutting_plane::{CutStatus, SearchSpace, CutChoices, IntroCutChoices};
use crate::ell_calc::EllCalc;
// #[macro_use]
// extern crate ndarray;
use ndarray::prelude::*;

type Arr = Array1<f64>;
type Params = (f64, f64, f64);
type Returns = (i32, Params);

/**
 * @brief Ellipsoid Search Space
 *
 *        Ell = {x | (x - xc)' mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
#[derive(Debug, Clone)]
pub struct Ell {
    pub no_defer_trick: bool,

    mq: Array2<f64>,
    xc: Arr,
    kappa: f64,
    n: usize,
    helper: EllCalc,
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
    pub fn new_with_matrix(kappa: f64, mq: Array2<f64>, xc: Arr) -> Ell {
        let n = xc.len();
        let helper = EllCalc::new(n as f64);

        Ell {
            kappa,
            mq,
            xc,
            n,
            helper,
            no_defer_trick: false,
        }
    }

    /**
     * @brief Construct a new Ell object
     *
     * @param[in] val
     * @param[in] x
     */
    pub fn new(val: &Arr, xc: Arr) -> Ell {
        Ell::new_with_matrix(1.0, Array2::from_diag(val), xc)
    }

    // /**
    //  * @brief Set the xc object
    //  *
    //  * @param[in] xc
    //  */
    // pub fn set_xc(&mut self, xc: Arr) { self.xc = xc; }
}

impl<T> SearchSpace for Ell {
    /**
     * @brief copy the whole array anyway
     *
     * @return Arr
     */
    fn xc(&self) -> Arr {
        self.xc.clone()
    }

    /**
     * @brief Update ellipsoid core function using the cut
     *
     *        grad' * (x - xc) + beta <= 0
     *
     * @tparam T
     * @param[in] cut
     * @return (i32, f64)
     */
    fn update<T>(&mut self, cut: &(Arr, T)) -> (CutStatus, f64) {
        let (grad, beta) = cut;
        let mut mq_g = Array1::zeros(self.n); // initial x0
        let mut omega = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                mq_g[i] += self.mq[[i, j]] * grad[j];
            }
            omega += mq_g[i] * grad[i];
        }

        self.helper.tsq = self.kappa * omega;
        let mut status = self.helper.update::<T>(beta);
        let status = CutStatus::Success;
        if status != CutStatus::Success {
            return (status, self.helper.tsq);
        }

        self.xc -= &((self.helper.rho / omega) * &mq_g); // n
                                                         // n*(n+1)/2 + n
                                                         // self.mq -= (self.sigma / omega) * xt::linalg::outer(mq_g, mq_g);

        let r = self.helper.sigma / omega;
        for i in 0..self.n {
            let r_mq_g = r * mq_g[i];
            for j in 0..i {
                self.mq[[i, j]] -= r_mq_g * mq_g[j];
                self.mq[[j, i]] = self.mq[[i, j]];
            }
            self.mq[[i, i]] -= r_mq_g * mq_g[i];
        }

        self.kappa *= self.helper.delta;

        if self.no_defer_trick {
            self.mq *= self.kappa;
            self.kappa = 1.0;
        }
        (status, self.helper.tsq)
    }
} // } Ell
