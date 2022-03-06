// mod lib;
use crate::lib::{CutStatus, SearchSpace, UpdateByCutChoices};
use crate::ell_calc::EllCalc;
// #[macro_use]
// extern crate ndarray;
use ndarray::prelude::*;

type Arr = Array1<f64>;

/**
 * @brief Ellipsoid Search Space
 *
 *  Ell = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
#[derive(Debug, Clone)]
pub struct Ell {
    pub no_defer_trick: bool,

    mq: Array2<f64>,
    xc: Array1<f64>,
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
    pub fn new(val: Arr, xc: Arr) -> Ell {
        Ell::new_with_matrix(1.0, Array2::from_diag(&val), xc)
    }

    // /**
    //  * @brief Set the xc object
    //  *
    //  * @param[in] xc
    //  */
    // pub fn set_xc(&mut self, xc: Arr) { self.xc = xc; }

    /**
     * @brief Update ellipsoid core function using the cut
     *
     *  $grad^T * (x - xc) + beta <= 0$
     *
     * @tparam T
     * @param[in] cut
     * @return (i32, f64)
     */
    fn update_single(&mut self, grad: &Array1<f64>, beta: &f64) -> (CutStatus, f64) {
        // let (grad, beta) = cut;
        let mut mq_g = Array1::zeros(self.n); // initial x0
        let mut omega = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                mq_g[i] += self.mq[[i, j]] * grad[j];
            }
            omega += mq_g[i] * grad[i];
        }

        self.helper.tsq = self.kappa * omega;
        let status = self.helper.calc_dc(*beta);
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

    /**
     * @brief Update ellipsoid core function using the cut
     *
     *  $grad^T * (x - xc) + beta <= 0$
     *
     * @tparam T
     * @param[in] cut
     * @return (i32, f64)
     */
    fn update_parallel(
        &mut self,
        grad: &Array1<f64>,
        beta: &(f64, Option<f64>),
    ) -> (CutStatus, f64) {
        // let (grad, beta) = cut;
        let mut mq_g = Array1::zeros(self.n); // initial x0
        let mut omega = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                mq_g[i] += self.mq[[i, j]] * grad[j];
            }
            omega += mq_g[i] * grad[i];
        }

        self.helper.tsq = self.kappa * omega;

        let (b0, b1_opt) = *beta;
        let status = if let Some(b1) = b1_opt {
            self.helper.calc_ll_core(b0, b1)
        } else {
            self.helper.calc_dc(b0)
        };
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
}

impl SearchSpace for Ell {
    type ArrayType = Array1<f64>;

    /**
     * @brief copy the whole array anyway
     *
     * @return Arr
     */
    fn xc(&self) -> Self::ArrayType {
        self.xc.clone()
    }

    fn update<T>(&mut self, cut: &(Self::ArrayType, T)) -> (CutStatus, f64)
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_by(self, grad)
    }
}

impl UpdateByCutChoices<Ell> for f64 {
    type ArrayType = Arr;

    fn update_by(&self, ell: &mut Ell, grad: &Self::ArrayType) -> (CutStatus, f64) {
        let beta = self;
        ell.update_single(grad, beta)
    }
}

impl UpdateByCutChoices<Ell> for (f64, Option<f64>) {
    type ArrayType = Arr;

    fn update_by(&self, ell: &mut Ell, grad: &Self::ArrayType) -> (CutStatus, f64) {
        let beta = self;
        ell.update_parallel(grad, beta)
    }
}
