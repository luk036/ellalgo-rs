// mod cutting_plane;
use crate::cutting_plane::{CutStatus, SearchSpace};
use crate::ell_calc::{EllCalc, UpdateByCutChoices};
// #[macro_use]
// extern crate ndarray;
use ndarray::prelude::*;

type Arr = Array1<f64>;

/**
 * @brief Ellipsoid Search Space
 *
 *        EllStable = {x | (x - xc)' mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
#[derive(Debug, Clone)]
pub struct EllStable {
    pub no_defer_trick: bool,

    mq: Array2<f64>,
    xc: Arr,
    kappa: f64,
    n: usize,
    helper: EllCalc,
}

impl EllStable {
    /**
     * @brief Construct a new EllStable object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    pub fn new_with_matrix(kappa: f64, mq: Array2<f64>, xc: Arr) -> EllStable {
        let n = xc.len();
        let helper = EllCalc::new(n as f64);

        EllStable {
            kappa,
            mq,
            xc,
            n,
            helper,
            no_defer_trick: false,
        }
    }

    /**
     * @brief Construct a new EllStable object
     *
     * @param[in] val
     * @param[in] x
     */
    pub fn new(val: &Arr, xc: Arr) -> EllStable {
        EllStable::new_with_matrix(1.0, Array2::from_diag(val), xc)
    }

    // /**
    //  * @brief Set the xc object
    //  *
    //  * @param[in] xc
    //  */
    // pub fn set_xc(&mut self, xc: Arr) { self.xc = xc; }
}

impl SearchSpace for EllStable {
    /**
     * @brief copy the whole array anyway
     *
     * @return Arr
     */
    fn xc(&self) -> Arr {
        self.xc.clone()
    }

    fn update<T: UpdateByCutChoices>(&mut self, cut: (Arr, T)) -> (CutStatus, f64) {
        let (grad, beta) = cut;
        // calculate inv(L)*grad: (n-1)*n/2 multiplications
        let mut inv_ml_g = grad; // initial x0
        for i in 0..self.n {
            for j in 0..i {
                self.mq[[i, j]] = self.mq[[j, i]] * inv_ml_g[j];
                // keep for rank-one update
                inv_ml_g[i] -= self.mq[[i, j]];
            }
        }

        // calculate inv(D)*inv(L)*grad: n
        let mut inv_md_inv_ml_g = inv_ml_g.clone(); // initially
        for i in 0..self.n {
            inv_md_inv_ml_g[i] *= self.mq[[i, i]];
        }

        // calculate omega: n
        let mut g_mq_g = inv_md_inv_ml_g.clone(); // initially
        let mut omega = 0.0; // initially
        for i in 0..self.n {
            g_mq_g[i] *= inv_ml_g[i];
            omega += g_mq_g[i];
        }

        self.helper.tsq = self.kappa * omega;
        let status = self.helper.update::<T>(beta);
        if status != CutStatus::Success {
            return (status, self.helper.tsq);
        }

        // calculate mq*grad = inv(L')*inv(D)*inv(L)*grad : (n-1)*n/2
        let mut mq_g = inv_md_inv_ml_g.clone(); // initially
        for i in (1..self.n).rev() {
            // backward subsituition
            for j in i..self.n {
                mq_g[i - 1] -= self.mq[[i, j]] * mq_g[j]; // ???
            }
        }

        // calculate xc: n
        self.xc -= &((self.helper.rho / omega) * &mq_g); // n

        // rank-one update: 3*n + (n-1)*n/2
        // let r = self.sigma / omega;
        let mu = self.helper.sigma / (1.0 - self.helper.sigma);
        let mut oldt = omega / mu; // initially
        let m = self.n - 1;
        for j in 0..m {
            // p=sqrt(k)*vv[j];
            // let p = inv_ml_g[j];
            // let mup = mu * p;
            let t = oldt + g_mq_g[j];
            // self.mq[[j, j]] /= t; // update invD
            let beta2 = inv_md_inv_ml_g[j] / t;
            self.mq[[j, j]] *= oldt / t; // update invD
            for l in (j + 1)..self.n {
                // v(l) -= p * self.mq(j, l);
                self.mq[[j, l]] += beta2 * self.mq[[l, j]];
            }
            oldt = t;
        }

        // let p = inv_ml_g(n1);
        // let mup = mu * p;
        let t = oldt + g_mq_g[m];
        self.mq[[m, m]] *= oldt / t; // update invD
        self.kappa *= self.helper.delta;

        // if self.no_defer_trick
        // {
        //     self.mq *= self.kappa;
        //     self.kappa = 1.;
        // }
        (status, self.helper.tsq)
    }
} // } Ell
