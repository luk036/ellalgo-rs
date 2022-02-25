// mod cutting_plane;
use crate::cutting_plane::{CutStatus, SearchSpace, UpdateByCutChoices};
// #[macro_use]
// extern crate ndarray;
use ndarray::prelude::*;

type Arr = Array1<f64>;

/**
 * @brief Ellipsoid Method for special 1D case
 *
 *  Ell = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
#[derive(Debug, Clone)]
pub struct Ell1D {
    r: f64,
    xc: f64,
}

impl Ell1D {
    /**
     * @brief Construct a new Ell1D object
     *
     * @param[in] l
     * @param[in] u
     */
    pub fn new(l: f64, u: f64) -> Self {
        let r = (u - l) / 2.0;
        let xc = l + r;
        Ell1D { r, xc }
    }

    /**
     * @brief Set the xc object
     *
     * @param[in] xc
     */
    fn set_xc(&mut self, xc: f64) { self.xc = xc; }
}

impl SearchSpace for Ell1D {
    type ArrayType = f64;

    /**
     * @brief
     *
     * @return f64
     */
    fn xc(&self) -> f64 { self.xc }

    /**
     * @brief Update ellipsoid core function using the cut
     *
     *  $grad^T * (x - xc) + beta <= 0$
     *
     * @tparam T
     * @param[in] cut
     * @return (i32, f64)
     */
    fn update<T: UpdateByCutChoices>(&mut self, cut: (Self::ArrayType, T)) -> (CutStatus, f64) {

        let (g, beta) = cut;
        let temp = self.r * g;
        let tau = if a < 0.0 { -temp } else { temp };
        let tsq = tau * tau;

        if beta == 0.0 {
            self.r /= 2.0;
            self.xc += g > 0.0 ? -self.r : self.r;
            return (CutStatus::Success, tsq);
        }
        if beta > tau {
            return (CutStatus::NoSoln, tsq);  // no sol'n
        }
        if beta < -tau {
            return (CutStatus::NoEffect, tsq);  // no effect
        }

        let bound = self.xc - beta / g;
        let u = g > 0.0 ? bound : self.xc + self.r;
        let l = g > 0.0 ? self.xc - self.r : bound;

        self.r = (u - l) / 2.0;
        self.xc = l + self.r;
        return (CutStatus::Success, tsq);
    }
}
