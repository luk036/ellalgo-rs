use crate::cutting_plane::{CutStatus, SearchSpace, UpdateByCutChoices};
// #[macro_use]
// extern crate ndarray;

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

    /**
     * @brief Update ellipsoid core function using the cut
     *
     *  $grad^T * (x - xc) + beta <= 0$
     *
     * @tparam T
     * @param[in] cut
     * @return (i32, f64)
     */
    fn update_single(&mut self, grad: &f64, b0: &f64) -> (CutStatus, f64) {
        let g = *grad;
        let beta = *b0;
        let temp = self.r * g;
        let tau = if g < 0.0 { -temp } else { temp };
        let tsq = tau * tau;

        if beta == 0.0 {
            self.r /= 2.0;
            self.xc += if g > 0.0 { -self.r } else { self.r };
            return (CutStatus::Success, tsq);
        }
        if beta > tau {
            return (CutStatus::NoSoln, tsq);  // no sol'n
        }
        if beta < -tau {
            return (CutStatus::NoEffect, tsq);  // no effect
        }

        let bound = self.xc - beta / g;
        let u = if g > 0.0 { bound } else { self.xc + self.r };
        let l = if g > 0.0 { self.xc - self.r } else { bound };

        self.r = (u - l) / 2.0;
        self.xc = l + self.r;
        return (CutStatus::Success, tsq);
    }
}

impl SearchSpace for Ell1D {
    type ArrayType = f64;

    /**
     * @brief
     *
     * @return f64
     */
    fn xc(&self) -> f64 { self.xc }

    fn update<T>(&mut self, cut: &(Self::ArrayType, T)) -> (CutStatus, f64)
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_by(self, grad)
    }
}

impl UpdateByCutChoices<Ell1D> for f64 {
    type ArrayType = f64;

    fn update_by(&self, ell: &mut Ell1D, grad: &Self::ArrayType) -> (CutStatus, f64) {
        let beta = self;
        ell.update_single(grad, beta)
    }
}

// TODO: Support Parallel Cut
// impl UpdateByCutChoices<Ell1D> for (f64, Option<f64>) {
//     type ArrayType = Arr;
//     fn update_by(&self, ell: &mut Ell1D, grad: &Self::ArrayType) -> (CutStatus, f64) {
//         let beta = self;
//         ell.update_parallel(grad, &beta)
//     }
// }

