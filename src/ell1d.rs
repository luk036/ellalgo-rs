use crate::cutting_plane::{CutStatus, SearchSpace, UpdateByCutChoice};
// #[macro_use]
// extern crate ndarray;

/// The [`Ell1D`] struct represents an ellipsoid in one dimension.
///
/// Properties:
///
/// * `r`: The `r` property represents the radius of the ellipsoid in the 1D case. It determines the
///         size of the ellipsoid along the x-axis.
/// * `xc`: The property `xc` represents the center of the ellipsoid in the 1D case. It is a scalar
///         value that specifies the position of the center along the x-axis.
#[derive(Debug, Clone)]
pub struct Ell1D {
    r: f64,
    xc: f64,
}

impl Ell1D {
    /// The `new` function creates a new instance of the [`Ell1D`] struct with the given lower and upper
    /// bounds.
    ///
    /// Arguments:
    ///
    /// * `l`: The parameter `l` represents the lower bound of the range.
    /// * `u`: The parameter `u` represents the upper bound of the range.
    ///
    /// Returns:
    ///
    /// The `new` function is returning an instance of the [`Ell1D`] struct.
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::ell1d::Ell1D;
    /// let ell1d = Ell1D::new(0.0, 10.0);
    /// ```
    pub fn new(l: f64, u: f64) -> Self {
        let r = (u - l) / 2.0;
        let xc = l + r;
        Ell1D { r, xc }
    }

    /// The function `set_xc` sets the value of the `xc` variable in a Rust struct.
    ///
    /// Arguments:
    ///
    /// * `xc`: The parameter `xc` is of type `f64`, which means it is a floating-point number.
    fn set_xc(&mut self, xc: f64) {
        self.xc = xc;
    }

    /// The function `update_single` updates an ellipsoid core using a single cut.
    ///
    /// Arguments:
    ///
    /// * `grad`: The `grad` parameter is the gradient value, which is of type `f64`. It represents the
    ///           gradient of the function at a particular point.
    /// * `beta0`: The parameter `beta0` represents the value of the constant term in the inequality constraint
    ///           equation. In the code, it is referred to as `beta`.
    fn update_single(&mut self, grad: f64, beta0: f64) -> (CutStatus, f64) {
        let g = *grad;
        let beta = *beta0;
        let temp = self.r * g;
        let tau = if g < 0.0 { -temp } else { temp };
        let tsq = tau * tau;

        if beta == 0.0 {
            self.r /= 2.0;
            self.xc += if g > 0.0 { -self.r } else { self.r };
            return (CutStatus::Success, tsq);
        }
        if beta > tau {
            return (CutStatus::NoSoln, tsq); // no sol'n
        }
        if beta < -tau {
            return (CutStatus::NoEffect, tsq); // no effect
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

    /// The function `xc` returns a copy of the `xc` array.
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::ell1d::Ell1D;
    /// let ell1d = Ell1D::new(0.0, 10.0);
    /// let center = ell1d.xc();
    /// assert_eq!(center, 5.0);
    /// ```
    fn xc(&self) -> f64 {
        self.xc
    }

    /// The `update` function updates the decision variable based on the given cut.
    /// 
    /// Arguments:
    /// 
    /// * `cut`: A tuple containing two elements:
    /// 
    /// Returns:
    /// 
    /// The `update` function returns a value of type `CutStatus`.
    fn update<T>(&mut self, cut: &(Self::ArrayType, T)) -> (CutStatus, f64)
    where
        T: UpdateByCutChoice<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_by(self, grad)
    }
}

impl UpdateByCutChoice<Ell1D> for f64 {
    type ArrayType = f64;

    fn update_by(&self, ell: &mut Ell1D, grad: &Self::ArrayType) -> (CutStatus, f64) {
        let beta = self;
        ell.update_single(grad, beta)
    }
}

// TODO: Support Parallel Cut
// impl UpdateByCutChoice<Ell1D> for (f64, Option<f64>) {
//     type ArrayType = Arr;
//     fn update_by(&self, ell: &mut Ell1D, grad: &Self::ArrayType) -> (CutStatus, f64) {
//         let beta = self;
//         ell.update_parallel(grad, beta)
//     }
// }
