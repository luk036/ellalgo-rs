// mod lib;
use crate::cutting_plane::{CutStatus, SearchSpace, SearchSpaceQ, UpdateByCutChoices};
use crate::ell_calc::EllCalc;
// #[macro_use]
// extern crate ndarray;
// use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Array2;

/// The code defines a struct called "Ell" that represents an ellipsoid search space in the Ellipsoid
/// method.
///
///   Ell = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
///
/// Properties:
///
/// * `no_defer_trick`: A boolean flag indicating whether the defer trick should be used. The defer
/// trick is a technique used in the Ellipsoid method to improve efficiency by deferring the update of
/// the ellipsoid until a certain condition is met.
/// * `mq`: A matrix representing the shape of the ellipsoid. It is a 2-dimensional array of f64 values.
/// * `xc`: The `xc` property represents the center of the ellipsoid search space. It is a 1-dimensional
/// array of floating-point numbers.
/// * `kappa`: A scalar value that determines the size of the ellipsoid. A larger value of kappa results
/// in a larger ellipsoid.
/// * `ndim`: The `ndim` property represents the number of dimensions of the ellipsoid search space.
/// * `helper`: The `helper` property is an instance of the `EllCalc` struct, which is used to perform
/// calculations related to the ellipsoid search space. It provides methods for calculating the distance
/// constant (`dc`), the center constant (`cc`), and the quadratic constant (`q`) used in the ell
/// * `tsq`: The `tsq` property represents the squared Mahalanobis distance threshold of the ellipsoid.
/// It is used to determine whether a point is inside or outside the ellipsoid.
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
    /// The function `new_with_matrix` constructs a new `Ell` object with the given parameters.
    ///
    /// Arguments:
    ///
    /// * `kappa`: The `kappa` parameter is a floating-point number that represents the curvature of the
    /// ellipse. It determines the shape of the ellipse, with higher values resulting in a more elongated
    /// shape and lower values resulting in a more circular shape.
    /// * `mq`: The `mq` parameter is of type `Array2<f64>`, which represents a 2-dimensional array of `f64`
    /// (floating-point) values. It is used to store the matrix `mq` in the `Ell` object.
    /// * `xc`: The parameter `xc` represents the center of the ellipsoid in n-dimensional space. It is an
    /// array of length `ndim`, where each element represents the coordinate of the center along a specific
    /// dimension.
    ///
    /// Returns:
    ///
    /// an instance of the `Ell` struct.
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

    /// Creates a new [`Ell`].
    ///
    /// The function `new` creates a new `Ell` object with the given values.
    ///
    /// Arguments:
    ///
    /// * `val`: An array of f64 values representing the diagonal elements of a matrix.
    /// * `xc`: `xc` is an `Array1<f64>` which represents the center of the ellipse. It contains the x and y
    /// coordinates of the center point.
    ///
    /// Returns:
    ///
    /// The function `new` returns an instance of the [`Ell`] struct.
    pub fn new(val: Array1<f64>, xc: Array1<f64>) -> Ell {
        Ell::new_with_matrix(1.0, Array2::from_diag(&val), xc)
    }

    /// The function `new_with_scalar` constructs a new [`Ell`] object with a scalar value and a vector.
    ///
    /// Arguments:
    ///
    /// * `val`: The `val` parameter is a scalar value of type `f64`. It represents the value of the scalar
    /// component of the `Ell` object.
    /// * `xc`: The parameter `xc` is an array of type `Array1<f64>`. It represents the center coordinates
    /// of the ellipse.
    ///
    /// Returns:
    ///
    /// an instance of the [`Ell`] struct.
    pub fn new_with_scalar(val: f64, xc: Array1<f64>) -> Ell {
        Ell::new_with_matrix(val, Array2::eye(xc.len()), xc)
    }

    /// Update ellipsoid core function using the cut
    ///
    ///  $grad^T * (x - xc) + beta <= 0$
    ///
    /// The `update_core` function in Rust updates the ellipsoid core based on a given gradient and beta
    /// value using a cut strategy.
    ///
    /// Arguments:
    ///
    /// * `grad`: A reference to an Array1<f64> representing the gradient vector.
    /// * `beta`: The `beta` parameter is a value that is used in the inequality constraint of the ellipsoid
    /// core function. It represents the threshold for the constraint, and the function checks if the dot
    /// product of the gradient and the difference between `x` and `xc` plus `beta` is less than or
    /// * `cut_strategy`: The `cut_strategy` parameter is a closure that takes two arguments: `beta` and
    /// `tsq`. It returns a tuple containing a `CutStatus` and a tuple `(rho, sigma, delta)`. The
    /// `cut_strategy` function is used to determine the values of `rho`, `
    ///
    /// Returns:
    ///
    /// a value of type `CutStatus`.
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

/// The `impl SearchSpace for Ell` block is implementing the `SearchSpace` trait for the `Ell` struct.
impl SearchSpace for Ell {
    type ArrayType = Array1<f64>;

    /// The function `xc` returns a copy of the `xc` array.
    fn xc(&self) -> Self::ArrayType {
        self.xc.clone()
    }

    /// The `tsq` function returns the value of the `tsq` field of the struct.
    ///
    /// Returns:
    ///
    /// The method `tsq` is returning a value of type `f64`.
    fn tsq(&self) -> f64 {
        self.tsq
    }

    /// The `update_dc` function updates the decision variable based on the given cut.
    /// 
    /// Arguments:
    /// 
    /// * `cut`: A tuple containing two elements:
    /// 
    /// Returns:
    /// 
    /// The `update_dc` function returns a value of type `CutStatus`.
    fn update_dc<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_dc_by(self, grad)
    }

    /// The `update_cc` function updates the cut choices using the gradient and beta values.
    /// 
    /// Arguments:
    /// 
    /// * `cut`: The `cut` parameter is a tuple containing two elements. The first element is of type
    /// `Self::ArrayType`, and the second element is of type `T`.
    /// 
    /// Returns:
    /// 
    /// The function `update_cc` returns a value of type `CutStatus`.
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

    /// The function `xc` returns a copy of the `xc` array.
    fn xc(&self) -> Self::ArrayType {
        self.xc.clone()
    }

    /// The `tsq` function returns the value of the `tsq` field of the struct.
    ///
    /// Returns:
    ///
    /// The method `tsq` is returning a value of type `f64`.
    fn tsq(&self) -> f64 {
        self.tsq
    }

    /// The `update_q` function updates the decision variable based on the given cut.
    /// 
    /// Arguments:
    /// 
    /// * `cut`: A tuple containing two elements:
    /// 
    /// Returns:
    /// 
    /// The `update_dc` function returns a value of type `CutStatus`.
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
