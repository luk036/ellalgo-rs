// mod lib;
use crate::cutting_plane::{CutStatus, SearchSpace, SearchSpaceQ, UpdateByCutChoices};
use crate::ell_calc::EllCalc;
// #[macro_use]
// extern crate ndarray;
// use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Array2;

/// The code defines a struct called [`EllStable`] that represents the stable version of an ellipsoid
/// search space in the Ellipsoid method.
///
///   EllStable = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
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
pub struct EllStable {
    pub no_defer_trick: bool,

    mq: Array2<f64>,
    xc: Array1<f64>,
    kappa: f64,
    ndim: usize,
    helper: EllCalc,
    tsq: f64,
}

impl EllStable {
    /// The function `new_with_matrix` constructs a new [`EllStable`] object with the given parameters.
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
    /// an instance of the [`EllStable`] struct.
    pub fn new_with_matrix(kappa: f64, mq: Array2<f64>, xc: Array1<f64>) -> EllStable {
        let ndim = xc.len();
        let helper = EllCalc::new(ndim as f64);

        EllStable {
            kappa,
            mq,
            xc,
            ndim,
            helper,
            no_defer_trick: false,
            tsq: 0.0,
        }
    }

    /// Creates a new [`EllStable`].
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
    /// The function `new` returns an instance of the [`EllStable`] struct.
    pub fn new(val: Array1<f64>, xc: Array1<f64>) -> EllStable {
        EllStable::new_with_matrix(1.0, Array2::from_diag(&val), xc)
    }

    /// The function `new_with_scalar` constructs a new [`EllStable`] object with a scalar value and a vector.
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
    /// an instance of the [`EllStable`] struct.
    pub fn new_with_scalar(val: f64, xc: Array1<f64>) -> EllStable {
        EllStable::new_with_matrix(val, Array2::eye(xc.len()), xc)
    }

    /// Update ellipsoid core function using the cut
    ///
    ///  $grad^T * (x - xc) + beta <= 0$
    ///
    /// The `update_core` function in Rust updates the ellipsoid core based on a given gradient and beta
    /// value using a cut strategy.
    ///
    /// Reference:
    /// Gill, Murray, and Wright, "Practical Optimization", p43. Author: Brian Borchers (borchers@nmt.edu)
    ///
    /// Arguments:
    ///
    /// * `grad`: A reference to an Array1<f64> representing the gradient vector.
    /// * `beta`: The `beta` parameter is a value that is used in the inequality constraint of the ellipsoid
    /// core function. It represents the threshold for the constraint, and the function checks if the dot
    /// product of the gradient and the difference between `x` and `xc` plus `beta` is less than or
    /// * `f_core`: The `f_core` parameter is a closure that takes two arguments: `beta` and `tsq`. It
    /// returns a tuple containing a `CutStatus` and a tuple `(rho, sigma, delta)`. The `f_core` function is
    /// used to determine the values of `rho`, `
    ///
    /// Returns:
    ///
    /// The `update_core` function returns a value of type `CutStatus`.
    fn update_core<T, F>(&mut self, grad: &Array1<f64>, beta: &T, f_core: F) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Array1<f64>>,
        F: FnOnce(&T, &f64) -> (CutStatus, (f64, f64, f64)),
    {
        // calculate inv(L)*grad: (n-1)*n/2 multiplications
        let mut inv_ml_g = grad.clone(); // initial x0
        for i in 1..self.ndim {
            for j in 0..i {
                self.mq[[i, j]] = self.mq[[j, i]] * inv_ml_g[j];
                // keep for rank-one update
                inv_ml_g[i] -= self.mq[[i, j]];
            }
        }

        // calculate inv(D)*inv(L)*grad: n
        let mut inv_md_inv_ml_g = inv_ml_g.clone(); // initially
        for i in 0..self.ndim {
            inv_md_inv_ml_g[i] *= self.mq[[i, i]];
        }

        // calculate omega: n
        let mut gg_t = inv_md_inv_ml_g.clone(); // initially
        let mut omega = 0.0; // initially
        for i in 0..self.ndim {
            gg_t[i] *= inv_ml_g[i];
            omega += gg_t[i];
        }

        self.tsq = self.kappa * omega;
        let (status, (rho, sigma, delta)) = f_core(beta, &self.tsq);

        if status != CutStatus::Success {
            return status;
        }

        // calculate mq*grad = inv(L')*inv(D)*inv(L)*grad : (n-1)*n/2
        let mut g_t = inv_md_inv_ml_g.clone(); // initially
        for i in (1..self.ndim).rev() {
            // backward subsituition
            for j in i..self.ndim {
                g_t[i - 1] -= self.mq[[i - 1, j]] * g_t[j]; // ???
            }
        }

        // calculate xc: n
        self.xc -= &((rho / omega) * &g_t); // n

        // Rank-one update: 3*n + (n-1)*n/2
        //
        // Reference:
        // Gill, Murray, and Wright, "Practical Optimization", p43. Author: Brian Borchers (borchers@nmt.edu)
        //
        // let r = self.sigma / omega;
        let mu = sigma / (1.0 - sigma);
        let mut oldt = omega / mu; // initially
        let m = self.ndim - 1;
        for j in 0..m {
            // p=sqrt(k)*vv[j];
            // let p = inv_ml_g[j];
            // let mup = mu * p;
            let t = oldt + gg_t[j];
            // self.mq[[j, j]] /= t; // update invD
            let beta2 = inv_md_inv_ml_g[j] / t;
            self.mq[[j, j]] *= oldt / t; // update invD
            for l in (j + 1)..self.ndim {
                // v(l) -= p * self.mq(j, l);
                self.mq[[j, l]] += beta2 * self.mq[[l, j]];
            }
            oldt = t;
        }

        // let p = inv_ml_g(n1);
        // let mup = mu * p;
        let t = oldt + gg_t[m];
        self.mq[[m, m]] *= oldt / t; // update invD
        self.kappa *= delta;

        // if self.no_defer_trick
        // {
        //     self.mq *= self.kappa;
        //     self.kappa = 1.;
        // }
        status
    }
}

impl SearchSpace for EllStable {
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

    /// The `update_deep_cut` function updates the decision variable based on the given cut.
    ///
    /// Arguments:
    ///
    /// * `cut`: A tuple containing two elements:
    ///
    /// Returns:
    ///
    /// The `update_deep_cut` function returns a value of type `CutStatus`.
    fn update_deep_cut<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_deep_cut_by(self, grad)
    }

    /// The `update_central_cut` function updates the cut choices using the gradient and beta values.
    ///
    /// Arguments:
    ///
    /// * `cut`: The `cut` parameter is a tuple containing two elements. The first element is of type
    /// `Self::ArrayType`, and the second element is of type `T`.
    ///
    /// Returns:
    ///
    /// The function `update_central_cut` returns a value of type `CutStatus`.
    fn update_central_cut<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_central_cut_by(self, grad)
    }
}

impl SearchSpaceQ for EllStable {
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
    /// The `update_deep_cut` function returns a value of type `CutStatus`.
    fn update_q<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
    {
        let (grad, beta) = cut;
        beta.update_q_by(self, grad)
    }
}

impl UpdateByCutChoices<EllStable> for f64 {
    type ArrayType = Array1<f64>;

    fn update_deep_cut_by(&self, ellip: &mut EllStable, grad: &Self::ArrayType) -> CutStatus {
        let beta = self;
        let helper = ellip.helper.clone();
        ellip.update_core(grad, beta, |beta, tsq| helper.calc_deep_cut(beta, tsq))
    }

    fn update_central_cut_by(&self, ellip: &mut EllStable, grad: &Self::ArrayType) -> CutStatus {
        let beta = self;
        let helper = ellip.helper.clone();
        ellip.update_core(grad, beta, |_beta, tsq| helper.calc_central_cut(tsq))
    }

    fn update_q_by(&self, ellip: &mut EllStable, grad: &Self::ArrayType) -> CutStatus {
        let beta = self;
        let helper = ellip.helper.clone();
        ellip.update_core(grad, beta, |beta, tsq| helper.calc_deep_cut_q(beta, tsq))
    }
}

impl UpdateByCutChoices<EllStable> for (f64, Option<f64>) {
    type ArrayType = Array1<f64>;

    fn update_deep_cut_by(&self, ellip: &mut EllStable, grad: &Self::ArrayType) -> CutStatus {
        let beta = self;
        let helper = ellip.helper.clone();
        ellip.update_core(grad, beta, |beta, tsq| {
            helper.calc_single_or_parallel_deep_cut(beta, tsq)
        })
    }

    fn update_central_cut_by(&self, ellip: &mut EllStable, grad: &Self::ArrayType) -> CutStatus {
        let beta = self;
        let helper = ellip.helper.clone();
        ellip.update_core(grad, beta, |beta, tsq| {
            helper.calc_single_or_parallel_central_cut(beta, tsq)
        })
    }

    fn update_q_by(&self, ellip: &mut EllStable, grad: &Self::ArrayType) -> CutStatus {
        let beta = self;
        let helper = ellip.helper.clone();
        ellip.update_core(grad, beta, |beta, tsq| {
            helper.calc_single_or_parallel_q(beta, tsq)
        })
    }
}
