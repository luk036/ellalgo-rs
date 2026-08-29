//! LMI problem facade: owns data and drives the cutting-plane loop.
//!
//! The [`LMIProblem`] struct bundles the LMI coefficient matrices with a
//! lazily-created oracle and drives the cutting-plane feasibility method,
//! hiding the 3-step recipe (build oracle -> build search space -> call
//! driver) behind a single call.

use crate::arr::Arr;
use crate::cutting_plane::{cutting_plane_feas, Options, SingleCut};
use crate::ell_stable::EllStable;
use crate::oracles::lmi_oracle::LMIOracle;

/// LMI feasibility problem facade.
///
/// Owns the LMI data (`mat_f` and the constant term `mat_b`) and the
/// lazily-created [`LMIOracle`], then drives the cutting-plane method through
/// the standard [`cutting_plane_feas`] driver.
///
/// The LMI feasibility problem is:
///
/// ```text
/// find  x
/// s.t.  B − Σₖ Fₖ xₖ ⪰ 0   (positive semidefinite)
/// ```
pub struct LMIProblem {
    mat_f: Vec<Arr>,
    mat_b: Arr,
    omega: LMIOracle,
}

impl LMIProblem {
    /// Construct a new `LMIProblem`.
    ///
    /// # Arguments
    ///
    /// * `mat_f` - List of symmetric coefficient matrices `[F₁, F₂, ..., Fₙ]`
    /// * `mat_b` - Constant matrix `B` defining the LMI constraint
    pub fn new(mat_f: Vec<Arr>, mat_b: Arr) -> Self {
        let omega = LMIOracle::new(mat_f.clone(), mat_b.clone());
        LMIProblem {
            mat_f,
            mat_b,
            omega,
        }
    }

    /// Solve the LMI feasibility problem.
    ///
    /// Builds an [`EllStable`] search space with the given initial ellipsoid
    /// parameters and runs the cutting-plane feasibility method.
    ///
    /// # Arguments
    ///
    /// * `val` - Either a scalar (kappa) or per-axis values for the initial
    ///   ellipsoid
    /// * `x_center` - Initial center point
    /// * `options` - Algorithm control parameters
    ///
    /// # Returns
    ///
    /// A tuple `(solution, number of iterations)` where the solution is
    /// `None` if no feasible point was found.
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::arr::Arr;
    /// use ellalgo_rs::cutting_plane::Options;
    /// use ellalgo_rs::lmi_problem::LMIProblem;
    ///
    /// let f1 = Arr::with_data(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
    /// let f2 = Arr::with_data(vec![0.0, 1.0, 1.0, 0.0], 2, 2);
    /// let b = Arr::with_data(vec![2.0, 0.0, 0.0, 2.0], 2, 2);
    /// let mut problem = LMIProblem::new(vec![f1, f2], b);
    /// let (x, _) = problem.solve_feas(10.0, Arr::new(2), Options::default());
    /// assert!(x.is_some());
    /// ```
    pub fn solve_feas(
        &mut self,
        val: f64,
        x_center: Arr,
        options: Options,
    ) -> (Option<Arr>, usize) {
        let mut space = EllStable::new_with_scalar(val, x_center);
        cutting_plane_feas::<SingleCut, _, _>(&mut self.omega, &mut space, &options)
    }

    /// Access the coefficient matrices (for inspection).
    pub fn mat_f(&self) -> &[Arr] {
        &self.mat_f
    }

    /// Access the constant term matrix.
    pub fn mat_b(&self) -> &Arr {
        &self.mat_b
    }
}
