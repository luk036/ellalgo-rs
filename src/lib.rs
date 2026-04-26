//! # ellalgo-rs
//!
//! A Rust implementation of the Ellipsoid Method for convex optimization.
//!
//! This crate provides an efficient implementation of the ellipsoid method algorithm,
//! which is a polynomial-time algorithm for solving linear programming problems and
//! more general convex optimization problems.
//!
//! ## Key Features
//!
//! - **Ellipsoid Search Space**: Multiple implementations including standard and stable variants
//! - **Cutting Plane Algorithms**: Support for feasibility and optimization problems
//! - **Parallel Cuts**: Efficient handling of parallel constraints
//! - **Oracle Framework**: Flexible interface for defining problem-specific oracles
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim};
//! use ellalgo_rs::ell::Ell;
//! use ndarray::prelude::*;
//!
//! type Arr = Array1<f64>;
//!
//! struct MyOracle;
//!
//! impl OracleOptim<Arr> for MyOracle {
//!     type CutChoice = f64;
//!
//!     fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
//!         ((array![0.0, 0.0], 0.0), true)
//!     }
//! }
//!
//! let mut ellip = Ell::new_with_scalar(10.0, array![0.0, 0.0]);
//! let mut oracle = MyOracle;
//! let mut gamma = f64::NEG_INFINITY;
//! let options = Options::default();
//!
//! let (xbest, num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);
//! println!("Best solution: {:?}", xbest);
//! println!("Iterations: {}", num_iters);
//! ```
//!
//! ## Modules
//!
//! - [`cutting_plane`] - Core cutting plane algorithms and traits
//! - [`ell`] - Main ellipsoid search space implementation
//! - [`ell_calc`] - Core calculations for ellipsoid updates
//! - [`ell_stable`] - Numerically stable ellipsoid implementation
//! - [`oracles`] - Problem-specific oracle implementations
//! - [`error`] - Error types for the library
//! - [`quasicvx`] - Quasiconvex optimization examples
//! - [`power_iteration`] - Power iteration for eigenvalue computation
//!
//! ## Feature Flags
//!
//! - `std`: Enables standard library features (enabled by default)
//! - `logging`: Enables logging support

// pub mod conceptual;
pub mod cutting_plane;
pub use crate::cutting_plane::{
    CutStatus, OracleFeas, OracleOptim, SearchSpace, UpdateByCutChoice,
};

pub mod ell;
pub mod ell_calc;
pub mod ell_stable;
pub mod error;
pub mod oracles;
pub mod power_iteration;
pub mod quasicvx;

#[cfg(test)]
pub mod example1;

#[cfg(test)]
pub mod example1_rr;

#[cfg(test)]
pub mod example3;

#[cfg(test)]
pub mod example4;

#[cfg(test)]
pub mod ell_calc_additional_tests;

#[cfg(test)]
pub mod ell_test;

#[cfg(feature = "std")]
pub mod logging;
