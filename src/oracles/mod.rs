//! Problem-specific oracles for the ellipsoid method.
//!
//! This module provides various oracle implementations that can be used with
//! the cutting plane algorithms to solve different types of optimization problems.
//!
//! ## Available Oracles
//!
//! - [`ldlt_mgr::LDLTMgr`] - LDL^T factorization manager for positive definiteness checking
//! - [`lmi_oracle::LMIOracle`] - Linear Matrix Inequality (LMI) feasibility oracle
//! - [`lowpass_oracle::LowpassOracle`] - Lowpass filter design oracle
//! - [`profit_oracle::ProfitOracle`] - Profit maximization oracle (Cobb-Douglas production)
//! - [`profit_oracle::ProfitRbOracle`] - Robust profit maximization oracle with interval uncertainty
//! - [`profit_oracle::ProfitOracleQ`] - Discrete profit maximization oracle
//!
//! ## Usage
//!
//! Each oracle implements either [`OracleFeas`] for feasibility problems or
//! [`OracleOptim`] for optimization problems from the [`crate::cutting_plane`] module.
//!
//! ```rust
//! use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim};
//! use ellalgo_rs::ell::Ell;
//! use ellalgo_rs::oracles::profit_oracle::ProfitOracle;
//! use ndarray::array;
//!
//! // Create oracle for profit maximization problem
//! let params = (20.0, 40.0, 30.5);
//! let elasticities = array![0.1, 0.4];
//! let price_out = array![10.0, 35.0];
//! let oracle = ProfitOracle::new(params, elasticities, price_out);
//! ```

pub mod ldlt_mgr;
pub mod lmi_oracle;
pub mod lowpass_oracle;
pub mod profit_oracle;
