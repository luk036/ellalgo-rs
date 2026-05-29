//! Problem-specific oracles for the ellipsoid method.
//!
//! This module provides various oracle implementations that can be used with
//! the cutting plane algorithms to solve different types of optimization problems.
//!
//! ## Available Oracles
//!
//! - [`lowpass_oracle::LowpassOracle`] - Lowpass filter design oracle
//! - [`profit_oracle::ProfitOracle`] - Profit maximization oracle (Cobb-Douglas production)
//! - [`profit_oracle::ProfitRbOracle`] - Robust profit maximization oracle with interval uncertainty
//! - [`profit_oracle::ProfitOracleQ`] - Discrete profit maximization oracle
//!
//! ## Usage
//!
//! Each oracle implements either [`crate::cutting_plane::OracleFeas`] for feasibility problems or
//! [`crate::cutting_plane::OracleOptim`] for optimization problems.
//!
//! ```rust,ignore
//! use ellalgo_rs::arr::Arr;
//! use ellalgo_rs::oracles::profit_oracle::ProfitOracle;
//!
//! let params = (20.0, 40.0, 30.5);
//! let elasticities = Arr::from(vec![0.1, 0.4]);
//! let price_out = Arr::from(vec![10.0, 35.0]);
//! let oracle = ProfitOracle::new(params, elasticities, price_out);
//! ```

pub mod lowpass_oracle;
pub mod maxcut_oracle;
pub mod profit_oracle;
pub mod svm_oracle;
