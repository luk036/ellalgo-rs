//! Logging module for ellalgo-rs.
//!
//! This module provides optional logging capabilities for applications using ellalgo-rs.
//! It is only available when the `std` feature is enabled.
//!
//! ## Usage
//!
//! This module re-exports the `log` crate for use with ellalgo-rs.
//! Users can then use any logger implementation (`env_logger`, tracing, etc.)
//! in their application code.
//!
//! ```ignore
//! // In your application
//! use ellalgo_rs::log;
//!
//! fn main() {
//!     // Initialize your preferred logger
//!     // Then use log macros
//!     log::info!("Application started");
//! }
//! ```
//!
//! ## Example with `env_logger`
//!
//! Add `env_logger` to your Cargo.toml:
//! ```toml
//! [dependencies]
//! env_logger = "0.11"
//! ```
//!
//! Then in your application:
//! ```ignore
//! use ellalgo_rs::log;
//! use env_logger::Env;
//!
//! fn main() {
//!     env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
//!     log::info!("Application started");
//! }
//! ```

// Re-export log crate for convenience
pub use log;

// Re-export commonly used log level filter
#[cfg(feature = "std")]
pub use log::LevelFilter;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_re_exports() {
        // Verify log crate is re-exported
        let _ = log::LevelFilter::Info;
        let _ = log::LevelFilter::Debug;
        let _ = log::LevelFilter::Warn;
        let _ = log::LevelFilter::Error;
    }
}
