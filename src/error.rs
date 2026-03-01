//! Error types for the ellalgo-rs library
//!
//! This module provides comprehensive error handling for the ellipsoid method
//! and cutting plane algorithms.

use std::fmt;

/// Errors that can occur during ellipsoid method optimization
#[derive(Debug, Clone, PartialEq)]
pub enum EllipsoidError {
    /// The algorithm failed to converge within the maximum number of iterations
    NonConvergence {
        /// Number of iterations performed
        iterations: usize,
        /// Maximum allowed iterations
        max_iters: usize,
        /// Final objective value (if available)
        final_value: Option<f64>,
    },

    /// The problem is infeasible - no solution satisfies all constraints
    Infeasible {
        /// Reason for infeasibility
        reason: String,
    },

    /// Numerical instability detected
    NumericalInstability {
        /// Description of the instability
        details: String,
        /// The problematic value (if available)
        value: Option<f64>,
    },

    /// Invalid input parameters
    InvalidParameters {
        /// Description of the invalid parameters
        details: String,
    },

    /// Matrix operation failed
    MatrixError {
        /// Description of the matrix error
        details: String,
    },
}

impl fmt::Display for EllipsoidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EllipsoidError::NonConvergence {
                iterations,
                max_iters,
                final_value,
            } => {
                write!(
                    f,
                    "Algorithm failed to converge after {} iterations (max: {})",
                    iterations, max_iters
                )?;
                if let Some(val) = final_value {
                    write!(f, ". Final objective value: {}", val)?;
                }
                Ok(())
            }
            EllipsoidError::Infeasible { reason } => {
                write!(f, "Problem is infeasible: {}", reason)
            }
            EllipsoidError::NumericalInstability { details, value } => {
                write!(f, "Numerical instability detected: {}", details)?;
                if let Some(val) = value {
                    write!(f, " (value: {})", val)?;
                }
                Ok(())
            }
            EllipsoidError::InvalidParameters { details } => {
                write!(f, "Invalid parameters: {}", details)
            }
            EllipsoidError::MatrixError { details } => {
                write!(f, "Matrix operation failed: {}", details)
            }
        }
    }
}

impl std::error::Error for EllipsoidError {}

/// Result type alias for ellipsoid method operations
pub type EllipsoidResult<T> = Result<T, EllipsoidError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = EllipsoidError::NonConvergence {
            iterations: 100,
            max_iters: 200,
            final_value: Some(0.5),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("failed to converge"));
        assert!(msg.contains("100"));
        assert!(msg.contains("200"));

        let err = EllipsoidError::Infeasible {
            reason: "Constraints conflict".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("infeasible"));
        assert!(msg.contains("Constraints conflict"));
    }

    #[test]
    fn test_error_clone() {
        let err = EllipsoidError::NumericalInstability {
            details: "Division by zero".to_string(),
            value: Some(0.0),
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }
}
