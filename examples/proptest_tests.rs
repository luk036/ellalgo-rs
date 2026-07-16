//! Property-based tests using proptest for ellalgo-rs
//!
//! Run with: `cargo test --example proptest_tests`

use proptest::prelude::*;

proptest! {
    #[test]
    fn ell_construct_positive_kappa(kappa in 0.001f64..100.0) {
        let ell = Ell::new_with_scalar(kappa, Arr::new(2));
        assert!(ell.kappa > 0.0);
    }

    #[test]
    fn ell_construct_tsq_nonnegative(kappa in 0.001f64..100.0) {
        let ell = Ell::new_with_scalar(kappa, Arr::new(2));
        assert!(ell.tsq >= 0.0);
    }

    #[test]
    fn ell_single_dimension(kappa in 0.001f64..100.0) {
        let ell = Ell::new_with_scalar(kappa, Arr::new(1));
        assert!(ell.kappa > 0.0);
    }
}

fn main() {
    println!("Run `cargo test --example proptest_tests` to execute proptest tests.");
}
