//! Property-based tests for ellalgo-rs using quickcheck.

use quickcheck::quickcheck;
use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};

use ellalgo_rs::cutting_plane::{CutStatus, SearchSpace};
use ellalgo_rs::ell::Ell;
use ellalgo_rs::ell_calc::{EllCalc, EllCalcCore};
use ndarray::prelude::*;
use ndarray::Array1;

type Arr = Array1<f64>;
type TestFn = fn() -> TestResult;

// Custom Arbitrary implementations

#[derive(Debug, Clone)]
struct PosFloat(f64);

impl Arbitrary for PosFloat {
    fn arbitrary(g: &mut Gen) -> Self {
        let mut value = f64::arbitrary(g);
        while value <= 0.0 || value.is_nan() || value.is_infinite() {
            value = f64::arbitrary(g);
        }
        PosFloat(value.abs() % 10.0 + 0.001)
    }
}

#[derive(Debug, Clone)]
struct TestEllCalcCore(EllCalcCore);

impl Arbitrary for TestEllCalcCore {
    fn arbitrary(g: &mut Gen) -> Self {
        let n = usize::arbitrary(g) % 8 + 1;
        TestEllCalcCore(EllCalcCore::new(n as f64))
    }
}

#[derive(Debug, Clone)]
struct SmallArray(Arr);

impl Arbitrary for SmallArray {
    fn arbitrary(g: &mut Gen) -> Self {
        let n = usize::arbitrary(g) % 4 + 1;
        let mut arr = Array1::zeros(n);
        for elem in arr.iter_mut() {
            let mut val = f64::arbitrary(g);
            while val.is_nan() || val.is_infinite() {
                val = f64::arbitrary(g);
            }
            *elem = (val % 20.0) - 10.0;
        }
        SmallArray(arr)
    }
}

#[derive(Debug, Clone)]
struct TestEll(Ell);

impl Arbitrary for TestEll {
    fn arbitrary(g: &mut Gen) -> Self {
        let n = usize::arbitrary(g) % 4 + 1;
        let kappa = PosFloat::arbitrary(g).0;
        let xc = Array1::zeros(n);
        TestEll(Ell::new_with_scalar(kappa, xc))
    }
}

// Property tests

fn prop_ell_construct_positive_kappa() -> TestResult {
    let test_ell = TestEll::arbitrary(&mut Gen::new(10));
    let ell = test_ell.0;
    TestResult::from_bool(ell.kappa > 0.0)
}

fn prop_ell_construct_tsq_nonnegative() -> TestResult {
    let test_ell = TestEll::arbitrary(&mut Gen::new(10));
    let ell = test_ell.0;
    TestResult::from_bool(ell.tsq >= 0.0)
}

fn prop_ell_central_cut_keeps_kappa_positive() -> TestResult {
    let mut test_ell = TestEll::arbitrary(&mut Gen::new(10));
    let ell = &mut test_ell.0;
    let n = ell.xc.len();
    let grad: Arr = Array1::from_vec(vec![0.01; n]);
    let cut = (grad, 0.0);
    let status = ell.update_central_cut(&cut);
    if status == CutStatus::Success {
        TestResult::from_bool(ell.kappa > 0.0)
    } else {
        TestResult::passed()
    }
}

fn prop_ell_bias_cut_keeps_kappa_positive() -> TestResult {
    let mut test_ell = TestEll::arbitrary(&mut Gen::new(10));
    let ell = &mut test_ell.0;
    let n = ell.xc.len();
    let grad: Arr = Array1::from_vec(vec![0.01; n]);
    let tsq: f64 = ell.kappa * ell.mq.dot(&grad).dot(&grad);
    let beta: f64 = (tsq.sqrt() * 0.5).abs();
    let cut = (grad, beta);
    let status = ell.update_bias_cut(&cut);
    if status == CutStatus::Success {
        TestResult::from_bool(ell.kappa > 0.0)
    } else {
        TestResult::passed()
    }
}

fn prop_ell_kappa_stays_positive() -> TestResult {
    let mut test_ell = TestEll::arbitrary(&mut Gen::new(10));
    let ell = &mut test_ell.0;
    let n = ell.xc.len();
    let grad: Arr = Array1::from_vec(vec![0.1; n]);
    let cut = (grad, 0.0);
    let _ = ell.update_central_cut(&cut);
    TestResult::from_bool(ell.kappa > 0.0)
}

fn prop_ellcalccore_nf_matches() -> TestResult {
    let test_core = TestEllCalcCore::arbitrary(&mut Gen::new(10));
    let core = test_core.0;
    TestResult::from_bool(core.n_f > 0.0)
}

fn prop_ellcalccore_half_n() -> TestResult {
    let test_core = TestEllCalcCore::arbitrary(&mut Gen::new(10));
    let core = test_core.0;
    TestResult::from_bool((core.half_n - core.n_f / 2.0).abs() < 1e-10)
}

fn prop_ellcalccore_n_plus_1() -> TestResult {
    let test_core = TestEllCalcCore::arbitrary(&mut Gen::new(10));
    let core = test_core.0;
    TestResult::from_bool((core.n_plus_1 - (core.n_f + 1.0)).abs() < 1e-10)
}

fn prop_calc_central_cut_valid() -> TestResult {
    let test_core = TestEllCalcCore::arbitrary(&mut Gen::new(10));
    let core = test_core.0;
    let tsq: f64 = 0.1;
    let (rho, sigma, delta) = core.calc_central_cut(tsq);
    TestResult::from_bool(rho > 0.0 && sigma > 0.0 && delta > 0.0)
}

fn prop_calc_bias_cut_valid() -> TestResult {
    let test_core = TestEllCalcCore::arbitrary(&mut Gen::new(10));
    let core = test_core.0;
    let tsq: f64 = 1.0;
    let beta: f64 = tsq.sqrt() * 0.5;
    let (rho, sigma, delta) = core.calc_bias_cut(beta, tsq.sqrt());
    TestResult::from_bool(rho > 0.0 && sigma > 0.0 && delta > 0.0)
}

fn prop_calc_parallel_bias_cut_valid() -> TestResult {
    let test_core = TestEllCalcCore::arbitrary(&mut Gen::new(10));
    let core = test_core.0;
    let tsq: f64 = 1.0;
    let (rho, sigma, delta) = core.calc_parallel_bias_cut(0.3, 0.6, tsq);
    TestResult::from_bool(rho > 0.0 && sigma > 0.0 && delta > 0.0)
}

fn prop_ell_dimension_consistency() -> TestResult {
    let test_ell = TestEll::arbitrary(&mut Gen::new(10));
    let ell = test_ell.0;
    let mq_shape = ell.mq.shape();
    let xc_shape = ell.xc.shape();
    TestResult::from_bool(mq_shape[0] == mq_shape[1] && mq_shape[0] == xc_shape[0])
}

fn prop_parallel_central_cut_keeps_kappa_positive() -> TestResult {
    let mut test_ell = TestEll::arbitrary(&mut Gen::new(10));
    let ell = &mut test_ell.0;
    let n = ell.xc.len();
    let grad: Arr = Array1::from_vec(vec![0.01; n]);
    let cut = (grad, (0.0, Some(0.05)));
    let status = ell.update_central_cut(&cut);
    if status == CutStatus::Success {
        TestResult::from_bool(ell.kappa > 0.0)
    } else {
        TestResult::passed()
    }
}

fn prop_bias_cut_no_effect() -> TestResult {
    let mut test_ell = TestEll::arbitrary(&mut Gen::new(10));
    let ell = &mut test_ell.0;
    let n = ell.xc.len();
    let grad: Arr = Array1::from_vec(vec![0.1; n]);
    let tsq: f64 = ell.kappa * grad.dot(&ell.mq.dot(&grad));
    let beta: f64 = tsq.sqrt() * 1.5;
    let cut = (grad, beta);
    let status = ell.update_bias_cut(&cut);
    TestResult::from_bool(status == CutStatus::NoEffect || status == CutStatus::NoSoln)
}

fn prop_multiple_cuts_keep_kappa_positive() -> TestResult {
    let mut ell = Ell::new_with_scalar(1.0, Array1::zeros(4));
    for _ in 0..5 {
        let grad: Arr = Array1::from_vec(vec![0.1; 4]);
        let cut = (grad, 0.0);
        let status = ell.update_central_cut(&cut);
        if status == CutStatus::Success && ell.kappa <= 0.0 {
            return TestResult::failed();
        }
    }
    TestResult::passed()
}

fn prop_ell_various_kappa() -> TestResult {
    let kappa_vals = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
    for &kappa in &kappa_vals {
        let ell = Ell::new_with_scalar(kappa, Array1::zeros(2));
        if ell.kappa <= 0.0 {
            return TestResult::failed();
        }
    }
    TestResult::passed()
}

fn prop_ell_identity_mq() -> TestResult {
    let ell = Ell::new_with_scalar(1.0, Array1::zeros(3));
    let expected_mq: Array2<f64> = Array2::eye(3);
    let mut is_identity = true;
    for i in 0..3 {
        for j in 0..3 {
            if (ell.mq[[i, j]] - expected_mq[[i, j]]).abs() > 1e-10 {
                is_identity = false;
                break;
            }
        }
    }
    TestResult::from_bool(is_identity)
}

fn prop_ellcalc_parallel_flag() -> TestResult {
    let ell_calc = EllCalc::new(4);
    TestResult::from_bool(ell_calc.use_parallel_cut)
}

fn prop_ell_single_dimension() -> TestResult {
    let ell = Ell::new_with_scalar(1.0, Array1::zeros(1));
    TestResult::from_bool(ell.kappa > 0.0)
}

fn prop_array_dimension_match() -> TestResult {
    let test_arr = SmallArray::arbitrary(&mut Gen::new(10));
    let arr = test_arr.0;
    let ell = Ell::new_with_scalar(1.0, arr.clone());
    TestResult::from_bool(ell.xc.len() == arr.len())
}

fn main() {
    println!("Running quickcheck property-based tests for ellalgo-rs...\n");

    let tests: Vec<(&str, TestFn)> = vec![
        (
            "ell_construct_positive_kappa",
            prop_ell_construct_positive_kappa,
        ),
        (
            "ell_construct_tsq_nonnegative",
            prop_ell_construct_tsq_nonnegative,
        ),
        (
            "ell_central_cut_keeps_kappa_positive",
            prop_ell_central_cut_keeps_kappa_positive,
        ),
        (
            "ell_bias_cut_keeps_kappa_positive",
            prop_ell_bias_cut_keeps_kappa_positive,
        ),
        ("ell_kappa_stays_positive", prop_ell_kappa_stays_positive),
        ("ellcalccore_nf_matches", prop_ellcalccore_nf_matches),
        ("ellcalccore_half_n", prop_ellcalccore_half_n),
        ("ellcalccore_n_plus_1", prop_ellcalccore_n_plus_1),
        ("calc_central_cut_valid", prop_calc_central_cut_valid),
        ("calc_bias_cut_valid", prop_calc_bias_cut_valid),
        (
            "calc_parallel_bias_cut_valid",
            prop_calc_parallel_bias_cut_valid,
        ),
        ("ell_dimension_consistency", prop_ell_dimension_consistency),
        (
            "parallel_central_cut_keeps_kappa_positive",
            prop_parallel_central_cut_keeps_kappa_positive,
        ),
        ("bias_cut_no_effect", prop_bias_cut_no_effect),
        (
            "multiple_cuts_keep_kappa_positive",
            prop_multiple_cuts_keep_kappa_positive,
        ),
        ("ell_various_kappa", prop_ell_various_kappa),
        ("ell_identity_mq", prop_ell_identity_mq),
        ("ellcalc_parallel_flag", prop_ellcalc_parallel_flag),
        ("ell_single_dimension", prop_ell_single_dimension),
        ("array_dimension_match", prop_array_dimension_match),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (name, test_fn) in tests {
        print!("Test {}: ", name);
        match QuickCheck::new().tests(100).quicktest(test_fn as TestFn) {
            Ok(n) => {
                println!("  Passed {}/100", n);
                passed += 1;
            }
            Err(_) => {
                println!("  FAILED");
                failed += 1;
            }
        }
    }

    println!("\n========================================");
    println!("Results: {} passed, {} failed", passed, failed);

    if failed == 0 {
        println!("Quickcheck integration verified!");
    } else {
        println!("Some tests failed - review the properties!");
    }
}

quickcheck! {
    fn qc_ell_construct_positive_kappa() -> TestResult { prop_ell_construct_positive_kappa() }
    fn qc_ell_construct_tsq_nonnegative() -> TestResult { prop_ell_construct_tsq_nonnegative() }
    fn qc_ell_central_cut_keeps_kappa_positive() -> TestResult { prop_ell_central_cut_keeps_kappa_positive() }
    fn qc_ell_bias_cut_keeps_kappa_positive() -> TestResult { prop_ell_bias_cut_keeps_kappa_positive() }
    fn qc_ell_kappa_stays_positive() -> TestResult { prop_ell_kappa_stays_positive() }
    fn qc_ellcalccore_nf_matches() -> TestResult { prop_ellcalccore_nf_matches() }
    fn qc_ellcalccore_half_n() -> TestResult { prop_ellcalccore_half_n() }
    fn qc_ellcalccore_n_plus_1() -> TestResult { prop_ellcalccore_n_plus_1() }
    fn qc_calc_central_cut_valid() -> TestResult { prop_calc_central_cut_valid() }
    fn qc_calc_bias_cut_valid() -> TestResult { prop_calc_bias_cut_valid() }
    fn qc_calc_parallel_bias_cut_valid() -> TestResult { prop_calc_parallel_bias_cut_valid() }
    fn qc_ell_dimension_consistency() -> TestResult { prop_ell_dimension_consistency() }
    fn qc_parallel_central_cut_keeps_kappa_positive() -> TestResult { prop_parallel_central_cut_keeps_kappa_positive() }
    fn qc_bias_cut_no_effect() -> TestResult { prop_bias_cut_no_effect() }
    fn qc_multiple_cuts_keep_kappa_positive() -> TestResult { prop_multiple_cuts_keep_kappa_positive() }
    fn qc_ell_various_kappa() -> TestResult { prop_ell_various_kappa() }
    fn qc_ell_identity_mq() -> TestResult { prop_ell_identity_mq() }
    fn qc_ellcalc_parallel_flag() -> TestResult { prop_ellcalc_parallel_flag() }
    fn qc_ell_single_dimension() -> TestResult { prop_ell_single_dimension() }
    fn qc_array_dimension_match() -> TestResult { prop_array_dimension_match() }
}
