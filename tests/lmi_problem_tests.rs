//! Tests for the LMIProblem facade and LMI oracle factory.

use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{Options, OracleFeas};
use ellalgo_rs::lmi_problem::LMIProblem;
use ellalgo_rs::oracles::lmi_factory::{make_lmi0_oracle, make_lmi_old_oracle, make_lmi_oracle};

/// Return the standard 2x2 LMI test data (F, B).
fn sample_problem() -> (Vec<Arr>, Arr) {
    let mat_f = vec![
        Arr::with_data(vec![-7.0, -11.0, -11.0, 3.0], 2, 2),
        Arr::with_data(vec![7.0, -18.0, -18.0, 8.0], 2, 2),
        Arr::with_data(vec![-2.0, -8.0, -8.0, 1.0], 2, 2),
    ];
    let mat_b = Arr::with_data(vec![33.0, -9.0, -9.0, 26.0], 2, 2);
    (mat_f, mat_b)
}

#[test]
fn test_make_lmi_oracle_factory_produces_working_oracle() {
    let (mat_f, mat_b) = sample_problem();
    let mut omega = make_lmi_oracle(mat_f, mat_b);
    let x = Arr::new(3);
    assert!(omega.assess_feas(&x).is_none()); // origin is feasible (B is PD)
}

#[test]
fn test_make_lmi_old_oracle_equivalent() {
    let (mat_f, mat_b) = sample_problem();
    let x = Arr::new(3);
    let lazy_cut = make_lmi_oracle(mat_f.clone(), mat_b.clone()).assess_feas(&x);
    let old_cut = make_lmi_old_oracle(mat_f, mat_b).assess_feas(&x);
    assert!(lazy_cut.is_none() && old_cut.is_none());
}

#[test]
fn test_make_lmi0_oracle_factory() {
    let mat_f = vec![
        Arr::with_data(vec![1.0, 0.0, 0.0, 0.0], 2, 2),
        Arr::with_data(vec![0.0, 1.0, 1.0, 0.0], 2, 2),
        Arr::with_data(vec![0.0, 0.0, 0.0, 1.0], 2, 2),
    ];
    let mut omega = make_lmi0_oracle(mat_f);
    assert!(omega.assess_feas(&Arr::from(vec![1.0, 0.0, 1.0])).is_none());
    assert!(omega
        .assess_feas(&Arr::from(vec![-1.0, 0.0, -1.0]))
        .is_some());
}

#[test]
fn test_lmi_problem_facade_solves_feasibility() {
    let (mat_f, mat_b) = sample_problem();
    let mut problem = LMIProblem::new(mat_f, mat_b);
    let (x, niter) = problem.solve_feas(10.0, Arr::new(3), Options::default());
    assert!(x.is_some());
    assert!(niter < 2000);
}

#[test]
fn test_lmi_problem_facade_exposes_data() {
    let (mat_f, mat_b) = sample_problem();
    let problem = LMIProblem::new(mat_f.clone(), mat_b.clone());
    assert_eq!(problem.mat_f().len(), 3);
    assert_eq!(*problem.mat_b(), mat_b);
}
