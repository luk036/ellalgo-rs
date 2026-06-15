//! Port of Python test_stress.py — Stress tests with lowpass oracle.
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options};
use ellalgo_rs::ell::Ell;
use ellalgo_rs::oracles::lowpass_oracle::create_lowpass_case;

#[test]
fn test_stress_lowpass_high_dimension() {
    let n = 128;
    let mut omega = create_lowpass_case(n);
    let mut v = Ell::new_with_scalar(1.0, vec![0.0; n].into());
    let mut sp_sq = omega.sp_sq;
    let options = Options::new(50000, 1e-14);
    let (_x, _num_iters) = cutting_plane_optim(&mut omega, &mut v, &mut sp_sq, &options);
}

#[test]
fn test_stress_lowpass_many_iterations() {
    let n = 32;
    let mut omega = create_lowpass_case(n);
    let mut v = Ell::new_with_scalar(1.0, vec![0.0; n].into());
    let mut sp_sq = 1e-12;
    let options = Options::new(50000, 1e-14);
    let (_x, _num_iters) = cutting_plane_optim(&mut omega, &mut v, &mut sp_sq, &options);
}
