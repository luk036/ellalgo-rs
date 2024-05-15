use std::f64::consts::PI;
use ndarray::{Array1, Array2, Axis};
use ndarray::prelude::*;
use ndarray_linalg::Lapack;
use ndarray_linalg::Eig;

fn spectral_fact(r: &Array1<f64>) -> Array1<f64> {
    // length of the impulse response sequence
    let n = r.len();

    // over-sampling factor
    let mult_factor = 100; // should have mult_factor*(n) >> n
    let m = mult_factor * n;

    // compute 1/2*ln(R(w))
    let w = Array1::linspace(0.0, 2.0 * PI, m);
    let mbn = w.outer_product(&Array1::linspace(1.0, (n - 1) as f64, n - 1));
    let man = 2.0 * mbn.map(|x| x.cos());
    let mr = Array2::from_columns(&[Array1::ones(m), man]).dot(r);
    let alpha = 0.5 * mr.map(|x| x.abs().ln());

    // find the Hilbert transform
    let mut alphatmp = alpha.fft();
    let ind = m / 2;
    alphatmp[(ind as usize)..(m as usize)].assign(&-alphatmp[(ind as usize)..(m as usize)]);
    alphatmp[0] = 0.0;
    alphatmp[ind as usize] = 0.0;
    let phi = alphatmp.ifft().map(|x| x.re);

    // now retrieve the original sampling
    let index = (0..m).step_by(mult_factor);
    let alpha1 = alpha.index_axis(Axis(0), &index);
    let phi1 = phi.index_axis(Axis(0), &index);

    // compute the impulse response (inverse Fourier transform)
    let h = alpha1.zip_with(&phi1, |a, p| (a + 1.0j * p).exp().re).slice(s![..n]).to_owned();

    h
}

fn inverse_spectral_fact(h: &Array1<f64>) -> Array1<f64> {
    let n = h.len();
    let mut r = Array1::zeros(n);
    for t in 0..n {
        r[t] = h[t..].dot(&h[..n - t]);
    }
    r
}

#[cfg(test)]
fn test_spectral_fact() {
    let h = array![0.76006445, 0.54101887, 0.42012073, 0.3157191, 0.10665804, 0.04326203, 0.01315678];
    let r = inverse_spectral_fact(&h);
    let h2 = spectral_fact(&r);
    assert_eq!(h.len(), h2.len());
    assert!(h2.all_close(&h, 1e-6));
}

