use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

fn spectral_fact(r: &[f64]) -> Vec<f64> {
    // length of the impulse response sequence
    let nr = r.len();
    let n = (nr + 1) / 2;
    // over-sampling factor
    let mult_factor = 30; // should have mult_factor*(n) >> n
    let m = mult_factor * n;
    // computation method:
    // H(exp(jTw)) = alpha(w) + j*phi(w)
    // where alpha(w) = 1/2*ln(R(w)) and phi(w) = Hilbert_trans(alpha(w))
    // compute 1/2*ln(R(w))
    let w: Vec<f64> = (0..m).map(|i| 2.0 * std::f64::consts::PI * i as f64 / m as f64).collect();
    let mut R = vec![Complex::zero(); m * (2 * n - 1)];
    for i in 0..m {
        for j in -(n - 1)..n {
            let k = i * (2 * n - 1) + j + n - 1;
            R[k] = Complex::from_polar(&1.0, &(-w[i] * j as f64)) * r[j + n - 1];
        }
    }
    let mut R = R.iter().map(|&x| x.re.abs()).collect::<Vec<f64>>(); // remove numerical noise from the imaginary part
    let alpha = R.iter().map(|&x| 0.5 * x.ln()).collect::<Vec<f64>>();
    // find the Hilbert transform
    let mut alpha_tmp = vec![Complex::zero(); m];
    for i in 0..m {
        alpha_tmp[i] = Complex::new(alpha[i], 0.0);
    }
    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(m);
    fft.process(&mut alpha_tmp);
    for i in 1..m / 2 {
        alpha_tmp[i] = -alpha_tmp[i];
        alpha_tmp[m - i] = alpha_tmp[i].conj();
    }
    alpha_tmp[0] = Complex::zero();
    alpha_tmp[m / 2] = Complex::zero();
    let mut phi_tmp = vec![Complex::zero(); m];
    let ifft = planner.plan_ifft(m);
    ifft.process(&mut alpha_tmp, &mut phi_tmp);
    let phi = phi_tmp.iter().map(|&x| x.re).collect::<Vec<f64>>();
    // now retrieve the original sampling
    let index = (0..m).filter(|&i| i % mult_factor == 0).collect::<Vec<usize>>();
    let alpha1 = index.iter().map(|&i| alpha[i]).collect::<Vec<f64>>();
    let phi1 = index.iter().map(|&i| phi[i]).collect::<Vec<f64>>();
    // compute the impulse response (inverse Fourier transform)
    let mut h = vec![Complex::zero(); n];
    for i in 0..n {
        h[i] = Complex::from_polar(&1.0, &(alpha1[i] + phi1[i])) * Complex::from(r[i]);
    }
    let mut planner = FFTplanner::new(true);
    let ifft = planner.plan_ifft(n);
    ifft.process(&mut h);
    h.iter().map(|&x| x.re).collect::<Vec<f64>>()
}

