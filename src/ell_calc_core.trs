use std::f64::sqrt;

struct EllCalcCore {
    n_f: f64,
    half_n: f64,
    n_plus_1: f64,
    cst0: f64,
    cst1: f64,
    cst2: f64,
    cst3: f64,
}

impl EllCalcCore {
    fn new(n_f: f64) -> Self {
        let half_n = n_f / 2.0;
        let n_plus_1 = n_f + 1.0;
        let n_sq = n_f * n_f;
        let cst0 = 1.0 / n_plus_1;
        let cst1 = n_sq / (n_sq - 1.0);
        let cst2 = 2.0 * cst0;
        let cst3 = n_f * cst0;

        EllCalcCore {
            n_f,
            half_n,
            n_plus_1,
            cst0,
            cst1,
            cst2,
            cst3,
        }
    }

    fn calc_parallel_central_cut(&self, beta1: f64, tsq: f64) -> (f64, f64, f64) {
        let b1sq = beta1 * beta1;
        let a1sq = b1sq / tsq;
        let temp = self.half_n * a1sq;
        let mu_plus_1 = temp + sqrt(1.0 - a1sq + temp * temp);
        let mu_plus_2 = mu_plus_1 + 1.0;
        let rho = beta1 / mu_plus_2;
        let sigma = 2.0 / mu_plus_2;
        let temp2 = self.n_f * mu_plus_1;
        let delta = temp2 / (temp2 - 1.0);

        (rho, sigma, delta)
    }

    fn calc_parallel_deep_cut(&self, beta0: f64, beta1: f64, tsq: f64) -> (f64, f64, f64) {
        let b0b1 = beta0 * beta1;
        let bsum = beta0 + beta1;
        let bsumsq = bsum * bsum;
        let gamma = tsq + self.n_f * b0b1;
        let h = tsq + b0b1 + self.half_n * bsumsq;
        let temp2 = h + sqrt(h * h - gamma * self.n_plus_1 * bsumsq);
        let inv_mu_plus_2 = gamma / temp2;
        let inv_mu = gamma / (temp2 - 2.0 * gamma);
        let rho = bsum * inv_mu_plus_2;
        let sigma = 2.0 * inv_mu_plus_2;
        let delta = 1.0 + (-2.0 * b0b1 + bsumsq * inv_mu_plus_2) * inv_mu / tsq;

        (rho, sigma, delta)
    }
}
