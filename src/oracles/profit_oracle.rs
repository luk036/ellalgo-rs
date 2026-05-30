use crate::arr::Arr;
use crate::cutting_plane::{OracleOptim, OracleOptimQ, SingleCut};

type Cut = (Arr, SingleCut);

#[derive(Debug)]
pub struct ProfitOracle {
    idx: i32,
    log_p_scale: f64,
    log_k: f64,
    price_out: Arr,
    pub elasticities: Arr,
    log_cobb: f64,
    vx: f64,
    q: Arr,
}

impl ProfitOracle {
    pub fn new(params: (f64, f64, f64), elasticities: Arr, price_out: Arr) -> Self {
        let (unit_price, scale, limit) = params;
        let log_p_scale = (unit_price * scale).ln();
        let log_k = limit.ln();
        ProfitOracle {
            idx: -1,
            log_p_scale,
            log_k,
            price_out,
            elasticities,
            log_cobb: 0.0,
            vx: 0.0,
            q: Arr::new(2),
        }
    }

    fn assess_feas(&mut self, y: &Arr, gamma: &mut f64) -> Option<(Arr, f64)> {
        let num_constraints = 2;
        for _ in 0..num_constraints {
            self.idx += 1;
            if self.idx == num_constraints {
                self.idx = 0;
            }
            let fj = match self.idx {
                0 => y[0] - self.log_k,
                1 => {
                    self.log_cobb = self.log_p_scale + self.elasticities.dot(y);
                    self.q = &self.price_out * &y.map(f64::exp);
                    self.vx = self.q[0] + self.q[1];
                    (*gamma + self.vx).ln() - self.log_cobb
                }
                _ => unreachable!(),
            };
            if fj > 0.0 {
                return Some((
                    match self.idx {
                        0 => Arr::from(vec![1.0, 0.0]),
                        1 => &self.q / (*gamma + self.vx) - &self.elasticities,
                        _ => unreachable!(),
                    },
                    fj,
                ));
            }
        }
        None
    }
}

impl OracleOptim<Arr> for ProfitOracle {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, y: &Arr, gamma: &mut f64) -> ((Arr, SingleCut), bool) {
        if let Some((g, fj)) = self.assess_feas(y, gamma) {
            return ((g, SingleCut(fj)), false);
        }
        let exp_val = self.log_cobb.exp();
        *gamma = exp_val - self.vx;
        let grad = (&self.q / exp_val) - &self.elasticities;
        ((grad, SingleCut(0.0)), true)
    }
}

#[derive(Debug)]
pub struct ProfitRbOracle {
    uie: [f64; 2],
    omega: ProfitOracle,
    elasticities: Arr,
}

impl ProfitRbOracle {
    pub fn new(
        params: (f64, f64, f64),
        elasticities: Arr,
        price_out: Arr,
        vparams: (f64, f64, f64, f64, f64),
    ) -> Self {
        let (e1, e2, e3, e4, e5) = vparams;
        let uie = [e1, e2];
        let params_rb = (params.0 - e3, params.1, params.2 - e4);
        let omega = ProfitOracle::new(
            params_rb,
            elasticities.clone(),
            &price_out + &Arr::from(vec![e5, e5]),
        );
        ProfitRbOracle {
            uie,
            omega,
            elasticities,
        }
    }
}

impl OracleOptim<Arr> for ProfitRbOracle {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, y: &Arr, gamma: &mut f64) -> (Cut, bool) {
        let mut a_rb = self.elasticities.clone();
        for i in 0..2 {
            a_rb[i] += if y[i] > 0.0 {
                -self.uie[i]
            } else {
                self.uie[i]
            };
        }
        self.omega.elasticities = a_rb;
        self.omega.assess_optim(y, gamma)
    }
}

pub struct ProfitOracleQ {
    omega: ProfitOracle,
    yd: Arr,
}

impl ProfitOracleQ {
    pub fn new(params: (f64, f64, f64), elasticities: Arr, price_out: Arr) -> Self {
        ProfitOracleQ {
            yd: Arr::from(vec![0.0, 0.0]),
            omega: ProfitOracle::new(params, elasticities, price_out),
        }
    }
}

impl OracleOptimQ<Arr> for ProfitOracleQ {
    type CutChoice = SingleCut;

    fn assess_optim_q(&mut self, y: &Arr, gamma: &mut f64, retry: bool) -> (Cut, bool, Arr, bool) {
        if !retry {
            if let Some((g, fj)) = self.omega.assess_feas(y, gamma) {
                return ((g, SingleCut(fj)), false, y.clone(), true);
            }
            let mut x_disc = y.map(|x| x.exp().round());
            if x_disc[0] == 0.0 {
                x_disc[0] = 1.0;
            }
            if x_disc[1] == 0.0 {
                x_disc[1] = 1.0;
            }
            self.yd = x_disc.map(f64::ln);
        }
        let ((grad, SingleCut(beta)), shrunk) = self.omega.assess_optim(&self.yd, gamma);
        let beta = beta + grad.dot(&(&self.yd - y));
        ((grad, SingleCut(beta)), shrunk, self.yd.clone(), !retry)
    }
}

#[cfg(test)]
mod tests {
    use super::{ProfitOracle, ProfitOracleQ, ProfitRbOracle};
    use crate::arr::Arr;
    use crate::cutting_plane::{cutting_plane_optim, cutting_plane_optim_q, Options, OracleOptim};
    use crate::ell::Ell;

    #[test]
    pub fn test_profit_oracle() {
        let params = (20.0, 40.0, 30.5);
        let elasticities = Arr::from(vec![0.1, 0.4]);
        let price_out = Arr::from(vec![10.0, 35.0]);

        let mut ellip = Ell::new(Arr::from(vec![100.0, 100.0]), Arr::from(vec![0.0, 0.0]));
        let mut omega = ProfitOracle::new(params, elasticities, price_out);
        let mut gamma = 0.0;
        let options = Options::default();
        let (y_opt, niter) = cutting_plane_optim(&mut omega, &mut ellip, &mut gamma, &options);
        assert!(y_opt.is_some());
        if let Some(y) = y_opt {
            assert!(y[0] <= 30.5f64.ln());
        }
        assert_eq!(niter, 83, "regression test");
    }

    #[test]
    pub fn test_profit_oracle_rb() {
        let params = (20.0, 40.0, 30.5);
        let elasticities = Arr::from(vec![0.1, 0.4]);
        let price_out = Arr::from(vec![10.0, 35.0]);
        let vparams = (0.003, 0.007, 1.0, 1.0, 1.0);

        let mut ellip = Ell::new(Arr::from(vec![100.0, 100.0]), Arr::from(vec![0.0, 0.0]));
        let mut omega = ProfitRbOracle::new(params, elasticities, price_out, vparams);
        let mut gamma = 0.0;
        let options = Options::default();
        let (y_opt, niter) = cutting_plane_optim(&mut omega, &mut ellip, &mut gamma, &options);
        assert!(y_opt.is_some());
        if let Some(y) = y_opt {
            assert!(y[0] <= 30.5f64.ln());
        }
        assert_eq!(niter, 90, "regression test");
    }

    #[test]
    pub fn test_profit_oracle_q() {
        let params = (20.0, 40.0, 30.5);
        let elasticities = Arr::from(vec![0.1, 0.4]);
        let price_out = Arr::from(vec![10.0, 35.0]);

        let mut ellip = Ell::new(Arr::from(vec![100.0, 100.0]), Arr::from(vec![0.0, 0.0]));
        let mut omega = ProfitOracleQ::new(params, elasticities, price_out);
        let mut gamma = 0.0;
        let options = Options::default();
        let (y_opt, niter) = cutting_plane_optim_q(&mut omega, &mut ellip, &mut gamma, &options);
        assert!(y_opt.is_some());
        if let Some(y) = y_opt {
            assert!(y[0] <= 30.5f64.ln());
        }
        assert_eq!(niter, 29, "regression test");
    }

    #[test]
    fn test_profit_oracle_direct() {
        let params = (20.0, 40.0, 30.5);
        let elasticities = Arr::from(vec![0.1, 0.4]);
        let price_out = Arr::from(vec![10.0, 35.0]);
        let mut omega = ProfitOracle::new(params, elasticities, price_out);
        let mut gamma = 0.0;
        let y_vec = Arr::from(vec![3.5, 2.0]);
        let (cut, feasible) = omega.assess_optim(&y_vec, &mut gamma);
        assert!(!feasible);
        assert_eq!(cut.1 .0, 3.5 - 30.5f64.ln());
        let y2 = Arr::from(vec![3.0, 2.0]);
        let (cut2, feasible2) = omega.assess_optim(&y2, &mut gamma);
        assert!(feasible2);
        assert_eq!(cut2.1 .0, 0.0);
    }
}
