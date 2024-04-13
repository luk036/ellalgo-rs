use std::f64::consts::PI;
// use ndarray::{stack, Axis, Array, Array1, Array2};
use crate::cutting_plane::{OracleFeas, OracleOptim};
use ndarray::{Array, Array1};

type Arr = Array1<f64>;
pub type Cut = (Arr, (f64, Option<f64>));

pub struct LowpassOracle {
    pub more_alt: bool,
    pub idx1: usize,
    pub spectrum: Vec<Arr>,
    pub nwpass: usize,
    pub nwstop: usize,
    pub lp_sq: f64,
    pub up_sq: f64,
    pub sp_sq: f64,
    pub idx2: usize,
    pub idx3: usize,
    pub fmax: f64,
    pub kmax: usize,
}

impl LowpassOracle {
    pub fn new(ndim: usize, wpass: f64, wstop: f64, lp_sq: f64, up_sq: f64, sp_sq: f64) -> Self {
        let mdim = 15 * ndim;
        let w: Array1<f64> = Array::linspace(0.0, std::f64::consts::PI, mdim);
        // let tmp: Array2<f64> = Array::from_shape_fn((mdim, ndim - 1), |(i, j)| 2.0 * (w[i] * (j + 1) as f64).cos());
        // let spectrum: Array2<f64> = stack![Axis(1), Array::ones(mdim).insert_axis(Axis(1)), tmp];

        let mut spectrum = vec![Arr::zeros(ndim); mdim];
        for i in 0..mdim {
            spectrum[i][0] = 1.0;
            for j in 1..ndim {
                spectrum[i][j] = 2.0 * (w[i] * j as f64).cos();
            }
        }
        // spectrum.iter_mut().for_each(|row| row.insert(0, 1.0));

        let nwpass = (wpass * (mdim - 1) as f64).floor() as usize + 1;
        let nwstop = (wstop * (mdim - 1) as f64).floor() as usize + 1;

        Self {
            more_alt: true,
            idx1: 0,
            spectrum,
            nwpass,
            nwstop,
            lp_sq,
            up_sq,
            sp_sq,
            idx2: nwpass,
            idx3: nwstop,
            fmax: f64::NEG_INFINITY,
            kmax: 0,
        }
    }
}

impl OracleFeas<Arr> for LowpassOracle {
    type CutChoices = (f64, Option<f64>); // parallel cut

    fn assess_feas(&mut self, x: &Arr) -> Option<Cut> {
        self.more_alt = true;

        let mdim = self.spectrum.len();
        let ndim = self.spectrum[0].len();
        for _ in 0..self.nwpass {
            self.idx1 += 1;
            if self.idx1 == self.nwpass {
                self.idx1 = 0;
            }
            let col_k = &self.spectrum[self.idx1];
            // let v = col_k.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let v = col_k.dot(x);
            if v > self.up_sq {
                let f = (v - self.up_sq, Some(v - self.lp_sq));
                return Some((col_k.clone(), f));
            }
            if v < self.lp_sq {
                let f = (-v + self.lp_sq, Some(-v + self.up_sq));
                return Some((col_k.iter().map(|&a| -a).collect(), f));
            }
        }

        self.fmax = f64::NEG_INFINITY;
        self.kmax = 0;
        for _ in self.nwstop..mdim {
            self.idx3 += 1;
            if self.idx3 == mdim {
                self.idx3 = self.nwstop;
            }
            let col_k = &self.spectrum[self.idx3];
            // let v = col_k.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let v = col_k.dot(x);
            if v > self.sp_sq {
                return Some((col_k.clone(), (v - self.sp_sq, Some(v))));
            }
            if v < 0.0 {
                return Some((
                    col_k.iter().map(|&a| -a).collect(),
                    (-v, Some(-v + self.sp_sq)),
                ));
            }
            if v > self.fmax {
                self.fmax = v;
                self.kmax = self.idx3;
            }
        }

        for _ in self.nwpass..self.nwstop {
            self.idx2 += 1;
            if self.idx2 == self.nwstop {
                self.idx2 = self.nwpass;
            }
            let col_k = &self.spectrum[self.idx2];
            // let v = col_k.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let v = col_k.dot(x);
            if v < 0.0 {
                // single cut
                return Some((col_k.iter().map(|&a| -a).collect(), (-v, None)));
            }
        }

        self.more_alt = false;

        if x[0] < 0.0 {
            let mut grad = Arr::zeros(ndim);
            grad[0] = -1.0;
            return Some((grad, (-x[0], None)));
        }

        None
    }
}

impl OracleOptim<Arr> for LowpassOracle {
    type CutChoices = (f64, Option<f64>); // parallel cut

    fn assess_optim(&mut self, x: &Arr, sp_sq: &mut f64) -> (Cut, bool) {
        self.sp_sq = *sp_sq;

        if let Some(cut) = self.assess_feas(x) {
            return (cut, false);
        }

        let cut = (self.spectrum[self.kmax].clone(), (0.0, Some(self.fmax)));
        *sp_sq = self.fmax;
        (cut, true)
    }
}

pub fn create_lowpass_case(ndim: usize) -> LowpassOracle {
    let delta0_wpass = 0.025;
    let delta0_wstop = 0.125;
    let delta1 = 20.0 * (delta0_wpass * PI).log10();
    let delta2 = 20.0 * (delta0_wstop * PI).log10();

    let low_pass = 10.0f64.powf(-delta1 / 20.0);
    let up_pass = 10.0f64.powf(delta1 / 20.0);
    let stop_pass = 10.0f64.powf(delta2 / 20.0);

    let lp_sq = low_pass * low_pass;
    let up_sq = up_pass * up_pass;
    let sp_sq = stop_pass * stop_pass;

    LowpassOracle::new(ndim, 0.12, 0.20, lp_sq, up_sq, sp_sq)
}

#[cfg(test)]
mod tests {
    use super::*;
    // use super::{ProfitOracle, ProfitOracleQ, ProfitRbOracle};
    use crate::cutting_plane::{cutting_plane_optim, Options};
    use crate::ell::Ell;

    fn run_lowpass() -> (bool, usize) {
        let ndim = 32;
        let r0 = Arr::zeros(ndim);
        let mut ellip = Ell::new_with_scalar(40.0, r0);
        // ellip.helper.use_parallel_cut = use_parallel_cut;
        let mut omega = create_lowpass_case(ndim);
        let mut sp_sq = omega.sp_sq;
        let options = Options {
            max_iters: 50000,
            tolerance: 1e-14,
        };
        let (h, num_iters) = cutting_plane_optim(&mut omega, &mut ellip, &mut sp_sq, &options);
        (h.is_some(), num_iters)
    }

    #[test]
    fn test_lowpass() {
        let (_feasible, _num_iters) = run_lowpass();
        // assert!(feasible);
        // assert!(num_iters >= 23000);
        // assert!(num_iters <= 24000);
    }
}
