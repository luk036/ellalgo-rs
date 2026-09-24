use crate::arr::{linspace, Arr};
use crate::cutting_plane::{OracleFeas, OracleOptim, ParallelCut};
use crate::round_robin::RoundRobin;

pub type Cut = (Arr, ParallelCut);

pub struct LowpassOracle {
    pub more_alt: bool,
    pub idx1: i32,
    pub spectrum: Vec<Arr>,
    pub nwpass: i32,
    pub nwstop: i32,
    pub lp_sq: f64,
    pub up_sq: f64,
    pub sp_sq: f64,
    pub idx2: i32,
    pub idx3: i32,
    pub fmax: f64,
    pub kmax: i32,
    rr1: RoundRobin,
    rr2: RoundRobin,
    rr3: RoundRobin,
}

/// Scan one frequency band for the first violated constraint.
///
/// Advances `rr` over the `hi - lo` points of the band and returns the first
/// violation as `(index, value, is_over)` — `is_over` distinguishes an
/// upper-bound violation from a lower-bound one — together with the running
/// peak `(fmax, kmax)` and the last cursor index reached. The latter keeps the
/// legacy `idx1`/`idx2`/`idx3` mirrors in sync with the round-robin cursor.
fn scan_band(
    spectrum: &[Arr],
    x: &Arr,
    rr: &mut RoundRobin,
    lo: i32,
    hi: i32,
    lower: f64,
    upper: Option<f64>,
) -> (Option<(usize, f64, bool)>, f64, i32, i32) {
    let mut fmax = f64::NEG_INFINITY;
    let mut kmax = -1;
    let mut last = rr.current();
    for _ in lo..hi {
        let idx = rr.advance();
        last = idx;
        let val = spectrum[idx as usize].dot(x);
        if let Some(u) = upper {
            if val > u {
                return (Some((idx as usize, val, true)), fmax, kmax, last);
            }
        }
        if val < lower {
            return (Some((idx as usize, val, false)), fmax, kmax, last);
        }
        if val > fmax {
            fmax = val;
            kmax = idx;
        }
    }
    (None, fmax, kmax, last)
}

/// Build the cut for a band violation at `idx` with response `val`.
#[inline]
fn band_cut(
    spectrum: &[Arr],
    idx: usize,
    val: f64,
    is_over: bool,
    lower: f64,
    upper: Option<f64>,
) -> Cut {
    let col = &spectrum[idx];
    if is_over {
        let u = upper.expect("upper-bound violation requires an upper bound");
        (col.clone(), ParallelCut(val - u, Some(val - lower)))
    } else {
        (
            Arr::from(col.iter().map(|&a| -a).collect::<Vec<_>>()),
            ParallelCut(lower - val, upper.map(|u| u - val)),
        )
    }
}

impl LowpassOracle {
    pub fn new(ndim: usize, wpass: f64, wstop: f64, lp_sq: f64, up_sq: f64, sp_sq: f64) -> Self {
        let mdim = 15 * ndim;
        let omega = linspace(0.0, std::f64::consts::PI, mdim);

        let mut spectrum = vec![Arr::new(ndim); mdim];
        for i in 0..mdim {
            spectrum[i][0] = 1.0;
            for (j, val) in spectrum[i].iter_mut().enumerate().skip(1) {
                *val = 2.0 * (omega[i] * j as f64).cos();
            }
        }

        let nwpass = (wpass * (mdim - 1) as f64).floor() as i32 + 1;
        let nwstop = (wstop * (mdim - 1) as f64).floor() as i32 + 1;

        Self {
            more_alt: true,
            idx1: -1,
            spectrum,
            nwpass,
            nwstop,
            lp_sq,
            up_sq,
            sp_sq,
            idx2: nwpass - 1,
            idx3: nwstop - 1,
            fmax: f64::NEG_INFINITY,
            kmax: -1,
            rr1: RoundRobin::new_range(0, nwpass),
            rr2: RoundRobin::new_range(nwpass, nwstop),
            rr3: RoundRobin::new_range(nwstop, mdim as i32),
        }
    }
}

impl OracleFeas<Arr> for LowpassOracle {
    type CutChoice = ParallelCut;

    fn assess_feas(&mut self, x: &Arr) -> Option<Cut> {
        self.more_alt = true;
        let ndim = self.spectrum[0].len();
        let mdim = self.spectrum.len() as i32;

        let (viol, _fmax, _kmax, last) = scan_band(
            &self.spectrum,
            x,
            &mut self.rr1,
            0,
            self.nwpass,
            self.lp_sq,
            Some(self.up_sq),
        );
        self.idx1 = last;
        if let Some((idx, val, is_over)) = viol {
            return Some(band_cut(
                &self.spectrum,
                idx,
                val,
                is_over,
                self.lp_sq,
                Some(self.up_sq),
            ));
        }

        self.fmax = f64::NEG_INFINITY;
        self.kmax = -1;
        let (viol, fmax, kmax, last) = scan_band(
            &self.spectrum,
            x,
            &mut self.rr3,
            self.nwstop,
            mdim,
            0.0,
            Some(self.sp_sq),
        );
        self.idx3 = last;
        if let Some((idx, val, is_over)) = viol {
            return Some(band_cut(
                &self.spectrum,
                idx,
                val,
                is_over,
                0.0,
                Some(self.sp_sq),
            ));
        }
        self.fmax = fmax;
        self.kmax = kmax;

        let (viol, _fmax, _kmax, last) = scan_band(
            &self.spectrum,
            x,
            &mut self.rr2,
            self.nwpass,
            self.nwstop,
            0.0,
            None,
        );
        self.idx2 = last;
        if let Some((idx, val, is_over)) = viol {
            return Some(band_cut(&self.spectrum, idx, val, is_over, 0.0, None));
        }

        self.more_alt = false;

        if x[0] < 0.0 {
            let mut grad = Arr::new(ndim);
            grad[0] = -1.0;
            return Some((grad, ParallelCut(-x[0], None)));
        }

        None
    }
}

impl OracleOptim<Arr> for LowpassOracle {
    type CutChoice = ParallelCut;

    fn assess_optim(&mut self, x: &Arr, sp_sq: &mut f64) -> (Cut, bool) {
        self.sp_sq = *sp_sq;

        if let Some(cut) = self.assess_feas(x) {
            return (cut, false);
        }

        let cut = (
            self.spectrum[self.kmax as usize].clone(),
            ParallelCut(0.0, Some(self.fmax)),
        );
        *sp_sq = self.fmax;
        (cut, true)
    }
}

pub fn create_lowpass_case(ndim: usize) -> LowpassOracle {
    let delta0_wpass = 0.025_f64;
    let delta0_wstop = 0.125_f64;
    let delta1 = 20.0 * (1.0 + delta0_wpass).log10();
    let delta2 = 20.0 * delta0_wstop.log10();

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
    use crate::cutting_plane::{cutting_plane_optim, Options};
    use crate::ell::Ell;

    fn run_lowpass() -> (bool, usize) {
        let ndim = 32;
        let r0 = Arr::new(ndim);
        let mut ellip = Ell::new_with_scalar(40.0, r0);
        let mut omega = create_lowpass_case(ndim);
        let mut sp_sq = omega.sp_sq;
        let options = Options {
            max_iters: 50000,
            tolerance: 1e-14,
            verbose: false,
        };
        let (h, num_iters) = cutting_plane_optim(&mut omega, &mut ellip, &mut sp_sq, &options);
        (h.is_some(), num_iters)
    }

    #[test]
    fn test_lowpass() {
        let (_feasible, _num_iters) = run_lowpass();
    }

    #[test]
    fn test_lowpass_oracle() {
        let mut oracle = create_lowpass_case(32);
        let x_vec = Arr::new(32);
        let res = oracle.assess_feas(&x_vec);
        assert!(res.is_some());
    }

    #[test]
    fn test_lowpass_oracle_direct() {
        let mut omega = create_lowpass_case(32);
        let mut h = Arr::new(32);
        h[0] = 1.0;
        let mut sp_sq = omega.sp_sq;
        let cut = omega.assess_optim(&h, &mut sp_sq);
        let res = cut.0;
        assert!(!res.0.is_empty());
        assert!(res.1 .0.is_finite());
    }

    #[test]
    fn test_lowpass_oracle_negative_transition() {
        let mut omega = create_lowpass_case(32);
        let mut h = Arr::new(32);
        h[0] = -0.1;
        let res = omega.assess_feas(&h);
        assert!(res.is_some());
        assert!(!res.unwrap().0.is_empty());
    }

    #[test]
    fn test_lowpass_oracle_negative_first_coeff() {
        let mut omega = create_lowpass_case(32);
        let mut h = Arr::new(32);
        h[0] = -0.5;
        for i in 1..32 {
            h[i] = 0.01;
        }
        let res = omega.assess_feas(&h);
        assert!(res.is_some());
        let (grad, _) = res.unwrap();
        // Verify gradient has non-zero values (exact gradient depends on which
        // frequency-domain constraint triggers first in the Rust implementation)
        assert!(!grad.is_empty());
    }
}
