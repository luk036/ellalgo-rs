use std::f64::consts::PI;
use std::cmp::Ordering;

type Arr = Vec<f64>;
type Cut = (Arr, (f64, f64));

struct LowpassOracle {
    more_alt: bool,
    idx1: usize,
    spectrum: Vec<Arr>,
    nwpass: usize,
    nwstop: usize,
    lp_sq: f64,
    up_sq: f64,
    idx2: usize,
    idx3: usize,
    fmax: f64,
    kmax: usize,
}

impl LowpassOracle {
    fn new(ndim: usize, wpass: f64, wstop: f64, lp_sq: f64, up_sq: f64) -> Self {
        let mdim = 15 * ndim;
        let w = (0..=mdim).map(|i| i as f64 * PI / mdim as f64).collect::<Vec<_>>();

        let mut spectrum = vec![vec![0.0; ndim]; mdim];
        for i in 0..mdim {
            for j in 0..ndim {
                spectrum[i][j] = 2.0 * (w[i] * (j + 1) as f64).cos();
            }
        }
        spectrum.iter_mut().for_each(|row| row.insert(0, 1.0));

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
            idx2: nwpass,
            idx3: nwstop,
            fmax: f64::NEG_INFINITY,
            kmax: 0,
        }
    }

    fn assess_feas(&mut self, x: &Arr, sp_sq: f64) -> Option<Cut> {
        self.more_alt = true;

        let mdim = self.spectrum.len();
        let ndim = self.spectrum[0].len();
        for _ in 0..self.nwpass {
            self.idx1 += 1;
            if self.idx1 == self.nwpass {
                self.idx1 = 0;
            }
            let col_k = &self.spectrum[self.idx1];
            let v = col_k.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            if v > self.up_sq {
                let f = (v - self.up_sq, v - self.lp_sq);
                return Some((col_k.clone(), f));
            }
            if v < self.lp_sq {
                let f = (-v + self.lp_sq, -v + self.up_sq);
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
            let v = col_k.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            if v > sp_sq {
                return Some((col_k.clone(), (v - sp_sq, v)));
            }
            if v < 0.0 {
                return Some((col_k.iter().map(|&a| -a).collect(), (-v, -v + sp_sq)));
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
            let v = col_k.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            if v < 0.0 {
                return Some((col_k.iter().map(|&a| -a).collect(), -v));
            }
        }

        self.more_alt = false;

        if x[0] < 0.0 {
            let mut grad = vec![0.0; ndim];
            grad[0] = -1.0;
            return Some((grad, -x[0]));
        }

        None
    }

    fn assess_optim(&mut self, x: &Arr, sp_sq: f64) -> (Option<Cut>, Option<f64>) {
        if let Some(cut) = self.assess_feas(x, sp_sq) {
            return (Some(cut), None);
        }

        let cut = (self.spectrum[self.kmax].clone(), (0.0, self.fmax));
        (Some(cut), Some(self.fmax))
    }
}

fn create_lowpass_case(ndim: usize) -> (LowpassOracle, f64) {
    let delta0_wpass = 0.025;
    let delta0_wstop = 0.125;
    let delta1 = 20.0 * (delta0_wpass * PI).log10() / 20.0;
    let delta2 = 20.0 * (delta0_wstop * PI).log10() / 20.0;

    let low_pass = 10.0f64.powf(-delta1);
    let up_pass = 10.0f64.powf(delta1);
    let stop_pass = 10.0f64.powf(delta2);

    let lp_sq = low_pass * low_pass;
    let up_sq = up_pass * up_pass;
    let sp_sq = stop_pass * stop_pass;

    let omega = LowpassOracle::new(ndim, 0.12, 0.20, lp_sq, up_sq);
    (omega, sp_sq)
}

fn run_lowpass(use_parallel_cut: bool) -> (bool, usize) {
    let N = 32;
    let r0 = vec![0.0; N];
    let mut ellip = Ell::new(40.0, r0);
    ellip.helper.use_parallel_cut = use_parallel_cut;
    let (omega, sp_sq) = create_lowpass_case(N);
    let mut options = Options::default();
    options.max_iters = 50000;
    options.tolerance = 1e-14;
    let (h, _, num_iters) = cutting_plane_optim(&omega, &mut ellip, sp_sq, &options);
    (h.is_some(), num_iters)
}

fn test_lowpass() {
    let (feasible, num_iters) = run_lowpass(true);
    assert!(feasible);
    assert!(num_iters >= 23000);
    assert!(num_iters <= 24000);
}
