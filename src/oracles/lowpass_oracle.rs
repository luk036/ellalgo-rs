use ndarray::Array2;
use ndarray::Array1;
use ndarray::Axis;
use ndarray::s;
use ndarray::stack;
use ndarray::Array;
use ndarray::ArrayView;

type Arr = Array2<f64>;
type Cut = (Arr, f64);

struct LowpassOracle {
    A: Arr,
    nwpass: usize,
    nwstop: usize,
    Lpsq: f64,
    Upsq: f64,
    more_alt: bool,
    idx1: usize,
    idx2: usize,
    idx3: usize,
}

impl LowpassOracle {
    fn new(A: Arr, nwpass: usize, nwstop: usize, Lpsq: f64, Upsq: f64) -> Self {
        LowpassOracle {
            A,
            nwpass,
            nwstop,
            Lpsq,
            Upsq,
            more_alt: true,
            idx1: 0,
            idx2: nwpass,
            idx3: nwstop,
        }
    }

    fn assess_optim(&mut self, x: ArrayView<f64, ndarray::Ix1>, Spsq: f64) -> (Cut, Option<f64>) {
        let n = x.len();
        self.more_alt = true;

        for _k in 0..self.nwpass {
            self.idx1 += 1;
            if self.idx1 == self.nwpass {
                self.idx1 = 0;
            }
            let v = self.A.slice(s![self.idx1, ..]).dot(&x);
            if v > self.Upsq {
                let g = self.A.slice(s![self.idx1, ..]);
                let f = (v - self.Upsq, v - self.Lpsq);
                return ((g, f), None);
            }

            if v < self.Lpsq {
                let g = -self.A.slice(s![self.idx1, ..]);
                let f = (-v + self.Lpsq, -v + self.Upsq);
                return ((g, f), None);
            }
        }

        let mut fmax = f64::NEG_INFINITY;
        let mut imax = 0;
        let mdim = self.A.shape()[0];
        for _k in self.nwstop..mdim {
            self.idx3 += 1;
            if self.idx3 == mdim {
                self.idx3 = self.nwstop;
            }
            let v = self.A.slice(s![self.idx3, ..]).dot(&x);
            if v > Spsq {
                let g = self.A.slice(s![self.idx3, ..]);
                let f = (v - Spsq, v);
                return ((g, f), None);
            }

            if v < 0.0 {
                let g = -self.A.slice(s![self.idx3, ..]);
                let f = (-v, -v + Spsq);
                return ((g, f), None);
            }

            if v > fmax {
                fmax = v;
                imax = self.idx3;
            }
        }

        for _k in self.nwpass..self.nwstop {
            self.idx2 += 1;
            if self.idx2 == self.nwstop {
                self.idx2 = self.nwpass;
            }
            let v = self.A.slice(s![self.idx2, ..]).dot(&x);
            if v < 0.0 {
                let f = -v;
                let g = -self.A.slice(s![self.idx2, ..]);
                return ((g, f), None);
            }
        }

        self.more_alt = false;

        if x[0] < 0.0 {
            let g = Array1::zeros(n);
            g[[0]] = -1.0;
            let f = -x[0];
            return ((g, f), None);
        }

        let Spsq = fmax;
        let f = (0.0, fmax);
        let g = self.A.slice(s![imax, ..]);
        ((g, f), Some(Spsq))
    }
}

fn create_lowpass_case(N: usize) -> (LowpassOracle, f64) {
    let delta0_wpass = 0.025;
    let delta0_wstop = 0.125;
    let delta1 = 20.0 * (delta0_wpass).log10();
    let delta2 = 20.0 * (delta0_wstop).log10();

    let m = 15 * N;
    let w = Array::linspace(0.0, std::f64::consts::PI, m);
    let An = Array::from_shape_fn((m, N - 1), |(i, j)| 2.0 * (w[i] * (j + 1) as f64).cos());
    let A = stack![Axis(1), Array::ones(m).insert_axis(Axis(1)), An];
    let nwpass = (0.12 * (m - 1) as f64).floor() as usize + 1;
    let nwstop = (0.20 * (m - 1) as f64).floor() as usize + 1;
    let Lp = (10.0f64).powf(-delta1 / 20.0);
    let Up = (10.0f64).powf(delta1 / 20.0);
    let Sp = (10.0f64).powf(delta2 / 20.0);
    let Lpsq = Lp * Lp;
    let Upsq = Up * Up;
    let Spsq = Sp * Sp;

    let omega = LowpassOracle::new(A, nwpass, nwstop, Lpsq, Upsq);
    (omega, Spsq)
}
