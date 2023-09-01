use itertools::chain;
use ndarray::Array2;

type Arr = Array2<f64>;
type Cut = (Arr, f64);

struct LowpassOracle {
    more_alt: bool,
    i_Anr: usize,
    i_As: usize,
    i_Ap: usize,
    count: usize,
    Ap: Arr,
    As: Arr,
    Anr: Arr,
    Lpsq: f64,
    Upsq: f64,
}

impl LowpassOracle {
    fn new(Ap: Arr, As: Arr, Anr: Arr, Lpsq: f64, Upsq: f64) -> Self {
        LowpassOracle {
            more_alt: true,
            i_Anr: 0,
            i_As: 0,
            i_Ap: 0,
            count: 0,
            Ap,
            As,
            Anr,
            Lpsq,
            Upsq,
        }
    }

    fn assess_optim(&mut self, x: Arr, Spsq: f64) -> (Cut, Option<f64>) {
        let n = x.len();
        self.more_alt = true;

        let N = self.Ap.shape()[0];
        let mut i_Ap = self.i_Ap;
        for k in chain(i_Ap..N, 0..i_Ap) {
            let v = self.Ap.slice(s![k, ..]).dot(&x);
            if v > self.Upsq {
                let g = self.Ap.slice(s![k, ..]);
                let f = (v - self.Upsq, v - self.Lpsq);
                self.i_Ap = k + 1;
                return ((g, f), None);
            }

            if v < self.Lpsq {
                let g = -self.Ap.slice(s![k, ..]);
                let f = (-v + self.Lpsq, -v + self.Upsq);
                self.i_Ap = k + 1;
                return ((g, f), None);
            }
        }

        let N = self.As.shape()[0];
        let mut fmax = f64::NEG_INFINITY;
        let mut imax = 0;
        let mut i_As = self.i_As;
        for k in chain(i_As..N, 0..i_As) {
            let v = self.As.slice(s![k, ..]).dot(&x);
            if v > Spsq {
                let g = self.As.slice(s![k, ..]);
                let f = (v - Spsq, v);
                self.i_As = k + 1;
                return ((g, f), None);
            }

            if v < 0.0 {
                let g = -self.As.slice(s![k, ..]);
                let f = (-v, -v + Spsq);
                self.i_As = k + 1;
                return ((g, f), None);
            }

            if v > fmax {
                fmax = v;
                imax = k;
            }
        }

        let N = self.Anr.shape()[0];
        let mut i_Anr = self.i_Anr;
        for k in chain(i_Anr..N, 0..i_Anr) {
            let v = self.Anr.slice(s![k, ..]).dot(&x);
            if v < 0.0 {
                let f = -v;
                let g = -self.Anr.slice(s![k, ..]);
                self.i_Anr = k + 1;
                return ((g, f), None);
            }
        }

        self.more_alt = false;

        if x[0] < 0.0 {
            let mut g = Array2::zeros(n);
            g[[0, 0]] = -1.0;
            let f = -x[0];
            return ((g, f), None);
        }

        let Spsq = fmax;
        let f = (0.0, fmax);
        let g = self.As.slice(s![imax, ..]);
        ((g, f), Some(Spsq))
    }
}

