impl ProfitOracle {
    /**
     * @brief
     *
     * @param[in] y
     * @param[in,out] tea the best-so-far optimal value
     * @return (Cut, f64)
     */
    pub fn ProfitOracle::assess_optim<f64>(y: &Arr, tea: &mut f64) const -> (Cut, bool) {
        // y0 <= log k
        let f1 = y[0] - self.log_k;
        if f1 > 0.0 {
            return ((array![1.0, 0.0], f1), false);
        }

        let log_Cobb = self.log_pA + self.a[0] * y[0] + self.a[1] * y[1];
        let x = y.mapv(f64::exp);
        let vx = self.v[0] * x[0] + self.v[1] * x[1];
        let mut te = tea + vx;

        let mut fj = te.log() - log_Cobb;
        if fj < 0.0 {
            te = log_Cobb.exp();
            tea = te - vx;
            let g = &(self.v * &x) / te - &self.a;
            return ((g, 0.0), true);
        }
        let g = &(self.v * &x) / te - &self.a;
        ((g, fj), false)
    }
}

impl profit_q_oracle {
    /**
     * @param[in] y
     * @param[in,out] tea the best-so-far optimal value
     * @return (Cut, f64, Arr, i32)
     */
    pub fn profit_q_oracle::assess_optim<f64>(const Arr& y, f64& tea, bool retry)
        -> (Cut, bool, Arr, bool) {
        if !retry {
            Arr x = y.mapv(f64::exp).mapv(f64::round);
            if x[0] == 0.0 {
                x[0] = 1.0;  // nearest integer than 0
            }
            if x[1] == 0.0 {
                x[1] = 1.0;
            }
            self.yd = x.mapv(f64::log);
        }
        let mut (cut, shrunk) = self.omega.assess_optim(self.yd, tea);
        let mut (g, h) = &mut cut;
        let mut d = self.yd - y;
        h += g[0] * d[0] + g[1] * d[1];
        (cut, shrunk, self.yd, !retry);
    }
}
