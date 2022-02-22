// mod cutting_plane;
use crate::cutting_plane::CutStatus;

/**
 * @brief Ellipsoid Search Space
 *
 *  EllCalc = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
#[derive(Debug, Clone)]
pub struct EllCalc {
    pub use_parallel_cut: bool,

    pub rho: f64,
    pub sigma: f64,
    pub delta: f64,
    pub tsq: f64,

    n_float: f64,
    n_plus_1: f64,
    half_n: f64,
    c1: f64,
    c2: f64,
    c3: f64,
}

impl EllCalc {
    /**
     * @brief Construct a new EllCalc object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    pub fn new(n_float: f64) -> EllCalc {
        let n_plus_1 = n_float + 1.0;
        let half_n = n_float / 2.0;
        let n_sq = n_float * n_float;
        let c1 = n_sq / (n_sq - 1.0);
        let c2 = 2.0 / n_plus_1;
        let c3 = n_float / n_plus_1;

        EllCalc {
            n_float,
            n_plus_1,
            half_n,
            c1,
            c2,
            c3,
            rho: 0.0,
            sigma: 0.0,
            delta: 0.0,
            tsq: 0.0,
            use_parallel_cut: true,
        }
    }

    // pub fn update_cut(&mut self, beta: f64) -> CutStatus { self.calc_dc(beta) }

    /**
     * @brief
     *
     * @param[in] b0
     * @param[in] b1
     * @return i32
     */
    pub fn calc_ll_core(&mut self, b0: f64, b1: f64) -> CutStatus {
        // let b1sq = b1 * b1;
        let b1sqn = b1 * (b1 / self.tsq);
        let t1n = 1.0 - b1sqn;
        if t1n < 0.0 || !self.use_parallel_cut {
            return self.calc_dc(b0);
        }

        let bdiff = b1 - b0;
        if bdiff < 0.0 {
            return CutStatus::NoSoln; // no sol'n
        }

        if b0 == 0.0 {
            // central cut
            self.calc_ll_cc(b1, b1sqn);
            return CutStatus::Success;
        }

        let b0b1n = b0 * (b1 / self.tsq);
        if self.n_float * b0b1n < -1.0 {
            return CutStatus::NoEffect; // no effect
        }

        // let t0 = self.tsq - b0 * b0;
        let t0n = 1.0 - b0 * (b0 / self.tsq);
        // let t1 = self.tsq - b1sq;
        let bsum = b0 + b1;
        let bsumn = bsum / self.tsq;
        let bav = bsum / 2.0;
        let tempn = self.half_n * bsumn * bdiff;
        let xi = (t0n * t1n + tempn * tempn).sqrt();
        self.sigma = self.c3 + (1.0 - b0b1n - xi) / (bsumn * bav) / self.n_plus_1;
        self.rho = self.sigma * bav;
        self.delta = self.c1 * ((t0n + t1n) / 2.0 + xi / self.n_float);
        CutStatus::Success
    }

    /**
     * @brief
     *
     * @param[in] b1
     * @param[in] b1sq
     * @return void
     */
    pub fn calc_ll_cc(&mut self, b1: f64, b1sqn: f64) {
        let temp = self.half_n * b1sqn;
        let xi = (1.0 - b1sqn + temp * temp).sqrt();
        self.sigma = self.c3 + self.c2 * (1.0 - xi) / b1sqn;
        self.rho = self.sigma * b1 / 2.0;
        self.delta = self.c1 * (1.0 - b1sqn / 2.0 + xi / self.n_float);
    }

    /**
     * @brief Deep Cut
     *
     * @param[in] beta
     * @return i32
     */
    pub fn calc_dc(&mut self, beta: f64) -> CutStatus {
        let tau = (self.tsq).sqrt();

        let bdiff = tau - beta;
        if bdiff < 0.0 {
            return CutStatus::NoSoln; // no sol'n
        }

        if beta == 0.0 {
            self.calc_cc(tau);
            return CutStatus::Success;
        }

        let gamma = tau + self.n_float * beta;
        if gamma < 0.0 {
            return CutStatus::NoEffect; // no effect
        }

        // self.mu = (bdiff / gamma) * self.half_n_minus_1;
        self.rho = gamma / self.n_plus_1;
        self.sigma = 2.0 * self.rho / (tau + beta);
        self.delta = self.c1 * (1.0 - beta * (beta / self.tsq));
        CutStatus::Success
    }

    /**
     * @brief Central Cut
     *
     * @param[in] tau
     * @return i32
     */
    pub fn calc_cc(&mut self, tau: f64) {
        // self.mu = self.half_n_minus_1;
        self.sigma = self.c2;
        self.rho = tau / self.n_plus_1;
        self.delta = self.c1;
    }
}

pub trait UpdateByCutChoices {
    fn update_by(self, ell: &mut EllCalc) -> CutStatus;
}

impl EllCalc {
    pub fn update<A>(&mut self, args: A) -> CutStatus
    where
        A: UpdateByCutChoices,
    {
        args.update_by(self)
    }
}

impl UpdateByCutChoices for f64 {
    fn update_by(self, ell: &mut EllCalc) -> CutStatus {
        ell.calc_dc(self)
    }
}

impl UpdateByCutChoices for (f64, Option<f64>) {
    fn update_by(self, ell: &mut EllCalc) -> CutStatus {
        let (b0, b1_opt) = self;
        if let Some(b1) = b1_opt {
            ell.calc_ll_core(b0, b1)
        } else {
            ell.calc_dc(b0)
        }
    }
}
