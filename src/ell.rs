mod cutting_plane;
use crate::cutting_plane::CutStatus;

type Arr = [f64; 100];

/**
 * @brief Ellipsoid Search Space
 *
 *        Ell = {x | (x - xc)' mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
#[derive(Debug, Clone)]
pub struct Ell {
    type Params = (f64, f64, f64);
    type Returns = (i32, Params);

    pub use_parallel_cut: bool;
    pub no_defer_trick: bool;

    mq: Arr;
    xc: Arr;

    kappa: f64,
    mu: f64,
    rho: f64,
    sigma: f64,
    delta: f64,
    tsq: f64,

    usize n: usize,
    n_float: f64,
    n_plus_1: f64,
    n_minus_1: f64,
    half_n: f64,
    half_n_plus_1: f64,
    half_n_minus_1: f64,
    n_sq: f64,
    c1: f64,
    c2: f64,
    c3: f64,
}

impl Ell {
    /**
     * @brief Construct a new Ell object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    pub fn new_with_matrix(kappa: f64, mq: Arr, xc: Arr) -> Ell {
        let n = xc.len(),
        let n_float = n as f64,
        let n_plus_1 = n_float + 1.0,
        let n_minus_1 = n_float - 1.0,
        let half_n = n_float / 2.0,
        let half_n_plus_1 = n_plus_1 / 2.0,
        let half_n_minus_1 = n_minus_1 / 2.0,
        let n_sq = n_float * n_float,
        let c1 = n_sq / (n_sq - 1),
        let c2 = 2.0 / n_plus_1,
        let c3 = n_float / n_plus_1,

        Ell {
          kappa,
          mq,
          xc,
          n,
          n_float,
          n_plus_1,
          n_minus_1,
          half_n,
          half_n_plus_1,
          half_n_minus_1,
          n_sq,
          c1,
          c2,
          c3,
          mu: 0.0,
          rho: 0.0,
          sigma: 0.0,
          delta: 0.0,
          tsq: 0.0,
          use_parallel_cut: true;
          no_defer_trick: false;
        }
    }

    /**
     * @brief Construct a new Ell object
     *
     * @param[in] val
     * @param[in] x
     */
    pub fn new(val: Arr, xc: Arr) -> Ell {
        Ell::new_with_matrix(1.0, nparray::diag(val), xc}
    }

    /**
     * @brief Construct a new Ell object
     *
     * @param[in] alpha
     * @param[in] x
     */
    pub fn new(kappa: Arr, xc: Arr) -> Ell{
        Ell::new_with_matrix(kappa, nparray::eye(val), xc}
    }

    /**
     * @brief copy the whole array anyway
     *
     * @return Arr
     */
    pub fn xc(&self) -> Arr { self.xc }

    /**
     * @brief Set the xc object
     *
     * @param[in] xc
     */
    pub fn set_xc(&mut self, xc: &Arr) { self.xc = xc; }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * @tparam T
     * @param[in] cut cutting-plane
     * @return (i32, f64)
     */
    template <typename T> let mut update(const (Arr, T)& cut)
        -> (CutStatus, f64);

    pub fn update_cut(&mut self, beta: f64) -> CutStatus { return self.calc_dc(beta); }

    pub fn update_cut_ll(&mut self, beta: Arr) -> CutStatus {  // parallel cut
        if beta.shape()[0] < 2 {
            return self.calc_dc(beta[0]);
        }
        self.calc_ll_core(beta[0], beta[1])
    }

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
            return CutStatus::NoSoln;  // no sol'n
        }

        if b0 == 0.0  // central cut
        {
            self.calc_ll_cc(b1, b1sqn);
            return CutStatus::Success;
        }

        let b0b1n = b0 * (b1 / self.tsq);
        if self.n_float * b0b1n < -1.0 {
            return CutStatus::NoEffect;  // no effect
        }

        // let t0 = self.tsq - b0 * b0;
        let t0n = 1.0 - b0 * (b0 / self.tsq);
        // let t1 = self.tsq - b1sq;
        let bsum = b0 + b1;
        let bsumn = bsum / self.tsq;
        let bav = bsum / 2.;
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
        self.rho = self.sigma * b1 / 2;
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
            return CutStatus::NoSoln;  // no sol'n
        }

        if beta == 0.0 {
            self.calc_cc(tau);
            return CutStatus::Success;
        }

        let gamma = tau + self.n_float * beta;
        if gamma < 0.0 {
            return CutStatus::NoEffect;  // no effect
        }

        self.mu = (bdiff / gamma) * self.half_n_minus_1;
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
        self.mu = self.half_n_minus_1;
        self.sigma = self.c2;
        self.rho = tau / self.n_plus_1;
        self.delta = self.c1;
    }

    /**
     * @brief Update ellipsoid core function using the cut
     *
     *        grad' * (x - xc) + beta <= 0
     *
     * @tparam T
     * @param[in] cut
     * @return (i32, f64)
     */
    template <typename T> let mut update(&mut self, const (Arr, T)& cut)
        -> (CutStatus, f64) {
        let (grad, beta) = cut;
        // n^2
        // let mq_g = Arr(xt::linalg::dot(self.mq, grad));  // n^2
        // let omega = xt::linalg::dot(grad, mq_g)();        // n

        let mut mq_g = ndarray::zeros({self.n});  // initial x0
        let mut omega = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                mq_g(i) += self.mq(i, j) * grad(j);
            }
            omega += mq_g(i) * grad(i);
        }

        self.tsq = self.kappa * omega;
        let mut status = self.update_cut(beta);
        if status != CutStatus::Success {
            return (status, self.tsq);
        }

        self.xc -= (self.rho / omega) * mq_g;  // n
        // n*(n+1)/2 + n
        // self.mq -= (self.sigma / omega) * xt::linalg::outer(mq_g, mq_g);
        let r = self.sigma / omega;
        for i in 0..self.n {
            let r_mq_g = r * mq_g(i);
            for j in 0..i {
                self.mq(i, j) -= r_mq_g * mq_g(j);
                self.mq(j, i) = self.mq(i, j);
            }
            self.mq(i, i) -= r_mq_g * mq_g(i);
        }

        self.kappa *= self.delta;

        if self.no_defer_trick {
            self.mq *= self.kappa;
            self.kappa = 1.;
        }
        (status, self.tsq)
    }
}  // } Ell
