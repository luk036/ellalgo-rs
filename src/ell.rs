mod cutting_plane;
use crate::cutting_plane::CutStatus;

type Arr = [f64; 100];

/**
 * @brief Ellipsoid Search Space
 *
 *        ell = {x | (x - xc)' mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
pub struct ell {
    // using params_t = (f64, f64, f64);
    // using return_t = (i32, params_t);
    bool use_parallel_cut = true;
    bool no_defer_trick = false;

    f64 _mu();
    f64 _rho();
    f64 _sigma();
    f64 _delta();
    f64 _tsq();

    const i32 _n;

    const f64 _nFloat;
    const f64 _nPlus1;
    const f64 _nMinus1;
    const f64 _halfN;
    const f64 _halfNplus1;
    const f64 _halfNminus1;
    const f64 _nSq;
    const f64 _c1;
    const f64 _c2;
    const f64 _c3;

    f64 _kappa;
    Arr _mq;
    Arr _xc;
}

impl ell {
    /**
     * @brief Construct a new ell object
     *
     * @param[in] E
     */
    let mut operator=(const ell& E) -> ell& = delete;

    /**
     * @brief Construct a new ell object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    template <typename V, typename U> ell(V&& kappa, Arr&& mq, U&& x) 
        : _n{i32(x.size())},
          _nFloat{f64(_n)},
          _nPlus1{_nFloat + 1.},
          _nMinus1{_nFloat - 1.},
          _halfN{_nFloat / 2.},
          _halfNplus1{_nPlus1 / 2.},
          _halfNminus1{_nMinus1 / 2.},
          _nSq{_nFloat * _nFloat},
          _c1{_nSq / (_nSq - 1)},
          _c2{2.0 / _nPlus1},
          _c3{_nFloat / _nPlus1},
          _kappa{std::forward<V>(kappa)},
          _mq{std::move(mq)},
          _xc{std::forward<U>(x)} {}

  public:
    /**
     * @brief Construct a new ell object
     *
     * @param[in] val
     * @param[in] x
     */
    ell(const Arr& val, Arr x) : ell{1., xt::diag(val), std::move(x)} {}

    /**
     * @brief Construct a new ell object
     *
     * @param[in] alpha
     * @param[in] x
     */
    ell(alpha, Arr x: f64) : ell{alpha, xt::eye(x.size()), std::move(x)} {}

    /**
     * @brief copy the whole array anyway
     *
     * @return Arr
     */
    pub fn xc(&self) -> &Arr { return _xc; }

    /**
     * @brief Set the xc object
     *
     * @param[in] xc
     */
    pub fn set_xc(&mut self, xc: &Arr) { _xc = xc; }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * @tparam T
     * @param[in] cut cutting-plane
     * @return (i32, f64)
     */
    template <typename T> let mut update(const (Arr, T)& cut)
        -> (CutStatus, f64);

    let mut _update_cut(beta: f64) -> CutStatus { return self.calc_dc(beta); }

    CutStatus _update_cut(const Arr& beta) {  // parallel cut
        if beta.shape([0] < 2) {
            return self.calc_dc(beta[0]);
        }
        return self.calc_ll_core(beta[0], beta[1]);
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
        if self.nFloat * b0b1n < -1. {
            return CutStatus::NoEffect;  // no effect
        }

        // let t0 = self.tsq - b0 * b0;
        let t0n = 1.0 - b0 * (b0 / self.tsq);
        // let t1 = self.tsq - b1sq;
        let bsum = b0 + b1;
        let bsumn = bsum / self.tsq;
        let bav = bsum / 2.;
        let tempn = self.halfN * bsumn * bdiff;
        let xi = (t0n * t1n + tempn * tempn).sqrt();
        self.sigma = self.c3 + (1.0 - b0b1n - xi) / (bsumn * bav) / self.nPlus1;
        self.rho = self.sigma * bav;
        self.delta = self.c1 * ((t0n + t1n) / 2.0 + xi / self.nFloat);
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
        let temp = self.halfN * b1sqn;
        let xi = (1.0 - b1sqn + temp * temp).sqrt();
        self.sigma = self.c3 + self.c2 * (1.0 - xi) / b1sqn;
        self.rho = self.sigma * b1 / 2;
        self.delta = self.c1 * (1.0 - b1sqn / 2.0 + xi / self.nFloat);
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

        let gamma = tau + self.nFloat * beta;
        if gamma < 0.0 {
            return CutStatus::NoEffect;  // no effect
        }

        self.mu = (bdiff / gamma) * self.halfNminus1;
        self.rho = gamma / self.nPlus1;
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
        self.mu = self.halfNminus1;
        self.sigma = self.c2;
        self.rho = tau / self.nPlus1;
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
        // let [grad, beta] = cut;
        let grad = std::get<0>(cut);
        let beta = std::get<1>(cut);
        // n^2
        // let mq_g = Arr(xt::linalg::dot(self.mq, grad));  // n^2
        // let omega = xt::linalg::dot(grad, mq_g)();        // n

        let mut mq_g = zeros({self.n});  // initial x0
        let mut omega = 0.;
        for i in 0..self.n {
            for (let mut j = 0; j != self.n; ++j) {
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
}  // } ell
