use super::lib::OracleOptim;
use ndarray::prelude::*;

type Arr = Array1<f64>;
type Cut = (Arr, f64);

/**
 * @brief Oracle for a profit maximization problem.
 *
 *  This example is taken from [Aliabadi and Salahi, 2013]:
 *
 *    max     p(A x1^alpha x2^beta) - v1*x1 - v2*x2
 *    s.t.    x1 \le k
 *
 *  where:
 *
 *    p(A x1^alpha x2^beta): Cobb-Douglas production function
 *    p: the market price per unit
 *    A: the scale of production
 *    alpha, beta: the output elasticities
 *    x: input quantity
 *    v: output price
 *    k: a given constant that restricts the quantity of x1
 */
#[derive(Debug)]
pub struct ProfitOracle {
    log_pA: f64,
    log_k: f64,
    v: Arr,
    a: Arr
}

impl ProfitOracle {
    /**
     * @brief Construct a new profit oracle object
     *
     * @param[in] p the market price per unit
     * @param[in] A the scale of production
     * @param[in] k a given constant that restricts the quantity of x1
     * @param[in] a the output elasticities
     * @param[in] v output price
     */
    pub fn new(p: f64, A: f64, k: f64, const Arr& a, const Arr& v)
        : _log_pA{std::log(p * A)}, _log_k{std::log(k)}, _v{v}, _a{a} {}

}

impl OracleOptim for ProfitOracle {
    type ArrayType = Arr;
    type CutChoices = f64; // single cut

    /**
     * @brief
     *
     * @param[in] z
     * @param[in,out] t
     * @return std::tuple<Cut, double>
     */
    fn assess_optim(&mut self, z: &Arr, t: &mut f64) -> ((Arr, f64), bool) {
};

/**
 * @brief Oracle for a profit maximization problem (robust version)
 *
 *    This example is taken from [Aliabadi and Salahi, 2013]:
 *
 *        max     p0(A x1^alpha0 x2^beta0) - v1*x1 - v2*x2
 *        s.t.    x1 \le k0
 *
 *    where:
 *        alpha0 = alpha \pm e1
 *        beta0 = beta \pm e2
 *        p0 = p \pm e3
 *        k0 = k \pm e4
 *        v0 = v \pm e5
 *
 * @see ProfitOracle
 */
pub struct ProfitOracleRB {
    using Arr = xt::xarray<f64, xt::layout_type::row_major>;

  private:
    const Arr _uie;
    Arr _a;
    ProfitOracle _P;

  public:
    /**
     * @brief Construct a new profit rb oracle object
     *
     * @param[in] p the market price per unit
     * @param[in] A the scale of production
     * @param[in] k a given constant that restricts the quantity of x1
     * @param[in] a the output elasticities
     * @param[in] v output price
     * @param[in] e paramters for uncertainty
     * @param[in] e3 paramters for uncertainty
     */
    ProfitOracleRB(p: f64, A: f64, k: f64, const Arr& a, const Arr& v, const Arr& e,
                     f64 e3)
        : _uie{e}, _a{a}, _P(p - e3, A, k - e3, a, v + e3) {}

    /**
     * @brief Make object callable for cutting_plane_dc()
     *
     * @param[in] y input quantity (in log scale)
     * @param[in,out] t the best-so-far optimal value
     * @return Cut and the updated best-so-far value
     *
     * @see cutting_plane_dc
     */
    pub fn assess_optim<f64>(const Arr& y, f64& t) {
        pub fn a_rb = self.a;
        a_rb[0] += y[0] > 0.0 ? -self.uie[0] : self.uie[0];
        a_rb[1] += y[1] > 0.0 ? -self.uie[1] : self.uie[1];
        self.P._a = a_rb;
        return self.P(y, t);
    }
};

/**
 * @brief Oracle for profit maximization problem (discrete version)
 *
 *    This example is taken from [Aliabadi and Salahi, 2013]
 *
 *        max     p(A x1^alpha x2^beta) - v1*x1 - v2*x2
 *        s.t.    x1 \le k
 *
 *    where:
 *
 *        p(A x1^alpha x2^beta): Cobb-Douglas production function
 *        p: the market price per unit
 *        A: the scale of production
 *        alpha, beta: the output elasticities
 *        x: input quantity (must be integer value)
 *        v: output price
 *        k: a given constant that restricts the quantity of x1
 *
 * @see ProfitOracle
 */
pub struct ProfitOracleQ {
    using Arr = xt::xarray<f64, xt::layout_type::row_major>;
    using Cut = (Arr, f64);

  private:
    ProfitOracle _P;
    Arr _yd;

  public:
    /**
     * @brief Construct a new profit q oracle object
     *
     * @param[in] p the market price per unit
     * @param[in] A the scale of production
     * @param[in] k a given constant that restricts the quantity of x1
     * @param[in] a the output elasticities
     * @param[in] v output price
     */
    ProfitOracleQ(p: f64, A: f64, k: f64, const Arr& a, const Arr& v) : _P{p, A, k, a, v} {}

    /**
     * @brief Make object callable for cutting_plane_q()
     *
     * @param[in] y input quantity (in log scale)
     * @param[in,out] t the best-so-far optimal value
     * @param[in] retry whether it is a retry
     * @return Cut and the updated best-so-far value
     *
     * @see cutting_plane_q
     */
    pub fn assess_optim<f64>(const Arr& y, f64& t, bool retry) -> (Cut, bool, Arr, bool);
};
