use crate::cutting_plane::{OracleOptim, OracleOptimQ};
use ndarray::prelude::*;

type Arr = Array1<f64>;
type Cut = (Arr, f64);

/**
 * @brief Oracle for a profit maximization problem.
 *
 *  This example is taken from [Aliabadi and Salahi, 2013]:
 *
 *    max   p(A x1**alpha x2**beta) - v1 * x1 - v2 * x2
 *    s.t.  x1 <= k
 *
 *  where:
 *
 *    p(scale x1**alpha x2**beta): Cobb-Douglas production function
 *    p: the market price per unit
 *    scale: the scale of production
 *    alpha, beta: the output elasticities
 *    x: input quantity
 *    v: output price
 *    k: a given constant that restricts the quantity of x1
 */
#[derive(Debug)]
pub struct ProfitOracle {
    log_p_scale: f64,
    log_k: f64,
    v: Arr,
    pub a: Arr,
}

impl ProfitOracle {
    /**
     * @brief Construct a new profit oracle object
     *
     * @param[in] p the market price per unit
     * @param[in] scale the scale of production
     * @param[in] k a given constant that restricts the quantity of x1
     * @param[in] a the output elasticities
     * @param[in] v output price
     */
    pub fn new(p: f64, scale: f64, k: f64, a: Arr, v: Arr) -> Self {
        ProfitOracle {
            log_p_scale: (p * scale).ln(),
            log_k: k.ln(),
            v,
            a,
        }
    }
}

impl OracleOptim<Arr> for ProfitOracle {
    type CutChoices = f64; // single cut

    fn assess_optim(&mut self, y: &Arr, tea: &mut f64) -> ((Arr, f64), bool) {
        // y0 <= log k
        let f1 = y[0] - self.log_k;
        if f1 > 0.0 {
            return ((array![1.0, 0.0], f1), false);
        }

        let log_cobb = self.log_p_scale + self.a[0] * y[0] + self.a[1] * y[1];
        let x = y.mapv(f64::exp);
        let vx = self.v[0] * x[0] + self.v[1] * x[1];
        let mut te = *tea + vx;

        let fj = te.ln() - log_cobb;
        if fj < 0.0 {
            te = log_cobb.exp();
            *tea = te - vx;
            let g = (&self.v * &x) / te - &self.a;
            return ((g, 0.0), true);
        }
        let g = (&self.v * &x) / te - &self.a;
        ((g, fj), false)
    }
}

/**
 * @brief Oracle for a profit maximization problem (robust version)
 *
 *    This example is taken from [Aliabadi and Salahi, 2013]:
 *
 * @see ProfitOracle
 */
#[derive(Debug)]
pub struct ProfitOracleRB {
    uie: Arr,
    omega: ProfitOracle,
    a: Arr,
}

impl ProfitOracleRB {
    /**
     * @brief Construct a new profit rb oracle object
     *
     * @param[in] p the market price per unit
     * @param[in] scale the scale of production
     * @param[in] k a given constant that restricts the quantity of x1
     * @param[in] a the output elasticities
     * @param[in] v output price
     * @param[in] e paramters for uncertainty
     * @param[in] e3 paramters for uncertainty
     */
    pub fn new(p: f64, scale: f64, k: f64, aa: Arr, v: Arr, e: Arr, e3: f64) -> Self {
        ProfitOracleRB {
            uie: e,
            omega: ProfitOracle::new(p - e3, scale, k - e3, aa.clone(), &v + &array![e3, e3]),
            a: aa,
        }
    }
}

impl OracleOptim<Arr> for ProfitOracleRB {
    type CutChoices = f64; // single cut

    /**
     * @brief Make object callable for cutting_plane_dc()
     *
     * @param[in] y input quantity (in log scale)
     * @param[in,out] tea the best-so-far optimal value
     * @return Cut and the updated best-so-far value
     *
     * @see cutting_plane_dc
     */
    fn assess_optim(&mut self, y: &Arr, tea: &mut f64) -> ((Arr, f64), bool) {
        let mut a_rb = self.a.clone();
        a_rb[0] += if y[0] > 0.0 {
            -self.uie[0]
        } else {
            self.uie[0]
        };
        a_rb[1] += if y[1] > 0.0 {
            -self.uie[1]
        } else {
            self.uie[1]
        };
        self.omega.a = a_rb;
        self.omega.assess_optim(y, tea)
    }
}

/**
 * @brief Oracle for profit maximization problem (discrete version)
 *
 *    This example is taken from [Aliabadi and Salahi, 2013]
 *
 * @see ProfitOracle
 */
pub struct ProfitOracleQ {
    omega: ProfitOracle,
    yd: Arr,
}

impl ProfitOracleQ {
    /**
     * @brief Construct a new profit q oracle object
     *
     * @param[in] p the market price per unit
     * @param[in] scale the scale of production
     * @param[in] k a given constant that restricts the quantity of x1
     * @param[in] a the output elasticities
     * @param[in] v output price
     */
    pub fn new(p: f64, scale: f64, k: f64, a: Arr, v: Arr) -> Self {
        ProfitOracleQ {
            yd: array![0.0, 0.0],
            omega: ProfitOracle::new(p, scale, k, a, v),
        }
    }
}

impl OracleOptimQ<Arr> for ProfitOracleQ {
    type CutChoices = f64; // single cut

    /**
     * @brief Make object callable for cutting_plane_optim_q()
     *
     * @param[in] y input quantity (in log scale)
     * @param[in,out] tea the best-so-far optimal value
     * @param[in] retry whether it is a retry
     * @return Cut and the updated best-so-far value
     *
     * @see cutting_plane_optim_q
     */
    fn assess_optim_q(&mut self, y: &Arr, tea: &mut f64, retry: bool) -> (Cut, bool, Arr, bool) {
        if !retry {
            let mut x = y.mapv(f64::exp).mapv(f64::round);
            if x[0] == 0.0 {
                x[0] = 1.0; // nearest integer than 0
            }
            if x[1] == 0.0 {
                x[1] = 1.0;
            }
            self.yd = x.mapv(f64::ln);
        }
        let (mut cut, shrunk) = self.omega.assess_optim(&self.yd, tea);
        let g = &cut.0;
        let h = &mut cut.1;
        // let (g, mut h) = cut;
        let d = &self.yd - y;
        *h += g[0] * d[0] + g[1] * d[1];
        (cut, shrunk, self.yd.clone(), !retry)
    }
}

#[cfg(test)]
mod tests {
    use super::ProfitOracle;
    // use super::{ProfitOracle, ProfitOracleQ, ProfitOracleRB};
    use crate::cutting_plane::{cutting_plane_optim, Options};
    use crate::ell::Ell;
    use ndarray::array;

    #[test]
    pub fn test_profit_oracle() {
        let unit_price = 20.0;
        let scale = 40.0;
        let limit = 30.5;
        let a = array![0.1, 0.4];
        let v = array![10.0, 35.0];

        let mut ellip = Ell::new(array![100.0, 100.0], array![0.0, 0.0]);
        let mut omega = ProfitOracle::new(unit_price, scale, limit, a, v);
        let mut tea = 0.0;
        let options = Options {
            max_iters: 2000,
            tol: 1e-8,
        };
        let (y_opt, niter) = cutting_plane_optim(&mut omega, &mut ellip, &mut tea, &options);
        assert!(y_opt.is_some());
        if let Some(y) = y_opt {
            assert!(y[0] <= limit.ln());
        }
        assert_eq!(niter, 57);
    }
}
