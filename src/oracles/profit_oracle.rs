use crate::cutting_plane::{OracleOptim, OracleOptimQ};
use ndarray::prelude::*;

type Arr = Array1<f64>;
type Cut = (Arr, f64);

/// The ProfitOracle struct represents an oracle for a profit maximization problem with specific
/// parameters.
///
///  This example is taken from [Aliabadi and Salahi, 2013]:
///
///    max   p(A x1**alpha x2**beta) - v1 * x1 - v2 * x2
///    s.t.  x1 <= k
///
///  where:
///
///    p(scale x1**alpha x2**beta): Cobb-Douglas production function
///    p: the market price per unit
///    scale: the scale of production
///    alpha, beta: the output elasticities
///    x: input quantity
///    v: output price
///    k: a given constant that restricts the quantity of x1
///
/// Reference:
///
/// * Aliabadi, Hossein, and Maziar Salahi. "Robust Geometric Programming Approach to Profit Maximization
/// with Interval Uncertainty." Computer Science Journal Of Moldova 61.1 (2013): 86-96.
///
/// Properties:
///
/// * `log_p_scale`: The natural logarithm of the scale parameter of the Cobb-Douglas production
/// function. It represents the overall scale of production.
/// * `log_k`: The natural logarithm of the constant k that restricts the quantity of x1.
/// * `price_out`: The `price_out` property represents the output prices `v1` and `v2` in the profit
/// maximization problem. It is of type `Arr`, which is likely a shorthand for an array or vector data
/// structure.
/// * `elasticities`: An array representing the output elasticities (alpha and beta) in the profit
/// maximization problem.
#[derive(Debug)]
pub struct ProfitOracle {
    idx: i32,
    log_p_scale: f64,
    log_k: f64,
    price_out: Arr,
    pub elasticities: Arr,
    log_cobb: f64,
    vx: f64,
    q: Arr,
}

impl ProfitOracle {
    /// The function `new` constructs a new ProfitOracle object with given parameters.
    ///
    /// Arguments:
    ///
    /// * `p`: The parameter `p` represents the market price per unit. It is a floating-point number (f64)
    /// that indicates the price at which the product is sold in the market.
    /// * `scale`: The scale parameter represents the scale of production. It determines the quantity of
    /// output produced.
    /// * `k`: The parameter `k` is a given constant that restricts the quantity of `x1`. It is used in the
    /// calculation of the profit oracle object.
    /// * `elasticities`: The parameter "elasticities" represents the output elasticities, which are
    /// coefficients that measure the responsiveness of the quantity of output to changes in the inputs. It
    /// is expected to be an array or list of values.
    /// * `price_out`: The `price_out` parameter represents the output price. It is of type `Arr`, which
    /// suggests that it is an array or collection of values. The specific type of `Arr` is not specified in
    /// the code snippet, so it could be an array, a vector, or any other collection type
    ///
    /// Returns:
    ///
    /// The `new` function is returning an instance of the `ProfitOracle` struct.
    pub fn new(params: (f64, f64, f64), elasticities: Arr, price_out: Arr) -> Self {
        let (unit_price, scale, limit) = params;
        let log_p_scale = (unit_price * scale).ln();
        let log_k = limit.ln();
        ProfitOracle {
            idx: -1,
            log_p_scale,
            log_k,
            price_out,
            elasticities,
            log_cobb: 0.0,
            vx: 0.0,
            q: Arr::zeros(2),
        }
    }

    /// The function assess_feas calculates the gradient and objective function value for an optimization
    /// problem in Rust.
    ///
    /// Arguments:
    ///
    /// * `y`: A reference to an array of f64 values.
    /// * `gamma`: The parameter `gamma` is a mutable reference to a `f64` variable.
    fn assess_feas(&mut self, y: &Arr, gamma: &mut f64) -> Option<(Arr, f64)> {
        let num_constraints = 2;
        for _ in 0..num_constraints {
            self.idx += 1;
            if self.idx == num_constraints {
                self.idx = 0; // round robin
            }
            let fj = match self.idx {
                0 => y[0] - self.log_k, // y0 <= log k
                1 => {
                    self.log_cobb = self.log_p_scale + self.elasticities.dot(y);
                    self.q = &self.price_out * y.mapv(f64::exp);
                    self.vx = self.q[0] + self.q[1];
                    (*gamma + self.vx).ln() - self.log_cobb
                }
                _ => unreachable!(),
            };
            if fj > 0.0 {
                return Some((
                    match self.idx {
                        0 => array![1.0, 0.0],
                        1 => &self.q / (*gamma + self.vx) - &self.elasticities,
                        _ => unreachable!(),
                    },
                    fj,
                ));
            }
        }

        None
    }
}

impl OracleOptim<Arr> for ProfitOracle {
    type CutChoice = f64; // single cut

    /// The function assess_optim calculates the gradient and objective function value for an optimization
    /// problem in Rust.
    ///
    /// Arguments:
    ///
    /// * `y`: A reference to an array of f64 values.
    /// * `gamma`: The parameter `gamma` is a mutable reference to a `f64` variable.
    fn assess_optim(&mut self, y: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
        if let Some(cut) = self.assess_feas(y, gamma) {
            return (cut, false);
        }

        let te = self.log_cobb.exp();
        *gamma = te - self.vx;
        let grad = (&self.q / te) - &self.elasticities;
        ((grad, 0.0), true)
    }
}

/// The `ProfitRbOracle` struct is an implementation of an oracle for a profit maximization problem with
/// robustness. It is used to solve a profit maximization problem where the parameters have interval
/// uncertainty. The implementation is based on the approach described in the paper "Robust Geometric
/// Programming Approach to Profit Maximization with Interval Uncertainty" by Aliabadi and Salahi.
/// The `ProfitRbOracle` struct is an implementation of an oracle for a profit maximization problem with
/// robustness.
///
/// Reference:
///
/// * Aliabadi, Hossein, and Maziar Salahi. "Robust Geometric Programming Approach to Profit Maximization
/// with Interval Uncertainty." Computer Science Journal Of Moldova 61.1 (2013): 86-96.
#[derive(Debug)]
pub struct ProfitRbOracle {
    uie: [f64; 2],
    omega: ProfitOracle,
    elasticities: Arr,
}

impl ProfitRbOracle {
    /// The function `new` constructs a new ProfitRbOracle object with given parameters.
    ///
    /// Arguments:
    ///
    /// * `p`: The market price per unit.
    /// * `scale`: The `scale` parameter represents the scale of production. It determines the level of
    /// output or production for a given set of inputs. It can be thought of as the size or capacity of the
    /// production process.
    /// * `k`: A given constant that restricts the quantity of x1.
    /// * `aa`: The parameter `aa` represents the output elasticities. It is an array that contains the
    /// elasticities of each output variable.
    /// * `price_out`: The parameter `price_out` represents the output price. It is of type `Arr`, which
    /// suggests that it is an array or vector of values.
    /// * `e`: Parameters for uncertainty.
    /// * `e3`: The parameter `e3` represents the uncertainty in the market price per unit. It is used to
    /// adjust the market price in the calculation of the `omega` variable.
    ///
    /// Returns:
    ///
    /// The `new` function is returning an instance of the `ProfitRbOracle` struct.
    pub fn new(
        params: (f64, f64, f64),
        aa: Arr,
        price_out: Arr,
        vparams: (f64, f64, f64, f64, f64),
    ) -> Self {
        let (e1, e2, e3, e4, e5) = vparams;
        let uie = [e1, e2];
        let params_rb = (params.0 - e3, params.1, params.2 - e4);
        let omega = ProfitOracle::new(
            params_rb,
            aa.clone(),
            price_out + Arr::from_vec(vec![e5, e5]),
        );
        ProfitRbOracle {
            uie,
            omega,
            elasticities: aa,
        }
    }
}

impl OracleOptim<Arr> for ProfitRbOracle {
    type CutChoice = f64; // single cut

    /// The `assess_optim` function takes an input quantity `y` and updates the best-so-far optimal value
    /// `gamma` based on the elasticities and returns a cut and the updated best-so-far value.
    ///
    /// Arguments:
    ///
    /// * `y`: The parameter `y` is an input quantity represented as an array (`Arr`) in log scale.
    /// * `gamma`: The parameter `gamma` is the best-so-far optimal value. It is passed as a mutable reference
    ///    (`&mut f64`) so that its value can be updated within the function.
    fn assess_optim(&mut self, y: &Arr, gamma: &mut f64) -> (Cut, bool) {
        let mut a_rb = self.elasticities.clone();
        for i in 0..2 {
            a_rb[i] += if y[i] > 0.0 {
                -self.uie[i]
            } else {
                self.uie[i]
            };
        }
        self.omega.elasticities = a_rb;
        self.omega.assess_optim(y, gamma)
    }
}

/// The ProfitOracleQ struct is an oracle for the profit maximization problem in a discrete version.
///
/// Properties:
///
/// * `omega`: The `omega` property is an instance of the `ProfitOracle` struct. It is used to calculate
///    the profit for a given input.
/// * `yd`: The variable `yd` is an array that represents the discrete version of y values
pub struct ProfitOracleQ {
    omega: ProfitOracle,
    yd: Arr,
}

impl ProfitOracleQ {
    /// The function `new` constructs a new `ProfitOracleQ` object with given parameters.
    ///
    /// Arguments:
    ///
    /// * `p`: The parameter `p` represents the market price per unit. It is a floating-point number (f64)
    /// that indicates the price at which a unit of the product is sold in the market.
    /// * `scale`: The "scale" parameter represents the scale of production, which refers to the level of
    /// output or production of a particular good or service. It indicates the quantity of goods or services
    /// produced within a given time period.
    /// * `k`: A given constant that restricts the quantity of x1. It is used to limit the quantity of a
    /// particular input (x1) in the production process.
    /// * `elasticities`: The parameter `elasticities` represents the output elasticities. Output elasticity measures the
    /// responsiveness of the quantity of output to a change in the input quantity. It indicates how much
    /// the quantity of output changes in response to a change in the quantity of input.
    /// * `price_out`: The parameter `price_out` represents the output price. It is an array that contains
    /// the prices of the outputs.
    ///
    /// Returns:
    ///
    /// The `new` function is returning an instance of the `ProfitOracleQ` struct.
    pub fn new(params: (f64, f64, f64), elasticities: Arr, price_out: Arr) -> Self {
        ProfitOracleQ {
            yd: array![0.0, 0.0],
            omega: ProfitOracle::new(params, elasticities, price_out),
        }
    }
}

impl OracleOptimQ<Arr> for ProfitOracleQ {
    type CutChoice = f64; // single cut

    /// The `assess_optim_q` function takes in an input quantity `y` in log scale, updates the best-so-far
    /// optimal value `gamma`, and returns a cut and the updated best-so-far value.
    ///
    /// Arguments:
    ///
    /// * `y`: The parameter `y` is an input quantity in log scale. It is of type `Arr`, which is likely an
    /// array or vector type.
    /// * `gamma`: The parameter `gamma` represents the best-so-far optimal value. It is a mutable reference to
    /// a `f64` value, which means it can be modified within the function.
    /// * `retry`: A boolean value indicating whether it is a retry or not.
    fn assess_optim_q(&mut self, y: &Arr, gamma: &mut f64, retry: bool) -> (Cut, bool, Arr, bool) {
        if !retry {
            if let Some(cut) = self.omega.assess_feas(y, gamma) {
                return (cut, false, y.clone(), true);
            }

            // let mut xd = y.mapv(f64::exp).mapv(f64::round);
            let mut xd = y.map(|x| x.exp().round());
            if xd[0] == 0.0 {
                xd[0] = 1.0; // nearest integer than 0
            }
            if xd[1] == 0.0 {
                xd[1] = 1.0;
            }
            self.yd = xd.mapv(f64::ln);
        }
        let ((grad, beta), shrunk) = self.omega.assess_optim(&self.yd, gamma);
        let beta = beta + grad.dot(&(&self.yd - y));
        ((grad, beta), shrunk, self.yd.clone(), !retry)
    }
}

#[cfg(test)]
mod tests {
    use super::{ProfitOracle, ProfitOracleQ, ProfitRbOracle};
    use crate::cutting_plane::{cutting_plane_optim, cutting_plane_optim_q, Options, OracleOptim};
    use crate::ell::Ell;
    use ndarray::array;

    #[test]
    pub fn test_profit_oracle() {
        let unit_price = 20.0;
        let scale = 40.0;
        let limit = 30.5;
        let params = (unit_price, scale, limit);
        let elasticities = array![0.1, 0.4];
        let price_out = array![10.0, 35.0];

        let mut ellip = Ell::new(array![100.0, 100.0], array![0.0, 0.0]);
        let mut omega = ProfitOracle::new(params, elasticities, price_out);
        let mut gamma = 0.0;
        let options = Options::default();
        let (y_opt, niter) = cutting_plane_optim(&mut omega, &mut ellip, &mut gamma, &options);
        assert!(y_opt.is_some());
        if let Some(y) = y_opt {
            assert!(y[0] <= limit.ln());
        }
        assert_eq!(niter, 83, "regression test");
    }

    #[test]
    pub fn test_profit_oracle_rb() {
        let unit_price = 20.0;
        let scale = 40.0;
        let limit = 30.5;
        let params = (unit_price, scale, limit);
        let elasticities = array![0.1, 0.4];
        let price_out = array![10.0, 35.0];
        let e1 = 0.003;
        let e2 = 0.007;
        let e3 = 1.0;
        let e4 = 1.0;
        let e5 = 1.0;
        let vparams = (e1, e2, e3, e4, e5);

        let mut ellip = Ell::new(array![100.0, 100.0], array![0.0, 0.0]);
        let mut omega = ProfitRbOracle::new(params, elasticities, price_out, vparams);
        let mut gamma = 0.0;
        let options = Options::default();
        let (y_opt, niter) = cutting_plane_optim(&mut omega, &mut ellip, &mut gamma, &options);
        assert!(y_opt.is_some());
        if let Some(y) = y_opt {
            assert!(y[0] <= limit.ln());
        }
        assert_eq!(niter, 90, "regression test");
    }

    #[test]
    pub fn test_profit_oracle_q() {
        let unit_price = 20.0;
        let scale = 40.0;
        let limit = 30.5;
        let params = (unit_price, scale, limit);
        let elasticities = array![0.1, 0.4];
        let price_out = array![10.0, 35.0];

        let mut ellip = Ell::new(array![100.0, 100.0], array![0.0, 0.0]);
        let mut omega = ProfitOracleQ::new(params, elasticities, price_out);
        let mut gamma = 0.0;
        let options = Options::default();
        let (y_opt, niter) = cutting_plane_optim_q(&mut omega, &mut ellip, &mut gamma, &options);
        assert!(y_opt.is_some());
        if let Some(y) = y_opt {
            assert!(y[0] <= limit.ln());
        }
        assert_eq!(niter, 29, "regression test");
    }

    #[test]
    fn test_profit_oracle_direct() {
        let unit_price = 20.0;
        let scale = 40.0;
        let limit = 30.5;
        let params = (unit_price, scale, limit);
        let elasticities = array![0.1, 0.4];
        let price_out = array![10.0, 35.0];
        let mut omega = ProfitOracle::new(params, elasticities, price_out);
        let mut gamma = 0.0;
        // Infeasible case
        let y = array![3.5, 2.0];
        let (cut, feasible) = omega.assess_optim(&y, &mut gamma);
        assert!(!feasible);
        assert_eq!(cut.1, 3.5 - limit.ln());
        // Feasible but not optimal case
        let y2 = array![3.0, 2.0];
        let (cut2, feasible2) = omega.assess_optim(&y2, &mut gamma);
        assert!(feasible2);
        assert_eq!(cut2.1, 0.0);
    }
}
