use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim};
use ellalgo_rs::ell::Ell;
use ndarray::Array2;

struct PortfolioOracle<'a> {
    expected_returns: &'a Arr,
    risk_matrix: Array2<f64>,
    max_risk: f64,
    budget: f64,
}

impl<'a> PortfolioOracle<'a> {
    fn new(
        expected_returns: &'a Arr,
        risk_matrix: Array2<f64>,
        max_risk: f64,
        budget: f64,
    ) -> Self {
        Self {
            expected_returns,
            risk_matrix,
            max_risk,
            budget,
        }
    }
}

impl<'a> OracleOptim<Arr> for PortfolioOracle<'a> {
    type CutChoice = f64;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
        let n = xc.len();
        let mut gradient = Arr::new(n);

        let mut budget_violation = 0.0;
        for i in 0..n {
            budget_violation += xc[i];
        }
        budget_violation -= self.budget;
        if budget_violation > 0.0 {
            for i in 0..n {
                gradient[i] = 1.0;
            }
            return ((gradient, budget_violation), false);
        }

        let mut risk = 0.0;
        for i in 0..n {
            for j in 0..n {
                risk += xc[i] * self.risk_matrix[(i, j)] * xc[j];
            }
        }
        let risk_violation = risk - self.max_risk;
        if risk_violation > 0.0 {
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += (self.risk_matrix[(i, j)] + self.risk_matrix[(j, i)]) * xc[j];
                }
                gradient[i] = sum;
            }
            return ((gradient, risk_violation), false);
        }

        let ret = self.expected_returns.dot(xc);
        let obj = -ret;
        if obj < *gamma {
            *gamma = obj;
            for i in 0..n {
                gradient[i] = -self.expected_returns[i];
            }
            return ((gradient, 0.0), true);
        }

        ((gradient, 0.0), false)
    }
}

fn main() {
    let expected_returns = Arr::from(vec![0.08, 0.12, 0.10, 0.06]);
    let risk_matrix = Array2::eye(4);
    let max_risk = 0.02;
    let budget = 1.0;

    let mut ellip = Ell::new_with_scalar(1.0, Arr::new(4));
    let mut oracle = PortfolioOracle::new(&expected_returns, risk_matrix, max_risk, budget);
    let mut gamma = f64::INFINITY;
    let options = Options::new(1000, 1e-8);

    let (xbest, num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    println!("Best portfolio weights: {:?}", xbest);
    println!("Optimal objective: {:.4}", gamma);
    println!("Iterations: {}", num_iters);

    if let Some(x) = xbest {
        let ret = expected_returns.dot(&x);
        println!("Expected return: {:.4}", ret);
    }
}
