use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim};
use ellalgo_rs::ell::Ell;
use ndarray::prelude::*;

type Arr = Array1<f64>;

/// Portfolio optimization example using ellipsoid method.
///
/// This example solves a simple portfolio allocation problem:
/// - Maximize expected return
/// - Subject to budget constraint
/// - Subject to risk constraint
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

impl OracleOptim<Arr> for PortfolioOracle<'_> {
    type CutChoice = f64;

    fn assess_optim(&mut self, xc: &Arr, gamma: &mut f64) -> ((Arr, f64), bool) {
        let n = xc.len();

        // Check budget constraint: sum(x) <= budget
        let sum_weights: f64 = xc.sum();
        let budget_violation = sum_weights - self.budget;
        if budget_violation > 0.0 {
            return ((Array1::ones(n), budget_violation), false);
        }

        // Check risk constraint: x^T * R * x <= max_risk
        let weighted = self.risk_matrix.dot(xc);
        let risk_value = xc.dot(&weighted);
        let risk_violation = risk_value - self.max_risk;
        if risk_violation > 0.0 {
            let gradient = &self.risk_matrix + &self.risk_matrix.t();
            let grad = gradient.dot(xc);
            return ((grad, risk_violation), false);
        }

        // Objective: maximize expected return (minimize negative)
        let expected_return = self.expected_returns.dot(xc);
        *gamma = -expected_return;
        ((-self.expected_returns.clone(), 0.0), true)
    }
}

fn main() {
    let expected_returns = array![0.08, 0.12, 0.15, 0.10];
    let risk_matrix = array![
        [0.0100, 0.0018, 0.0011, 0.0008],
        [0.0018, 0.0200, 0.0015, 0.0012],
        [0.0011, 0.0015, 0.0250, 0.0018],
        [0.0008, 0.0012, 0.0018, 0.0180],
    ];

    let mut oracle = PortfolioOracle::new(
        &expected_returns,
        risk_matrix,
        0.025, // max_risk
        1.0,   // budget
    );

    let mut ellip = Ell::new_with_scalar(1.0, Array1::zeros(4));
    let mut gamma = f64::NEG_INFINITY;
    let options = Options::default();

    let (xbest, num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

    println!("Portfolio Optimization Result:");
    println!("Iterations: {}", num_iters);

    if let Some(weights) = xbest {
        println!("Optimal weights:");
        for (i, &w) in weights.iter().enumerate() {
            println!("  Asset {}: {:.4}", i + 1, w);
        }
        let expected_return = expected_returns.dot(&weights);
        println!("Expected return: {:.2}%%", expected_return * 100.0);
    }
}
