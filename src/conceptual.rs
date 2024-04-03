/// Defines an injectable strategy for building update_bias_cuts.
trait CalcStrategy {
    fn calc_parallel_bias_cut(&self, beta: f64, tsq: f64) -> (CutStatus, f64, f64, f64);
}

struct EllCalc;

impl CalcStrategy for EllCalc {
    fn calc_parallel_bias_cut(&self, beta: f64, tsq: f64) -> (CutStatus, f64, f64, f64) {
        println!("Walking update_bias_cut from {} to {}: 4 km, 30 min", beta, tsq);
    }
}

struct EllCalcQ;

impl CalcStrategy for EllCalcQ {
    fn calc_parallel_bias_cut(&self, beta: f64, tsq: f64) -> (CutStatus, f64, f64, f64) {
        println!(
            "Public transport update_bias_cut from {} to {}: 3 km, 5 min",
            beta, tsq
        );
    }
}

struct EllStable<Calc: CalcStrategy> {
    helper: Calc,
}

impl<Calc: CalcStrategy> EllStable<T> {
    pub fn new(helper: Calc) -> Self {
        Self { helper }
    }

    pub fn update_bias_cut(&self, beta: f64, tsq: f64) -> (CutStatus, f64, f64, f64) {
        self.helper.calc_parallel_bias_cut(beta, tsq);
    }
}

fn main() {
    let ellip = EllStable::new(EllCalc);
    ellip.update_bias_cut("Home", "Club");
    ellip.update_bias_cut("Club", "Work");

    let ellip = EllStable::new(EllCalcQ);
    ellip.update_bias_cut("Home", "Club");
    ellip.update_bias_cut("Club", "Work");
}
