/// Defines an injectable strategy for building update_dcs.
trait CalcStrategy {
    fn calc_ll_dc(&self, beta: f64, tsq: f64) -> (CutStatus, f64, f64, f64);
}

struct EllCalc;

impl CalcStrategy for EllCalc {
    fn calc_ll_dc(&self, beta: f64, tsq: f64) -> (CutStatus, f64, f64, f64) {
        println!("Walking update_dc from {} to {}: 4 km, 30 min", beta, tsq);
    }
}

struct EllCalcQ;

impl CalcStrategy for EllCalcQ {
    fn calc_ll_dc(&self, beta: f64, tsq: f64) -> (CutStatus, f64, f64, f64) {
        println!(
            "Public transport update_dc from {} to {}: 3 km, 5 min",
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

    pub fn update_dc(&self, beta: f64, tsq: f64) -> (CutStatus, f64, f64, f64) {
        self.helper.calc_ll_dc(beta, tsq);
    }
}

fn main() {
    let ellip = EllStable::new(EllCalc);
    ellip.update_dc("Home", "Club");
    ellip.update_dc("Club", "Work");

    let ellip = EllStable::new(EllCalcQ);
    ellip.update_dc("Home", "Club");
    ellip.update_dc("Club", "Work");
}
