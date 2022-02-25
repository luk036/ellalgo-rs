use super::cutting_plane::OracleFeas;
use ndarray::prelude::*;

type Arr = Array1<f64>;

#[derive(Debug)]
struct MyOracle {}

impl OracleFeas for MyOracle {
    type ArrayType = Arr;
    type CutChoices = f64;

    /**
     * @brief
     *
     * @param[in] z
     * @return std::optional<Cut>
     */
    fn asset_feas(&mut self, z: &Arr) -> Option<(Arr, f64)> {
        let x = z[0];
        let y = z[1];

        // constraint 1: x + y <= 3
        let fj = x + y - 3.0;
        if fj > 0.0 {
            return Some((array![1.0, 1.0], fj));
        }
        // constraint 2: x - y >= 1
        let fj = -x + y + 1.0;
        if fj > 0.0 {
            return Some((array![-1.0, 1.0], fj));
        }
        return None;
    }
}

mod tests {
    use super::*;
    use crate::cutting_plane::{cutting_plane_feas, Options};
    use crate::ell::Ell;
    use ndarray::array;

    // use super::ell_stable::EllStable;

    #[test]
    pub fn test_example2() {
        let mut ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut oracle = MyOracle {};
        let options = Options {
            max_iter: 2000,
            tol: 1e-12,
        };
        let (feasible, _niter, _status) = cutting_plane_feas(&mut oracle, &mut ell, &options);
        assert!(feasible);
    }
}
