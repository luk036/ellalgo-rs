use super::cutting_plane::OracleOptim;
use super::ell_stable::EllStable;
use ndarray::prelude::*;

type Arr = Array1<f64>;

#[derive(Debug)]
pub struct MyOracle {}

impl OracleOptim for MyOracle {
    /**
     * @brief
     *
     * @param[in] z
     * @param[in,out] t
     * @return std::tuple<Cut, double>
     */
    fn asset_optim(&mut self, z: &Arr, t: &mut f64) -> ((Arr, f64), bool) {
        let x = z[0];
        let y = z[1];

        // constraint 1: x + y <= 3
        let fj = x + y - 3.0;
        if fj > 0.0 {
            return ((array![1.0, 1.0], fj), false);
        }
        // constraint 2: x - y >= 1
        let fj = -x + y + 1.0;
        if fj > 0.0 {
            return ((array![-1.0, 1.0], fj), false);
        }
        // objective: maximize x + y
        let f0 = x + y;
        let fj = *t - f0;
        if fj < 0.0 {
            *t = f0;
            return ((array![-1.0, -1.0], 0.0), true);
        }
        ((array![-1.0, -1.0], fj), false)
    }
}

mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    pub fn test_feasible() {
        let mut ell = EllStable::new_wtih_scalar(10.0, array![0.0, 0.0]);
        let oracle = MyOracle {};
        let mut t = -1.e100;  // std::numeric_limits<double>::min()
        let (x_opt, _niter, _status) = cutting_plane_optim(&oracle, &mut ell, &mut t);
        if let Some(x) = x_opt {
            assert!(x[0] >= 0.0);
        }
        else {
            assert!(false); // not feasible
        }
    }

    #[test]
    pub fn test_infeasible1() {
        let mut ell = EllStable::new_wtih_scalar(10.0, array![100.0, 100.0]);  // wrong initial guess
                                             // or ellipsoid is too small
        let oracle = MyOracle {};
        let t = -1.e100;  // std::numeric_limits<double>::min()
        let (x_opt, _niter, status) = cutting_plane_optim(&oracle, &mut ell, &mut t);
        if let Some(_x) = x_opt {
            assert!(false);
        }
        else {
            assert_eq!(status, CutStatus::NoSoln);  // no sol'n
        }
    }

    #[test]
    pub fn test_infeasible2() {
        let mut ell = EllStable::new_wtih_scalar(10.0, array![0.0, 0.0]);
        let mut oracle = MyOracle {};
        // wrong initial guess
        let (x_opt, _niter, status) = cutting_plane_optim(&oracle, &mut ell, &mut 100);
        if let Some(_x) = x_opt {
            assert!(false);
        }
        else {
            assert_eq!(status, CutStatus::NoSoln);  // no sol'n
        }
    }
}
