#[cfg(test)]
mod tests {
    use crate::cutting_plane::CutStatus;
    use crate::ell_calc::EllCalc;

    #[test]
    fn test_ell_calc_central_cut_no_effect() {
        let ell_calc = EllCalc::new(4);
        let (status, (rho, sigma, delta)) = ell_calc.calc_central_cut(0.0);
        assert_eq!(status, CutStatus::Success);
        assert_eq!(rho, 0.0);
        // For n=4, cst2 = 2.0 / (4.0 + 1.0) = 0.4
        assert_eq!(sigma, 0.4);
        // For n=4, cst1 = 4.0*4.0 / (4.0*4.0 - 1.0) = 16.0 / 15.0
        assert_eq!(delta, 16.0 / 15.0);
    }

    #[test]
    fn test_ell_calc_central_cut_positive_beta() {
        let ell_calc = EllCalc::new(4);
        let (status, (rho, sigma, delta)) = ell_calc.calc_central_cut(0.01);
        assert_eq!(status, CutStatus::Success);
        assert!(rho > 0.0);
        assert!(sigma > 0.0);
        assert!(delta > 0.0);
    }

    #[test]
    fn test_ell_calc_central_cut_positive_tsq() {
        let ell_calc = EllCalc::new(4);
        let (status, (rho, sigma, delta)) = ell_calc.calc_central_cut(0.01);
        assert_eq!(status, CutStatus::Success);
        assert!(rho > 0.0);
        assert!(sigma > 0.0);
        assert!(delta > 0.0);
    }

    #[test]
    fn test_ell_calc_parallel_central_cut_no_effect() {
        let ell_calc = EllCalc::new(4);
        let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_central_cut(0.0, 0.0);
        assert_eq!(status, CutStatus::Success);
        assert_eq!(rho, 0.0);
        // For n=4, cst2 = 0.4
        assert_eq!(sigma, 0.4);
        // For n=4, cst1 = 16.0 / 15.0
        assert_eq!(delta, 16.0 / 15.0);
    }

    #[test]
    fn test_ell_calc_parallel_central_cut_positive_beta() {
        let ell_calc = EllCalc::new(4);
        let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_central_cut(0.05, 0.01);
        assert_eq!(status, CutStatus::Success);
        assert!(rho > 0.0);
        assert!(sigma > 0.0);
        assert!(delta > 0.0);
    }

    #[test]
    fn test_ell_calc_parallel_central_cut_negative_beta() {
        let ell_calc = EllCalc::new(4);
        let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_central_cut(-0.1, 0.01);
        assert_eq!(status, CutStatus::NoSoln);
        assert_eq!(rho, 0.0);
        assert_eq!(sigma, 0.0);
        assert_eq!(delta, 0.0);
    }

    #[test]
    fn test_ell_calc_parallel_central_cut_two_cuts() {
        let ell_calc = EllCalc::new(4);
        let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_central_cut(0.1, 0.2);
        assert_eq!(status, CutStatus::Success);
        assert!(rho > 0.0);
        assert!(sigma > 0.0);
        assert!(delta > 0.0);
    }

    #[test]
    fn test_ell_calc_parallel_central_cut_two_cuts_negative() {
        let ell_calc = EllCalc::new(4);
        let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_central_cut(-0.1, 0.2);
        assert_eq!(status, CutStatus::NoSoln);
        assert_eq!(rho, 0.0);
        assert_eq!(sigma, 0.0);
        assert_eq!(delta, 0.0);
    }
}
