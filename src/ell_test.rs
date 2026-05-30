#[cfg(test)]
mod tests {
    use crate::arr::Arr;
    use crate::cutting_plane::{CutStatus, ParallelCut, SearchSpace, SingleCut};
    use crate::ell::Ell;
    use approx_eq::assert_approx_eq;

    fn create_test_ell() -> Ell {
        Ell::new_with_scalar(0.01, Arr::new(4))
    }

    #[test]
    fn test_construct() {
        let ellip = create_test_ell();
        assert!(!ellip.no_defer_trick);
        assert_approx_eq!(ellip.kappa, 0.01);
        assert_eq!(ellip.mq, Arr::eye(4));
        assert_eq!(ellip.xc, Arr::new(4));
        assert_approx_eq!(ellip.tsq, 0.0);
    }

    #[test]
    fn test_update_central_cut() {
        let mut ellip = create_test_ell();
        let cut = (0.5 * Arr::ones(4), SingleCut(0.0));
        let status = ellip.update_central_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_eq!(ellip.xc, -0.01 * Arr::ones(4));
        assert_eq!(ellip.mq, &Arr::eye(4) - &(0.1 * Arr::full(4, 4, 1.0)));
        assert_approx_eq!(ellip.kappa, 0.16 / 15.0);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_bias_cut() {
        let mut ellip = create_test_ell();
        let cut = (0.5 * Arr::ones(4), SingleCut(0.05));
        let status = ellip.update_bias_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(ellip.xc[0], -0.03);
        assert_approx_eq!(ellip.mq.at(0, 0), 0.8);
        assert_approx_eq!(ellip.kappa, 0.008);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_parallel_central_cut() {
        let mut ellip = create_test_ell();
        let cut = (0.5 * Arr::ones(4), ParallelCut(0.0, Some(0.05)));
        let status = ellip.update_central_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_eq!(ellip.xc, -0.01 * Arr::ones(4));
        assert_eq!(ellip.mq, &Arr::eye(4) - &(0.2 * Arr::full(4, 4, 1.0)));
        assert_approx_eq!(ellip.kappa, 0.012);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_parallel() {
        let mut ellip = create_test_ell();
        let cut = (0.5 * Arr::ones(4), ParallelCut(0.01, Some(0.04)));
        let status = ellip.update_bias_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(ellip.xc[0], -0.0116);
        assert_approx_eq!(ellip.mq.at(0, 0), 1.0 - 0.232);
        assert_approx_eq!(ellip.kappa, 0.01232);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_parallel_no_effect() {
        let mut ellip = create_test_ell();
        let cut = (0.5 * Arr::ones(4), ParallelCut(-0.04, Some(0.0625)));
        let status = ellip.update_bias_cut(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_eq!(ellip.xc, Arr::new(4));
        assert_eq!(ellip.mq, Arr::eye(4));
        assert_approx_eq!(ellip.kappa, 0.01);
    }

    #[test]
    fn test_update_q_no_effect() {
        let mut ellip = create_test_ell();
        let cut = (0.5 * Arr::ones(4), ParallelCut(-0.04, Some(0.0625)));
        let status = ellip.update_q(&cut);
        assert_eq!(status, CutStatus::NoEffect);
        assert_eq!(ellip.xc, Arr::new(4));
        assert_eq!(ellip.mq, Arr::eye(4));
        assert_approx_eq!(ellip.kappa, 0.01);
    }

    #[test]
    fn test_update_q() {
        let mut ellip = create_test_ell();
        let cut = (0.5 * Arr::ones(4), ParallelCut(0.01, Some(0.04)));
        let status = ellip.update_q(&cut);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(ellip.xc[0], -0.0116);
        assert_approx_eq!(ellip.mq.at(0, 0), 1.0 - 0.232);
        assert_approx_eq!(ellip.kappa, 0.01232);
        assert_approx_eq!(ellip.tsq, 0.01);
    }

    #[test]
    fn test_update_central_cut_mq() {
        let mut ellip = create_test_ell();
        let cut = (0.5 * Arr::ones(4), SingleCut(0.0));
        let _ = ellip.update_central_cut(&cut);
        let mq_expected = &Arr::eye(4) - &(0.1 * Arr::full(4, 4, 1.0));
        for i in 0..4 {
            for j in 0..4 {
                assert_approx_eq!(ellip.mq.at(i, j), mq_expected.at(i, j));
            }
        }
    }

    #[test]
    fn test_no_defer_trick() {
        let mut ellip = create_test_ell();
        ellip.no_defer_trick = true;
        let cut = (0.5 * Arr::ones(4), SingleCut(0.0));
        let _ = ellip.update_central_cut(&cut);
        assert_approx_eq!(ellip.kappa, 1.0);
        let mq_expected = &(&Arr::eye(4) - &(0.1 * Arr::full(4, 4, 1.0))) * (0.16 / 15.0);
        for i in 0..4 {
            for j in 0..4 {
                assert_approx_eq!(ellip.mq.at(i, j), mq_expected.at(i, j));
            }
        }
    }
}
