use cutting_plane::OracleOptim;

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
    fn asset_optim<f64>(&mut self, x: &Arr, t: &mut f64) -> ((Arr, f64), bool) {
        auto x = z[0];
        auto y = z[1];

        // constraint 1: x + y <= 3
        auto fj = x + y - 3.0;
        if (fj > 0.0) {
            return {{Arr{1.0, 1.0}, fj}, false};
        }

        // constraint 2: x - y >= 1
        fj = -x + y + 1.0;
        if (fj > 0.0) {
            return {{Arr{-1.0, 1.0}, fj}, false};
        }

        // objective: maximize x + y
        auto f0 = x + y;
        fj = t - f0;
        if (fj < 0.0) {
            t = f0;
            return {{Arr{-1.0, -1.0}, 0.0}, true};
        }
        return {{Arr{-1.0, -1.0}, fj}, false};
    }

TEST_CASE("Example 1, test feasible") {
    ell_stable E{10.0, Arr{0.0, 0.0}};
    const auto P = my_oracle;
    auto t = -1.e100;  // std::numeric_limits<double>::min()
    const auto result = cutting_plane_dc(P, E, t);
    const auto& x = std::get<0>(result);
    const auto& ell_info = std::get<1>(result);
    CHECK(x[0] >= 0.0);
    CHECK(ell_info.feasible);
}

TEST_CASE("Example 1, test infeasible 1") {
    ell_stable E{10.0, Arr{100.0, 100.0}};  // wrong initial guess
                                         // or ellipsoid is too small
    const auto P = my_oracle;
    auto t = -1.e100;  // std::numeric_limits<double>::min()
    const auto result = cutting_plane_dc(P, E, t);
    const auto& ell_info = std::get<1>(result);
    CHECK(!ell_info.feasible);
    CHECK(ell_info.status == CUTStatus::nosoln);  // no sol'n
}

TEST_CASE("Example 1, test infeasible 2") {
    ell_stable E{10.0, Arr{0.0, 0.0}};
    const auto P = my_oracle;
    const auto result = cutting_plane_dc(P, E, 100.0);  // wrong initial guess
    const auto& ell_info = std::get<1>(result);
    CHECK(!ell_info.feasible);
}
