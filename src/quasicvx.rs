// -*- coding: utf-8 -*-
#include <doctest/doctest.h>  // for ResultBuilder, Approx, CHECK

#include <cmath>                        // for exp
#include <ellalgo/cutting_plane.hpp>    // for cutting_plane_dc
#include <ellalgo/ell.hpp>              // for ell
#include <ellalgo/ell_stable.hpp>       // for ell_stable
#include <tuple>                        // for get, tuple
#include <xtensor/xaccessible.hpp>      // for xconst_accessible
#include <xtensor/xarray.hpp>           // for xarray_container
#include <xtensor/xlayout.hpp>          // for layout_type, layout_type::row...
#include <xtensor/xtensor_forward.hpp>  // for xarray

#include "ellalgo/cut_config.hpp"  // for CInfo

using Arr = xt::xarray<double, xt::layout_type::row_major>;
using Cut = std::tuple<Arr, double>;

/**
 * @brief
 *
 * @param[in] z
 * @param[in,out] t
 * @return std::tuple<Cut, double>
 */
auto my_quasicvx_oracle(const Arr& z, double& t) -> std::tuple<Cut, bool> {
    auto sqrtx = z[0];
    auto ly = z[1];

    // constraint 1: exp(x) <= y, or sqrtx**2 <= ly
    auto fj = sqrtx * sqrtx - ly;
    if (fj > 0.0) {
        return {{Arr{2 * sqrtx, -1.0}, fj}, false};
    }

    // constraint 2: x - y >= 1
    auto tmp2 = std::exp(ly);
    auto tmp3 = t * tmp2;
    fj = -sqrtx + tmp3;
    if (fj < 0.0)  // feasible
    {
        t = sqrtx / tmp2;
        return {{Arr{-1.0, sqrtx}, 0}, true};
    }

    return {{Arr{-1.0, tmp3}, fj}, false};
}

TEST_CASE("Quasiconvex 1, test feasible") {
    ell E{10.0, Arr{0.0, 0.0}};

    const auto P = my_quasicvx_oracle;
    auto t = 0.0;
    const auto result = cutting_plane_dc(P, E, t);
    const auto& x = std::get<0>(result);
    const auto& ell_info = std::get<1>(result);
    CHECK(ell_info.feasible);
    CHECK(-t == doctest::Approx(-0.4288673397));
    CHECK(x[0] * x[0] == doctest::Approx(0.5029823096));
    CHECK(std::exp(x[1]) == doctest::Approx(1.6536872635));
}

TEST_CASE("Quasiconvex 1, test feasible (stable)") {
    ell_stable E{10.0, Arr{0.0, 0.0}};
    const auto P = my_quasicvx_oracle;
    auto t = 0.0;
    const auto result = cutting_plane_dc(P, E, t);
    const auto& ell_info = std::get<1>(result);
    CHECK(ell_info.feasible);
    // CHECK(-t == doctest::Approx(-0.4288673397));
    // CHECK(x[0] * x[0] == doctest::Approx(0.5029823096));
    // CHECK(std::exp(x[1]) == doctest::Approx(1.6536872635));
}
