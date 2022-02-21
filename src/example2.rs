/* -*- coding: utf-8 -*- */
#include <doctest/doctest.h>  // for ResultBuilder, TestCase, CHECK

#include <ellalgo/cutting_plane.hpp>    // for cutting_plane_feas
#include <ellalgo/ell_stable.hpp>       // for ell_stable
#include <tuple>                        // for get, tuple
#include <xtensor/xaccessible.hpp>      // for xconst_accessible
#include <xtensor/xlayout.hpp>          // for layout_type, layout_type::row...
#include <xtensor/xtensor_forward.hpp>  // for xarray

#include "ellalgo/cut_config.hpp"  // for CInfo
// #include <optional>

using Arr = xt::xarray<double, xt::layout_type::row_major>;
using Cut = std::tuple<Arr, double>;

/**
 * @brief
 *
 * @param[in] z
 * @return std::optional<Cut>
 */
auto my_oracle2(const Arr& z) -> Cut* {
    static auto cut1 = Cut{Arr{1.0, 1.0}, 0.0};
    static auto cut2 = Cut{Arr{-1.0, 1.0}, 0.0};

    auto x = z[0];
    auto y = z[1];

    // constraint 1: x + y <= 3
    auto fj = x + y - 3.0;
    if (fj > 0.0) {
        std::get<1>(cut1) = fj;
        return &cut1;
    }

    // constraint 2: x - y >= 1
    fj = -x + y + 1.0;
    if (fj > 0.0) {
        std::get<1>(cut2) = fj;
        return &cut2;
    }

    return nullptr;
}

TEST_CASE("Example 2") {
    ell_stable E{10.0, Arr{0.0, 0.0}};

    const auto P = my_oracle2;
    auto ell_info = cutting_plane_feas(P, E);
    CHECK(ell_info.feasible);
}
