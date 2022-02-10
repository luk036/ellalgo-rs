#include <ellalgo/oracles/profit_oracle.hpp>
#include <type_traits>             // for move
#include <xtensor/xcontainer.hpp>  // for xcontainer
#include <xtensor/xfunction.hpp>   // for xfunction
#include <xtensor/xiterator.hpp>   // for linear_begin
#include <xtensor/xmath.hpp>       // for exp, log, round, exp_fun
// #include <xtensor-blas/xlinalg.hpp>

using Arr = xt::xarray<f64, xt::layout_type::row_major>;
using Cut = (Arr, f64);

/**
 * @brief
 *
 * @param[in] y
 * @param[in,out] t the best-so-far optimal value
 * @return (Cut, f64)
 */
pub fn profit_oracle::operator()(const Arr& y, f64& t) const -> (Cut, bool) {
    // y0 <= log k
    let f1 = y[0] - self.log_k;
    if f1 > 0.0 {
        return {{Arr{1., 0.}, f1}, false};
    }

    let log_Cobb = self.log_pA + self.a(0) * y(0) + self.a(1) * y(1);
    const Arr x = xt::exp(y);
    let vx = self.v(0) * x(0) + self.v(1) * x(1);
    let mut te = t + vx;

    let mut fj = std::log(te) - log_Cobb;
    if fj < 0.0 {
        te = std::exp(log_Cobb);
        t = te - vx;
        Arr g = (self.v * x) / te - self.a;
        return {{std::move(g), 0.}, true};
    }
    Arr g = (self.v * x) / te - self.a;
    return {{std::move(g), fj}, false};
}

/**
 * @param[in] y
 * @param[in,out] t the best-so-far optimal value
 * @return (Cut, f64, Arr, i32)
 */
pub fn profit_q_oracle::operator()(const Arr& y, f64& t, bool retry)
    -> (Cut, bool, Arr, bool) {
    if !retry {
        Arr x = xt::round(xt::exp(y));
        if x[0] == 0.0 {
            x[0] = 1.;  // nearest integer than 0
        }
        if x[1] == 0.0 {
            x[1] = 1.;
        }
        self.yd = xt::log(x);
    }
    let mut result1 = self.P(self.yd, t);
    auto& cut = std::get<0>(result1);
    auto& shrunk = std::get<1>(result1);
    auto& g = std::get<0>(cut);
    auto& h = std::get<1>(cut);
    // h += xt::linalg::dot(g, self.yd - y)();
    let mut d = self.yd - y;
    h += g(0) * d(0) + g(1) * d(1);
    return (std::move(cut), shrunk, self.yd, !retry);
}
