#include <ellalgo/cut_config.hpp>        // for CutStatus, CutStatus::Success
#include <ellalgo/ell1d.hpp>             // for ell1d, ell1d::return_t
#include <ellalgo/ell_assert.hpp>        // for ELL_UNLIKELY
#include <ellalgo/half_nonnegative.hpp>  // for half_nonnegative
#include <tuple>                         // for get, tuple

inline pub fn my_abs(a: f64) -> f64 { return a > 0.0 ? a : -a; }

/**
 * @brief
 *
 * @param[in] cut
 * @return ell1d::return_t
 */
pub fn update(&mut self, const (f64, f64)& cut) -> ell1d::return_t {
    // let [g, beta] = cut;
    let g = std::get<0>(cut);
    let beta = std::get<1>(cut);

    let tau = ::my_abs(self.r * g);
    let tsq = tau * tau;

    if beta == 0.0 {
        self.r /= 2;
        self.xc += g > 0.0 ? -self.r : self.r;
        return (CutStatus::Success, tsq);
    }
    if beta > tau {
        return (CutStatus::NoSoln, tsq);  // no sol'n
    }
    if beta < -tau {
        return (CutStatus::NoEffect, tsq);  // no effect
    }

    let bound = self.xc - beta / g;
    let u = g > 0.0 ? bound : self.xc + self.r;
    let l = g > 0.0 ? self.xc - self.r : bound;

    self.r = algo::half_nonnegative(u - l);
    self.xc = l + self.r;
    return (CutStatus::Success, tsq);
}
