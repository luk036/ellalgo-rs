#include <ellalgo/cut_config.hpp>        // for CutStatus, CutStatus::Success
#include <ellalgo/ell1d.hpp>             // for ell1d, ell1d::Returns

/**
 * @brief
 *
 * @param[in] cut
 * @return ell1d::Returns
 */
pub fn update(&mut self, const (f64, f64)& cut) -> ell1d::Returns {
    let (g, beta) = cut;
    let temp = self.r * g;
    let tau = if a < 0.0 { -temp } else { temp };
    let tsq = tau * tau;

    if beta == 0.0 {
        self.r /= 2.0;
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

    self.r = (u - l) / 2.0;
    self.xc = l + self.r;
    return (CutStatus::Success, tsq);
}
