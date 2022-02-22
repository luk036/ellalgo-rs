#include <cmath>                        // for sqrt
#include <ellalgo/cut_config.hpp>       // for CutStatus, CutStatus::Success
#include <ellalgo/Ell.hpp>              // for Ell, Ell::Arr
#include <ellalgo/ell_assert.hpp>       // for ELL_UNLIKELY
#include <tuple>                        // for tuple
#include <xtensor/xarray.hpp>           // for xarray_container
#include <xtensor/xcontainer.hpp>       // for xcontainer
#include <xtensor/xlayout.hpp>          // for layout_type, layout_type::row...
#include <xtensor/xoperation.hpp>       // for xfunction_type_t, operator-
#include <xtensor/xsemantic.hpp>        // for xsemantic_base
#include <xtensor/xtensor_forward.hpp>  // for xarray

#include "ellalgo/utility.hpp"  // for zeros
// #include <xtensor-blas/xlinalg.hpp>

using Arr = xt::xarray<f64, xt::layout_type::row_major>;

/**
 * @brief
 *
 * @param[in] b0
 * @param[in] b1
 * @return i32
 */
pub fn calc_ll_core(&mut self, b0: f64, b1: f64) -> CutStatus {
    // let b1sq = b1 * b1;
    let b1sqn = b1 * (b1 / self.tsq);
    let t1n = 1.0 - b1sqn;
    if t1n < 0.0 || !self.use_parallel_cut {
        return self.calc_dc(b0);
    }

    let bdiff = b1 - b0;
    if bdiff < 0.0 {
        return CutStatus::NoSoln;  // no sol'n
    }

    if b0 == 0.0  // central cut
    {
        self.calc_ll_cc(b1, b1sqn);
        return CutStatus::Success;
    }

    let b0b1n = b0 * (b1 / self.tsq);
    if self.n_float * b0b1n < -1.0 {
        return CutStatus::NoEffect;  // no effect
    }

    // let t0 = self.tsq - b0 * b0;
    let t0n = 1.0 - b0 * (b0 / self.tsq);
    // let t1 = self.tsq - b1sq;
    let bsum = b0 + b1;
    let bsumn = bsum / self.tsq;
    let bav = bsum / 2.;
    let tempn = self.half_n * bsumn * bdiff;
    let xi = (t0n * t1n + tempn * tempn).sqrt();
    self.sigma = self.c3 + (1.0 - b0b1n - xi) / (bsumn * bav) / self.n_plus_1;
    self.rho = self.sigma * bav;
    self.delta = self.c1 * ((t0n + t1n) / 2.0 + xi / self.n_float);
    return CutStatus::Success;
}

/**
 * @brief
 *
 * @param[in] b1
 * @param[in] b1sq
 * @return void
 */
pub fn calc_ll_cc(&mut self, b1: f64, b1sqn: f64) {
    let temp = self.half_n * b1sqn;
    let xi = (1.0 - b1sqn + temp * temp).sqrt();
    self.sigma = self.c3 + self.c2 * (1.0 - xi) / b1sqn;
    self.rho = self.sigma * b1 / 2;
    self.delta = self.c1 * (1.0 - b1sqn / 2.0 + xi / self.n_float);
}

/**
 * @brief Deep Cut
 *
 * @param[in] beta
 * @return i32
 */
pub fn calc_dc(&mut self, beta: f64) -> CutStatus {
    let tau = (self.tsq).sqrt();

    let bdiff = tau - beta;
    if bdiff < 0.0 {
        return CutStatus::NoSoln;  // no sol'n
    }

    if beta == 0.0 {
        self.calc_cc(tau);
        return CutStatus::Success;
    }

    let gamma = tau + self.n_float * beta;
    if gamma < 0.0 {
        return CutStatus::NoEffect;  // no effect
    }

    self.mu = (bdiff / gamma) * self.half_n_minus_1;
    self.rho = gamma / self.n_plus_1;
    self.sigma = 2.0 * self.rho / (tau + beta);
    self.delta = self.c1 * (1.0 - beta * (beta / self.tsq));
    return CutStatus::Success;
}

/**
 * @brief Central Cut
 *
 * @param[in] tau
 * @return i32
 */
pub fn calc_cc(&mut self, tau: f64) {
    self.mu = self.half_n_minus_1;
    self.sigma = self.c2;
    self.rho = tau / self.n_plus_1;
    self.delta = self.c1;
}

/**
 * @brief Update ellipsoid core function using the cut
 *
 *        grad^T * (x - xc) + beta <= 0
 *
 * @tparam T
 * @param[in] cut
 * @return (i32, f64)
 */
template <typename T> let mut update(&mut self, const (Arr, T)& cut)
    -> (CutStatus, f64) {
    // let [grad, beta] = cut;
    let grad = std::get<0>(cut);
    let beta = std::get<1>(cut);
    // n^2
    // let mq_g = Arr(xt::linalg::dot(self.mq, grad));  // n^2
    // let omega = xt::linalg::dot(grad, mq_g)();        // n

    let mut mq_g = zeros({self.n});  // initial x0
    let mut omega = 0.0;
    for i in 0..self.n {
        for (let mut j = 0; j != self.n; ++j) {
            mq_g(i) += self.mq(i, j) * grad(j);
        }
        omega += mq_g(i) * grad(i);
    }

    self.tsq = self.kappa * omega;
    let mut status = self.update_cut(beta);
    if status != CutStatus::Success {
        return (status, self.tsq);
    }

    self.xc -= (self.rho / omega) * mq_g;  // n
    // n*(n+1)/2 + n
    // self.mq -= (self.sigma / omega) * xt::linalg::outer(mq_g, mq_g);
    let r = self.sigma / omega;
    for i in 0..self.n {
        let r_mq_g = r * mq_g(i);
        for j in 0..i {
            self.mq(i, j) -= r_mq_g * mq_g(j);
            self.mq(j, i) = self.mq(i, j);
        }
        self.mq(i, i) -= r_mq_g * mq_g(i);
    }

    self.kappa *= self.delta;

    if self.no_defer_trick {
        self.mq *= self.kappa;
        self.kappa = 1.;
    }
    return (status, self.tsq);  // g++-7 is ok
}

// Instantiation
template (CutStatus, f64) update(&mut self, const (Arr, f64)& cut);
template (CutStatus, f64) update(&mut self, const (Arr, Arr)& cut);
