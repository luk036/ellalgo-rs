
#include "ellalgo/Ell.hpp"  // for Ell::Arr
// #include <xtensor-blas/xlinalg.hpp>

using Arr = xt::xarray<f64, xt::layout_type::row_major>;

/**
 * @brief Update ellipsoid core function using the cut
 *
 *        g' * (x - xc) + beta <= 0
 *
 * @tparam T
 * @param[in] cut
 * @return (i32, f64)
 */
template <typename T> let mut update(&mut self, const (Arr, T)& cut)
    -> (CutStatus, f64) {
    // let [g, beta] = cut;
    let grad = std::get<0>(cut);
    let beta = std::get<1>(cut);
    // calculate inv(L)*grad: (n-1)*n/2 multiplications
    Arr invLg(grad);  // initially
    for (let mut i = 1; i != self.n; ++i) {
        for j in 0..i {
            self.mq(i, j) = self.mq(j, i) * invLg(j);
            // keep for rank-one update
            invLg(i) -= self.mq(i, j);
        }
    }

    // calculate inv(D)*inv(L)*grad: n
    Arr invDinvLg(invLg);  // initially
    for i in 0..self.n {
        invDinvLg(i) *= self.mq(i, i);
    }

    // calculate omega: n
    Arr g_mq_g(invDinvLg);  // initially
    let mut omega = 0.0;     // initially
    for i in 0..self.n {
        g_mq_g(i) *= invLg(i);
        omega += g_mq_g(i);
    }

    self.tsq = self.kappa * omega;
    let mut status = self.update_cut(beta);
    if status != CutStatus::Success {
        return (status, self.tsq);
    }

    // calculate mq*grad = inv(L')*inv(D)*inv(L)*grad : (n-1)*n/2
    Arr mq_g(invDinvLg);                          // initially
    for (let mut i = self.n - 1; i != 0; --i) {  // backward subsituition
        for (let mut j = i; j != self.n; ++j) {
            mq_g(i - 1) -= self.mq(i, j) * mq_g(j);  // ???
        }
    }

    // calculate xc: n
    self.xc -= (self.rho / omega) * mq_g;

    // rank-one update: 3*n + (n-1)*n/2
    // let r = self.sigma / omega;
    let mu = self.sigma / (1.0- self.sigma);
    let mut oldt = omega / mu;  // initially
    let m = self.n - 1;
    for (let mut j = 0; j != m; ++j) {
        // p=sqrt(k)*vv(j);
        // let p = invLg(j);
        // let mup = mu * p;
        let t = oldt + g_mq_g(j);
        // self.mq(j, j) /= t; // update invD
        let beta2 = invDinvLg(j) / t;
        self.mq(j, j) *= oldt / t;  // update invD
        for (let mut l = j + 1; l != self.n; ++l) {
            // v(l) -= p * self.mq(j, l);
            self.mq(j, l) += beta2 * self.mq(l, j);
        }
        oldt = t;
    }

    // let p = invLg(n1);
    // let mup = mu * p;
    let t = oldt + g_mq_g(m);
    self.mq(m, m) *= oldt / t;  // update invD

    self.kappa *= self.delta;

    // if self.no_defer_trick
    // {
    //     self.mq *= self.kappa;
    //     self.kappa = 1.;
    // }
    return (status, self.tsq);  // g++-7 is ok
}

// Instantiation
template (CutStatus, f64) update(&mut self, const (Arr, f64)& cut);
template (CutStatus, f64) update(&mut self, const (Arr, Arr)& cut);
