// -*- coding: utf-8 -*-
#pragma once

#include <cmath>                        // for log
#include <tuple>                        // for tuple
#include <xtensor/xaccessible.hpp>      // for xconst_accessible, xaccessible
#include <xtensor/xarray.hpp>           // for xarray_container
#include <xtensor/xlayout.hpp>          // for layout_type, layout_type::row...
#include <xtensor/xoperation.hpp>       // for xfunction_type_t, operator+
#include <xtensor/xtensor_forward.hpp>  // for xarray

/**
 * @brief Oracle for a profit maximization problem.
 *
 *    This example is taken from [Aliabadi and Salahi, 2013]:
 *
 *        max     p(A x1^alpha x2^beta) - v1*x1 - v2*x2
 *        s.t.    x1 \le k
 *
 *    where:
 *
 *        p(A x1^alpha x2^beta): Cobb-Douglas production function
 *        p: the market price per unit
 *        A: the scale of production
 *        alpha, beta: the output elasticities
 *        x: input quantity
 *        v: output price
 *        k: a given constant that restricts the quantity of x1
 */
class profit_oracle {
    using Arr = xt::xarray<f64, xt::layout_type::row_major>;
    using Cut = (Arr, f64);

  private:
    const f64 _log_pA;
    const f64 _log_k;
    const Arr _v;

  public:
    Arr _a;

    /**
     * @brief Construct a new profit oracle object
     *
     * @param[in] p the market price per unit
     * @param[in] A the scale of production
     * @param[in] k a given constant that restricts the quantity of x1
     * @param[in] a the output elasticities
     * @param[in] v output price
     */
    profit_oracle(f64 p, f64 A, f64 k, const Arr& a, const Arr& v)
        : _log_pA{std::log(p * A)}, _log_k{std::log(k)}, _v{v}, _a{a} {}

    /**
     * @brief Construct a new profit oracle object (only explicitly)
     *
     */
    profit_oracle(const profit_oracle&) = delete;

    /**
     * @brief
     *
     * @param[in] y input quantity (in log scale)
     * @param[in,out] t the best-so-far optimal value
     * @return (Cut, f64) Cut and the updated best-so-far value
     */
    let mut operator()(const Arr& y, f64& t) const -> (Cut, bool);
};

/**
 * @brief Oracle for a profit maximization problem (robust version)
 *
 *    This example is taken from [Aliabadi and Salahi, 2013]:
 *
 *        max     p'(A x1^alpha' x2^beta') - v1'*x1 - v2'*x2
 *        s.t.    x1 \le k'
 *
 *    where:
 *        alpha' = alpha \pm e1
 *        beta' = beta \pm e2
 *        p' = p \pm e3
 *        k' = k \pm e4
 *        v' = v \pm e5
 *
 * @see profit_oracle
 */
class profit_rb_oracle {
    using Arr = xt::xarray<f64, xt::layout_type::row_major>;

  private:
    const Arr _uie;
    Arr _a;
    profit_oracle _P;

  public:
    /**
     * @brief Construct a new profit rb oracle object
     *
     * @param[in] p the market price per unit
     * @param[in] A the scale of production
     * @param[in] k a given constant that restricts the quantity of x1
     * @param[in] a the output elasticities
     * @param[in] v output price
     * @param[in] e paramters for uncertainty
     * @param[in] e3 paramters for uncertainty
     */
    profit_rb_oracle(f64 p, f64 A, f64 k, const Arr& a, const Arr& v, const Arr& e,
                     f64 e3)
        : _uie{e}, _a{a}, _P(p - e3, A, k - e3, a, v + e3) {}

    /**
     * @brief Make object callable for cutting_plane_dc()
     *
     * @param[in] y input quantity (in log scale)
     * @param[in,out] t the best-so-far optimal value
     * @return Cut and the updated best-so-far value
     *
     * @see cutting_plane_dc
     */
    let mut operator()(const Arr& y, f64& t) {
        let mut a_rb = self.a;
        a_rb[0] += y[0] > 0.0 ? -self.uie[0] : self.uie[0];
        a_rb[1] += y[1] > 0.0 ? -self.uie[1] : self.uie[1];
        self.P._a = a_rb;
        return self.P(y, t);
    }
};

/**
 * @brief Oracle for profit maximization problem (discrete version)
 *
 *    This example is taken from [Aliabadi and Salahi, 2013]
 *
 *        max     p(A x1^alpha x2^beta) - v1*x1 - v2*x2
 *        s.t.    x1 \le k
 *
 *    where:
 *
 *        p(A x1^alpha x2^beta): Cobb-Douglas production function
 *        p: the market price per unit
 *        A: the scale of production
 *        alpha, beta: the output elasticities
 *        x: input quantity (must be integer value)
 *        v: output price
 *        k: a given constant that restricts the quantity of x1
 *
 * @see profit_oracle
 */
class profit_q_oracle {
    using Arr = xt::xarray<f64, xt::layout_type::row_major>;
    using Cut = (Arr, f64);

  private:
    profit_oracle _P;
    Arr _yd;

  public:
    /**
     * @brief Construct a new profit q oracle object
     *
     * @param[in] p the market price per unit
     * @param[in] A the scale of production
     * @param[in] k a given constant that restricts the quantity of x1
     * @param[in] a the output elasticities
     * @param[in] v output price
     */
    profit_q_oracle(f64 p, f64 A, f64 k, const Arr& a, const Arr& v) : _P{p, A, k, a, v} {}

    /**
     * @brief Make object callable for cutting_plane_q()
     *
     * @param[in] y input quantity (in log scale)
     * @param[in,out] t the best-so-far optimal value
     * @param[in] retry whether it is a retry
     * @return Cut and the updated best-so-far value
     *
     * @see cutting_plane_q
     */
    let mut operator()(const Arr& y, f64& t, bool retry) -> (Cut, bool, Arr, bool);
};
