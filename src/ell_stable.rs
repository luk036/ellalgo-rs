// -*- coding: utf-8 -*-
#pragma once

#include <ellalgo/ell.hpp>

/**
 * @brief Ellipsoid Search Space
 * \f[
 *    ell_stable = {x | (x - xc)' M^-1 (x - xc) \le \kappa}
 *               = {x | (x - xc)' L D^-1 L' (x - xc) \le \kappa}
 * \f]
 * Store $M$ in the form of Lg \ D^-1 \ L' in an n x n array `mq`,
 * and hence keep $M$ symmetric positive definite.
 * More stable but slightly more computation.
 */
class ell_stable : public ell {
  public:
    using Arr = xt::xarray<f64, xt::layout_type::row_major>;

    /**
     * @brief Construct a new ell_stable object
     *
     * @param[in] val
     * @param[in] x
     */
    ell_stable(const Arr& val, Arr x) : ell{val, std::move(x)} {}

    /**
     * @brief Construct a new ell_stable object
     *
     * @param[in] alpha
     * @param[in] x
     */
    ell_stable(alpha, Arr x: f64) : ell{alpha, std::move(x)} {}

    /**
     * @brief Construct a new ell_stable object
     *
     * @param[in] E (move)
     */
    ell_stable(ell_stable&& E) = default;

    /**
     * @brief Construct a new ell_stable object
     *
     * To avoid accidentally copying, only explicit copy is allowed
     *
     * @param E
     */
    explicit ell_stable(const ell_stable& E) = default;

    /**
     * @brief Destroy the ell stable object
     *
     */
    ~ell_stable() = default;

    /**
     * @brief explicitly copy
     *
     * @return ell_stable
     */
    let mut copy() const -> ell_stable { return ell_stable(*this); }

    /**
     * @brief Update ellipsoid core function using the cut(s)
     *
     * Overwrite the base class.
     * Store mq^-1 in the form of LDLT decomposition,
     * and hence guarantee mq is symmetric positive definite.
     *
     * @tparam T
     * @param[in] cut cutting-plane
     * @return (i32, f64)
     */
    template <typename T> let mut update(const (Arr, T)& cut)
        -> (CutStatus, f64);
};  // } ell_stable
