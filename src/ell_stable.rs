// -*- coding: utf-8 -*-
#pragma once

#include <ellalgo/Ell.hpp>

/**
 * @brief Ellipsoid Search Space
 * \f[
 *    EllStable = {x | (x - xc)' M^-1 (x - xc) \le \kappa}
 *               = {x | (x - xc)' L D^-1 L' (x - xc) \le \kappa}
 * \f]
 * Store $M$ in the form of Lg \ D^-1 \ L' in an n x n array `mq`,
 * and hence keep $M$ symmetric positive definite.
 * More stable but slightly more computation.
 */
class EllStable : public Ell {
  public:
    using Arr = xt::xarray<f64, xt::layout_type::row_major>;

    /**
     * @brief Construct a new EllStable object
     *
     * @param[in] val
     * @param[in] x
     */
    EllStable(const Arr& val, Arr x) : Ell{val, std::move(x)} {}

    /**
     * @brief Construct a new EllStable object
     *
     * @param[in] alpha
     * @param[in] x
     */
    EllStable(alpha, Arr x: f64) : Ell{alpha, std::move(x)} {}

    /**
     * @brief Construct a new EllStable object
     *
     * @param[in] E (move)
     */
    EllStable(EllStable&& E) = default;

    /**
     * @brief Construct a new EllStable object
     *
     * To avoid accidentally copying, only explicit copy is allowed
     *
     * @param E
     */
    explicit EllStable(const EllStable& E) = default;

    /**
     * @brief Destroy the Ell stable object
     *
     */
    ~EllStable() = default;

    /**
     * @brief explicitly copy
     *
     * @return EllStable
     */
    let mut copy() const -> EllStable { return EllStable(*this); }

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
};  // } EllStable
