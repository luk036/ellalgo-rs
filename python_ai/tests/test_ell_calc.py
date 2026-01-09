"""
Tests for ell_calc.py (Ellipsoid calculations).
"""

import numpy as np
import pytest

from ellalgo import CutStatus, EllCalc, EllCalcCore


def test_ell_calc_core_initialization():
    """Test EllCalcCore initialization."""
    core = EllCalcCore(4.0)

    assert core.n_f == 4.0
    assert core.n_plus_1 == 5.0
    assert core.half_n == 2.0
    assert core.inv_n == 0.25
    assert core.cst1 == 16.0 / 15.0  # n²/(n²-1) = 16/15
    assert core.cst2 == 2.0 / 5.0  # 2/(n+1) = 2/5


def test_ell_calc_core_central_cut():
    """Test central cut calculation."""
    core = EllCalcCore(4.0)

    rho, sigma, delta = core.calc_central_cut(1.0)

    assert rho == 0.0
    assert sigma == 2.0 / 5.0  # 2/(n+1)
    assert delta == 16.0 / 25.0  # n²/(n+1)² = 16/25


def test_ell_calc_initialization():
    """Test EllCalc initialization."""
    calc = EllCalc(4)

    assert calc.core.n_f == 4.0
    assert calc.core.n_plus_1 == 5.0


def test_ell_calc_bias_cut():
    """Test bias cut calculation."""
    calc = EllCalc(2)

    # Valid bias cut
    status, (rho, sigma, delta) = calc.calc_bias_cut(-0.5, 1.0)

    assert status == CutStatus.SUCCESS
    assert rho < 0.0  # Negative for negative beta
    assert 0.0 < sigma < 1.0
    assert 0.0 < delta < 1.0

    # No solution (beta >= 0)
    status, _ = calc.calc_bias_cut(0.5, 1.0)
    assert status == CutStatus.NO_SOLN

    # No effect (beta <= -τ)
    status, _ = calc.calc_bias_cut(-2.0, 1.0)
    assert status == CutStatus.NO_EFFECT

    # No effect (tsq <= 0)
    status, _ = calc.calc_bias_cut(-0.5, 0.0)
    assert status == CutStatus.NO_EFFECT


def test_ell_calc_central_cut():
    """Test central cut calculation."""
    calc = EllCalc(3)

    # Valid central cut
    status, (rho, sigma, delta) = calc.calc_central_cut(1.0)

    assert status == CutStatus.SUCCESS
    assert rho == 0.0
    assert 0.0 < sigma < 1.0
    assert 0.0 < delta < 1.0

    # No effect (tsq <= 0)
    status, _ = calc.calc_central_cut(0.0)
    assert status == CutStatus.NO_EFFECT


def test_ell_calc_parallel_bias_cut():
    """Test parallel bias cut calculation."""
    calc = EllCalc(2)

    # Valid parallel cut
    status, (rho, sigma, delta) = calc.calc_parallel_bias_cut(-0.5, 0.5, 1.0)

    assert status == CutStatus.SUCCESS
    assert abs(rho) < 0.5
    assert 0.0 < sigma < 1.0
    assert 0.0 < delta < 1.0

    # No solution (beta0 * beta1 <= 0)
    status, _ = calc.calc_parallel_bias_cut(-0.5, -0.5, 1.0)
    assert status == CutStatus.NO_SOLN

    # No effect (tsq <= 0)
    status, _ = calc.calc_parallel_bias_cut(-0.5, 0.5, 0.0)
    assert status == CutStatus.NO_EFFECT


def test_ell_calc_single_or_parallel():
    """Test single or parallel cut calculations."""
    calc = EllCalc(2)

    # Single cut
    status1, params1 = calc.calc_single_or_parallel_bias_cut((-0.5, None), 1.0)
    status2, params2 = calc.calc_bias_cut(-0.5, 1.0)

    assert status1 == status2
    assert np.allclose(params1, params2)

    # Parallel cut
    status3, params3 = calc.calc_single_or_parallel_bias_cut((-0.5, 0.5), 1.0)
    status4, params4 = calc.calc_parallel_bias_cut(-0.5, 0.5, 1.0)

    assert status3 == status4
    assert np.allclose(params3, params4)


def test_ell_calc_bias_cut_q():
    """Test bias Q cut calculation."""
    calc = EllCalc(2)

    # Valid Q cut
    status, (rho, sigma, delta) = calc.calc_bias_cut_q(-0.5, 1.0)

    assert status == CutStatus.SUCCESS
    assert rho < 0.0
    assert 0.0 < sigma < 1.0
    assert 0.0 < delta < 1.0

    # Compare with regular bias cut (should be different)
    status2, (rho2, sigma2, delta2) = calc.calc_bias_cut(-0.5, 1.0)

    assert status == status2
    assert not np.allclose(rho, rho2)  # Different rho
    assert not np.allclose(sigma, sigma2)  # Different sigma
    assert np.allclose(delta, delta2)  # Same delta


def test_ell_calc_parallel_q():
    """Test parallel Q cut calculation."""
    calc = EllCalc(2)

    # Valid parallel Q cut
    status, (rho, sigma, delta) = calc.calc_parallel_q(-0.5, 0.5, 1.0)

    assert status == CutStatus.SUCCESS
    assert abs(rho) < 0.5
    assert 0.0 < sigma < 1.0
    assert 0.0 < delta < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
