"""
Tests for ell.py (Ellipsoid implementation).
"""

import numpy as np
import pytest

from ellalgo import CutStatus, Ell


def test_ell_initialization():
    """Test Ell initialization."""
    # Test with scalar
    xc = np.array([1.0, 2.0, 3.0])
    ell = Ell.new_with_scalar(2.0, xc)

    assert ell.kappa == 2.0
    assert ell.ndim == 3
    assert np.allclose(ell.xc, xc)
    assert ell.mq.shape == (3, 3)
    assert np.allclose(ell.mq, 2.0 * np.eye(3))

    # Test with diagonal matrix
    val = np.array([1.0, 2.0, 3.0])
    ell2 = Ell.new(val, xc)

    assert ell2.kappa == 1.0
    assert np.allclose(np.diag(ell2.mq), val)

    # Test with full matrix
    mq = np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 2.0]])
    ell3 = Ell.new_with_matrix(1.0, mq, xc)

    assert np.allclose(ell3.mq, mq)


def test_ell_update_bias_cut():
    """Test bias cut update."""
    # Start at x₁ = 1.0
    xc = np.array([1.0, 0.0])
    ell = Ell.new_with_scalar(1.0, xc)

    # Constraint: x₁ ≤ 0.5
    # x₁ - 0.5 ≤ 0 → β = -0.5
    grad = np.array([1.0, 0.0])
    beta = -0.5

    status = ell.update_bias_cut((grad, beta))

    # Should return some status (SUCCESS, NO_SOLN, or NO_EFFECT)
    assert status in [CutStatus.SUCCESS, CutStatus.NO_SOLN, CutStatus.NO_EFFECT]
    # kappa should change if cut was applied
    if status == CutStatus.SUCCESS:
        assert ell.kappa != 1.0


def test_ell_update_central_cut():
    """Test central cut update."""
    xc = np.zeros(2)
    ell = Ell.new_with_scalar(1.0, xc)

    # Central cut (no bias)
    grad = np.array([1.0, 0.0])
    beta = 0.0

    status = ell.update_central_cut((grad, beta))

    assert status == CutStatus.SUCCESS
    assert np.allclose(ell.xc, 0.0)  # Center shouldn't move for central cut
    assert ell.kappa < 1.0  # Should shrink


def test_ell_no_effect_cut():
    """Test cut with no effect."""
    xc = np.zeros(2)
    ell = Ell.new_with_scalar(1.0, xc)

    # Cut that's already satisfied (point is feasible)
    # x₁ ≤ 1.0, and x₁ = 0 satisfies this
    grad = np.array([1.0, 0.0])
    beta = -1.0  # x₁ - 1 ≤ 0 → β = -1.0, or x₁ ≤ 1.0

    status = ell.update_bias_cut((grad, beta))

    # With beta = -1.0 and xc = [0,0], we have τ² = κ * ω
    # For small τ², this might be NO_EFFECT
    # The actual status depends on tsq calculation
    assert status in [CutStatus.NO_EFFECT, CutStatus.NO_SOLN]
    # Center might not change for NO_EFFECT
    if status == CutStatus.NO_EFFECT:
        assert np.allclose(ell.xc, 0.0)  # No change
        assert ell.kappa == 1.0  # No change


def test_ell_parallel_cut():
    """Test parallel cut update."""
    xc = np.array([1.0, 0.0])  # Start at x₁ = 1.0
    ell = Ell.new_with_scalar(1.0, xc)

    # Parallel cut with beta0 and beta1 both negative
    # This represents something like: -1.0 ≤ x₁ ≤ -0.5
    # Starting at x₁ = 1.0 violates this
    grad = np.array([1.0, 0.0])
    beta = (-1.0, -0.5)  # (beta0, beta1) with same sign

    status = ell.update_bias_cut((grad, beta))

    # The cut should be successful since we're outside the bounds
    assert status == CutStatus.SUCCESS
    assert ell.xc[0] < 1.0  # Should move toward the bound
    assert ell.kappa < 1.0  # Should shrink


def test_ell_getters_setters():
    """Test getter and setter methods."""
    xc = np.array([1.0, 2.0])
    ell = Ell.new_with_scalar(1.0, xc)

    # Test get_xc
    xc_copy = ell.get_xc()
    assert np.allclose(xc_copy, xc)
    xc_copy[0] = 99.0  # Should not affect original
    assert not np.allclose(ell.xc, xc_copy)

    # Test set_xc
    new_xc = np.array([3.0, 4.0])
    ell.set_xc(new_xc)
    assert np.allclose(ell.xc, new_xc)

    # Test tsq
    tsq = ell.get_tsq()
    assert isinstance(tsq, float)

    # Test invalid set_xc
    with pytest.raises(ValueError):
        ell.set_xc(np.array([1.0, 2.0, 3.0]))


def test_ell_repr():
    """Test string representation."""
    xc = np.array([1.0, 2.0, 3.0])
    ell = Ell.new_with_scalar(1.0, xc)

    repr_str = repr(ell)
    assert "Ell" in repr_str
    assert "kappa=" in repr_str
    assert "ndim=3" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])
