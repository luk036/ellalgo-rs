"""
Tests for cutting_plane.py (core algorithms).
"""

from typing import Optional, Tuple

import numpy as np
import pytest

from ellalgo import (
    CutStatus,
    Options,
    binary_search,
    cutting_plane,
    cutting_plane_feas,
    cutting_plane_optim,
)


class SimpleFeasOracle:
    """Simple feasibility oracle for testing."""

    def __init__(self, target_norm: float = 1.0):
        self.target_norm = target_norm
        self.call_count = 0

    def assess_feas(self, xc: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Check if point is inside target norm ball."""
        self.call_count += 1
        norm_sq = np.sum(xc**2)

        if norm_sq <= self.target_norm**2:
            return None  # Feasible

        # Return cut
        grad = 2.0 * xc
        beta = self.target_norm**2 - norm_sq
        return grad, beta


class SimpleOptimOracle:
    """Simple optimization oracle for testing."""

    def __init__(self):
        self.best_value = float("inf")
        self.call_count = 0

    def assess_optim(
        self,
        xc: np.ndarray,
        gamma: float,
    ) -> Tuple[Tuple[np.ndarray, float], bool]:
        """Minimize ||x||²."""
        self.call_count += 1

        current_value = np.sum(xc**2)
        if current_value < self.best_value:
            self.best_value = current_value

        # Check optimality (near origin)
        if current_value < 1e-6:
            return (np.zeros_like(xc), 0.0), True

        # Return gradient cut
        grad = 2.0 * xc
        beta = -0.1  # Bias cut
        return (grad, beta), False


class SimpleBSOracle:
    """Simple binary search oracle for testing."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.call_count = 0

    def assess_bs(self, gamma: float) -> bool:
        """Check if gamma >= threshold."""
        self.call_count += 1
        return gamma >= self.threshold


def test_cut_status():
    """Test CutStatus enum."""
    assert CutStatus.SUCCESS.value == "success"
    assert CutStatus.NO_SOLN.value == "no_solution"
    assert CutStatus.NO_EFFECT.value == "no_effect"
    assert CutStatus.UNKNOWN.value == "unknown"

    # Test string representation
    assert str(CutStatus.SUCCESS) == "CutStatus.SUCCESS"


def test_options():
    """Test Options class."""
    # Default constructor
    opts1 = Options()
    assert opts1.max_iters == 2000
    assert opts1.tolerance == 1e-20

    # Custom constructor
    opts2 = Options(max_iters=100, tolerance=1e-10)
    assert opts2.max_iters == 100
    assert opts2.tolerance == 1e-10

    # Default method
    opts3 = Options.default()
    assert opts3.max_iters == 2000
    assert opts3.tolerance == 1e-20


def test_cutting_plane_feas_success():
    """Test successful feasibility cutting plane."""
    from ellalgo import Ell

    # Start outside unit ball
    xc = np.array([2.0, 2.0])
    ell = Ell.new_with_scalar(4.0, xc)

    oracle = SimpleFeasOracle(target_norm=1.0)
    options = Options(max_iters=100, tolerance=1e-6)

    status, solution, iterations = cutting_plane_feas(oracle, ell, options)

    assert status == CutStatus.SUCCESS
    assert np.linalg.norm(solution) <= 1.0 + 1e-3
    assert iterations < options.max_iters
    assert oracle.call_count > 0


def test_cutting_plane_feas_max_iters():
    """Test feasibility cutting plane hitting max iterations."""
    from ellalgo import Ell

    # Use very small max iterations
    xc = np.array([2.0, 2.0])
    ell = Ell.new_with_scalar(4.0, xc)

    oracle = SimpleFeasOracle(target_norm=0.1)  # Very small target
    options = Options(max_iters=5, tolerance=1e-6)

    status, solution, iterations = cutting_plane_feas(oracle, ell, options)

    assert status == CutStatus.UNKNOWN
    assert iterations == options.max_iters


def test_cutting_plane_optim():
    """Test optimization cutting plane."""
    from ellalgo import Ell

    # Start away from origin
    xc = np.array([2.0, 2.0])
    ell = Ell.new_with_scalar(4.0, xc)

    oracle = SimpleOptimOracle()
    options = Options(max_iters=100, tolerance=1e-6)

    status, solution, objective, iterations = cutting_plane_optim(oracle, ell, options)

    assert status == CutStatus.SUCCESS
    assert np.sum(solution**2) < 1e-3  # Near origin
    assert objective < 1e-3
    assert iterations < options.max_iters
    assert oracle.call_count > 0


def test_cutting_plane_general():
    """Test general cutting plane function."""
    from ellalgo import Ell

    # Test with feasibility oracle
    xc = np.array([2.0, 2.0])
    ell = Ell.new_with_scalar(4.0, xc)

    oracle = SimpleFeasOracle(target_norm=1.0)
    options = Options(max_iters=100, tolerance=1e-6)

    status, solution, objective, iterations = cutting_plane(oracle, ell, options)

    assert status == CutStatus.SUCCESS
    assert np.linalg.norm(solution) <= 1.0 + 1e-3
    assert objective is None  # No objective for feasibility
    assert iterations < options.max_iters

    # Test with optimization oracle
    ell2 = Ell.new_with_scalar(4.0, xc)
    oracle2 = SimpleOptimOracle()

    status2, solution2, objective2, iterations2 = cutting_plane(oracle2, ell2, options)

    assert status2 == CutStatus.SUCCESS
    assert objective2 is not None
    assert objective2 < 1e-3


def test_binary_search():
    """Test binary search algorithm."""
    oracle = SimpleBSOracle(threshold=0.75)
    options = Options(max_iters=50, tolerance=1e-8)

    status, value, iterations = binary_search(oracle, 0.0, 1.0, options)

    assert status == CutStatus.SUCCESS
    assert abs(value - 0.75) < 1e-6
    assert iterations < options.max_iters
    assert oracle.call_count > 0


def test_binary_search_max_iters():
    """Test binary search hitting max iterations."""
    oracle = SimpleBSOracle(threshold=0.123456789)
    options = Options(max_iters=10, tolerance=1e-12)  # High precision, few iterations

    status, value, iterations = binary_search(oracle, 0.0, 1.0, options)

    assert status == CutStatus.UNKNOWN
    assert iterations == options.max_iters


def test_binary_search_convergence():
    """Test binary search convergence."""
    oracle = SimpleBSOracle(threshold=0.5)
    options = Options(max_iters=100, tolerance=1e-3)

    status, value, iterations = binary_search(oracle, 0.0, 1.0, options)

    assert status == CutStatus.SUCCESS
    assert abs(value - 0.5) < 1e-2  # Within tolerance
    assert iterations > 0


if __name__ == "__main__":
    pytest.main([__file__])
