"""
Tests for oracle implementations.
"""

import numpy as np
import pytest

from ellalgo.oracles import LmiOracle, ProfitOracle


def test_lmi_oracle_initialization():
    """Test LMI oracle initialization."""
    # Valid initialization
    F0 = np.eye(3)
    F1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    F2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    oracle = LmiOracle(F0, [F1, F2])

    assert oracle.F0.shape == (3, 3)
    assert len(oracle.F_matrices) == 2
    assert np.allclose(oracle.F0, np.eye(3))

    # Invalid: non-square F0
    with pytest.raises(ValueError):
        LmiOracle(np.ones((3, 2)), [F1])

    # Invalid: wrong size F matrix
    with pytest.raises(ValueError):
        LmiOracle(F0, [np.eye(2)])


def test_lmi_oracle_assess_feas_feasible():
    """Test LMI oracle with feasible point."""
    # Simple LMI: I + x₁·I ≽ 0  (equivalent to 1 + x₁ ≥ 0)
    F0 = np.eye(2)
    F1 = np.eye(2)

    oracle = LmiOracle(F0, [F1])

    # Feasible point: x₁ = 0.5
    xc = np.array([0.5])
    result = oracle.assess_feas(xc)

    assert result is None  # Feasible

    # Another feasible point: x₁ = -0.5
    xc2 = np.array([-0.5])
    result2 = oracle.assess_feas(xc2)

    assert result2 is None  # Still feasible (1 - 0.5 = 0.5 > 0)


def test_lmi_oracle_assess_feas_infeasible():
    """Test LMI oracle with infeasible point."""
    # LMI: I + x₁·I ≽ 0
    F0 = np.eye(2)
    F1 = np.eye(2)

    oracle = LmiOracle(F0, [F1])

    # Infeasible point: x₁ = -2.0
    xc = np.array([-2.0])
    result = oracle.assess_feas(xc)

    assert result is not None
    grad, beta = result

    assert len(grad) == 1
    assert beta < 0.0  # Negative eigenvalue
    assert grad[0] > 0.0  # Should be positive


def test_lmi_oracle_multiple_variables():
    """Test LMI oracle with multiple variables."""
    # LMI: I + x₁·A + x₂·B ≽ 0
    F0 = np.eye(2)
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    B = np.array([[0.0, 1.0], [1.0, 0.0]])

    oracle = LmiOracle(F0, [A, B])

    # Test at origin (feasible)
    xc = np.array([0.0, 0.0])
    result = oracle.assess_feas(xc)

    assert result is None  # I ≽ 0 is feasible

    # Test infeasible point
    xc2 = np.array([-2.0, 0.0])
    result2 = oracle.assess_feas(xc2)

    assert result2 is not None
    grad2, beta2 = result2

    assert len(grad2) == 2
    assert beta2 < 0.0


def test_profit_oracle_initialization():
    """Test profit oracle initialization."""
    # Valid initialization
    elasticities = np.array([0.3, 0.7])
    input_prices = np.array([1.0, 2.0])

    oracle = ProfitOracle(
        price=10.0,
        scale=1.0,
        elasticities=elasticities,
        input_prices=input_prices,
        capacity=5.0,
    )

    assert oracle.price == 10.0
    assert oracle.scale == 1.0
    assert np.allclose(oracle.elasticities, elasticities)
    assert np.allclose(oracle.input_prices, input_prices)
    assert oracle.capacity == 5.0

    # Invalid: wrong elasticities length
    with pytest.raises(ValueError):
        ProfitOracle(10.0, 1.0, np.array([0.3]), input_prices, 5.0)

    # Invalid: wrong input_prices length
    with pytest.raises(ValueError):
        ProfitOracle(10.0, 1.0, elasticities, np.array([1.0]), 5.0)


def test_profit_oracle_compute_profit():
    """Test profit computation."""
    oracle = ProfitOracle(
        price=10.0,
        scale=1.0,
        elasticities=np.array([0.5, 0.5]),
        input_prices=np.array([1.0, 2.0]),
        capacity=10.0,
    )

    # Test at valid point
    x = np.array([4.0, 9.0])
    profit = oracle._compute_profit(x)

    # Manual calculation:
    # production = 1.0 * sqrt(4*9) = 1.0 * 6 = 6
    # revenue = 10 * 6 = 60
    # cost = 1*4 + 2*9 = 4 + 18 = 22
    # profit = 60 - 22 = 38
    expected_profit = 38.0
    assert abs(profit - expected_profit) < 1e-10

    # Test at invalid point (zero or negative inputs)
    x_invalid = np.array([0.0, 1.0])
    profit_invalid = oracle._compute_profit(x_invalid)
    assert profit_invalid == -np.inf


def test_profit_oracle_capacity_constraint():
    """Test profit oracle capacity constraint."""
    oracle = ProfitOracle(
        price=10.0,
        scale=1.0,
        elasticities=np.array([0.3, 0.7]),
        input_prices=np.array([1.0, 2.0]),
        capacity=5.0,
    )

    # Point violating capacity constraint
    xc = np.array([6.0, 1.0])  # x₁ = 6 > capacity = 5
    gamma = 100.0

    (grad, beta), is_optimal = oracle.assess_optim(xc, gamma)

    assert not is_optimal
    assert np.allclose(grad, [1.0, 0.0])  # Gradient of x₁
    assert abs(beta - (-1.0)) < 1e-10  # β = capacity - x₁ = 5 - 6 = -1

    # Point satisfying capacity constraint
    xc2 = np.array([3.0, 1.0])
    (grad2, beta2), is_optimal2 = oracle.assess_optim(xc2, gamma)

    # Should not be optimal yet
    assert not is_optimal2
    assert len(grad2) == 2


def test_profit_oracle_gradient():
    """Test profit oracle gradient computation."""
    oracle = ProfitOracle(
        price=10.0,
        scale=1.0,
        elasticities=np.array([0.4, 0.6]),
        input_prices=np.array([2.0, 3.0]),
        capacity=10.0,
    )

    x = np.array([2.0, 3.0])
    grad = oracle._compute_gradient(x)

    # Check gradient shape and properties
    assert len(grad) == 2
    assert not np.allclose(grad, 0.0)  # Should be non-zero

    # Numerical gradient check
    eps = 1e-6
    for i in range(2):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps

        f_plus = -oracle._compute_profit(x_plus)  # Negative profit
        f_minus = -oracle._compute_profit(x_minus)

        num_grad = (f_plus - f_minus) / (2 * eps)
        assert abs(grad[i] - num_grad) < 1e-4


def test_profit_oracle_repr():
    """Test profit oracle string representation."""
    oracle = ProfitOracle(
        price=10.0,
        scale=1.5,
        elasticities=np.array([0.3, 0.7]),
        input_prices=np.array([1.0, 2.0]),
        capacity=5.0,
    )

    repr_str = repr(oracle)
    assert "ProfitOracle" in repr_str
    assert "price=10.0" in repr_str
    assert "scale=1.5" in repr_str
    assert "α=0.30" in repr_str or "α=0.3" in repr_str
    assert "β=0.70" in repr_str or "β=0.7" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])
