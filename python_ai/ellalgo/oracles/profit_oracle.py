"""
Profit maximization oracle.

This oracle handles profit maximization problems with Cobb-Douglas
production functions and linear constraints.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ..cutting_plane import OracleOptim


class ProfitOracle(OracleOptim):
    """
    Oracle for profit maximization with Cobb-Douglas production.

    Maximizes: p(A x₁^α x₂^β) - v₁x₁ - v₂x₂
    Subject to: x₁ ≤ k

    where:
        p: market price per unit
        A: scale of production
        α, β: output elasticities
        v: input prices
        k: capacity constraint
    """

    def __init__(
        self,
        price: float,
        scale: float,
        elasticities: NDArray[np.float64],
        input_prices: NDArray[np.float64],
        capacity: float,
    ) -> None:
        """
        Initialize profit oracle.

        Args:
            price: Market price per unit (p)
            scale: Scale of production (A)
            elasticities: Output elasticities [α, β]
            input_prices: Input prices [v₁, v₂]
            capacity: Capacity constraint (k)
        """
        if len(elasticities) != 2:
            raise ValueError("elasticities must have length 2")
        if len(input_prices) != 2:
            raise ValueError("input_prices must have length 2")

        self.price = price
        self.scale = scale
        self.elasticities = elasticities.copy()
        self.input_prices = input_prices.copy()
        self.capacity = capacity

        # Precompute logarithms
        self.log_price_scale = np.log(price * scale)
        self.log_capacity = np.log(capacity)

    def _compute_profit(self, x: NDArray[np.float64]) -> float:
        """Compute profit at point x."""
        if x[0] <= 0 or x[1] <= 0:
            return -np.inf

        # Cobb-Douglas production: A x₁^α x₂^β
        production = (
            self.scale * (x[0] ** self.elasticities[0]) * (x[1] ** self.elasticities[1])
        )
        revenue = self.price * production
        cost = self.input_prices[0] * x[0] + self.input_prices[1] * x[1]

        return revenue - cost

    def _compute_gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute gradient of negative profit (for minimization)."""
        if x[0] <= 0 or x[1] <= 0:
            return np.array([0.0, 0.0])

        # Gradient of production function
        prod = (
            self.scale * (x[0] ** self.elasticities[0]) * (x[1] ** self.elasticities[1])
        )

        grad_prod = np.array(
            [
                self.elasticities[0] * prod / x[0],
                self.elasticities[1] * prod / x[1],
            ]
        )

        # Gradient of profit = price * grad_prod - input_prices
        grad_profit = self.price * grad_prod - self.input_prices

        # Negative gradient for minimization
        return -grad_profit

    def assess_optim(
        self,
        xc: NDArray[np.float64],
        gamma: float,
    ) -> Tuple[Tuple[NDArray[np.float64], float], bool]:
        """
        Assess optimization at point xc.

        Args:
            xc: Current point [x₁, x₂]
            gamma: Current best objective value (negative profit)

        Returns:
            ((gradient, beta), is_optimal)
        """
        # Check capacity constraint
        if xc[0] > self.capacity:
            # Violates x₁ ≤ k constraint
            grad = np.array([1.0, 0.0])  # Gradient of x₁
            beta = self.capacity - xc[0]  # x₁ - k ≤ 0 → β = k - x₁
            return (grad, beta), False

        # Compute current profit (negative for minimization)
        current_profit = self._compute_profit(xc)
        current_objective = -current_profit  # Negative profit for minimization

        if current_objective < gamma:
            # Found better solution
            gamma = current_objective

        # Check optimality (simplified condition)
        grad = self._compute_gradient(xc)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < 1e-6:
            # Near stationary point
            return (grad, 0.0), True

        # Return cut for improvement
        beta = -0.1  # Small negative value for bias cut
        return (grad, beta), False

    def __repr__(self) -> str:
        return (
            f"ProfitOracle(price={self.price}, scale={self.scale:.2f}, "
            f"α={self.elasticities[0]:.2f}, β={self.elasticities[1]:.2f})"
        )
