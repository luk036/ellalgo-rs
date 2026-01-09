"""
Core cutting plane algorithms for the Ellipsoid Method.

This module implements the main cutting plane algorithm and related traits.
"""

from enum import Enum
from typing import Generic, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")
ArrayType = NDArray[np.float64]


class CutStatus(Enum):
    """Status of a cut operation."""

    SUCCESS = "success"
    NO_SOLN = "no_solution"
    NO_EFFECT = "no_effect"
    UNKNOWN = "unknown"


class Options:
    """Options for cutting plane algorithm."""

    def __init__(self, max_iters: int = 2000, tolerance: float = 1e-20) -> None:
        """
        Initialize options.

        Args:
            max_iters: Maximum number of iterations
            tolerance: Error tolerance
        """
        self.max_iters = max_iters
        self.tolerance = tolerance

    @classmethod
    def default(cls) -> "Options":
        """Create default options."""
        return cls(2000, 1e-20)


@runtime_checkable
class UpdateByCutChoice(Protocol, Generic[T]):
    """Protocol for types that can update cut choices."""

    def update_bias_cut_by(self, space: T, grad: ArrayType) -> CutStatus:
        """Update using bias cut strategy."""
        ...

    def update_central_cut_by(self, space: T, grad: ArrayType) -> CutStatus:
        """Update using central cut strategy."""
        ...

    def update_q_by(self, space: T, grad: ArrayType) -> CutStatus:
        """Update using Q cut strategy."""
        ...


@runtime_checkable
class OracleFeas(Protocol):
    """Oracle for feasibility problems."""

    def assess_feas(
        self, xc: ArrayType
    ) -> Optional[Tuple[ArrayType, Union[float, Tuple[float, Optional[float]]]]]:
        """
        Assess feasibility at point xc.

        Args:
            xc: Current point

        Returns:
            None if feasible, otherwise (gradient, beta) cut information
        """
        ...


@runtime_checkable
class OracleFeas2(OracleFeas, Protocol):
    """Oracle for feasibility problems with update capability."""

    def update(self, gamma: float) -> None:
        """
        Update oracle parameters.

        Args:
            gamma: Update parameter
        """
        ...


@runtime_checkable
class OracleOptim(Protocol):
    """Oracle for optimization problems."""

    def assess_optim(
        self,
        xc: ArrayType,
        gamma: float,
    ) -> Tuple[Tuple[ArrayType, Union[float, Tuple[float, Optional[float]]]], bool]:
        """
        Assess optimization at point xc.

        Args:
            xc: Current point
            gamma: Current objective value

        Returns:
            ((gradient, beta), is_optimal)
        """
        ...


@runtime_checkable
class OracleOptimQ(Protocol):
    """Oracle for quantized optimization problems."""

    def assess_optim_q(
        self,
        xc: ArrayType,
        gamma: float,
        retry: bool,
    ) -> Tuple[
        Tuple[ArrayType, Union[float, Tuple[float, Optional[float]]]],
        bool,
        ArrayType,
        bool,
    ]:
        """
        Assess quantized optimization at point xc.

        Args:
            xc: Current point
            gamma: Current objective value
            retry: Whether this is a retry

        Returns:
            ((gradient, beta), is_optimal, new_xc, should_retry)
        """
        ...


@runtime_checkable
class OracleBS(Protocol):
    """Oracle for binary search."""

    def assess_bs(self, gamma: float) -> bool:
        """
        Assess binary search condition.

        Args:
            gamma: Current value

        Returns:
            True if condition satisfied
        """
        ...


@runtime_checkable
class SearchSpace(Protocol):
    """Protocol for search spaces."""

    def xc(self) -> ArrayType:
        """Get current center point."""
        ...

    def tsq(self) -> float:
        """Get squared Mahalanobis distance threshold."""
        ...

    def update_bias_cut(
        self,
        cut: Tuple[ArrayType, Union[float, Tuple[float, Optional[float]]]],
    ) -> CutStatus:
        """
        Update using bias cut.

        Args:
            cut: (gradient, beta) tuple

        Returns:
            Cut status
        """
        ...

    def update_central_cut(
        self,
        cut: Tuple[ArrayType, Union[float, Tuple[float, Optional[float]]]],
    ) -> CutStatus:
        """
        Update using central cut.

        Args:
            cut: (gradient, beta) tuple

        Returns:
            Cut status
        """
        ...

    def set_xc(self, x: ArrayType) -> None:
        """
        Set center point.

        Args:
            x: New center point
        """
        ...


@runtime_checkable
class SearchSpaceQ(Protocol):
    """Protocol for search spaces with Q updates."""

    def xc(self) -> ArrayType:
        """Get current center point."""
        ...

    def tsq(self) -> float:
        """Get squared Mahalanobis distance threshold."""
        ...

    def update_q(
        self,
        cut: Tuple[ArrayType, Union[float, Tuple[float, Optional[float]]]],
    ) -> CutStatus:
        """
        Update using Q cut.

        Args:
            cut: (gradient, beta) tuple

        Returns:
            Cut status
        """
        ...


def cutting_plane_feas(
    oracle: OracleFeas,
    space: SearchSpace,
    options: Optional[Options] = None,
) -> Tuple[CutStatus, ArrayType, int]:
    """
    Cutting plane algorithm for feasibility problems.

    Args:
        oracle: Feasibility oracle
        space: Search space
        options: Algorithm options

    Returns:
        (status, solution, iterations)
    """
    if options is None:
        options = Options.default()

    for i in range(options.max_iters):
        xc = space.xc()
        cut = oracle.assess_feas(xc)

        if cut is None:
            # Feasible solution found
            return CutStatus.SUCCESS, xc, i

        # Apply cut
        status = space.update_bias_cut(cut)
        if status != CutStatus.SUCCESS:
            return status, xc, i

        # Check convergence
        if space.tsq() < options.tolerance:
            return CutStatus.SUCCESS, space.xc(), i

    return CutStatus.UNKNOWN, space.xc(), options.max_iters


def cutting_plane_optim(
    oracle: OracleOptim,
    space: SearchSpace,
    options: Optional[Options] = None,
) -> Tuple[CutStatus, ArrayType, float, int]:
    """
    Cutting plane algorithm for optimization problems.

    Args:
        oracle: Optimization oracle
        space: Search space
        options: Algorithm options

    Returns:
        (status, solution, objective, iterations)
    """
    if options is None:
        options = Options.default()

    gamma = float("inf")

    for i in range(options.max_iters):
        xc = space.xc()
        (grad, beta), is_optimal = oracle.assess_optim(xc, gamma)

        if is_optimal:
            return CutStatus.SUCCESS, xc, gamma, i

        # Apply cut
        cut = (grad, beta)
        status = space.update_bias_cut(cut)
        if status != CutStatus.SUCCESS:
            return status, xc, gamma, i

        # Check convergence
        if space.tsq() < options.tolerance:
            return CutStatus.SUCCESS, space.xc(), gamma, i

    return CutStatus.UNKNOWN, space.xc(), gamma, options.max_iters


def cutting_plane(
    oracle: Union[OracleFeas, OracleOptim],
    space: SearchSpace,
    options: Optional[Options] = None,
) -> Tuple[CutStatus, ArrayType, Optional[float], int]:
    """
    General cutting plane algorithm.

    Args:
        oracle: Feasibility or optimization oracle
        space: Search space
        options: Algorithm options

    Returns:
        (status, solution, objective (if optimization), iterations)
    """
    if isinstance(oracle, OracleFeas):
        status, solution, iterations = cutting_plane_feas(oracle, space, options)
        return status, solution, None, iterations
    else:
        status, solution, objective, iterations = cutting_plane_optim(
            oracle, space, options
        )
        return status, solution, objective, iterations


def binary_search(
    oracle: OracleBS,
    lower: float,
    upper: float,
    options: Optional[Options] = None,
) -> Tuple[CutStatus, float, int]:
    """
    Binary search using oracle.

    Args:
        oracle: Binary search oracle
        lower: Lower bound
        upper: Upper bound
        options: Algorithm options

    Returns:
        (status, value, iterations)
    """
    if options is None:
        options = Options.default()

    for i in range(options.max_iters):
        if upper - lower < options.tolerance:
            return CutStatus.SUCCESS, (lower + upper) / 2.0, i

        mid = (lower + upper) / 2.0
        if oracle.assess_bs(mid):
            upper = mid
        else:
            lower = mid

    return CutStatus.UNKNOWN, (lower + upper) / 2.0, options.max_iters
