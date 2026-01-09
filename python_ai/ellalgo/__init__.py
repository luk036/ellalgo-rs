"""
Ellipsoid Method in Python

This is a Python implementation of the Ellipsoid Method (L. G. Khachiyan, 1979)
for linear programming and convex optimization.

Modules:
    ell: Ellipsoid search space implementation
    ell_calc: Ellipsoid calculation utilities
    cutting_plane: Core cutting plane algorithms
    oracles: Various oracle implementations for specific problems
"""

from .cutting_plane import CutStatus, Options, cutting_plane
from .ell import Ell
from .ell_calc import EllCalc, EllCalcCore

__version__ = "0.1.0"
__all__ = [
    "CutStatus",
    "Options",
    "cutting_plane",
    "Ell",
    "EllCalc",
    "EllCalcCore",
]
