"""
metrics/utils/typing.py

Type definitions and type aliases for the XAI evaluation library.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# =============================================================================
# Array Types
# =============================================================================

# Generic array-like input
ArrayLike = np.ndarray | list | tuple


# =============================================================================
# Feasibility Specification Types
# =============================================================================

# A feasible spec for a single feature can be:
# - A tuple (min, max) for continuous features with bounds
# - A sequence of allowed values for categorical features
FeasibleSpec = tuple[float, float] | Sequence[Any]

# Private alias (same as FeasibleSpec, for internal use)
# Some modules use _FeasibleSpec, others use FeasibleSpec
_FeasibleSpec = FeasibleSpec
