"""
metrics/metrics/feature_attribution/timeseries_multivariate/__init__.py

Feature attribution metrics for multivariate time-series data.
Input convention: (B, T, F) — batch × timesteps × features.
"""

from .complexity import (
    FAMVComplexityEntropyChannel,
    FAMVComplexityEntropyElement,
    FAMVSparsenessChannel,
    FAMVSparsenessElement,
)
from .faithfulness import (
    FAMVFaithfulnessCorrelation,
    FAMVPixelFlipping,
)
from .robustness import (
    FAMVAvgSensitivity,
    FAMVContinuity,
)

__all__ = [
    # Complexity
    "FAMVSparsenessElement",
    "FAMVSparsenessChannel",
    "FAMVComplexityEntropyElement",
    "FAMVComplexityEntropyChannel",
    # Faithfulness
    "FAMVFaithfulnessCorrelation",
    "FAMVPixelFlipping",
    # Robustness
    "FAMVAvgSensitivity",
    "FAMVContinuity",
]
