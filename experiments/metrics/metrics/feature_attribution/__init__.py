"""
metrics/metrics/feature_attribution/__init__.py

Feature attribution evaluation metrics.
"""

from .timeseries_multivariate.complexity import (
    FAMVComplexityEntropyChannel,
    FAMVComplexityEntropyElement,
    FAMVSparsenessChannel,
    FAMVSparsenessElement,
)
from .timeseries_multivariate.faithfulness import (
    FAMVFaithfulnessCorrelation,
    FAMVPixelFlipping,
)
from .timeseries_multivariate.robustness import (
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
