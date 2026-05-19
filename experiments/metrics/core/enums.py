"""
metrics/core/enums.py

Core enumerations for the XAI evaluation library.
"""

from enum import Enum, auto


class ExplanationType(Enum):
    """Type of explanation method being evaluated."""

    COUNTERFACTUAL = auto()
    FEATURE_ATTRIBUTION = auto()


class DataType(Enum):
    """Type of input data."""

    TABULAR = auto()
    TIMESERIES_UNIVARIATE = auto()
    TIMESERIES_MULTIVARIATE = auto()
    IMAGE = auto()
    TEXT = auto()


class TaskType(Enum):
    """Type of ML task."""

    CLASSIFICATION = auto()
    REGRESSION = auto()
