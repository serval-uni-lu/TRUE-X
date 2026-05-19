"""
metrics/core/__init__.py

Core components for the XAI evaluation library.
"""

from .base_metric import (
    BaseCounterfactualMetric,
    BaseFeatureAttributionMetric,
    BaseMetric,
    MetricMetadata,
)
from .enums import DataType, ExplanationType, TaskType
from .metric_config import MetricConfig, ParamSpec
from .metric_registry import (
    METRIC_REGISTRY,
    get_metric_cls,
    get_metric_metadata,
    get_metrics_by_category,
    has_metric,
    list_categories,
    list_metrics,
    list_tags,
    register_metric,
)
from .router import MetricRouter, get_router

__all__ = [
    # Enums
    "ExplanationType",
    "DataType",
    "TaskType",
    # Config
    "MetricConfig",
    "ParamSpec",
    # Base classes
    "BaseMetric",
    "BaseCounterfactualMetric",
    "BaseFeatureAttributionMetric",
    "MetricMetadata",
    # Registry
    "register_metric",
    "get_metric_cls",
    "get_metric_metadata",
    "list_metrics",
    "list_categories",
    "list_tags",
    "get_metrics_by_category",
    "has_metric",
    "METRIC_REGISTRY",
    # Router
    "MetricRouter",
    "get_router",
]
