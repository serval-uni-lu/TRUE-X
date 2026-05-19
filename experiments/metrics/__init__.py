"""
metrics: A comprehensive library for evaluating Explainable AI methods.

Basic usage:
    >>> from metrics import list_metrics, get_metric
    >>>
    >>> # List available metrics
    >>> print(list_metrics())
    ['cf_constraint_violations_tabular', 'cf_delta_entropy_tabular', ...]
    >>>
    >>> # Get and use a metric
    >>> metric = get_metric("cf_delta_sparsity_tabular")
    >>> result = metric.evaluate(model=model, x=x, explanation=x_cf)

For more control, use the router:
    >>> from metrics.core.router import MetricRouter
    >>> router = MetricRouter()
    >>> results = router.evaluate(metrics=["metric1", "metric2"], ...)
"""

__version__ = "0.1.0"
__author__ = "Kaouther Benguessoum"
__email__ = "kaouther.benguessoum@uni.lu"

# =============================================================================
# Public API
# =============================================================================

# Import registry to trigger metric registration
from metrics import registry as _registry  # noqa: F401
from metrics.core.base_metric import (
    BaseCounterfactualMetric,
    BaseFeatureAttributionMetric,
    BaseMetric,
    MetricMetadata,
)
from metrics.core.enums import DataType, ExplanationType, TaskType
from metrics.core.metric_config import MetricConfig, ParamSpec
from metrics.core.metric_registry import (
    get_metric_cls,
    get_metric_metadata,
    get_metrics_by_category,
    has_metric,
    list_categories,
    list_metrics,
    list_tags,
)
from metrics.core.router import MetricRouter, get_router

# =============================================================================
# Convenience Functions
# =============================================================================


def get_metric(metric_id: str, **kwargs):
    """
    Get a metric instance by its ID.

    Args:
        metric_id: The metric identifier (e.g., "cf_delta_sparsity_tabular")
        **kwargs: Arguments passed to the metric constructor

    Returns:
        Instantiated metric object

    Example:
        >>> metric = get_metric("cf_delta_sparsity_tabular", normalise=True)
        >>> result = metric.evaluate(model=model, x=x, explanation=x_cf)
    """
    cls = get_metric_cls(metric_id)
    return cls(**kwargs)


def get_metadata(metric_id: str) -> MetricMetadata:
    """
    Get metadata for a metric without instantiating it.

    Args:
        metric_id: The metric identifier

    Returns:
        MetricMetadata object with metric information

    Example:
        >>> meta = get_metadata("cf_delta_sparsity_tabular")
        >>> print(meta.category)  # "Complexity"
        >>> print(meta.tags)      # frozenset({'local'})
    """
    return get_metric_metadata(metric_id)


def evaluate(
    metric_id: str,
    *,
    model,
    x,
    explanation,
    **kwargs,
):
    """
    Evaluate a metric directly without creating a metric object.

    Args:
        metric_id: The metric identifier
        model: Trained model with .predict() method
        x: Original inputs
        explanation: Explanations (counterfactuals or attributions)
        **kwargs: Additional arguments for the metric

    Returns:
        Metric scores as numpy array

    Example:
        >>> scores = evaluate(
        ...     "cf_delta_sparsity_tabular",
        ...     model=model,
        ...     x=x_original,
        ...     explanation=x_cf,
        ... )
    """
    metric = get_metric(metric_id)
    return metric.evaluate(model=model, x=x, explanation=explanation, **kwargs)


# =============================================================================
# Public API Exports
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Enums
    "DataType",
    "ExplanationType",
    "TaskType",
    # Base classes
    "BaseMetric",
    "BaseCounterfactualMetric",
    "BaseFeatureAttributionMetric",
    "MetricMetadata",
    # Config
    "MetricConfig",
    "ParamSpec",
    # Registry functions
    "list_metrics",
    "list_categories",
    "list_tags",
    "get_metrics_by_category",
    "get_metric_cls",
    "get_metric_metadata",
    "has_metric",
    # Convenience functions
    "get_metric",
    "get_metadata",
    "evaluate",
    # Router
    "MetricRouter",
    "get_router",
]
