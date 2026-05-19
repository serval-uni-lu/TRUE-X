"""
metrics/registry.py

Public API entry point for metric registration and lookup.

Architecture note:
  - core/metric_registry.py — storage engine: the METRIC_REGISTRY dict,
    @register_metric decorator, and low-level query functions.
  - registry_loaders/* — side-effect import modules that trigger registration.
  - registry.py (this file) — public API facade on top of the registry engine.

This module:
1. Triggers registration of all metrics via aggregate loader import
2. Provides simple top-level API for common operations

Usage:
```python
from metrics.registry import list_metrics, get_metric, evaluate

# List all counterfactual metrics
cf_metrics = list_metrics(explanation_type=ExplanationType.COUNTERFACTUAL)

# Get a metric instance
metric = get_metric("cf_validity_margin_tabular", kind="regression")

# Evaluate multiple metrics at once
results = evaluate(
    metrics=["cf_validity_margin_tabular", "cf_validity_under_noise_tabular"],
    model=model,
    x=X_original,
    explanation=X_cf,
    target_range=(0.5, 1.0),
)
```
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import metrics.registry_loaders.all_metrics  # noqa: F401
from metrics.core.base_metric import BaseMetric
from metrics.core.enums import DataType, ExplanationType, TaskType
from metrics.core.metric_registry import (
    get_metric_cls,
    get_metric_metadata,
    get_metrics_by_category,
    has_metric,
    list_categories,
    list_tags,
)
from metrics.core.metric_registry import (
    list_metrics as _list_metrics,
)


def list_metrics(
    *,
    explanation_type: ExplanationType | None = None,
    data_type: DataType | None = None,
    task_type: TaskType | None = None,
    category: str | None = None,
    tags: Iterable[str] | None = None,
    tags_any: Iterable[str] | None = None,
) -> list[str]:
    """
    List all registered metric IDs, optionally filtered.

    Args:
        explanation_type: Filter by COUNTERFACTUAL or FEATURE_ATTRIBUTION
        data_type: Filter by TABULAR, TIMESERIES_*, IMAGE, TEXT
        task_type: Filter by CLASSIFICATION or REGRESSION
        category: Filter by category (e.g., "Robustness", "Faithfulness")
        tags: Filter by tags (metric must have ALL specified tags)
        tags_any: Filter by tags (metric must have AT LEAST ONE of these tags)

    Returns:
        Sorted list of matching metric IDs
    """
    return _list_metrics(
        explanation_type=explanation_type,
        data_type=data_type,
        task_type=task_type,
        category=category,
        tags=tags,
        tags_any=tags_any,
    )


def get_metric(metric_id: str, **params: Any) -> BaseMetric:
    """
    Instantiate a metric by its registered ID.

    Args:
        metric_id: The stable metric identifier (e.g., "cf_validity_margin_tabular")
        **params: Runtime parameters for the metric

    Returns:
        Instantiated metric object

    Example:
        metric = get_metric("cf_validity_margin_tabular", kind="regression")
    """
    cls = get_metric_cls(metric_id)
    return cls(**params)


def get_metadata(metric_id: str):
    """
    Get metadata for a metric without instantiating it.

    Returns MetricMetadata with: metric_id, display_name, category,
    explanation_type, supported_data_types, supported_task_types, tags, param_schema
    """
    return get_metric_metadata(metric_id)


def evaluate(
    *,
    metrics: Sequence[str | BaseMetric],
    model: Any,
    x: Any,
    explanation: Any,
    y_pred: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Evaluate multiple metrics and return results.

    Args:
        metrics: List of metric IDs or instantiated metric objects
        model: The ML model
        x: Original input(s)
        explanation: The explanation to evaluate
        y_pred: Optional pre-computed predictions
        **kwargs: Additional parameters passed to all metrics

    Returns:
        Dict mapping metric_id -> result

    Example:
        results = evaluate(
            metrics=["cf_validity_margin_tabular", "cf_validity_under_noise_tabular"],
            model=model,
            x=X_original,
            explanation=X_cf,
            target_range=(0.5, 1.0),
        )
    """
    results: dict[str, Any] = {}

    for m in metrics:
        metric = get_metric(m) if isinstance(m, str) else m
        key = metric.metric_id
        results[key] = metric.evaluate(
            model=model,
            x=x,
            explanation=explanation,
            y_pred=y_pred,
            **kwargs,
        )

    return results


__all__ = [
    "list_metrics",
    "get_metric",
    "get_metadata",
    "evaluate",
    "list_categories",
    "list_tags",
    "get_metrics_by_category",
    "has_metric",
    "ExplanationType",
    "DataType",
    "TaskType",
]
