"""
metrics/core/metric_registry.py

Central registry for all metrics.

Metrics register themselves using the @register_metric decorator.
The registry enables discovery and filtering by metadata WITHOUT instantiation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .enums import DataType, ExplanationType, TaskType

if TYPE_CHECKING:
    from .base_metric import BaseMetric, MetricMetadata


METRIC_REGISTRY: dict[str, type[BaseMetric]] = {}


def register_metric(metric_id: str):
    """
    Decorator to register a metric class under a stable metric_id.

    Usage:
    ```python
    @register_metric("cf_validity_margin_tabular")
    class CFValidityMarginTabular(BaseMetric):
        METADATA = MetricMetadata(
            metric_id="cf_validity_margin_tabular",  # Must match!
            ...
        )
    ```

    Enforces:
    - metric_id uniqueness (no duplicates)
    - METADATA class attribute exists and is valid
    - METADATA.metric_id matches the decorator argument
    """

    def decorator(cls: type[BaseMetric]) -> type[BaseMetric]:
        meta = cls.metadata()

        if meta.metric_id != metric_id:
            raise ValueError(
                f"{cls.__name__}: METADATA.metric_id='{meta.metric_id}' "
                f"must match decorator id '{metric_id}'."
            )

        if metric_id in METRIC_REGISTRY:
            existing = METRIC_REGISTRY[metric_id]
            raise ValueError(
                f"Metric '{metric_id}' is already registered by {existing.__name__}. "
                f"Cannot register {cls.__name__}."
            )

        METRIC_REGISTRY[metric_id] = cls
        return cls

    return decorator


def unregister_metric(metric_id: str) -> bool:
    """
    Remove a metric from the registry.

    Useful for testing. Returns True if removed, False if not found.
    """
    if metric_id in METRIC_REGISTRY:
        del METRIC_REGISTRY[metric_id]
        return True
    return False


def clear_registry() -> None:
    """Clear all registered metrics. Use only in tests."""
    METRIC_REGISTRY.clear()


def get_metric_cls(metric_id: str) -> type[BaseMetric]:
    """
    Get metric class by its registered ID.

    Args:
        metric_id: The stable metric identifier

    Returns:
        The metric class (not instantiated)

    Raises:
        KeyError: If metric not found
    """
    if metric_id not in METRIC_REGISTRY:
        available = ", ".join(sorted(METRIC_REGISTRY.keys())[:10])
        raise KeyError(
            f"Metric '{metric_id}' not found in registry. "
            f"Available: {available}{'...' if len(METRIC_REGISTRY) > 10 else ''}"
        )
    return METRIC_REGISTRY[metric_id]


def get_metric_metadata(metric_id: str) -> MetricMetadata:
    """
    Get metadata for a metric WITHOUT instantiating it.

    Args:
        metric_id: The stable metric identifier

    Returns:
        MetricMetadata for the metric
    """
    cls = get_metric_cls(metric_id)
    return cls.metadata()


def has_metric(metric_id: str) -> bool:
    """Check if a metric is registered."""
    return metric_id in METRIC_REGISTRY


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
    List registered metric IDs, optionally filtered by metadata.

    All filters are AND-ed together. Pass None to skip a filter.

    Args:
        explanation_type: Filter by explanation type (COUNTERFACTUAL, FEATURE_ATTRIBUTION)
        data_type: Filter by supported data type (TABULAR, TIMESERIES_*, etc.)
        task_type: Filter by supported task type (CLASSIFICATION, REGRESSION)
        category: Filter by category string (e.g., "Robustness", "Faithfulness")
        tags: Filter by tags - metric must have ALL of these tags
        tags_any: Filter by tags - metric must have AT LEAST ONE of these tags

    Returns:
        Sorted list of metric IDs matching all filters
    """
    results: list[str] = []
    tags_set = set(tags) if tags else None
    tags_any_set = set(tags_any) if tags_any else None

    for metric_id, cls in METRIC_REGISTRY.items():
        meta = cls.metadata()

        if explanation_type is not None and meta.explanation_type != explanation_type:
            continue
        if data_type is not None and data_type not in meta.supported_data_types:
            continue
        if task_type is not None and task_type not in meta.supported_task_types:
            continue
        if category is not None and meta.category != category:
            continue
        if tags_set is not None and not tags_set.issubset(meta.tags):
            continue
        if tags_any_set is not None and not tags_any_set.intersection(meta.tags):
            continue

        results.append(metric_id)

    return sorted(results)


def list_categories() -> list[str]:
    """List all unique categories from registered metrics."""
    categories = set()
    for cls in METRIC_REGISTRY.values():
        categories.add(cls.metadata().category)
    return sorted(categories)


def list_tags() -> list[str]:
    """List all unique tags from registered metrics."""
    tags: set[str] = set()
    for cls in METRIC_REGISTRY.values():
        tags.update(cls.metadata().tags)
    return sorted(tags)


def get_metrics_by_category() -> dict[str, list[str]]:
    """
    Group metric IDs by category.

    Returns:
        Dict mapping category name -> list of metric IDs
    """
    by_category: dict[str, list[str]] = {}
    for metric_id, cls in METRIC_REGISTRY.items():
        category = cls.metadata().category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(metric_id)

    for cat in by_category:
        by_category[cat].sort()

    return by_category
