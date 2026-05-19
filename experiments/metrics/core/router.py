"""
metrics/core/router.py

Central dispatcher for XAI metrics.

Provides a high-level API for:
- Discovery: Find metrics by criteria
- Instantiation: Create configured metric instances
- Execution: Run metrics and collect results
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from metrics.core.base_metric import BaseMetric
from metrics.core.enums import DataType, ExplanationType, TaskType
from metrics.core.metric_registry import (
    get_metric_cls,
    get_metric_metadata,
    get_metrics_by_category,
    list_categories,
    list_tags,
)
from metrics.core.metric_registry import (
    list_metrics as registry_list_metrics,
)


class MetricRouter:
    """
    Central dispatcher for XAI metrics.

    Provides three levels of interaction:
    1. Discovery: Find metrics using class-level metadata (no instantiation)
    2. Instantiation: Create metric instances with runtime config
    3. Execution: Run evaluate() and collect results

    Example usage:
    ```python
    router = MetricRouter()

    # Discovery
    cf_metrics = router.list_metrics(
        explanation_type=ExplanationType.COUNTERFACTUAL,
        data_type=DataType.TABULAR,
    )

    # Instantiation
    metric = router.create_metric("cf_validity_margin_tabular", kind="regression")

    # Execution
    results = router.evaluate(
        metrics=["cf_validity_margin_tabular", "cf_validity_under_noise_tabular"],
        model=model,
        x=X_original,
        explanation=X_cf,
        target_range=(0.5, 1.0),
    )
    ```
    """

    def list_metrics(
        self,
        *,
        explanation_type: ExplanationType | None = None,
        data_type: DataType | None = None,
        task_type: TaskType | None = None,
        category: str | None = None,
        tags: Iterable[str] | None = None,
        tags_any: Iterable[str] | None = None,
    ) -> list[str]:
        """
        List metric IDs matching the given filters.

        All filters are AND-ed. Pass None to skip a filter.
        """
        return registry_list_metrics(
            explanation_type=explanation_type,
            data_type=data_type,
            task_type=task_type,
            category=category,
            tags=tags,
            tags_any=tags_any,
        )

    def list_categories(self) -> list[str]:
        """List all unique metric categories."""
        return list_categories()

    def list_tags(self) -> list[str]:
        """List all unique metric tags."""
        return list_tags()

    def get_metrics_by_category(self) -> dict[str, list[str]]:
        """Group metric IDs by category."""
        return get_metrics_by_category()

    def get_metadata(self, metric_id: str):
        """Get metadata for a metric without instantiating it."""
        return get_metric_metadata(metric_id)

    def supports(
        self,
        metric_id: str,
        *,
        data_type: DataType | None = None,
        task_type: TaskType | None = None,
        explanation_type: ExplanationType | None = None,
    ) -> bool:
        """Check if a metric supports the given constraints."""
        cls = get_metric_cls(metric_id)
        return cls.supports(
            data_type=data_type,
            task_type=task_type,
            explanation_type=explanation_type,
        )

    def create_metric(
        self,
        metric_id: str,
        **params: Any,
    ) -> BaseMetric:
        """
        Instantiate a metric by its registered ID.

        Args:
            metric_id: The stable metric identifier
            **params: Runtime parameters (n_draws, sigma, kind, etc.)

        Returns:
            Instantiated metric object
        """
        cls = get_metric_cls(metric_id)
        return cls(**params)

    def create_metrics(
        self,
        metric_ids: Sequence[str],
        *,
        shared_params: dict[str, Any] | None = None,
        per_metric_params: dict[str, dict[str, Any]] | None = None,
    ) -> list[BaseMetric]:
        """
        Instantiate multiple metrics.

        Args:
            metric_ids: List of metric IDs to instantiate
            shared_params: Parameters applied to all metrics
            per_metric_params: Dict of metric_id -> params for metric-specific config

        Returns:
            List of instantiated metrics (same order as metric_ids)
        """
        shared = shared_params or {}
        per_metric = per_metric_params or {}

        metrics = []
        for mid in metric_ids:
            params = {**shared, **per_metric.get(mid, {})}
            metrics.append(self.create_metric(mid, **params))
        return metrics

    def evaluate(
        self,
        *,
        metrics: Sequence[str | BaseMetric],
        model: Any,
        x: Any,
        explanation: Any,
        y_pred: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run multiple metrics and return results.

        Args:
            metrics: List of metric IDs (str) or instantiated metrics
            model: The ML model
            x: Original input(s)
            explanation: The explanation (counterfactual, attributions, etc.)
            y_pred: Optional pre-computed predictions
            **kwargs: Additional params passed to all metrics' evaluate()

        Returns:
            Dict mapping metric_id -> result
        """
        results: dict[str, Any] = {}

        for m in metrics:
            metric = self.create_metric(m) if isinstance(m, str) else m
            key = metric.metric_id
            results[key] = metric.evaluate(
                model=model,
                x=x,
                explanation=explanation,
                y_pred=y_pred,
                **kwargs,
            )

        return results

    def evaluate_single(
        self,
        metric: str | BaseMetric,
        *,
        model: Any,
        x: Any,
        explanation: Any,
        y_pred: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run a single metric and return its result directly.

        Convenience method when you only need one metric.
        """
        m = self.create_metric(metric) if isinstance(metric, str) else metric
        return m.evaluate(
            model=model,
            x=x,
            explanation=explanation,
            y_pred=y_pred,
            **kwargs,
        )

    def evaluate_category(
        self,
        category: str,
        *,
        model: Any,
        x: Any,
        explanation: Any,
        y_pred: Any | None = None,
        filter_data_type: DataType | None = None,
        filter_task_type: TaskType | None = None,
        metric_params: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run all metrics in a category.

        Args:
            category: Category name (e.g., "Robustness")
            filter_data_type: Only run metrics supporting this data type
            filter_task_type: Only run metrics supporting this task type
            metric_params: Per-metric parameters
            **kwargs: Shared params for all metrics

        Returns:
            Dict mapping metric_id -> result
        """
        metric_ids = self.list_metrics(
            category=category,
            data_type=filter_data_type,
            task_type=filter_task_type,
        )

        params = metric_params or {}
        metrics = [self.create_metric(mid, **params.get(mid, {})) for mid in metric_ids]

        return self.evaluate(
            metrics=metrics,
            model=model,
            x=x,
            explanation=explanation,
            y_pred=y_pred,
            **kwargs,
        )


_default_router: MetricRouter | None = None


def get_router() -> MetricRouter:
    """Get the default MetricRouter instance (lazy-initialized)."""
    global _default_router
    if _default_router is None:
        _default_router = MetricRouter()
    return _default_router
