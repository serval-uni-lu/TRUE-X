"""metrics/core/base_metric.py"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast

from .enums import DataType, ExplanationType, TaskType
from .metric_config import MetricConfig


@dataclass(frozen=True)
class MetricMetadata:
    """
    Metadata that must be readable WITHOUT instantiating a metric.

    This is the SINGLE SOURCE OF TRUTH for metric identity and capabilities.
    """

    metric_id: str
    display_name: str
    category: str
    explanation_type: ExplanationType
    supported_data_types: tuple[DataType, ...]
    supported_task_types: tuple[TaskType, ...]
    tags: frozenset[str] = field(default_factory=frozenset)
    param_schema: MetricConfig | None = None

    def supports(
        self,
        *,
        data_type: DataType | None = None,
        task_type: TaskType | None = None,
        explanation_type: ExplanationType | None = None,
    ) -> bool:
        """Check if this metric supports the given constraints."""
        if explanation_type is not None and self.explanation_type != explanation_type:
            return False
        if data_type is not None and data_type not in self.supported_data_types:
            return False
        if task_type is not None and task_type not in self.supported_task_types:
            return False
        return True


class BaseMetric(ABC):
    """
    Abstract base class for all XAI evaluation metrics.

    Design:
    - METADATA (class attribute): Static identity and capabilities - used for
      registry/discovery WITHOUT instantiation
    - __init__ (instance): Runtime hyperparameters only (n_draws, sigma, etc.)

    Subclasses MUST define METADATA as a class attribute.
    Subclasses should NOT pass metadata fields to super().__init__().
    """

    METADATA: MetricMetadata

    def __init__(self, **runtime_params: Any) -> None:
        """
        Initialize a metric with runtime parameters.

        All identity/capability info comes from METADATA automatically.
        Only pass hyperparameters here (n_draws, sigma, seed, etc.)

        Args:
            **runtime_params: Metric-specific hyperparameters. Stored in self._params
                              and can be accessed for serialization/logging.
        """
        meta = self.__class__.metadata()
        self._metric_id = meta.metric_id
        self._name = meta.display_name
        self._category = meta.category
        self._explanation_type = meta.explanation_type
        self._supported_data_types = meta.supported_data_types
        self._supported_task_types = meta.supported_task_types
        self._tags = meta.tags
        self._param_schema = meta.param_schema

        self._params: dict[str, Any] = runtime_params

    @property
    def metric_id(self) -> str:
        return self._metric_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def explanation_type(self) -> ExplanationType:
        return self._explanation_type

    @property
    def supported_data_types(self) -> tuple[DataType, ...]:
        return self._supported_data_types

    @property
    def supported_task_types(self) -> tuple[TaskType, ...]:
        return self._supported_task_types

    @property
    def tags(self) -> frozenset[str]:
        return self._tags

    @property
    def param_schema(self) -> MetricConfig | None:
        return self._param_schema

    @property
    def params(self) -> dict[str, Any]:
        """Runtime parameters passed to __init__."""
        return self._params.copy()

    @classmethod
    def metadata(cls) -> MetricMetadata:
        """
        Return class-level metadata without instantiating the metric.

        Raises:
            NotImplementedError: If METADATA is not defined or wrong type.
        """
        meta = getattr(cls, "METADATA", None)
        if meta is None or not isinstance(meta, MetricMetadata):
            raise NotImplementedError(f"{cls.__name__} must define METADATA = MetricMetadata(...)")
        return cast(MetricMetadata, meta)

    @classmethod
    def metric_key(cls) -> str:
        """Stable identifier for registry/results."""
        return cls.metadata().metric_id

    @classmethod
    def supports(
        cls,
        *,
        data_type: DataType | None = None,
        task_type: TaskType | None = None,
        explanation_type: ExplanationType | None = None,
    ) -> bool:
        """Check if this metric class supports given constraints (no instantiation)."""
        return cls.metadata().supports(
            data_type=data_type,
            task_type=task_type,
            explanation_type=explanation_type,
        )

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get a runtime parameter by name."""
        return self._params.get(name, default)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return f"{self.__class__.__name__}({params_str})"

    @abstractmethod
    def evaluate(
        self,
        *,
        model: Any,
        x: Any,
        explanation: Any,
        y_pred: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Evaluate this metric.

        Args:
            model: The ML model (must support prediction).
            x: Original input(s). Shape depends on data type.
            explanation: The explanation to evaluate. For counterfactuals, this is x_cf.
            y_pred: Optional pre-computed predictions for x.
            **kwargs: Metric-specific parameters (target_range, feature_names, etc.)

        Returns:
            Metric result. Typically np.ndarray of shape (batch_size,) for per-sample
            metrics, or a scalar for aggregate metrics.
        """
        raise NotImplementedError


class BaseCounterfactualMetric(BaseMetric):
    """
    Base class for counterfactual evaluation metrics.

    Provides clearer naming: x_original and x_cf instead of x and explanation.
    Subclasses can override evaluate_cf() for cleaner signatures.
    """

    def evaluate(
        self,
        *,
        model: Any,
        x: Any,
        explanation: Any,
        y_pred: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegates to evaluate_cf with clearer parameter names."""
        return self.evaluate_cf(
            model=model,
            x_original=x,
            x_cf=explanation,
            y_pred=y_pred,
            **kwargs,
        )

    @abstractmethod
    def evaluate_cf(
        self,
        *,
        model: Any,
        x_original: Any,
        x_cf: Any,
        y_pred: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Evaluate counterfactual quality.

        Args:
            model: The ML model.
            x_original: Original input(s), shape (batch_size, n_features) for tabular.
            x_cf: Counterfactual(s), same shape as x_original.
            y_pred: Optional predictions for x_original.
            **kwargs: Metric-specific parameters.

        Returns:
            Metric result, typically shape (batch_size,).
        """
        raise NotImplementedError


class BaseFeatureAttributionMetric(BaseMetric):
    """
    Base class for feature attribution evaluation metrics.

    Provides clearer naming: attributions instead of explanation.
    Subclasses can override evaluate_attributions() for cleaner signatures.
    """

    def evaluate(
        self,
        *,
        model: Any,
        x: Any,
        explanation: Any,
        y_pred: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegates to evaluate_attributions with clearer parameter names."""
        return self.evaluate_attributions(
            model=model,
            x=x,
            attributions=explanation,
            y_pred=y_pred,
            **kwargs,
        )

    @abstractmethod
    def evaluate_attributions(
        self,
        *,
        model: Any,
        x: Any,
        attributions: Any,
        y_pred: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Evaluate feature attribution quality.

        Args:
            model: The ML model.
            x: Input(s). Shape depends on data type:
                - Tabular:                  (batch_size, n_features)
                - Multivariate time series: (batch_size, timesteps, n_features)  [B, T, F]
            attributions: Attribution scores, same shape as x.
            y_pred: Optional predictions for x.
            **kwargs: Metric-specific parameters (e.g., explain_fn, baseline_channels).

        Returns:
            Metric result, typically shape (batch_size,).
        """
        raise NotImplementedError
