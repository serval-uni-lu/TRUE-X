"""
metrics/metrics/feature_attribution/timeseries_multivariate/complexity/sparseness.py

Sparseness metrics (Gini-based) for feature attribution on multivariate time series.

Two aggregation levels:
  - element: Gini index over all (channel, timestep) pairs — shape (C*T,)
  - channel: Gini index over per-channel importance sums  — shape (C,)

Both are task-neutral (model output is not used).
Input convention: (B, C, T) — batch × channels × timesteps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metrics.core.base_metric import BaseFeatureAttributionMetric, MetricMetadata
from metrics.core.enums import DataType, ExplanationType, TaskType
from metrics.core.metric_config import MetricConfig
from metrics.core.metric_registry import register_metric
from metrics.utils.ts_helpers import gini_from_values


@register_metric("fa_ts_mv_sparseness_element")
class FAMVSparsenessElement(BaseFeatureAttributionMetric):
    """
    Element-level Sparseness for multivariate time-series attributions.

    Computes the Gini coefficient over all (channel × timestep) attribution
    values for each sample:

        vec = |attr[b]|.ravel()   # shape (C*T,)
        score[b] = Gini(vec)

    Higher score → more concentrated attribution (sparser, more interpretable).
    Lower score  → attribution spread uniformly across all positions.

    Does NOT use the model or the input x — pure attribution analysis.

    Input convention: (B, C, T) — batch × channels × timesteps.
    Returns: np.ndarray shape (B,).

    Tags:
        local — one score per sample
    """

    METADATA = MetricMetadata(
        metric_id="fa_ts_mv_sparseness_element",
        display_name="FA-TS — Sparseness (Element)",
        category="Complexity",
        explanation_type=ExplanationType.FEATURE_ATTRIBUTION,
        supported_data_types=(DataType.TIMESERIES_MULTIVARIATE,),
        supported_task_types=(TaskType.REGRESSION, TaskType.CLASSIFICATION),
        tags=frozenset({"local"}),
        param_schema=MetricConfig({}),
    )

    def __init__(self) -> None:
        super().__init__()

    def evaluate_attributions(
        self,
        *,
        model: Any = None,
        x: Any = None,
        attributions: np.ndarray,
        y_pred: Any = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute element-level Sparseness for a batch of attribution maps.

        Args:
            model:        Unused — kept for interface compatibility.
            x:            Unused — kept for interface compatibility.
            attributions: Attribution maps, shape (B, C, T).
            y_pred:       Unused.
            **kwargs:     Unused.

        Returns:
            np.ndarray shape (B,) — Gini coefficient per sample, in [0, 1].
        """
        E = np.asarray(attributions, dtype=np.float64)
        if E.ndim != 3:
            raise ValueError(f"attributions must be 3D (B, C, T), got shape {E.shape}")

        B = E.shape[0]
        scores = np.empty(B, dtype=np.float64)
        for b in range(B):
            vec = np.abs(E[b]).ravel()  # (C*T,)
            scores[b] = gini_from_values(vec)
        return scores


@register_metric("fa_ts_mv_sparseness_channel")
class FAMVSparsenessChannel(BaseFeatureAttributionMetric):
    """
    Channel-level Sparseness for multivariate time-series attributions.

    For each sample, collapses the attribution map along the time axis by
    summing absolute values, then computes the Gini coefficient over channels:

        channel_sums = |attr[b]|.sum(axis=1)   # shape (C,) — sum over T
        score[b] = Gini(channel_sums)

    Higher score → attribution concentrated on a few channels (sparser channels).
    Lower score  → attribution spread evenly across all channels.

    Does NOT use the model or the input x — pure attribution analysis.

    Input convention: (B, C, T) — batch × channels × timesteps.
    Returns: np.ndarray shape (B,).

    Tags:
        local — one score per sample
    """

    METADATA = MetricMetadata(
        metric_id="fa_ts_mv_sparseness_channel",
        display_name="FA-TS — Sparseness (Channel)",
        category="Complexity",
        explanation_type=ExplanationType.FEATURE_ATTRIBUTION,
        supported_data_types=(DataType.TIMESERIES_MULTIVARIATE,),
        supported_task_types=(TaskType.REGRESSION, TaskType.CLASSIFICATION),
        tags=frozenset({"local"}),
        param_schema=MetricConfig({}),
    )

    def __init__(self) -> None:
        super().__init__()

    def evaluate_attributions(
        self,
        *,
        model: Any = None,
        x: Any = None,
        attributions: np.ndarray,
        y_pred: Any = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute channel-level Sparseness for a batch of attribution maps.

        Args:
            model:        Unused — kept for interface compatibility.
            x:            Unused — kept for interface compatibility.
            attributions: Attribution maps, shape (B, C, T).
            y_pred:       Unused.
            **kwargs:     Unused.

        Returns:
            np.ndarray shape (B,) — Gini coefficient per sample, in [0, 1].
        """
        E = np.asarray(attributions, dtype=np.float64)
        if E.ndim != 3:
            raise ValueError(f"attributions must be 3D (B, C, T), got shape {E.shape}")

        B = E.shape[0]
        scores = np.empty(B, dtype=np.float64)
        for b in range(B):
            channel_sums = np.abs(E[b]).sum(axis=1)  # sum over T → (C,)
            scores[b] = gini_from_values(channel_sums)
        return scores
