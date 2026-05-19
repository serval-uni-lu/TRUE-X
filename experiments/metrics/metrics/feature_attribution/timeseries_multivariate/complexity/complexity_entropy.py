"""
metrics/metrics/feature_attribution/timeseries_multivariate/complexity/complexity_entropy.py

Complexity metrics (normalized Shannon entropy) for feature attribution
on multivariate time series.

Two aggregation levels:
  - element: entropy over all (channel, timestep) pairs — shape (C*T,)
  - channel: entropy over per-channel importance sums   — shape (C,)

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
from metrics.utils.ts_helpers import normalized_entropy


@register_metric("fa_ts_mv_complexity_entropy_element")
class FAMVComplexityEntropyElement(BaseFeatureAttributionMetric):
    """
    Element-level Complexity (normalized Shannon entropy) for multivariate
    time-series attributions.

    Computes the normalized Shannon entropy over all (channel × timestep)
    attribution values for each sample:

        vec = |attr[b]|.ravel()             # shape (C*T,)
        p   = vec / sum(vec)
        score[b] = -sum(p * log(p)) / log(C*T)   in [0, 1]

    Higher score → attribution spread across many positions (complex explanation).
    Lower score  → attribution concentrated at a few positions (simple explanation).

    Does NOT use the model or the input x — pure attribution analysis.

    Input convention: (B, C, T) — batch × channels × timesteps.
    Returns: np.ndarray shape (B,).

    Tags:
        local — one score per sample
    """

    METADATA = MetricMetadata(
        metric_id="fa_ts_mv_complexity_entropy_element",
        display_name="FA-TS — Complexity Entropy (Element)",
        category="Complexity",
        explanation_type=ExplanationType.FEATURE_ATTRIBUTION,
        supported_data_types=(DataType.TIMESERIES_MULTIVARIATE,),
        supported_task_types=(TaskType.REGRESSION, TaskType.CLASSIFICATION),
        tags=frozenset({"local"}),
        param_schema=MetricConfig(
            {
                "eps": {
                    "type": float,
                    "default": 1e-12,
                    "help": "Small constant added to probabilities before log to avoid log(0).",
                },
            }
        ),
    )

    def __init__(self, *, eps: float = 1e-12) -> None:
        if eps < 0:
            raise ValueError("eps must be >= 0")
        super().__init__(eps=eps)
        self.eps = float(eps)

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
        Compute element-level Complexity Entropy for a batch of attribution maps.

        Args:
            model:        Unused — kept for interface compatibility.
            x:            Unused — kept for interface compatibility.
            attributions: Attribution maps, shape (B, C, T).
            y_pred:       Unused.
            **kwargs:     Unused.

        Returns:
            np.ndarray shape (B,) — normalized entropy per sample, in [0, 1].
        """
        E = np.asarray(attributions, dtype=np.float64)
        if E.ndim != 3:
            raise ValueError(f"attributions must be 3D (B, C, T), got shape {E.shape}")

        B = E.shape[0]
        scores = np.empty(B, dtype=np.float64)
        for b in range(B):
            vec = np.abs(E[b]).ravel()  # (C*T,)
            scores[b] = normalized_entropy(vec, eps=self.eps)
        return scores


@register_metric("fa_ts_mv_complexity_entropy_channel")
class FAMVComplexityEntropyChannel(BaseFeatureAttributionMetric):
    """
    Channel-level Complexity (normalized Shannon entropy) for multivariate
    time-series attributions.

    For each sample, collapses the attribution map along the time axis by
    summing absolute values, then computes normalized entropy over channels:

        channel_sums = |attr[b]|.sum(axis=1)    # shape (C,) — sum over T
        p = channel_sums / sum(channel_sums)
        score[b] = -sum(p * log(p)) / log(C)    in [0, 1]

    Higher score → importance spread across many channels (complex at channel level).
    Lower score  → importance concentrated on a few channels (simple at channel level).

    Does NOT use the model or the input x — pure attribution analysis.

    Input convention: (B, C, T) — batch × channels × timesteps.
    Returns: np.ndarray shape (B,).

    Tags:
        local — one score per sample
    """

    METADATA = MetricMetadata(
        metric_id="fa_ts_mv_complexity_entropy_channel",
        display_name="FA-TS — Complexity Entropy (Channel)",
        category="Complexity",
        explanation_type=ExplanationType.FEATURE_ATTRIBUTION,
        supported_data_types=(DataType.TIMESERIES_MULTIVARIATE,),
        supported_task_types=(TaskType.REGRESSION, TaskType.CLASSIFICATION),
        tags=frozenset({"local"}),
        param_schema=MetricConfig(
            {
                "eps": {
                    "type": float,
                    "default": 1e-12,
                    "help": "Small constant added to probabilities before log to avoid log(0).",
                },
            }
        ),
    )

    def __init__(self, *, eps: float = 1e-12) -> None:
        if eps < 0:
            raise ValueError("eps must be >= 0")
        super().__init__(eps=eps)
        self.eps = float(eps)

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
        Compute channel-level Complexity Entropy for a batch of attribution maps.

        Args:
            model:        Unused — kept for interface compatibility.
            x:            Unused — kept for interface compatibility.
            attributions: Attribution maps, shape (B, C, T).
            y_pred:       Unused.
            **kwargs:     Unused.

        Returns:
            np.ndarray shape (B,) — normalized entropy per sample, in [0, 1].
        """
        E = np.asarray(attributions, dtype=np.float64)
        if E.ndim != 3:
            raise ValueError(f"attributions must be 3D (B, C, T), got shape {E.shape}")

        B = E.shape[0]
        scores = np.empty(B, dtype=np.float64)
        for b in range(B):
            channel_sums = np.abs(E[b]).sum(axis=1)  # sum over T → (C,)
            scores[b] = normalized_entropy(channel_sums, eps=self.eps)
        return scores
