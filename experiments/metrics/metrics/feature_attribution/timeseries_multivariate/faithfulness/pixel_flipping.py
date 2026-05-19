"""
metrics/metrics/feature_attribution/timeseries_multivariate/faithfulness/pixel_flipping.py

Pixel Flipping for feature attribution on multivariate time series.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from metrics.core.base_metric import BaseFeatureAttributionMetric, MetricMetadata
from metrics.core.enums import DataType, ExplanationType, TaskType
from metrics.core.metric_config import MetricConfig
from metrics.core.metric_registry import register_metric
from metrics.utils.ts_helpers import (
    mask_with_channel_baseline_bct,
    ts_class_scores,
    ts_regression_scores,
)


@register_metric("fa_ts_mv_pixel_flipping")
class FAMVPixelFlipping(BaseFeatureAttributionMetric):
    """
    Pixel Flipping for multivariate time-series attributions.

    Tests faithfulness by iteratively replacing the most important
    (channel, timestep) positions with baseline values and measuring
    how much the model output drops.

    Algorithm (per sample b):
      1. Flatten |E[b]| to (D,) and sort positions by descending importance.
      2. Starting from the original x[b], flip positions in batches of k:
           - Replace next k positions with baseline values.
           - Record |output_before - output_after|.
      3. Accumulate all drops and normalize:
           score = Σ |Δoutput| / (D × |output_original|)
         (denominator falls back to D if output_original == 0)

    For regression, output = scalar prediction.
    For classification, output = probability / score of the target class c.

    Higher score → removing important positions causes large output drops
               → attributions accurately identify what the model relies on
               → more faithful explanation.

    baseline_channels: per-channel baseline values, shape (C,).
    If not provided at init or call time, defaults to x.mean(axis=(0, 2)).

    Input convention: (B, C, T) — batch × channels × timesteps.
    Returns: np.ndarray shape (B,) — normalized AUC score per sample (>= 0).

    Tags:
        requires_model — model called at evaluation time
        local          — one score per sample
    """

    METADATA = MetricMetadata(
        metric_id="fa_ts_mv_pixel_flipping",
        display_name="FA-TS — Pixel Flipping",
        category="Faithfulness",
        explanation_type=ExplanationType.FEATURE_ATTRIBUTION,
        supported_data_types=(DataType.TIMESERIES_MULTIVARIATE,),
        supported_task_types=(TaskType.REGRESSION, TaskType.CLASSIFICATION),
        tags=frozenset({"requires_model", "local"}),
        param_schema=MetricConfig(
            {
                "kind": {
                    "type": str,
                    "default": "regression",
                    "choices": ["regression", "classification"],
                    "help": "Task type: 'regression' tracks scalar output; "
                    "'classification' tracks target class score.",
                },
                "features_in_step": {
                    "type": int,
                    "default": 1,
                    "help": "Number of positions to flip per step (k). "
                    "k=1 gives the finest-grained curve; larger k is faster.",
                },
                "baseline_channels": {
                    "type": np.ndarray,
                    "optional": True,
                    "help": "Per-channel baseline values, shape (C,). "
                    "If not given, computed as x.mean(axis=(0,2)) from the test batch.",
                },
            }
        ),
    )

    def __init__(
        self,
        *,
        kind: Literal["regression", "classification"] = "regression",
        features_in_step: int = 1,
        baseline_channels: np.ndarray | None = None,
    ) -> None:
        if kind not in {"regression", "classification"}:
            raise ValueError("kind must be 'regression' or 'classification'")
        if features_in_step < 1:
            raise ValueError("features_in_step must be >= 1")

        super().__init__(
            kind=kind, features_in_step=features_in_step, baseline_channels=baseline_channels
        )

        self.kind = kind
        self.features_in_step = int(features_in_step)
        self.baseline_channels: np.ndarray | None = (
            np.asarray(baseline_channels, dtype=np.float64)
            if baseline_channels is not None
            else None
        )

    def evaluate_attributions(
        self,
        *,
        model: Any,
        x: np.ndarray,
        attributions: np.ndarray,
        y_pred: Any = None,
        baseline_channels: np.ndarray | None = None,
        y: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute Pixel Flipping scores for a batch of (x, attribution) pairs.

        Args:
            model:             Callable — receives (B, C, T), returns predictions.
            x:                 Input time series, shape (B, C, T).
            attributions:      Attribution maps, shape (B, C, T).
            y_pred:            Unused (kept for interface compatibility).
            baseline_channels: Per-channel baseline values, shape (C,).
                               Overrides instance-level baseline_channels.
                               Falls back to x.mean(axis=(0,2)) if None.
            y:                 Target class indices, shape (B,) — classification only.
                               If None, uses argmax of model output.
            **kwargs:          Unused.

        Returns:
            np.ndarray shape (B,) — normalized AUC score per sample.
        """
        X = np.asarray(x, dtype=np.float64)
        E = np.asarray(attributions, dtype=np.float64)

        if X.ndim != 3:
            raise ValueError(f"x must be 3D (B, C, T), got shape {X.shape}")
        if E.shape != X.shape:
            raise ValueError(f"attributions shape {E.shape} must match x shape {X.shape}")

        B, C, T = X.shape
        D = C * T

        # Resolve baseline: call-time > init-time > batch mean fallback
        base = (
            np.asarray(baseline_channels, dtype=np.float64)
            if baseline_channels is not None
            else self.baseline_channels
        )
        if base is None:
            base = X.mean(axis=(0, 2))  # (C,) — mean over batch and time
        base = np.asarray(base, dtype=np.float64).ravel()
        if base.shape[0] != C:
            raise ValueError(
                f"baseline_channels length ({base.shape[0]}) must match n_channels ({C})"
            )

        # For classification: determine target class per sample
        if self.kind == "classification":
            S0 = ts_class_scores(model, X)  # (B, K)
            if S0.ndim != 2 or S0.shape[0] != B:
                raise ValueError(
                    f"classification model output must have shape (B, K); got {S0.shape}"
                )
            K = S0.shape[1]
            cls = np.argmax(S0, axis=1) if y is None else np.asarray(y, dtype=int).ravel()
            if cls.shape[0] != B:
                raise ValueError(f"y must have length B={B}; got length {cls.shape[0]}")
            if np.any((cls < 0) | (cls >= K)):
                raise ValueError(f"y contains class indices outside [0, {K - 1}]")
            y0 = S0[np.arange(B), cls]  # (B,) — original score for target class
        else:
            y0 = ts_regression_scores(model, X)  # (B,)

        scores = np.empty(B, dtype=np.float64)

        for b in range(B):
            e_flat = np.abs(E[b]).ravel()  # (D,)
            order = np.argsort(-e_flat)
            xb = X[b].copy()  # (C, T) — mutable working copy

            if self.kind == "regression":
                output_prev = float(ts_regression_scores(model, xb[np.newaxis])[0])
            else:
                Sb = ts_class_scores(model, xb[np.newaxis])[0]  # (K,)
                output_prev = float(Sb[int(cls[b])])

            total = 0.0
            for step in range(0, D, self.features_in_step):
                idx = order[step : step + self.features_in_step]
                xb = mask_with_channel_baseline_bct(xb, idx, base)  # (C, T)

                if self.kind == "regression":
                    output_curr = float(ts_regression_scores(model, xb[np.newaxis])[0])
                else:
                    Sm = ts_class_scores(model, xb[np.newaxis])[0]  # (K,)
                    output_curr = float(Sm[int(cls[b])])

                total += abs(output_prev - output_curr)
                output_prev = output_curr

            y0_b = float(y0[b])
            denom = D * abs(y0_b) if y0_b != 0.0 else float(D)
            scores[b] = total / denom

        return scores
