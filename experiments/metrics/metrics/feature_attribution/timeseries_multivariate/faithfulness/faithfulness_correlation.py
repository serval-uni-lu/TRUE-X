"""
metrics/metrics/feature_attribution/timeseries_multivariate/faithfulness/faithfulness_correlation.py

Faithfulness Correlation for feature attribution on multivariate time series.
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
    spearman_rank_corr,
    ts_class_scores,
    ts_regression_scores,
)


@register_metric("fa_ts_mv_faithfulness_correlation")
class FAMVFaithfulnessCorrelation(BaseFeatureAttributionMetric):
    """
    Faithfulness Correlation for multivariate time-series attributions.

    Tests whether attributions faithfully reflect the model's behavior by
    measuring how well attribution importance predicts output changes under
    random masking.

    Algorithm (per sample b):
      1. Flatten attribution map |E[b]| to a 1D vector of length D = C × T.
      2. Repeat M times:
           - Sample a random subset S of q positions from [0, D).
           - Replace those positions in x[b] with per-channel baseline values.
           - Record: attr_sum = Σ_{i∈S} |E[b]_i|
                     delta    = output(x[b]) - output(masked x[b])
      3. Compute Spearman ρ between the M attr_sums and M deltas.

    For regression, output = scalar prediction.
    For classification, output = probability / score of the target class c.

    Higher ρ → attributions reliably predict where masking hurts the model most
             → more faithful explanation.

    baseline_channels: per-channel baseline values, shape (C,).
    If not provided at init or call time, defaults to the training-set mean
    estimated from the test batch: x.mean(axis=(0, 2)).

    Input convention: (B, C, T) — batch × channels × timesteps.
    Returns: np.ndarray shape (B,) — Spearman ρ per sample, in [-1, 1].

    Tags:
        requires_model — model called at evaluation time
        local          — one score per sample
    """

    METADATA = MetricMetadata(
        metric_id="fa_ts_mv_faithfulness_correlation",
        display_name="FA-TS — Faithfulness Correlation",
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
                "subset_size": {
                    "type": int,
                    "default": 10,
                    "help": "Number of (channel, timestep) positions masked per run (q).",
                },
                "n_runs": {
                    "type": int,
                    "default": 100,
                    "help": "Number of random masking runs per sample (M).",
                },
                "baseline_channels": {
                    "type": np.ndarray,
                    "optional": True,
                    "help": "Per-channel baseline values, shape (C,). "
                    "If not given, computed as x.mean(axis=(0,2)) from the test batch.",
                },
                "seed": {
                    "type": int,
                    "optional": True,
                    "help": "Random seed for reproducibility.",
                },
            }
        ),
    )

    def __init__(
        self,
        *,
        kind: Literal["regression", "classification"] = "regression",
        subset_size: int = 10,
        n_runs: int = 100,
        baseline_channels: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        if kind not in {"regression", "classification"}:
            raise ValueError("kind must be 'regression' or 'classification'")
        if subset_size < 1:
            raise ValueError("subset_size must be >= 1")
        if n_runs < 1:
            raise ValueError("n_runs must be >= 1")

        super().__init__(
            kind=kind,
            subset_size=subset_size,
            n_runs=n_runs,
            baseline_channels=baseline_channels,
            seed=seed,
        )

        self.kind = kind
        self.subset_size = int(subset_size)
        self.n_runs = int(n_runs)
        self.baseline_channels: np.ndarray | None = (
            np.asarray(baseline_channels, dtype=np.float64)
            if baseline_channels is not None
            else None
        )
        self._rng = np.random.default_rng(seed)

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
        Compute Faithfulness Correlation for a batch of (x, attribution) pairs.

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
            np.ndarray shape (B,) — Spearman ρ per sample.
        """
        X = np.asarray(x, dtype=np.float64)
        E = np.asarray(attributions, dtype=np.float64)

        if X.ndim != 3:
            raise ValueError(f"x must be 3D (B, C, T), got shape {X.shape}")
        if E.shape != X.shape:
            raise ValueError(f"attributions shape {E.shape} must match x shape {X.shape}")

        B, C, T = X.shape
        D = C * T

        if self.subset_size > D:
            raise ValueError(f"subset_size ({self.subset_size}) cannot exceed D=C×T={D}")

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

        if self.kind == "regression":
            y0 = ts_regression_scores(model, X)  # (B,)
        else:
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
            y0 = S0[np.arange(B), cls]  # (B,) — score for target class

        rhos = np.empty(B, dtype=np.float64)

        for b in range(B):
            e_flat = np.abs(E[b]).ravel()  # (D,)
            xb = X[b]  # (C, T)

            attr_sums = np.empty(self.n_runs, dtype=np.float64)
            deltas = np.empty(self.n_runs, dtype=np.float64)

            for m in range(self.n_runs):
                idx = self._rng.choice(D, size=self.subset_size, replace=False)
                masked = mask_with_channel_baseline_bct(xb, idx, base)  # (C, T)

                if self.kind == "regression":
                    y1 = float(ts_regression_scores(model, masked[np.newaxis])[0])
                    delta = float(y0[b]) - y1
                else:
                    Sm = ts_class_scores(model, masked[np.newaxis])[0]  # (K,)
                    y1 = float(Sm[int(cls[b])])
                    delta = float(y0[b]) - y1

                attr_sums[m] = float(e_flat[idx].sum())
                deltas[m] = delta

            rhos[b] = spearman_rank_corr(attr_sums, deltas)

        return rhos
