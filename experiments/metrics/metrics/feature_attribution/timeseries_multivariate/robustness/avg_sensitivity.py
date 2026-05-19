"""
metrics/metrics/feature_attribution/timeseries_multivariate/robustness/avg_sensitivity.py

Average Sensitivity for feature attribution on multivariate time series.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from metrics.core.base_metric import BaseFeatureAttributionMetric, MetricMetadata
from metrics.core.enums import DataType, ExplanationType, TaskType
from metrics.core.metric_config import MetricConfig
from metrics.core.metric_registry import register_metric
from metrics.utils.ts_helpers import fro_ratio, ts_class_scores, ts_regression_scores


@register_metric("fa_ts_mv_avg_sensitivity")
class FAMVAvgSensitivity(BaseFeatureAttributionMetric):
    """
    Average Sensitivity for multivariate time-series attributions.

    For each sample, draw M Gaussian-noise perturbations of the input.
    Accept a perturbation only when the model output change stays small:
      - Regression:      |f(x + ε) - f(x)| <= tau
      - Classification:  argmax f(x + ε) == argmax f(x)

    Per-sample score = mean( ||E - Ê||_F / ||ε||_F ) over M accepted perturbations,
    where E is the original attribution map and Ê is the attribution on the
    perturbed input (recomputed via explain_fn).

    Higher score → explanation changes more under small input changes → less stable.

    Input convention: (B, C, T) — batch × channels × timesteps.
    Returns: np.ndarray shape (B,) — NaN if no perturbation was accepted.

    Tags:
        requires_model     — model predictions used for acceptance filtering
        requires_explainer — explain_fn must be provided at evaluation time
        stochastic         — results vary with seed
        local              — one score per sample
    """

    METADATA = MetricMetadata(
        metric_id="fa_ts_mv_avg_sensitivity",
        display_name="FA-TS — Average Sensitivity",
        category="Robustness",
        explanation_type=ExplanationType.FEATURE_ATTRIBUTION,
        supported_data_types=(DataType.TIMESERIES_MULTIVARIATE,),
        supported_task_types=(TaskType.REGRESSION, TaskType.CLASSIFICATION),
        tags=frozenset({"requires_model", "requires_explainer", "stochastic", "local"}),
        param_schema=MetricConfig(
            {
                "kind": {
                    "type": str,
                    "default": "regression",
                    "choices": ["regression", "classification"],
                    "help": "Task type: 'regression' uses tau threshold; "
                    "'classification' uses label-preserving filter.",
                },
                "sigma": {
                    "type": float,
                    "default": 0.01,
                    "help": "Standard deviation of Gaussian noise added to inputs.",
                },
                "tau": {
                    "type": float,
                    "default": 0.05,
                    "help": "Max allowed output change |Δy| to accept a perturbation "
                    "(regression only).",
                },
                "n_perturbations": {
                    "type": int,
                    "default": 10,
                    "help": "Number of accepted perturbations M to collect per sample.",
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
        sigma: float = 0.01,
        tau: float = 0.05,
        n_perturbations: int = 10,
        seed: int | None = None,
    ) -> None:
        if kind not in {"regression", "classification"}:
            raise ValueError("kind must be 'regression' or 'classification'")
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        if tau <= 0:
            raise ValueError("tau must be > 0")
        if n_perturbations < 1:
            raise ValueError("n_perturbations must be >= 1")

        super().__init__(
            kind=kind,
            sigma=sigma,
            tau=tau,
            n_perturbations=n_perturbations,
            seed=seed,
        )

        self.kind = kind
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.n_perturbations = int(n_perturbations)
        self._rng = np.random.default_rng(seed)

    def evaluate_attributions(
        self,
        *,
        model: Any,
        x: np.ndarray,
        attributions: np.ndarray,
        y_pred=None,
        explain_fn: Callable | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute Average Sensitivity for a batch of (x, attribution) pairs.

        Args:
            model:        Callable — receives (B, C, T), returns predictions.
            x:            Input time series, shape (B, C, T).
            attributions: Attribution maps, shape (B, C, T).
            y_pred:       Unused (kept for interface compatibility).
            explain_fn:   Required. Callable: (x_batch: ndarray(1,C,T)) -> ndarray(C,T)
                          or ndarray(1,C,T). Re-runs the explainer on perturbed inputs.
            **kwargs:     Unused.

        Returns:
            np.ndarray shape (B,) — per-sample sensitivity score.
            NaN for samples where no perturbation was accepted within
            20 * n_perturbations attempts.
        """
        if explain_fn is None:
            raise ValueError(
                "FAMVAvgSensitivity requires 'explain_fn': a callable "
                "explain_fn(x_batch) -> attributions_batch. "
                "Pass it as a keyword argument at evaluation time."
            )

        X = np.asarray(x, dtype=np.float64)
        E = np.asarray(attributions, dtype=np.float64)

        if X.ndim != 3:
            raise ValueError(f"x must be 3D (B, C, T), got shape {X.shape}")
        if E.shape != X.shape:
            raise ValueError(f"attributions shape {E.shape} must match x shape {X.shape}")

        B = X.shape[0]
        max_attempts = 20 * self.n_perturbations

        if self.kind == "regression":
            y0 = ts_regression_scores(model, X)
        else:
            S0 = ts_class_scores(model, X)
            y_cls = np.argmax(S0, axis=1)

        scores = np.full(B, np.nan, dtype=np.float64)

        for b in range(B):
            Eb = E[b]  # (C, T)
            xb = X[b : b + 1]  # (1, C, T)
            collected = 0
            total = 0.0
            attempts = 0

            if self.kind == "regression":
                yb = float(y0[b])
            else:
                cls_b = int(y_cls[b])

            while collected < self.n_perturbations and attempts < max_attempts:
                attempts += 1
                eps = self._rng.normal(0.0, self.sigma, size=xb.shape).astype(np.float64)
                xt = xb + eps  # (1, C, T)

                if self.kind == "regression":
                    yt = float(ts_regression_scores(model, xt)[0])
                    accepted = abs(yt - yb) <= self.tau
                else:
                    St = ts_class_scores(model, xt)
                    cls_t = int(np.argmax(St, axis=1)[0])
                    accepted = cls_t == cls_b

                if accepted:
                    Et_raw = np.asarray(explain_fn(xt), dtype=np.float64)
                    # Explainers may return a single sample (C, T) or a batch (1, C, T).
                    # Enforce strict shape contract to avoid silent metric corruption.
                    if Et_raw.ndim == 3:
                        if Et_raw.shape[0] != 1:
                            raise ValueError(
                                f"explain_fn must return (C, T) or (1, C, T); got {Et_raw.shape}"
                            )
                        Et = Et_raw[0]
                    elif Et_raw.ndim == 2:
                        Et = Et_raw
                    else:
                        raise ValueError(
                            f"explain_fn must return (C, T) or (1, C, T); got {Et_raw.shape}"
                        )
                    if Et.shape != Eb.shape:
                        raise ValueError(
                            f"explain_fn output shape {Et.shape} does not match "
                            f"attribution shape {Eb.shape}"
                        )
                    total += fro_ratio(Eb, Et, eps[0])
                    collected += 1

            if collected > 0:
                scores[b] = total / collected

        return scores
