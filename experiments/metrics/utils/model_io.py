"""
metrics/utils/model_io.py

Model input/output utilities for standardized prediction handling.
"""

from __future__ import annotations

import numpy as np


def predict_regression(model, X: np.ndarray) -> np.ndarray:
    """
    Get regression predictions from a model.

    Tries model.predict() and ensures 1D output.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]

    y = model.predict(X)
    y = np.asarray(y, dtype=np.float64)

    # Handle (B, 1) shaped output
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()

    if y.ndim != 1:
        raise ValueError(f"Expected 1D output for regression. Got shape: {y.shape}")

    return y.ravel()


def predict_classification(model, X: np.ndarray) -> np.ndarray:
    """
    Get classification probability predictions from a model.

    Tries model.predict_proba() first, falls back to model.predict().
    Returns shape (B, K) where K is number of classes.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]

    # Try predict_proba first (preferred for classification)
    if hasattr(model, "predict_proba"):
        S = model.predict_proba(X)
    else:
        S = model.predict(X)

    S = np.asarray(S, dtype=np.float64)

    if S.ndim != 2 or S.shape[1] < 2:
        raise ValueError(f"Expected shape (B, K≥2) for classification. Got: {S.shape}")

    return S


# Aliases for backward compatibility
predict_reg = predict_regression
predict_proba_clf = predict_classification


def resolve_prediction_progress(
    model,
    kind: str,
    *,
    X_path: np.ndarray,
    x_start: np.ndarray | None = None,
    x_cf: np.ndarray | None = None,
    target_value: float | None = None,
    target_class: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    if kind == "regression":
        y = predict_regression(model, X_path)
        y0 = y[0]
        sign = 1.0 if target_value is None else np.sign(target_value - y0)
        return np.asarray(sign * (y - y0), dtype=np.float64)
    elif kind == "classification":
        S = predict_classification(model, X_path)
        if target_class is None:
            if x_cf is None:
                raise ValueError("x_cf must be provided when target_class is None.")
            Scf = predict_classification(model, x_cf[None, :])
            target_class = int(np.argmax(Scf[0]))
        return np.asarray(S[:, target_class] - S[0, target_class], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported kind: {kind}")
