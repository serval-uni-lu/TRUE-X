"""
metrics/utils/ts_helpers.py

Shared helper functions for FA multivariate time-series metrics.

All functions assume (B, C, T) layout: batch × channels/features × timesteps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray


def ts_predict_array(model: Any, x: Array) -> Array:
    """
    Call model on a (B, C, T) input; return a numpy array.
    Supports both sklearn-style .predict() and callable models.
    """
    if hasattr(model, "predict"):
        y = model.predict(x)
    else:
        y = model(x)
    return np.asarray(y)


def ts_regression_scores(model: Any, x: Array) -> Array:
    """
    Return scalar regression predictions, shape (B,).
    Collapses (B, 1) → (B,) automatically.
    """
    y = ts_predict_array(model, x)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    return y.ravel()


def ts_class_scores(model: Any, x: Array, *, num_classes: int | None = None) -> Array:
    """
    Return class scores / probabilities, shape (B, K).
    If the model returns hard labels (B,), converts to one-hot (B, K).
    """
    s = ts_predict_array(model, x)
    if s.ndim == 2:
        if s.shape[1] < 2:
            raise ValueError(f"classification outputs must have shape (B, K>=2), got {s.shape}")
        return s.astype(np.float64)

    if s.ndim != 1:
        raise ValueError(f"classification outputs must be 1D labels or 2D scores, got {s.shape}")

    labels = np.asarray(s).ravel()
    if labels.size == 0:
        raise ValueError("classification label output is empty")

    if not np.issubdtype(labels.dtype, np.integer):
        if np.issubdtype(labels.dtype, np.floating) and np.all(np.equal(labels, np.floor(labels))):
            labels = labels.astype(int)
        else:
            raise ValueError("1D classification outputs must contain integer class labels")
    else:
        labels = labels.astype(int)

    classes = getattr(model, "classes_", None)
    if num_classes is None:
        if classes is not None:
            classes_arr = np.asarray(classes)
            if classes_arr.ndim != 1 or classes_arr.size < 2:
                raise ValueError(
                    "model.classes_ must be 1D with at least 2 classes for classification"
                )
            class_to_idx = {int(c): i for i, c in enumerate(classes_arr)}
            try:
                mapped = np.array([class_to_idx[int(c)] for c in labels], dtype=int)
            except KeyError as exc:
                raise ValueError(
                    f"label {int(exc.args[0])} is not present in model.classes_"
                ) from None
            num_classes = int(classes_arr.size)
            labels_idx = mapped
        else:
            raise ValueError(
                "Model returned hard labels (B,) without class-space metadata. "
                "Provide 2D class scores/probabilities, define model.classes_, or pass num_classes."
            )
    else:
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError(
                f"hard labels must be in [0, {num_classes - 1}] when num_classes is provided"
            )
        labels_idx = labels

    out = np.zeros((labels_idx.size, num_classes), dtype=np.float64)
    out[np.arange(labels_idx.size), labels_idx] = 1.0
    return out


def fro_ratio(E: Array, E_t: Array, eps_noise: Array, eps: float = 1e-12) -> float:
    """
    Frobenius-norm ratio: ||E - E_t||_F / ||eps_noise||_F.

    Used by AvgSensitivity as the per-perturbation score.
    E, E_t: attribution maps (C, T) for one sample.
    eps_noise: the noise array that was added to the input (C, T).
    """
    num = np.linalg.norm((E - E_t).ravel())
    den = np.linalg.norm(eps_noise.ravel()) + eps
    return float(num / den)


def spearman_rank_corr(a: Array, b: Array) -> float:
    """
    Spearman rank correlation between two arrays (numpy-only, no scipy).
    Inputs are flattened before ranking.
    Returns 0.0 if either array is constant or empty.
    """
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    if va.size == 0 or vb.size == 0 or np.std(va) == 0.0 or np.std(vb) == 0.0:
        return 0.0
    ra = np.argsort(np.argsort(va)).astype(np.float64)
    rb = np.argsort(np.argsort(vb)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    na, nb = np.linalg.norm(ra), np.linalg.norm(rb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(ra, rb) / (na * nb))


def gini_from_values(values: Array, eps: float = 1e-12) -> float:
    """
    Gini coefficient on non-negative 1D values, result in [0, 1].

    Formula (sorted ascending):
        G = (1 / (n * sum_v)) * sum_{i=1}^n (2i - n - 1) * v_sorted[i]

    Returns 0.0 if all values are zero or array is empty.
    """
    v = np.abs(np.asarray(values, dtype=np.float64).ravel())
    n = v.size
    if n == 0:
        return 0.0
    s = v.sum()
    if s <= eps:
        return 0.0
    v_sorted = np.sort(v)
    i = np.arange(1, n + 1, dtype=np.float64)
    num = np.sum((2.0 * i - n - 1.0) * v_sorted)
    return float(np.clip(num / (n * s), 0.0, 1.0))


def normalized_entropy(values: Array, eps: float = 1e-12) -> float:
    """
    Normalized Shannon entropy H / ln(D) in [0, 1].

    p_i = v_i / sum(v),  H = -sum p_i * ln(p_i),  result = H / ln(D).
    Returns 0.0 if D <= 1 or all values are zero.
    """
    v = np.abs(np.asarray(values, dtype=np.float64).ravel())
    D = v.size
    if D <= 1:
        return 0.0
    tot = v.sum()
    if tot <= eps:
        return 0.0
    p = v / tot
    H = -np.sum(p * np.log(p + eps))
    return float(H / np.log(D))


def mask_with_channel_baseline_bct(
    window_ct: Array,
    flat_indices: Array,
    baseline_c: Array,
) -> Array:
    """
    For a single window (C, T), replace entries at flat_indices with
    the per-channel baseline value.

    Flat index is row-major over (C, T):
        c = idx // T
        t = idx % T
        out[c, t] = baseline_c[c]

    Args:
        window_ct: shape (C, T) — single time-series window.
        flat_indices: 1D int array of positions in [0, C*T).
        baseline_c: shape (C,) — per-channel baseline values.

    Returns:
        Masked window, same shape as window_ct.
    """
    C, T = window_ct.shape
    out = window_ct.copy()
    for idx in np.asarray(flat_indices, dtype=int):
        c = int(idx) // T
        t = int(idx) % T
        out[c, t] = baseline_c[c]
    return out
