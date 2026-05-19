"""
metrics/utils/validation.py

Validation utilities and helper functions.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from metrics.utils.base_helpers import bool_mask_from_indices
from metrics.utils.typing import FeasibleSpec


def ensure_matching_shapes(
    a: np.ndarray,
    b: np.ndarray,
    *,
    name_a: str = "a",
    name_b: str = "b",
) -> None:
    """
    Raise ValueError if a and b have different shapes.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {name_a} {a.shape} vs {name_b} {b.shape}")


def zero_out_immutable(
    delta: np.ndarray, n_features: int, indices: Iterable[int] | None
) -> tuple[np.ndarray, int]:
    """
    Sets entries in delta[:, indices] to 0.0 if indices are provided.

    Returns:
        (updated_delta, n_active_features)
    """
    if indices is None:
        return delta, n_features

    mask = bool_mask_from_indices(indices, n_features)
    delta = delta.copy()
    delta[:, mask] = 0.0
    return delta, int((~mask).sum())


def nan_violation_mask(x: np.ndarray) -> np.ndarray:
    """Return boolean mask where True indicates non-finite values."""
    return np.asarray(~np.isfinite(x), dtype=bool)


def validate_feasible_tabular(
    xcf: np.ndarray,
    feasible_values: dict[int, FeasibleSpec],
    tol_vec: np.ndarray,
    weight_vec: np.ndarray,
    treat_nan_as_violation: bool = True,
) -> np.ndarray:
    """Validate counterfactuals against feasibility constraints."""
    B = xcf.shape[0]
    counts = np.zeros(B, dtype=np.float64)

    for j, spec in feasible_values.items():
        j = int(j)
        col = xcf[:, j]
        nan_mask = nan_violation_mask(col)
        if treat_nan_as_violation and np.any(nan_mask):
            counts += weight_vec[j] * nan_mask.astype(np.float64)

        valid = ~nan_mask
        if not np.any(valid):
            continue

        col_v = col[valid]
        tol_j = tol_vec[j]

        if isinstance(spec, tuple) and len(spec) == 2:
            lo, hi = float(spec[0]), float(spec[1])
            if lo > hi:
                lo, hi = hi, lo
            bad = (col_v < lo - tol_j) | (col_v > hi + tol_j)
        else:
            allowed = np.asarray(list(spec))
            if np.issubdtype(col_v.dtype, np.floating):
                bad = np.ones_like(col_v, dtype=bool)
                for v in allowed:
                    bad &= np.abs(col_v - v) > tol_j
            else:
                bad = ~np.isin(col_v, allowed)

        full_mask = np.zeros(B, dtype=bool)
        full_mask[valid] = bad
        counts += weight_vec[j] * full_mask.astype(np.float64)

    return counts


# ---------------------------------------------------------------------
# Regression validity & margins
# ---------------------------------------------------------------------


def valid_regression(
    y: np.ndarray,
    *,
    target_range: tuple[float, float] | None = None,
    target_value: float | None = None,
    tol: float = 0.0,
) -> np.ndarray:
    """
    Check whether regression outputs are valid.

    Validity definition:
    - If target_range is provided: y ∈ [lo, hi]
    - Else: |y - target_value| ≤ tol
    """
    y = np.asarray(y, dtype=np.float64)

    if target_range is not None:
        lo, hi = float(target_range[0]), float(target_range[1])
        return (y >= lo) & (y <= hi)

    if target_value is None:
        raise ValueError("Either target_range or target_value must be provided.")

    return np.asarray(np.abs(y - float(target_value)) <= float(tol), dtype=bool)


def signed_distance_to_interval(
    y: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Signed distance to an interval [lo, hi].

    - Positive inside interval (distance to nearest boundary)
    - Negative outside interval
    """
    y = np.asarray(y, dtype=np.float64)

    below = y < lo
    above = y > hi
    inside = ~(below | above)

    out = np.empty_like(y, dtype=np.float64)
    out[below] = -(lo - y[below])
    out[above] = -(y[above] - hi)
    out[inside] = np.minimum(y[inside] - lo, hi - y[inside])

    return out


# ---------------------------------------------------------------------
# Classification validity & margins
# ---------------------------------------------------------------------


def valid_classification_margin(
    S: np.ndarray,
    target_class: np.ndarray,
    *,
    score_margin: float = 0.0,
) -> np.ndarray:
    """
    Check whether classification predictions satisfy a margin constraint.

    Valid if:
        f_c(x) - max_{k != c} f_k(x) >= score_margin
    """
    S = np.asarray(S, dtype=np.float64)
    c = np.asarray(target_class, dtype=int).ravel()

    B, K = S.shape
    if np.any((c < 0) | (c >= K)):
        raise ValueError("target_class indices out of bounds")

    sc = S[np.arange(B), c]
    S_masked = S.copy()
    S_masked[np.arange(B), c] = -np.inf
    margin = sc - np.max(S_masked, axis=1)

    return np.asarray(margin >= float(score_margin), dtype=bool)


def multi_class_margin(
    S: np.ndarray,
    target_class: np.ndarray,
) -> np.ndarray:
    """
    Compute multiclass margin:

        f_c(x) - max_{k != c} f_k(x)
    """
    S = np.asarray(S, dtype=np.float64)
    c = np.asarray(target_class, dtype=int).ravel()

    B, K = S.shape
    if np.any((c < 0) | (c >= K)):
        raise ValueError("target_class indices out of bounds")

    sc = S[np.arange(B), c]
    S_masked = S.copy()
    S_masked[np.arange(B), c] = -np.inf
    return np.asarray(sc - np.max(S_masked, axis=1), dtype=np.float64)
