# metrics/utils/base_helpers.py

from collections.abc import Iterable

import numpy as np


def to_bf(x) -> np.ndarray:
    """
    Convert input to 2D float64 array.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[None, :]
    if x_arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x_arr.shape}")
    return x_arr


def bool_mask_from_indices(indices: Iterable[int] | None, length: int) -> np.ndarray:
    """
    Create boolean mask of shape (length,) with True at specified indices.
    """
    mask = np.zeros(length, dtype=bool)
    if indices is not None:
        for idx in indices:
            if 0 <= int(idx) < length:
                mask[int(idx)] = True
    return mask


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12, use_abs: bool = True) -> np.ndarray:
    """
    Safe division: a / (|b| + eps) or a / (b + eps).
    """
    b_safe = np.abs(b) if use_abs else b
    return np.asarray(a / (b_safe + eps), dtype=np.float64)
