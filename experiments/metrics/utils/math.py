"""
metrics/utils/math.py

Mathematical utility functions.
"""

from __future__ import annotations

import numpy as np


def entropy_rows(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute row-wise entropy of probability matrix p (rows sum to 1).

    Args:
        p: np.ndarray of shape (B, F)
        eps: numerical stability cutoff

    Returns:
        entropy: np.ndarray of shape (B,)
    """
    p = np.clip(p, eps, 1.0)
    return np.asarray(-np.sum(p * np.log(p), axis=1), dtype=np.float64)


def linspace01(n: int) -> np.ndarray:
    """
    Generate n evenly spaced points in [0, 1].

    Args:
        n: Number of points

    Returns:
        np.ndarray of shape (n,) with values from 0 to 1
    """
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


__all__ = ["entropy_rows", "linspace01"]
