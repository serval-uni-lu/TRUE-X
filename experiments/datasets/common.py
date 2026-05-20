# datasets/common.py
import numpy as np
from dataclasses import dataclass
from typing import Optional


def ensure_channels_first(x: np.ndarray, channels_first: bool) -> np.ndarray:
    """Convert (N, T, C) to (N, C, T) if channels_first is False; no-op otherwise."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3-D array (N, ?, ?), got shape {x.shape}")
    if channels_first:
        return x
    return x.transpose(0, 2, 1)


def one_hot_to_index(y: np.ndarray) -> np.ndarray:
    if y.ndim != 2:
        raise ValueError(f"Expected 2-D one-hot array, got shape {y.shape}")
    return y.argmax(axis=1)


@dataclass
class Standardizer:
    """Per-channel z-score normalizer for (N, C, T) arrays.

    Fit on training data only, then apply transform to val/test.
    """

    mean_: Optional[np.ndarray] = None  # (C,)
    std_:  Optional[np.ndarray] = None  # (C,)

    def fit(self, X: np.ndarray) -> "Standardizer":
        if X.ndim != 3:
            raise ValueError(f"Standardizer.fit expects (N, C, T), got shape {X.shape}")
        flat = X.transpose(1, 0, 2).reshape(X.shape[1], -1)  # (C, N*T)
        mu   = flat.mean(axis=1)
        sd   = flat.std(axis=1)
        sd[sd == 0] = 1.0
        self.mean_ = mu.astype(np.float32)
        self.std_  = sd.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"Standardizer.transform expects (N, C, T), got shape {X.shape}")
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer is not fitted. Call fit() first.")
        return (X - self.mean_[:, None]) / self.std_[:, None]


def make_windows(series: np.ndarray, window: int, shift: int) -> np.ndarray:
    """
    series: (T, C) -> (num_windows, window, C)
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if shift <= 0:
        raise ValueError(f"shift must be > 0, got {shift}")
    T, C = series.shape
    if T < window:
        raise ValueError(f"T={T} < window={window}")
    n          = (T - window) // shift + 1
    s0, s1     = series.strides
    shape      = (n, window, C)
    strides    = (s0 * shift, s0, s1)
    return np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides).copy()


def window_targets(target: np.ndarray, window: int, shift: int) -> np.ndarray:
    """
    target: (T,) per-timestep values -> (num_windows,) value at each window's end
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if shift <= 0:
        raise ValueError(f"shift must be > 0, got {shift}")
    T = len(target)
    if T < window:
        raise ValueError(f"T={T} < window={window}")
    n = (T - window) // shift + 1
    return target[np.arange(n) * shift + (window - 1)]
