# datasets/adapters/ecoating.py
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from datasets.builders import build_loaders, TimeSeriesDataset


def _sliding_windows(
    X_2d: np.ndarray, y_1d: np.ndarray, window: int, shift: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_2d: (T, C)  continuous multivariate series
    y_1d: (T,)    per-timestep regression target

    Returns:
        Xw: (N, window, C)
        Yw: (N,)  value at each window's last timestep
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if shift <= 0:
        raise ValueError(f"shift must be > 0, got {shift}")
    T, C = X_2d.shape
    if T < window:
        raise ValueError(f"Series too short: T={T} < window={window}")

    n       = (T - window) // shift + 1
    s0, s1  = X_2d.strides
    Xw      = np.lib.stride_tricks.as_strided(
        X_2d, shape=(n, window, C), strides=(s0 * shift, s0, s1)
    ).copy().astype(np.float32)
    Yw      = y_1d[np.arange(n) * shift + (window - 1)].astype(np.float32)
    return Xw, Yw


def _last_k_windows_with_targets(
    X_2d: np.ndarray, y_1d: np.ndarray, window: int, shift: int, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the last K windows with their end-of-window targets.

    Returns:
        Xw:   (K_eff, window, C)
        Yw:   (K_eff,)
        mask: (K_eff, window)  True for valid positions (zero-padded short series)
    """
    T, C = X_2d.shape
    if T < window:
        pad_len = window - T
        x    = np.concatenate([X_2d, np.zeros((pad_len, C), dtype=X_2d.dtype)], axis=0)
        mask = np.concatenate([np.ones(T, dtype=bool), np.zeros(pad_len, dtype=bool)], axis=0)
        return x[None, ...].astype(np.float32), np.array([y_1d[-1]], dtype=np.float32), mask[None, ...]

    max_n       = (T - window) // shift + 1
    k_eff       = min(k, max_n)
    start_first = T - (k_eff - 1) * shift - window

    Xw   = np.empty((k_eff, window, C), dtype=np.float32)
    Yw   = np.empty((k_eff,), dtype=np.float32)
    mask = np.ones((k_eff, window), dtype=bool)

    for i in range(k_eff):
        s     = start_first + i * shift
        e     = s + window
        Xw[i] = X_2d[s:e]
        Yw[i] = y_1d[e - 1]

    return Xw, Yw, mask


def load_ecoating_as_arrays(
    train_path: str,
    *,
    test_path: Optional[str] = None,
    time_col: str = "TIME",
    target_col: str = "TP2",
    drop_extra_train: Optional[List[str]] = None,
    drop_extra_test: Optional[List[str]] = None,
    window: int = 30,
    shift: int = 1,
    val_size: float = 0.2,
    random_state: int = 83,
    scale_features: bool = True,
    scale_target: bool = False,
    last_k_test_windows: int = 5,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    StandardScaler, List[str], Optional[StandardScaler],
]:
    """
    Load Ecoating CSVs and build sliding windows for regression on `target_col`.

    Returns:
        X_tr (N_tr, C, T), y_tr (N_tr,)
        X_va (N_va, C, T), y_va (N_va,)
        test_tuple:
            (X_te (N_te, C, T), y_te (N_te,), mask (N_te, T), n_test_windows)
            or None if test_path is not provided
        x_scaler  – fit on train features only
        feat_cols – ordered list of feature column names (= channels)
        y_scaler  – target scaler or None
    """
    df_tr = pd.read_csv(train_path)
    drop_extra_train = drop_extra_train or ["EPOCH"]

    excluded  = set([time_col, target_col] + drop_extra_train)
    feat_cols = [c for c in df_tr.columns if c not in excluded]

    if target_col not in df_tr.columns:
        raise ValueError(f"target_col '{target_col}' not found in {train_path}")
    if target_col in feat_cols:
        raise ValueError("Target column leaked into features — check exclusion logic.")

    y_tr_series = df_tr[target_col].to_numpy().astype(np.float32)   # (T,)
    X_tr_cont   = df_tr[feat_cols].to_numpy().astype(np.float32)    # (T, C)

    # Scale features on TRAIN only
    x_scaler: Optional[StandardScaler] = StandardScaler() if scale_features else None
    if x_scaler is not None:
        X_tr_cont = x_scaler.fit_transform(X_tr_cont)

    # Optionally scale target on TRAIN only
    y_scaler: Optional[StandardScaler] = None
    if scale_target:
        y_scaler    = StandardScaler().fit(y_tr_series.reshape(-1, 1))
        y_tr_series = y_scaler.transform(y_tr_series.reshape(-1, 1)).reshape(-1).astype(np.float32)

    # Windowing → (N, C, T)
    Xw_tr, Yw_tr = _sliding_windows(X_tr_cont, y_tr_series, window=window, shift=shift)
    Xw_tr = Xw_tr.transpose(0, 2, 1)

    # Train / val split
    X_tr, X_va, y_tr, y_va = train_test_split(
        Xw_tr, Yw_tr, test_size=val_size, random_state=random_state
    )

    # Optional test windows (last-K) with labels
    test_tuple: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = None
    if test_path is not None and os.path.exists(test_path):
        df_te            = pd.read_csv(test_path)
        drop_extra_test  = drop_extra_test or ["EPOCH"]
        excluded_te      = set([time_col, target_col] + drop_extra_test)
        feat_cols_test   = [c for c in df_te.columns if c not in excluded_te]

        feat_cols_aligned = [c for c in feat_cols if c in feat_cols_test]
        if not feat_cols_aligned:
            raise ValueError("No overlapping feature columns between train and test after exclusions.")

        X_te_cont = df_te[feat_cols_aligned].to_numpy().astype(np.float32)
        if x_scaler is not None:
            X_te_cont = x_scaler.transform(X_te_cont)

        y_te_series = df_te[target_col].to_numpy().astype(np.float32)

        Xw_te, Yw_te, mask = _last_k_windows_with_targets(
            X_te_cont, y_te_series, window=window, shift=shift, k=last_k_test_windows
        )
        n_test_windows = int(Xw_te.shape[0])
        Xw_te          = Xw_te.transpose(0, 2, 1)  # (N, C, T)
        test_tuple     = (Xw_te, Yw_te, mask, n_test_windows)

    # Always return a fitted scaler for API stability
    return (
        X_tr.astype(np.float32),
        y_tr.astype(np.float32),
        X_va.astype(np.float32),
        y_va.astype(np.float32),
        test_tuple,
        x_scaler if x_scaler is not None else StandardScaler(),
        feat_cols,
        y_scaler,
    )


def make_ecoating_loaders(
    *,
    root: Optional[str] = None,
    train_filename: str = "manual_30min_norm.csv",
    test_filename: Optional[str] = "iiot_30min_norm.csv",
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    time_col: str = "TIME",
    target_col: str = "TP2",
    drop_extra_train: Optional[List[str]] = None,
    drop_extra_test: Optional[List[str]] = None,
    window: int = 30,
    shift: int = 1,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
    val_size: float = 0.2,
    random_state: int = 83,
    scale_features: bool = True,
    scale_target: bool = False,
    last_k_test_windows: int = 5,
    return_extras: bool = False,
):
    """
    Build PyTorch DataLoaders for the Ecoating regression dataset.

    Returns (return_extras=False):
        train_loader, val_loader, test_loader (or None), x_scaler, feat_cols

    Returns (return_extras=True):
        train_loader, val_loader, test_loader (or None), x_scaler, feat_cols,
        y_scaler (or None), n_test_windows (or None)
    """
    if train_path is None:
        if root is None:
            raise ValueError("Either provide train_path or root.")
        train_path = os.path.join(root, train_filename)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Ecoating train file not found: {train_path}")
    if not (0.0 < val_size < 1.0):
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")
    if test_path is None and root is not None and test_filename is not None:
        candidate = os.path.join(root, test_filename)
        if os.path.exists(candidate):
            test_path = candidate

    X_tr, y_tr, X_va, y_va, test_tuple, x_scaler, feat_cols, y_scaler = load_ecoating_as_arrays(
        train_path=train_path,
        test_path=test_path,
        time_col=time_col,
        target_col=target_col,
        drop_extra_train=drop_extra_train,
        drop_extra_test=drop_extra_test,
        window=window,
        shift=shift,
        val_size=val_size,
        random_state=random_state,
        scale_features=scale_features,
        scale_target=scale_target,
        last_k_test_windows=last_k_test_windows,
    )

    train_loader, val_loader = build_loaders(
        X_tr, y_tr, X_va, y_va,
        task="regression",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    test_loader: Optional[DataLoader] = None
    n_test_windows: Optional[int] = None
    if test_tuple is not None:
        X_te, y_te, _mask, n_test_windows = test_tuple
        test_ds     = TimeSeriesDataset(X_te, y_te, task="regression")
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    if return_extras:
        return train_loader, val_loader, test_loader, x_scaler, feat_cols, y_scaler, n_test_windows
    return train_loader, val_loader, test_loader, x_scaler, feat_cols
