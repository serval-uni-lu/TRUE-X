# datasets/adapters/cmapss.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from datasets.common import ensure_channels_first, make_windows, window_targets
from datasets.builders import build_loaders, TimeSeriesDataset


_DEFAULT_DROP_COLS = [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23]


def _process_targets(T: int, early_rul: Optional[int] = 120) -> np.ndarray:
    if early_rul is None:
        return np.arange(T - 1, -1, -1)
    dur = T - early_rul
    if dur <= 0:
        return np.arange(T - 1, -1, -1)
    return np.concatenate([np.full(dur, early_rul), np.arange(early_rul - 1, -1, -1)])


_VALID_FD = {"FD001", "FD002", "FD003", "FD004"}


def _read_split(
    train_path: str, test_path: str, rul_path: str, drop_cols: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for p in (train_path, test_path, rul_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"CMAPSS file not found: {p}")
    tr  = pd.read_csv(train_path, sep=r"\s+", header=None)
    te  = pd.read_csv(test_path,  sep=r"\s+", header=None)
    rul = pd.read_csv(rul_path,   sep=r"\s+", header=None)[0].values  # (n_test_engines,)

    engine_tr = tr[0].values
    engine_te = te[0].values

    X_tr = tr.drop(columns=drop_cols).values
    X_te = te.drop(columns=drop_cols).values
    return engine_tr, X_tr, engine_te, X_te, rul


def load_cmapss_as_arrays(
    train_path: str,
    test_path: str,
    rul_path: str,
    window: int = 30,
    shift: int = 1,
    early_rul: int = 120,
    drop_cols: Optional[List[int]] = None,
    val_size: float = 0.2,
    seed: int = 83,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, np.ndarray, np.ndarray]:
    """
    Returns:
        X_tr (N_tr, C, T), y_tr (N_tr,)
        X_va (N_va, C, T), y_va (N_va,)
        scaler  – fit on TRAIN features only
        eng_tr  – per-row engine ids for raw train
        eng_te  – per-row engine ids for raw test
    """
    if not (0.0 < val_size < 1.0):
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")
    if drop_cols is None:
        drop_cols = _DEFAULT_DROP_COLS

    eng_tr, X_tr_raw, eng_te, _X_te_raw, _true_rul = _read_split(
        train_path, test_path, rul_path, drop_cols
    )

    # standardize using TRAIN only
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr_raw = scaler.transform(X_tr_raw)

    # per-engine → windowed train
    Xw_list, Yw_list = [], []
    for eid in np.unique(eng_tr):
        seq   = X_tr_raw[eng_tr == eid]                      # (T, F)
        T_len = seq.shape[0]
        y_full = _process_targets(T_len, early_rul=early_rul) # (T,)
        Xw     = make_windows(seq, window, shift)             # (Nw, W, F)
        Yw     = window_targets(y_full, window, shift)        # (Nw,)
        Xw_list.append(Xw)
        Yw_list.append(Yw)

    Xw = np.concatenate(Xw_list)  # (N, W, F)
    Yw = np.concatenate(Yw_list)  # (N,)

    X_tr, X_va, y_tr, y_va = train_test_split(Xw, Yw, test_size=val_size, random_state=seed)

    # (N, W, F) → (N, C=F, T=W)
    X_tr = ensure_channels_first(X_tr, channels_first=False)
    X_va = ensure_channels_first(X_va, channels_first=False)

    return (
        X_tr.astype(np.float32), y_tr.astype(np.float32),
        X_va.astype(np.float32), y_va.astype(np.float32),
        scaler, eng_tr, eng_te,
    )


def _last_k_test_windows_with_labels(
    X_te_rows_scaled: np.ndarray,
    eng_te_rows: np.ndarray,
    true_rul_per_engine: np.ndarray,
    *,
    window: int,
    shift: int,
    k: int,
    early_rul: Optional[int] = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X_te  (N, C, T)
        y_te  (N,)     per-window RUL at the end of each window
        mask  (N, T)   True for valid (non-padded) positions
    """
    engines_sorted = np.sort(np.unique(eng_te_rows))

    if len(engines_sorted) != len(true_rul_per_engine):
        raise ValueError("Mismatch between number of test engines and RUL entries.")

    eid_to_rul = {eid: int(true_rul_per_engine[i]) for i, eid in enumerate(engines_sorted)}

    Xw_all, Yw_all, masks_all = [], [], []

    for eid in engines_sorted:
        seq        = X_te_rows_scaled[eng_te_rows == eid]  # (T, F)
        T_len, F   = seq.shape
        rul_last   = eid_to_rul[eid]

        if T_len < window:
            pad  = window - T_len
            xpad = np.vstack([seq, np.zeros((pad, F), dtype=seq.dtype)])
            mask = np.hstack([np.ones(T_len, dtype=bool), np.zeros(pad, dtype=bool)])
            y    = float(min(rul_last, early_rul) if early_rul is not None else rul_last)
            Xw_all.append(xpad[None, ...])
            Yw_all.append(np.array([y], dtype=np.float32))
            masks_all.append(mask[None, ...])
        else:
            max_n       = (T_len - window) // shift + 1
            k_eff       = min(k, max_n)
            start_first = T_len - (k_eff - 1) * shift - window

            Xw   = np.empty((k_eff, window, F), dtype=np.float32)
            Yw   = np.empty((k_eff,), dtype=np.float32)
            mask = np.ones((k_eff, window), dtype=bool)

            for i in range(k_eff):
                s     = start_first + i * shift
                e     = s + window
                Xw[i] = seq[s:e]
                delta  = (T_len - 1) - (e - 1)
                rul    = rul_last + delta
                if early_rul is not None:
                    rul = min(rul, early_rul)
                Yw[i] = float(rul)

            Xw_all.append(Xw)
            Yw_all.append(Yw)
            masks_all.append(mask)

    X_te = np.concatenate(Xw_all, axis=0).transpose(0, 2, 1).astype(np.float32)  # (N, C, T)
    y_te = np.concatenate(Yw_all, axis=0).astype(np.float32)
    mask = np.concatenate(masks_all, axis=0)
    return X_te, y_te, mask


def make_cmapss_loaders(
    root: str,
    fd: str = "FD001",
    *,
    window: int = 30,
    shift: int = 1,
    early_rul: int = 120,
    batch_size: int = 64,
    num_workers: int = 4,
    return_test: bool = False,
    k_last: int = 5,
):
    """
    Build DataLoaders for a CMAPSS sub-dataset (FD001–FD004).

    Returns (return_test=False):
        train_loader, val_loader

    Returns (return_test=True):
        train_loader, val_loader, test_loader, true_rul, mask, scaler
    """
    if fd not in _VALID_FD:
        raise ValueError(f"fd must be one of {sorted(_VALID_FD)}, got '{fd}'")

    train_path = f"{root}/train_{fd}.txt"
    test_path  = f"{root}/test_{fd}.txt"
    rul_path   = f"{root}/RUL_{fd}.txt"

    X_tr, y_tr, X_va, y_va, scaler, _eng_tr, _eng_te = load_cmapss_as_arrays(
        train_path, test_path, rul_path,
        window=window, shift=shift, early_rul=early_rul,
    )

    train_loader, val_loader = build_loaders(
        X_tr, y_tr, X_va, y_va,
        task="regression",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if not return_test:
        return train_loader, val_loader

    # Re-read raw test rows and apply the TRAIN scaler
    _eng_tr, _X_tr_rows, eng_te_rows, X_te_rows, true_rul = _read_split(
        train_path, test_path, rul_path, drop_cols=_DEFAULT_DROP_COLS
    )
    X_te_rows_scaled = scaler.transform(X_te_rows)

    X_te, y_te, mask = _last_k_test_windows_with_labels(
        X_te_rows_scaled, eng_te_rows, true_rul,
        window=window, shift=shift, k=k_last, early_rul=early_rul,
    )

    test_ds = TimeSeriesDataset(X_te, y_te, task="regression")
    test_loader = DataLoader(
        test_ds,
        batch_size=max(256, batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, true_rul, mask, scaler
