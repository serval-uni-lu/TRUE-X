# datasets/adapters/cwru.py
import os
import re
import glob
import logging
from typing import Dict, List, Tuple, Optional, Literal, Sequence

import numpy as np
from scipy.io import loadmat

from datasets.common import Standardizer
from datasets.builders import build_loaders

logger = logging.getLogger(__name__)

FAULT_TYPE_MAP = {"normal": 0, "b": 1, "ir": 2, "or": 3}


def _parse_fault_type(stem: str) -> int:
    s = stem.lower()
    if "normal" in s or s.startswith("n_") or s.startswith("normal_"):
        return FAULT_TYPE_MAP["normal"]
    if s.startswith("b"):
        return FAULT_TYPE_MAP["b"]
    if s.startswith("ir"):
        return FAULT_TYPE_MAP["ir"]
    if s.startswith("or"):
        return FAULT_TYPE_MAP["or"]
    return FAULT_TYPE_MAP["normal"]


def _load_channel_vec(mat_path: str, channel_key_suffix: str) -> np.ndarray:
    """
    Load a 1D vibration array from a .mat file.
    Tries exact key match, then case-insensitive suffix match,
    then falls back to the longest 1D numeric vector found.
    """
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if channel_key_suffix in m:
        return np.asarray(m[channel_key_suffix], dtype=np.float64).ravel()
    for k, v in m.items():
        if isinstance(v, np.ndarray) and k.lower().endswith(channel_key_suffix.lower()):
            return np.asarray(v, dtype=np.float64).ravel()
    best = None
    for v in m.values():
        if isinstance(v, np.ndarray):
            arr = np.asarray(v, dtype=np.float64).ravel()
            if arr.ndim == 1 and arr.size > 16:
                if best is None or arr.size > best.size:
                    best = arr
    if best is None:
        raise KeyError(f"{channel_key_suffix} not found in {mat_path}")
    return best


def _sliding_windows(x: np.ndarray, win: int, stride: int) -> np.ndarray:
    """x: (T,) -> (num_windows, win)"""
    n = x.shape[0]
    if n < win:
        return np.empty((0, win), dtype=x.dtype)
    nwin = 1 + (n - win) // stride
    idx  = np.arange(nwin)[:, None] * stride + np.arange(win)[None, :]
    return x[idx].astype(np.float32, copy=False)


def load_cwru_as_arrays(
    root: str,
    *,
    sample_rate: Literal["12k", "48k"] = "12k",
    channels: Sequence[Literal["DE", "FE"]] = ("DE", "FE"),
    window_size: int = 2048,
    stride: int = 512,
    val_ratio: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Standardizer]:
    """
    Returns X_tr (N_tr, C, T), y_tr (N_tr,), X_va (N_va, C, T), y_va (N_va,), scaler.
    Splits at file level to prevent data leakage between train and val.
    """
    if not channels:
        raise ValueError("channels must be non-empty.")
    if not all(c in ("DE", "FE") for c in channels):
        raise ValueError("channels must contain only 'DE' and/or 'FE'")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    de_dir     = os.path.join(root, f"{sample_rate}_DE")
    fe_dir     = os.path.join(root, f"{sample_rate}_FE")
    normal_dir = os.path.join(root, "Normal")

    de_files = {os.path.basename(p): p for p in glob.glob(os.path.join(de_dir, "*.mat"))}
    fe_files = {os.path.basename(p): p for p in glob.glob(os.path.join(fe_dir, "*.mat"))}

    if os.path.isdir(normal_dir):
        for p in glob.glob(os.path.join(normal_dir, "*.mat")):
            de_files.setdefault(os.path.basename(p), p)

    stems = sorted({os.path.splitext(k)[0] for k in (*de_files.keys(), *fe_files.keys())})
    if not stems:
        raise FileNotFoundError(
            f"No .mat files found under {root} for {sample_rate}_DE / {sample_rate}_FE"
        )

    rng   = np.random.RandomState(random_state)
    idx   = np.arange(len(stems))
    rng.shuffle(idx)
    n_val       = max(1, int(len(idx) * val_ratio))
    val_stems   = {stems[i] for i in idx[:n_val]}
    train_stems = {stems[i] for i in idx[n_val:]}

    def _build_split(sel_stems: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for stem in sorted(sel_stems):
            name    = stem + ".mat"
            de_path = de_files.get(name)
            fe_path = fe_files.get(name)

            if "DE" in channels and de_path is None:
                continue
            if "FE" in channels and fe_path is None:
                continue

            waves, lengths = [], []
            if "DE" in channels:
                sig = _load_channel_vec(de_path, "DE_time")
                waves.append(sig)
                lengths.append(sig.shape[0])
            if "FE" in channels:
                sig = _load_channel_vec(fe_path, "FE_time")
                waves.append(sig)
                lengths.append(sig.shape[0])

            L = min(lengths)
            if L < window_size:
                continue

            ch_windows = [_sliding_windows(w[:L], window_size, stride) for w in waves]
            Nw = ch_windows[0].shape[0]
            if any(w.shape[0] != Nw for w in ch_windows):
                continue

            Xw = np.stack(ch_windows, axis=1)  # (Nw, C, T)
            yw = np.full((Nw,), _parse_fault_type(stem), dtype=np.int64)

            X_list.append(Xw.astype(np.float32, copy=False))
            y_list.append(yw)

        if not X_list:
            raise RuntimeError(
                "No windows produced — check paths, channels, or window/stride settings."
            )
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    X_tr, y_tr = _build_split(train_stems)
    X_va, y_va = _build_split(val_stems)

    scaler = Standardizer()
    if scale:
        scaler.fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_va = scaler.transform(X_va)

    logger.info(
        "[CWRU] train=%d windows | val=%d windows | shape (C=%d, T=%d)",
        X_tr.shape[0], X_va.shape[0], X_tr.shape[1], X_tr.shape[2],
    )
    return X_tr, y_tr, X_va, y_va, scaler


def make_cwru_loaders(
    root: str,
    *,
    sample_rate: Literal["12k", "48k"] = "12k",
    channels: Sequence[Literal["DE", "FE"]] = ("DE", "FE"),
    window_size: int = 2048,
    stride: int = 512,
    batch_size: int = 64,
    num_workers: int = 0,
    val_ratio: float = 0.1,
    random_state: int = 42,
    scale: bool = True,
):
    """
    Build DataLoaders for the CWRU bearing classification dataset.

    Returns:
        train_loader, val_loader, scaler (Standardizer)
    """
    X_tr, y_tr, X_va, y_va, scaler = load_cwru_as_arrays(
        root=root,
        sample_rate=sample_rate,
        channels=channels,
        window_size=window_size,
        stride=stride,
        val_ratio=val_ratio,
        random_state=random_state,
        scale=scale,
    )
    train_loader, val_loader = build_loaders(
        X_tr, y_tr, X_va, y_va,
        task="classification",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader, scaler
