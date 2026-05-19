# datasets/adapters/hydraulic.py
import os
import numpy as np
from typing import List, Tuple, Dict, Optional, Literal, Union
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

from datasets.common import Standardizer
from datasets.builders import build_loaders

T_CYCLE = 60.0


def _load_txt_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path)


def _interp_row(row: np.ndarray, n_i: int) -> np.ndarray:
    T_raw = row.shape[0]
    if T_raw == n_i:
        return row
    x  = np.linspace(0.0, T_CYCLE, T_raw)
    xi = np.linspace(0.0, T_CYCLE, n_i)
    return interp1d(x, row, kind="linear")(xi)


def _encode_labels(raw: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    raw: (N, L) integer-like values from profile columns.
    Returns:
      y_enc: (N, L) in 0..K-1 per column
      label_maps: {col_idx: np.array of unique sorted raw values}
    """
    L = raw.shape[1]
    y_list, maps = [], {}
    for j in range(L):
        v = raw[:, j].astype(int)
        uniq = np.sort(np.unique(v))
        enc = np.searchsorted(uniq, v)  # raw -> class index
        y_list.append(enc[:, None])
        maps[j] = uniq
    return np.concatenate(y_list, axis=1), maps

def load_hydraulic_as_arrays(
    data_dir: str,
    channels: List[str],
    label_cols: Union[int, List[int]],
    *,
    seq_len: int = 50,
    test_size: float = 0.1,
    random_state: int = 42,
    filter_unstable: bool = False,
    stable_col: int = 4,
    stratify_on: Literal["first", "joint"] = "first",
    scale: bool = True,
    squeeze_single_target: bool = True,
):
    """
    Returns:
      X_tr (N_tr,C,T), y_tr (N_tr,) or (N_tr,L)
      X_va (N_va,C,T), y_va (N_va,) or (N_va,L)
      scaler (Standardizer), label_maps (dict col->raw_values)
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Hydraulic data directory not found: {data_dir}")
    if not channels:
        raise ValueError("channels must be a non-empty list of sensor names.")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    label_cols_list = [label_cols] if isinstance(label_cols, int) else list(label_cols)

    data   = {}
    needed = set(channels) | {"profile"}
    for name in needed:
        data[name] = _load_txt_matrix(os.path.join(data_dir, f"{name}.txt"))

    prof      = data["profile"]
    keep_mask = np.ones(prof.shape[0], dtype=bool)
    if filter_unstable and prof.shape[1] > stable_col:
        keep_mask &= (prof[:, stable_col].astype(int) == 0)

    X_list = []
    for c in channels:
        raw      = data[c][keep_mask]
        interped = np.vstack([_interp_row(r, seq_len) for r in raw])  # (N, seq_len)
        X_list.append(interped[:, None, :])                           # (N, 1, T)
    X = np.concatenate(X_list, axis=1)                                # (N, C, T)

    y_raw              = prof[keep_mask][:, label_cols_list]
    y_multi, label_maps = _encode_labels(y_raw)
    y = y_multi[:, 0] if (y_multi.shape[1] == 1 and squeeze_single_target) else y_multi

    idx = np.arange(X.shape[0])
    if y.ndim == 1:
        y_strat = y
    elif stratify_on == "joint":
        bases   = np.array([y[:, j].max() + 1 for j in range(y.shape[1])], dtype=int)
        mult    = np.concatenate([[1], np.cumprod(bases[:-1])])
        y_strat = (y * mult).sum(axis=1)
    else:
        y_strat = y[:, 0]

    X_tr_idx, X_va_idx, y_tr, y_va = train_test_split(
        idx, y, test_size=test_size, random_state=random_state, stratify=y_strat
    )

    scaler = Standardizer()
    if scale:
        scaler.fit(X[X_tr_idx])
        X_tr = scaler.transform(X[X_tr_idx])
        X_va = scaler.transform(X[X_va_idx])
    else:
        X_tr, X_va = X[X_tr_idx], X[X_va_idx]

    return X_tr, y_tr, X_va, y_va, scaler, label_maps

def make_hydraulic_loaders(
    data_dir: str,
    channels: List[str],
    label_cols: Union[int, List[int]],
    seq_len: int = 50,
    batch_size: int = 64,
    num_workers: int = 4,
    *,
    test_size: float = 0.1,
    random_state: int = 42,
    filter_unstable: bool = False,
    stratify_on: Literal["first", "joint"] = "first",
    squeeze_single_target: bool = True,
):
    """
    Build DataLoaders for the Hydraulic classification dataset.

    Returns:
        train_loader, val_loader, scaler (Standardizer), label_maps (dict col->raw_values)
    """
    X_tr, y_tr, X_va, y_va, scaler, label_maps = load_hydraulic_as_arrays(
        data_dir, channels, label_cols,
        seq_len=seq_len,
        test_size=test_size,
        random_state=random_state,
        filter_unstable=filter_unstable,
        stratify_on=stratify_on,
        squeeze_single_target=squeeze_single_target,
    )
    train_loader, val_loader = build_loaders(
        X_tr, y_tr, X_va, y_va,
        task="classification",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader, scaler, label_maps
