# datasets/builders.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Tuple

Task = Literal["classification", "regression"]


class TimeSeriesDataset(Dataset):
    """
    Wraps (N, C, T) arrays into a PyTorch Dataset.

    y shape:
      - classification : int indices (N,) or (N, L) for multi-task
      - regression     : float32 (N,) or (N, L) for multi-task
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, task: Task):
        if X.ndim != 3:
            raise ValueError(f"X must be (N, C, T), got shape {X.shape}")
        self.X    = X.astype(np.float32, copy=False)
        self.y    = y
        self.task = task

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        if self.task == "classification":
            y = torch.as_tensor(self.y[idx]).long()
        else:
            y = torch.as_tensor(self.y[idx]).float()
        return {"sequence": x, "label": y}


def build_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: Task,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    ds_tr = TimeSeriesDataset(X_train, y_train, task)
    ds_va = TimeSeriesDataset(X_val,   y_val,   task)

    train_loader = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader
