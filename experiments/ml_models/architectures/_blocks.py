from __future__ import annotations

import torch
import torch.nn as nn


def conv1d_same(in_ch: int, out_ch: int, k: int, bias: bool = False):
    """1D convolution with 'same' output length across PyTorch versions."""
    try:
        return nn.Conv1d(in_ch, out_ch, kernel_size=k, padding="same", bias=bias)
    except TypeError:  # older PyTorch
        pad_total = k - 1
        left = pad_total // 2
        right = pad_total - left
        return nn.Sequential(
            nn.ConstantPad1d((left, right), 0.0),
            nn.Conv1d(in_ch, out_ch, kernel_size=k, bias=bias),
        )


class _PatchEmbedding(nn.Module):
    """Conv1d-based patch embedding: (B,C,T)->(B,L,d)"""
    def __init__(self, c_in, d_model, patch_len=16, stride=8, bias=False):
        super().__init__()
        if patch_len < 1 or stride < 1:
            raise ValueError("patch_len and stride must be >= 1")
        self.proj = nn.Conv1d(c_in, d_model, kernel_size=patch_len, stride=stride, bias=bias)

    def forward(self, x):  # (B,C,T)
        z = self.proj(x)   # (B,d,L)
        return z.transpose(1, 2)  # (B,L,d)
