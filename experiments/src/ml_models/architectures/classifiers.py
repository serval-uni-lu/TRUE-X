from __future__ import annotations

import torch
import torch.nn as nn

from ._blocks import conv1d_same, _PatchEmbedding


class ResNetBlock1D(nn.Module):
    """Residual block: 3 convs (same T) + optional 1x1 skip proj."""
    def __init__(self, in_ch, out_ch, kernel_sizes=(9, 5, 3)):
        super().__init__()
        k1, k2, k3 = kernel_sizes
        if not all(k % 2 == 1 for k in (k1, k2, k3)):
            raise ValueError(f"All kernel sizes must be odd, got {kernel_sizes}")
        self.c1 = conv1d_same(in_ch, out_ch, k1, bias=False)
        self.b1 = nn.BatchNorm1d(out_ch)
        self.c2 = conv1d_same(out_ch, out_ch, k2, bias=False)
        self.b2 = nn.BatchNorm1d(out_ch)
        self.c3 = conv1d_same(out_ch, out_ch, k3, bias=False)
        self.b3 = nn.BatchNorm1d(out_ch)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch))

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu_out = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        y = self.relu1(self.b1(self.c1(x)))
        y = self.relu2(self.b2(self.c2(y)))
        y = self.b3(self.c3(y))
        if self.proj is not None:
            identity = self.proj(identity)
        return self.relu_out(y + identity)


class Classifier_RESNET(nn.Module):
    """3-block ResNet (GAP + Linear)."""
    def __init__(self, input_shape, nb_classes, block_channels=(64, 128, 128), kernel_sizes=(9, 5, 3)):
        super().__init__()
        T, C = input_shape
        chs = [C] + list(block_channels)
        blocks = []
        for i in range(3):
            blocks.append(ResNetBlock1D(chs[i], chs[i + 1], kernel_sizes=kernel_sizes))
        self.backbone = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(chs[-1], nb_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        z = self.backbone(x)
        z = self.gap(z).squeeze(-1)
        return self.fc(z)


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch, nb_filters=32, kernel_sizes=(9, 19, 39), bottleneck=True, bottleneck_channels=32):
        super().__init__()
        if not all(k % 2 == 1 for k in kernel_sizes):
            raise ValueError(f"All kernel sizes must be odd, got {kernel_sizes}")
        if bottleneck and in_ch > 1:
            self.bottleneck = nn.Conv1d(in_ch, bottleneck_channels, 1, bias=False)
            mid = bottleneck_channels
        else:
            self.bottleneck = None
            mid = in_ch
        self.branches = nn.ModuleList([nn.Conv1d(mid, nb_filters, k, padding=k // 2, bias=False) for k in kernel_sizes])
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_ch, nb_filters, 1, bias=False))
        self.bn = nn.BatchNorm1d((len(kernel_sizes) + 1) * nb_filters)
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = (len(kernel_sizes) + 1) * nb_filters

    def forward(self, x):
        xb = self.bottleneck(x) if self.bottleneck is not None else x
        outs = [b(xb) for b in self.branches]
        outs.append(self.pool_branch(x))
        y = torch.cat(outs, dim=1)
        return self.relu(self.bn(y))


class InceptionResidualBlock(nn.Module):
    def __init__(self, in_ch, nb_filters=32, kernel_sizes=(9, 19, 39), bottleneck=True, bottleneck_channels=32):
        super().__init__()
        self.b1 = InceptionBlock1D(in_ch, nb_filters, kernel_sizes, bottleneck, bottleneck_channels)
        self.b2 = InceptionBlock1D(self.b1.out_channels, nb_filters, kernel_sizes, bottleneck, bottleneck_channels)
        self.b3 = InceptionBlock1D(self.b2.out_channels, nb_filters, kernel_sizes, bottleneck, bottleneck_channels)
        out_ch = self.b3.out_channels
        self.match = (in_ch != out_ch)
        if self.match:
            self.proj = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch))
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = out_ch

    def forward(self, x):
        r = x
        y = self.b1(x)
        y = self.b2(y)
        y = self.b3(y)
        if self.match:
            r = self.proj(r)
        return self.relu(y + r)


class InceptionTime(nn.Module):
    """Classifier: InceptionTime backbone + GAP + Linear."""
    def __init__(self, input_shape, nb_classes, n_residual_blocks=3, nb_filters=32, kernel_sizes=(9, 19, 39), bottleneck=True):
        super().__init__()
        T, C = input_shape
        ch = C
        blocks = []
        for _ in range(n_residual_blocks):
            rb = InceptionResidualBlock(ch, nb_filters, kernel_sizes, bottleneck)
            blocks.append(rb)
            ch = rb.out_channels
        self.backbone = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ch, nb_classes)

    def forward(self, x):
        z = self.backbone(x)
        z = self.gap(z).squeeze(-1)
        return self.fc(z)


class TST(nn.Module):
    """Transformer classifier over patch tokens. Expects (B,C,T)."""
    def __init__(self, input_shape, nb_classes, d_model=128, n_heads=8, num_layers=4, d_ff=256,
                 dropout=0.1, patch_len=16, stride=8, use_cls_token=True, emb_dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        T, C = input_shape
        if T < patch_len:
            raise ValueError(f"T={T} < patch_len={patch_len}")
        self.patch = _PatchEmbedding(C, d_model, patch_len, stride)
        L = (T - patch_len) // stride + 1
        self.use_cls = use_cls_token
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None
        pe_len = L + (1 if use_cls_token else 0)
        self.pos = nn.Parameter(torch.zeros(1, pe_len, d_model))
        self.drop = nn.Dropout(emb_dropout)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, nb_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        B = x.size(0)
        tok = self.patch(x)
        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)
            tok = torch.cat([cls, tok], dim=1)
        tok = tok + self.pos[:, :tok.size(1), :]
        tok = self.drop(tok)
        z = self.encoder(tok)
        rep = z[:, 0] if self.use_cls else z.mean(dim=1)
        rep = self.norm(rep)
        return self.fc(rep)


class Classifier_LSTM(nn.Module):
    """Plain LSTM classifier."""
    def __init__(self, input_shape, nb_classes=None, hidden_size=128, num_layers=2,
                 bidirectional=True, dropout=0.1, temporal_pool="last"):
        super().__init__()
        T, C = input_shape
        self.temporal_pool = temporal_pool
        self.bi = bidirectional
        self.lstm = nn.LSTM(C, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=(dropout if num_layers > 1 else 0.0))
        feat = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(feat)
        self.head = nn.Linear(feat, nb_classes)
        for n, p in self.lstm.named_parameters():
            if "weight_ih" in n:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in n:
                nn.init.orthogonal_(p)
            elif "bias" in n:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):  # (B,C,T)
        x = x.transpose(1, 2)  # (B,T,C)
        out, (h_n, _) = self.lstm(x)
        if self.temporal_pool == "last":
            if self.bi:
                fwd, bwd = h_n[-2], h_n[-1]
                feats = torch.cat([fwd, bwd], dim=1)
            else:
                feats = h_n[-1]
        elif self.temporal_pool == "mean":
            feats = out.mean(dim=1)
        elif self.temporal_pool == "max":
            feats, _ = out.max(dim=1)
        else:
            raise ValueError(self.temporal_pool)
        feats = self.norm(feats)
        return self.head(feats)


class LogisticClassifier(nn.Module):
    """Multinomial logistic regression over flattened window."""
    def __init__(self, input_shape, nb_classes, use_bias=True):
        super().__init__()
        T, C = input_shape
        self.fc = nn.Linear(C * T, nb_classes, bias=use_bias)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        z = x.reshape(x.size(0), -1)
        return self.fc(z)
