from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._blocks import conv1d_same, _PatchEmbedding


class EncoderTSC(nn.Module):
    """
    Convolutional encoder with *optional* InstanceNorm + PReLU + Dropout + attention split.
    Input:  (B, C, T)   Output: logits (B, nb_classes)
    """
    def __init__(
        self,
        input_shape,
        nb_classes,
        filters=(128, 256, 512),
        kernels=(5, 11, 21),
        dropout=0.2,
        use_instancenorm: bool = True,
    ):
        super().__init__()
        T, C_in = input_shape
        f1, f2, f3 = filters
        k1, k2, k3 = kernels

        Norm = nn.InstanceNorm1d if use_instancenorm else nn.BatchNorm1d

        self.conv1 = nn.Conv1d(C_in, f1, kernel_size=k1, padding=k1 // 2, bias=False)
        self.n1 = Norm(f1)
        self.a1 = nn.PReLU()
        self.d1 = nn.Dropout(dropout)
        self.p1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(f1,  f2, kernel_size=k2, padding=k2 // 2, bias=False)
        self.n2 = Norm(f2)
        self.a2 = nn.PReLU()
        self.d2 = nn.Dropout(dropout)
        self.p2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(f2,  f3, kernel_size=k3, padding=k3 // 2, bias=False)
        self.n3 = Norm(f3)
        self.a3 = nn.PReLU()
        self.d3 = nn.Dropout(dropout)

        if f3 % 2 != 0:
            raise ValueError("filters[2] must be even for attention split.")
        self.data_ch = f3 // 2
        self.softmax_t = nn.Softmax(dim=2)

        with torch.no_grad():
            dummy = torch.zeros(1, C_in, T)
            z = self._features(dummy)
            d = z[:, :self.data_ch, :]
            a = self.softmax_t(z[:, self.data_ch:, :])
            att = d * a
            flat = att.numel()
        self.fc = nn.Linear(flat, 256)
        self.ln = nn.LayerNorm(256)
        self.out = nn.Linear(256, nb_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")

    def _features(self, x):
        x = self.p1(self.d1(self.a1(self.n1(self.conv1(x)))))
        x = self.p2(self.d2(self.a2(self.n2(self.conv2(x)))))
        x = self.d3(self.a3(self.n3(self.conv3(x))))
        return x

    def forward(self, x):
        z = self._features(x)
        d = z[:, :self.data_ch, :]
        a = self.softmax_t(z[:, self.data_ch:, :])
        att = d * a
        flat = att.reshape(att.size(0), -1)
        h = torch.sigmoid(self.fc(flat))
        h = self.ln(h)
        return self.out(h)


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


class Classifier_FCN(nn.Module):
    """FCN: 3 conv blocks (no pooling) + GAP + Linear."""
    def __init__(self, input_shape, nb_classes, channels=(128, 256, 128), kernels=(9, 5, 3)):
        super().__init__()
        T, C_in = input_shape
        c1, c2, c3 = channels
        k1, k2, k3 = kernels
        self.c1 = conv1d_same(C_in, c1, k1, bias=False)
        self.b1 = nn.BatchNorm1d(c1)
        self.c2 = conv1d_same(c1, c2, k2, bias=False)
        self.b2 = nn.BatchNorm1d(c2)
        self.c3 = conv1d_same(c2, c3, k3, bias=False)
        self.b3 = nn.BatchNorm1d(c3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c3, nb_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


class _MCDCNNBranch(nn.Module):
    def __init__(self, use_bn=False, k1=5, f1=8, k2=5, f2=8, pool=2):
        super().__init__()
        self.c1 = conv1d_same(1, f1, k1)
        self.bn1 = nn.BatchNorm1d(f1) if use_bn else nn.Identity()
        self.p1 = nn.MaxPool1d(pool, pool)
        self.c2 = conv1d_same(f1, f2, k2)
        self.bn2 = nn.BatchNorm1d(f2) if use_bn else nn.Identity()
        self.p2 = nn.MaxPool1d(pool, pool)
        self.flat = nn.Flatten()

    def forward(self, x):  # (B,1,T)
        x = F.relu(self.bn1(self.c1(x)))
        x = self.p1(x)
        x = F.relu(self.bn2(self.c2(x)))
        x = self.p2(x)
        return self.flat(x)


class Classifier_MCDCNN(nn.Module):
    """Per-channel branches -> concat -> dense head -> logits."""
    def __init__(self, input_shape, nb_classes, branch_filters=(8, 8), branch_kernels=(5, 5), pool=2,
                 dense_units=732, dropout=0.0, use_bn=False):
        super().__init__()
        T, C_in = input_shape
        f1, f2 = branch_filters
        k1, k2 = branch_kernels
        self.branches = nn.ModuleList([_MCDCNNBranch(use_bn=use_bn, k1=k1, f1=f1, k2=k2, f2=f2, pool=pool) for _ in range(C_in)])
        with torch.no_grad():
            d = torch.zeros(1, C_in, T)
            feats = [self.branches[i](d[:, i:i+1, :]) for i in range(C_in)]
            concat_dim = torch.cat(feats, dim=1).shape[1]
        self.fc1 = nn.Linear(concat_dim, dense_units)
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(dense_units, nb_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        feats = [self.branches[i](x[:, i:i+1, :]) for i in range(x.size(1))]
        z = torch.cat(feats, dim=1)
        z = F.relu(self.fc1(z))
        z = self.do(z)
        return self.fc2(z)


class _TimeCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, pool=3):
        super().__init__()
        self.c = conv1d_same(in_ch, out_ch, k, bias=True)
        self.p = nn.AvgPool1d(pool, pool)

    def forward(self, x):
        x = torch.sigmoid(self.c(x))
        return self.p(x)


class Classifier_TIMECNN(nn.Module):
    """Sigmoid convs + avg-pool + dense head baseline."""
    def __init__(self, input_shape, nb_classes, filters=(6, 12), kernel_size=7, pool_size=3, dense_units=128, dropout=0.0):
        super().__init__()
        T, C_in = input_shape
        chs = [C_in] + list(filters)
        blocks = [_TimeCNNBlock(chs[i], chs[i+1], kernel_size, pool_size) for i in range(len(filters))]
        self.blocks = nn.Sequential(*blocks)
        with torch.no_grad():
            d = torch.zeros(1, C_in, T)
            z = self.blocks(d)
            flat = z.numel()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(flat, dense_units)
        self.do  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(dense_units, nb_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="sigmoid")
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        z = self.blocks(x)
        z = self.flat(z)
        z = torch.sigmoid(self.fc1(z))
        z = self.do(z)
        return self.fc2(z)


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
        tok = self.patch(x)                # (B,L,E)
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
    def __init__(
        self,
        input_shape,
        nb_classes=None,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.1,
        temporal_pool="last",
    ):
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


class Classifier_GRU(nn.Module):
    """Plain GRU classifier."""
    def __init__(
        self,
        input_shape,
        nb_classes=None,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.1,
        temporal_pool="last",
    ):
        super().__init__()
        T, C = input_shape
        self.temporal_pool = temporal_pool
        self.bi = bidirectional
        self.gru = nn.GRU(C, hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=(dropout if num_layers > 1 else 0.0))
        feat = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(feat)
        self.head = nn.Linear(feat, nb_classes)
        for n, p in self.gru.named_parameters():
            if "weight_ih" in n:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in n:
                nn.init.orthogonal_(p)
            elif "bias" in n:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        out, h_n = self.gru(x)
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
