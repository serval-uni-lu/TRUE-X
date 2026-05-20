from __future__ import annotations

import torch
import torch.nn as nn

from ._blocks import conv1d_same


class _NAMTSChannelNet(nn.Module):
    """
    Per-channel temporal subnet: depthwise conv stack -> GAP -> feature vector (B, d).
    Keeps channels independent for additive transparency.
    """
    def __init__(self, k1=9, k2=5, k3=3, d_hidden=32, dropout=0.0):
        super().__init__()
        self.c1 = conv1d_same(1, 16, k1, bias=False)
        self.b1 = nn.BatchNorm1d(16)
        self.a1 = nn.ReLU(inplace=True)
        self.d1 = nn.Dropout(dropout)
        self.c2 = conv1d_same(16, 32, k2, bias=False)
        self.b2 = nn.BatchNorm1d(32)
        self.a2 = nn.ReLU(inplace=True)
        self.d2 = nn.Dropout(dropout)
        self.c3 = conv1d_same(32, 32, k3, bias=False)
        self.b3 = nn.BatchNorm1d(32)
        self.a3 = nn.ReLU(inplace=True)
        self.d3 = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, d_hidden)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x_ch):  # x_ch: (B, 1, T)
        z = self.d1(self.a1(self.b1(self.c1(x_ch))))
        z = self.d2(self.a2(self.b2(self.c2(z))))
        z = self.d3(self.a3(self.b3(self.c3(z))))
        z = self.gap(z).squeeze(-1)          # (B, 32)
        h = torch.relu(self.fc(z))           # (B, d_hidden)
        return h


class NAMTS_Classifier(nn.Module):
    """
    Additive-by-channel classifier:
      logits_k(x) = b_k + sum_c  h_c(x_c)^T W_c,k
    """
    def __init__(self, input_shape, nb_classes, d_hidden=32, dropout=0.0):
        super().__init__()
        T, C = input_shape
        self.C = C
        self.nb_classes = nb_classes
        self.subnets = nn.ModuleList([_NAMTSChannelNet(d_hidden=d_hidden, dropout=dropout) for _ in range(C)])
        self.W = nn.Parameter(torch.randn(C, d_hidden, nb_classes) * 0.02)
        self.b = nn.Parameter(torch.zeros(nb_classes))

    def forward(self, x):  # (B, C, T)
        B = x.size(0)
        contrib = []
        for c in range(self.C):
            h_c = self.subnets[c](x[:, c:c+1, :])       # (B, d)
            logit_c = h_c @ self.W[c]                   # (B, K)
            contrib.append(logit_c)
        logits = torch.stack(contrib, dim=0).sum(dim=0) + self.b
        return logits


class NAMTS_Regressor(nn.Module):
    """
    Additive-by-channel regressor:
      y(x) = b + sum_c  w_c^T h_c(x_c)
    """
    def __init__(self, input_shape, d_hidden=32, dropout=0.0):
        super().__init__()
        T, C = input_shape
        self.C = C
        self.subnets = nn.ModuleList([_NAMTSChannelNet(d_hidden=d_hidden, dropout=dropout) for _ in range(C)])
        self.W = nn.Parameter(torch.randn(C, d_hidden) * 0.02)   # (C, d)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # (B, C, T)
        contrib = []
        for c in range(self.C):
            h_c = self.subnets[c](x[:, c:c+1, :])                # (B, d)
            y_c = (h_c * self.W[c].unsqueeze(0)).sum(dim=1)      # (B,)
            contrib.append(y_c)
        y = torch.stack(contrib, dim=0).sum(dim=0) + self.b      # (B,)
        return y


class _SoftDecisionTreeBase(nn.Module):
    """
    Differentiable binary decision tree with depth=D.
    Internal nodes: p_i(x) = sigmoid((w_i^T x + b_i)/tau)
    Leaf output = sum_leaves prob(leaf|x) * leaf_value
    """
    def __init__(self, in_features, depth=3, tau=2.0):
        super().__init__()
        self.in_features = in_features
        self.depth = int(depth)
        self.n_internal = 2 ** self.depth - 1
        self.n_leaves = 2 ** self.depth
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))
        self.W = nn.Parameter(torch.randn(self.n_internal, in_features) * 0.02)
        self.b = nn.Parameter(torch.zeros(self.n_internal))

        paths = []
        for leaf in range(self.n_leaves):
            bits = [(leaf >> d) & 1 for d in reversed(range(self.depth))]
            node_indices = []
            idx = 0
            for bit in bits:
                node_indices.append(idx)
                idx = 2 * idx + 1 + bit
            row = torch.full((self.n_internal,), -1, dtype=torch.long)
            for j, ni in enumerate(node_indices):
                row[ni] = bits[j]
            paths.append(row)
        self.register_buffer("path_matrix", torch.stack(paths, dim=0))     # (L, I)
        self.register_buffer("visit_mask", (self.path_matrix >= 0))        # (L, I)

    def _routing(self, x):  # x: (B, D)
        tau = torch.exp(self.log_tau).clamp(min=1e-3)
        logits = (x @ self.W.t()) + self.b            # (B, I)
        p = torch.sigmoid(logits / tau)               # (B, I) prob RIGHT
        return p

    def _leaf_probs(self, p):  # p: (B, I)
        B, I = p.shape
        L = self.n_leaves
        p_exp = p.unsqueeze(0).expand(L, -1, -1)               # (L, B, I)
        right_mask = (self.path_matrix == 1).unsqueeze(1)      # (L,1,I)
        left_mask  = (self.path_matrix == 0).unsqueeze(1)      # (L,1,I)
        visit_mask = self.visit_mask.unsqueeze(1)              # (L,1,I)
        chosen = torch.where(right_mask, p_exp, 1.0 - p_exp)
        chosen = torch.where(visit_mask, chosen, torch.ones_like(chosen))
        probs = chosen.prod(dim=2)                              # (L, B)
        return probs.transpose(0, 1)                            # (B, L)


class SoftDecisionTreeClassifier(nn.Module):
    def __init__(self, input_shape, nb_classes, depth=3, tau=2.0, use_bias=True):
        super().__init__()
        T, C = input_shape
        D = C * T
        self.base = _SoftDecisionTreeBase(D, depth=depth, tau=tau)
        self.leaf_logits = nn.Parameter(torch.randn(self.base.n_leaves, nb_classes) * 0.02)
        self.bias = nn.Parameter(torch.zeros(nb_classes)) if use_bias else None

    def forward(self, x):  # (B, C, T)
        B = x.size(0)
        xt = x.reshape(B, -1)
        p = self.base._routing(xt)
        leaf_p = self.base._leaf_probs(p)             # (B, L)
        logits = leaf_p @ self.leaf_logits            # (B, K)
        if self.bias is not None:
            logits = logits + self.bias
        return logits


class SoftDecisionTreeRegressor(nn.Module):
    def __init__(self, input_shape, depth=3, tau=2.0, use_bias=True):
        super().__init__()
        T, C = input_shape
        D = C * T
        self.base = _SoftDecisionTreeBase(D, depth=depth, tau=tau)
        self.leaf_values = nn.Parameter(torch.randn(self.base.n_leaves, 1) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None

    def forward(self, x):  # (B, C, T)
        B = x.size(0)
        xt = x.reshape(B, -1)
        p = self.base._routing(xt)
        leaf_p = self.base._leaf_probs(p)             # (B, L)
        y = (leaf_p @ self.leaf_values).squeeze(-1)   # (B,)
        if self.bias is not None:
            y = y + self.bias
        return y


class _TimeAttentionPool(nn.Module):
    """
    Per-channel temporal attention over the raw signal (depthwise 1D conv scorer).
    """
    def __init__(self, channels, kernel_size=9, use_bias=True, init_tau=2.0):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.scorer = nn.Conv1d(channels, channels, kernel_size,
                                padding=pad, groups=channels, bias=use_bias)  # depthwise
        nn.init.kaiming_normal_(self.scorer.weight, nonlinearity="relu")
        if self.scorer.bias is not None:
            nn.init.zeros_(self.scorer.bias)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(init_tau))))

    def forward(self, x):  # (B, C, T)
        tau = torch.exp(self.log_tau).clamp(min=1e-3)
        s = self.scorer(torch.relu(x))             # (B, C, T) scores
        a = torch.softmax(s / tau, dim=2)          # (B, C, T)
        m = (a * x).sum(dim=2)                     # (B, C) attentive per-channel mean
        return m, a


class AttnPoolClassifier(nn.Module):
    def __init__(self, input_shape, nb_classes, kernel_size=9, dropout=0.0, init_tau=2.0):
        super().__init__()
        T, C = input_shape
        self.attn = _TimeAttentionPool(C, kernel_size=kernel_size, init_tau=init_tau)
        self.do = nn.Dropout(dropout)
        self.head = nn.Linear(C, nb_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):  # (B, C, T)
        m, _ = self.attn(x)
        z = self.do(m)
        return self.head(z)


class AttnPoolRegressor(nn.Module):
    def __init__(self, input_shape, kernel_size=9, dropout=0.0, init_tau=2.0):
        super().__init__()
        T, C = input_shape
        self.attn = _TimeAttentionPool(C, kernel_size=kernel_size, init_tau=init_tau)
        self.do = nn.Dropout(dropout)
        self.head = nn.Linear(C, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):  # (B, C, T)
        m, _ = self.attn(x)
        z = self.do(m)
        return self.head(z).squeeze(-1)
