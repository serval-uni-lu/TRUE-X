from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from ._blocks import conv1d_same, _PatchEmbedding


class MLP_LSTM_Attention(nn.Module):
    """
    Small LSTM + additive attention + MLP head. Good for RUL/regression.
    input:  (B, C, T)  (we internally use (B, T, C) for LSTM)
    output: (B, output_dim)
    """
    def __init__(self, input_dim=14, output_dim=1, hidden_dim=32, n_hidden_layers=2,
                 lstm_hidden_dim=32, use_dropout=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * lstm_hidden_dim, 1)
        self.fc1 = nn.Linear(2 * lstm_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):            # x: (B, C, T)
        x = x.transpose(1, 2)        # -> (B, T, C)
        h, _ = self.lstm(x)          # (B, T, 2H)
        a = torch.softmax(self.attn(h), dim=1)   # (B, T, 1)
        ctx = (a * h).sum(dim=1)                 # (B, 2H)
        z = torch.tanh(self.fc1(ctx))
        if self.use_dropout:
            z = self.dropout(z)
        z = torch.tanh(self.fc2(z))
        if self.use_dropout:
            z = self.dropout(z)
        for layer in self.hidden_layers:
            z = z + torch.tanh(layer(z))        # residual MLP
        return self.out(z)                       # logits for regression (raw)


class VAE_RUL(nn.Module):
    """VAE-style encoder producing a latent vector used by a regressor."""
    def __init__(self, timesteps, input_dim, intermediate_dim, latent_dim):
        super().__init__()
        self.enc = nn.LSTM(input_dim, intermediate_dim, batch_first=True, bidirectional=True)
        self.mu = nn.Linear(2 * intermediate_dim, latent_dim)
        self.logvar = nn.Linear(2 * intermediate_dim, latent_dim)
        self.reg = nn.Sequential(nn.Linear(latent_dim, 200), nn.Tanh(), nn.Linear(200, 1))

    def encode(self, x):
        h, _ = self.enc(x)
        h = h[:, -1, :]
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):           # x: (B, C, T)
        x = x.transpose(1, 2)       # -> (B, T, C)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.reg(z)             # (B,1)
        return y, mu, logvar


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.c = conv1d_same(in_ch, out_ch, k, bias=False)
        self.b = nn.BatchNorm1d(out_ch)
        self.a = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.a(self.b(self.c(x)))


class CNN_LSTM_Regressor(nn.Module):
    """Conv stack (same length) -> LSTM -> head."""
    def __init__(self, input_shape, conv_channels=(64, 128), conv_kernels=(7, 5), lstm_hidden=128,
                 lstm_layers=2, bidirectional=True, lstm_dropout=0.1, head_hidden=None, head_dropout=0.1):
        super().__init__()
        T, C = input_shape
        if len(conv_channels) != len(conv_kernels):
            raise ValueError(f"conv_channels and conv_kernels must have the same length, got {len(conv_channels)} vs {len(conv_kernels)}")
        convs = []
        in_ch = C
        for ch, k in zip(conv_channels, conv_kernels):
            convs.append(ConvBlock1D(in_ch, ch, k))
            in_ch = ch
        self.cnn = nn.Sequential(*convs)
        feat = conv_channels[-1]
        self.lstm = nn.LSTM(feat, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            dropout=(lstm_dropout if lstm_layers > 1 else 0.0), bidirectional=bidirectional)
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        if head_hidden is None:
            self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(out_dim, 1))
        else:
            self.head = nn.Sequential(nn.Linear(out_dim, head_hidden), nn.ReLU(True),
                                      nn.Dropout(head_dropout), nn.Linear(head_hidden, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        z = self.cnn(x)        # (B,F,T)
        z = z.permute(0, 2, 1) # (B,T,F)
        _, (h_n, _) = self.lstm(z)
        if self.lstm.bidirectional:
            fwd, bwd = h_n[-2], h_n[-1]
            h = torch.cat([fwd, bwd], dim=1)
        else:
            h = h_n[-1]
        return self.head(h).squeeze(-1)


class AttentionLSTMRegressor(nn.Module):
    """LSTM + additive attention over time."""
    def __init__(self, input_shape, hidden_size=128, num_layers=2, bidirectional=True, lstm_dropout=0.1,
                 attn_hidden=64, head_hidden=None, head_dropout=0.1):
        super().__init__()
        T, C = input_shape
        self.lstm = nn.LSTM(C, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=(lstm_dropout if num_layers > 1 else 0.0))
        feat = hidden_size * (2 if bidirectional else 1)
        self.attn = nn.Sequential(nn.Linear(feat, attn_hidden), nn.Tanh(), nn.Linear(attn_hidden, 1, bias=False))
        if head_hidden is None:
            self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(feat, 1))
        else:
            self.head = nn.Sequential(nn.Linear(feat, head_hidden), nn.ReLU(True),
                                      nn.Dropout(head_dropout), nn.Linear(head_hidden, 1))
        for n, p in self.lstm.named_parameters():
            if "weight_ih" in n:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in n:
                nn.init.orthogonal_(p)
            elif "bias" in n:
                nn.init.zeros_(p)

    def forward(self, x):     # (B,C,T)
        x = x.transpose(1, 2)
        H, _ = self.lstm(x)   # (B,T,D)
        s = self.attn(H)      # (B,T,1)
        a = torch.softmax(s, dim=1)
        ctx = (a * H).sum(dim=1)
        return self.head(ctx).squeeze(-1)


class TSTRegressor(nn.Module):
    """Transformer regressor over patch tokens."""
    def __init__(self, input_shape, d_model=128, n_heads=8, num_layers=4, d_ff=256, dropout=0.1,
                 patch_len=16, stride=8, use_cls_token=True, emb_dropout=0.1, head_hidden=None, head_dropout=0.1):
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
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        if head_hidden is None:
            self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(d_model, 1))
        else:
            self.head = nn.Sequential(nn.Linear(d_model, head_hidden), nn.ReLU(True),
                                      nn.Dropout(head_dropout), nn.Linear(head_hidden, 1))
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
        return self.head(rep).squeeze(-1)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1, bias=True):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=0, dilation=dilation, bias=bias))

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation, dropout=0.1):
        super().__init__()
        self.c1 = CausalConv1d(in_ch, out_ch, k, dilation)
        self.a1 = nn.ReLU(True)
        self.d1 = nn.Dropout(dropout)
        self.c2 = CausalConv1d(out_ch, out_ch, k, dilation)
        self.a2 = nn.ReLU(True)
        self.d2 = nn.Dropout(dropout)
        self.down = None if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        y = self.d1(self.a1(self.c1(x)))
        y = self.d2(self.a2(self.c2(y)))
        res = x if self.down is None else self.down(x)
        return F.relu(y + res)


class TCNRegressor(nn.Module):
    def __init__(
        self,
        input_shape,
        channels=(64, 64, 128),
        kernel_size=3,
        dropout=0.1,
        head_hidden=64,
        head_dropout=0.1,
    ):
        super().__init__()
        T, C = input_shape
        blocks = []
        in_ch = C
        for i, out_ch in enumerate(channels):
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)
        feat = channels[-1]
        if head_hidden is None:
            self.head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(feat, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(feat, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, 1),
            )
        last = self.head[-1] if isinstance(self.head, nn.Sequential) else self.head
        if isinstance(last, nn.Linear):
            nn.init.xavier_uniform_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x):
        z = self.tcn(x)           # (B, F, T)
        last = z[:, :, -1]        # take last timestep
        return self.head(last).squeeze(-1)


class LSTM_FCN_Regressor(nn.Module):
    """FCN branch + LSTM branch -> concat -> head."""
    def __init__(self, input_shape, lstm_hidden=128, lstm_layers=1, bidirectional=True, lstm_dropout=0.1,
                 fcn_channels=(128, 256, 128), kernels=(9, 5, 3), head_hidden=None, head_dropout=0.1):
        super().__init__()
        T, C = input_shape
        c1, c2, c3 = fcn_channels
        k1, k2, k3 = kernels
        self.c1 = conv1d_same(C, c1, k1, bias=False)
        self.b1 = nn.BatchNorm1d(c1)
        self.c2 = conv1d_same(c1, c2, k2, bias=False)
        self.b2 = nn.BatchNorm1d(c2)
        self.c3 = conv1d_same(c2, c3, k3, bias=False)
        self.b3 = nn.BatchNorm1d(c3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.lstm = nn.LSTM(C, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=(lstm_dropout if lstm_layers > 1 else 0.0))
        lstm_feat = lstm_hidden * (2 if bidirectional else 1)
        concat = c3 + lstm_feat
        if head_hidden is None:
            self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(concat, 1))
        else:
            self.head = nn.Sequential(nn.Linear(concat, head_hidden), nn.ReLU(True),
                                      nn.Dropout(head_dropout), nn.Linear(head_hidden, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        z = F.relu(self.b1(self.c1(x)))
        z = F.relu(self.b2(self.c2(z)))
        z = F.relu(self.b3(self.c3(z)))
        z = self.gap(z).squeeze(-1)          # (B,c3)
        xl = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(xl)
        if self.lstm.bidirectional:
            fwd, bwd = h_n[-2], h_n[-1]
            h = torch.cat([fwd, bwd], dim=1)
        else:
            h = h_n[-1]
        feats = torch.cat([z, h], dim=1)
        return self.head(feats).squeeze(-1)


class GLU(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(d_model, 2 * d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(x)
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.glu = GLU(d_out, dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = F.elu(self.fc1(x))
        y = self.fc2(y)
        y = self.glu(y)
        y = self.drop(y)
        return self.norm(self.skip(x) + y)


class VariableSelectionNetwork(nn.Module):
    """Time-distributed VSN for continuous vars only."""
    def __init__(self, n_vars, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_vars)])
        self.scorers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ELU(), nn.Linear(d_model // 2, 1))
            for _ in range(n_vars)
        ])
        self.drop = nn.Dropout(dropout)
        self.n_vars = n_vars

    def forward(self, x):  # (B,C,T)
        B, C, T = x.shape
        xs, scores = [], []
        for i in range(C):
            xi = x[:, i:i+1, :].transpose(1, 2)   # (B,T,1)
            ei = self.proj[i](xi)                 # (B,T,D)
            si = self.scorers[i](ei)              # (B,T,1)
            xs.append(ei)
            scores.append(si)
        E = torch.stack(xs, dim=2)                # (B,T,C,D)
        S = torch.stack(scores, dim=2)            # (B,T,C,1)
        A = torch.softmax(S, dim=2)
        Z = (A * E).sum(dim=2)                    # (B,T,D)
        return self.drop(Z), A.squeeze(-1)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.d1 = nn.Dropout(dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.glu = GLU(d_model, dropout)
        self.d2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        z = self.n1(x)
        z, _ = self.attn(z, z, z, attn_mask=attn_mask, need_weights=False)
        x = x + self.d1(z)
        z = self.n2(x)
        z = self.ff(z)
        z = self.glu(z)
        return x + self.d2(z)


class TFT_Regressor(nn.Module):
    """Temporal Fusion Transformer (lite) for single-step regression from a window."""
    def __init__(self, input_shape, d_model=128, n_heads=4, n_attn_layers=2, d_ff=256,
                 lstm_hidden=128, lstm_layers=1, dropout=0.1, causal_attention=True, head_hidden=None):
        super().__init__()
        T, C = input_shape
        self.causal = causal_attention
        self.vsn = VariableSelectionNetwork(C, d_model, dropout=dropout)
        self.vsn_grn = GatedResidualNetwork(d_model, d_ff, d_model, dropout=dropout)
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=lstm_layers, batch_first=True,
                            bidirectional=False, dropout=(dropout if lstm_layers > 1 else 0.0))
        self.lstm_proj = nn.Linear(lstm_hidden, d_model)
        self.blocks = nn.ModuleList([SelfAttentionBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_attn_layers)])
        self.head_grn = GatedResidualNetwork(d_model, d_ff, d_model, dropout=dropout)
        self.head = nn.Linear(d_model, 1) if head_hidden is None else nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(head_hidden, 1)
        )

    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x):  # (B,C,T)
        B, C, T = x.shape
        z, _ = self.vsn(x)            # (B,T,D)
        z = self.vsn_grn(z)
        lstm_out, _ = self.lstm(z)    # (B,T,H)
        z = self.lstm_proj(lstm_out) + z
        mask = self._causal_mask(T, x.device) if self.causal else None
        for blk in self.blocks:
            z = blk(z, attn_mask=mask)
        h = self.head_grn(z[:, -1, :])
        return self.head(h).squeeze(-1)


class LSTM_Regressor(nn.Module):
    """Plain LSTM regressor with pooling choice."""
    def __init__(self, input_shape, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.1,
                 temporal_pool="last", head_hidden=None):
        super().__init__()
        T, C = input_shape
        self.temporal_pool = temporal_pool
        self.bi = bidirectional
        self.lstm = nn.LSTM(C, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=(dropout if num_layers > 1 else 0.0))
        feat = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(feat)
        if head_hidden is None:
            self.head = nn.Linear(feat, 1)
        else:
            self.head = nn.Sequential(nn.Linear(feat, head_hidden), nn.ReLU(True),
                                      nn.Dropout(dropout), nn.Linear(head_hidden, 1))
        for n, p in self.lstm.named_parameters():
            if "weight_ih" in n:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in n:
                nn.init.orthogonal_(p)
            elif "bias" in n:
                nn.init.zeros_(p)

    def forward(self, x):
        x = x.transpose(1, 2)
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
        return self.head(feats).squeeze(-1)


class BiLSTM_Regressor(nn.Module):
    """BiLSTM regressor (explicit bi-directional)."""
    def __init__(self, input_shape, hidden_size=128, num_layers=2, dropout=0.1, temporal_pool="last", head_hidden=None):
        super().__init__()
        T, C = input_shape
        self.temporal_pool = temporal_pool
        self.lstm = nn.LSTM(C, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=(dropout if num_layers > 1 else 0.0))
        feat = hidden_size * 2
        self.norm = nn.LayerNorm(feat)
        if head_hidden is None:
            self.head = nn.Linear(feat, 1)
        else:
            self.head = nn.Sequential(nn.Linear(feat, head_hidden), nn.ReLU(True),
                                      nn.Dropout(dropout), nn.Linear(head_hidden, 1))
        for n, p in self.lstm.named_parameters():
            if "weight_ih" in n:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in n:
                nn.init.orthogonal_(p)
            elif "bias" in n:
                nn.init.zeros_(p)

    def forward(self, x):
        x = x.transpose(1, 2)
        out, (h_n, _) = self.lstm(x)
        if self.temporal_pool == "last":
            fwd, bwd = h_n[-2], h_n[-1]
            feats = torch.cat([fwd, bwd], dim=1)
        elif self.temporal_pool == "mean":
            feats = out.mean(dim=1)
        elif self.temporal_pool == "max":
            feats, _ = out.max(dim=1)
        else:
            raise ValueError(self.temporal_pool)
        feats = self.norm(feats)
        return self.head(feats).squeeze(-1)


class LinearRegressor(nn.Module):
    """Linear regression over flattened window."""
    def __init__(self, input_shape, use_bias=True):
        super().__init__()
        T, C = input_shape
        self.fc = nn.Linear(C * T, 1, bias=use_bias)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        z = x.reshape(x.size(0), -1)
        return self.fc(z).squeeze(-1)
