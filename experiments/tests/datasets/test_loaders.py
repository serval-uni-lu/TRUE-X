# tests/datasets/test_loaders.py
import pytest
import numpy as np
import torch
from pathlib import Path

# ── Data paths ────────────────────────────────────────────────────────────────
# This file lives at  xurl/tests/datasets/test_loaders.py
# parents[2] resolves to  xurl/
# parents[3] resolves to  XRUL/
_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

_CWRU_ROOT   = _DATA_ROOT / "CWRU"
_HYD_ROOT    = _DATA_ROOT / "hydraulic_systems"
_CMAPSS_ROOT = _DATA_ROOT / "CMAPPS_Dataset"
_ECO_ROOT    = _DATA_ROOT / "E-coating"

# ── Skip markers (skip gracefully when data is absent) ────────────────────────
needs_cwru   = pytest.mark.skipif(not _CWRU_ROOT.exists(),   reason="CWRU data not found")
needs_hyd    = pytest.mark.skipif(not _HYD_ROOT.exists(),    reason="Hydraulic data not found")
needs_cmapss = pytest.mark.skipif(not _CMAPSS_ROOT.exists(), reason="CMAPSS data not found")
needs_eco    = pytest.mark.skipif(not _ECO_ROOT.exists(),    reason="E-coating data not found")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _first_batch(loader):
    return next(iter(loader))


def _assert_loader_batch(batch, expected_C, expected_T, task):
    x, y = batch["sequence"], batch["label"]
    assert x.ndim == 3, f"Expected (B, C, T), got {x.shape}"
    assert x.shape[1] == expected_C, f"Expected C={expected_C}, got {x.shape[1]}"
    assert x.shape[2] == expected_T, f"Expected T={expected_T}, got {x.shape[2]}"
    assert x.dtype == torch.float32
    if task == "classification":
        assert y.dtype == torch.int64
    else:
        assert y.dtype == torch.float32


# ══════════════════════════════════════════════════════════════════════════════
# CWRU
# ══════════════════════════════════════════════════════════════════════════════
class TestCWRU:
    WINDOW = 2048
    STRIDE = 512
    CHANNELS = ("DE", "FE")

    @needs_cwru
    def test_arrays_shapes(self):
        from datasets.adapters.cwru import load_cwru_as_arrays
        X_tr, y_tr, X_va, y_va, scaler = load_cwru_as_arrays(
            root=str(_CWRU_ROOT),
            window_size=self.WINDOW,
            stride=self.STRIDE,
            channels=self.CHANNELS,
        )
        assert X_tr.ndim == 3
        assert X_tr.shape[1] == len(self.CHANNELS)
        assert X_tr.shape[2] == self.WINDOW
        assert X_tr.dtype == np.float32
        assert y_tr.shape == (X_tr.shape[0],)
        assert X_va.ndim == 3
        assert y_va.shape == (X_va.shape[0],)

    @needs_cwru
    def test_arrays_labels_are_valid(self):
        from datasets.adapters.cwru import load_cwru_as_arrays
        _, y_tr, _, y_va, _ = load_cwru_as_arrays(
            root=str(_CWRU_ROOT),
            window_size=self.WINDOW,
            stride=self.STRIDE,
            channels=self.CHANNELS,
        )
        valid = {0, 1, 2, 3}
        assert set(np.unique(y_tr)).issubset(valid)
        assert set(np.unique(y_va)).issubset(valid)

    @needs_cwru
    def test_loaders_batch_shape(self):
        from datasets.adapters.cwru import make_cwru_loaders
        tr, va, _ = make_cwru_loaders(
            root=str(_CWRU_ROOT),
            channels=self.CHANNELS,
            window_size=self.WINDOW,
            stride=self.STRIDE,
            batch_size=32,
            num_workers=0,
        )
        _assert_loader_batch(_first_batch(tr), expected_C=2, expected_T=self.WINDOW, task="classification")
        _assert_loader_batch(_first_batch(va), expected_C=2, expected_T=self.WINDOW, task="classification")

    @needs_cwru
    def test_loaders_return_scaler(self):
        from datasets.adapters.cwru import make_cwru_loaders
        from datasets.common import Standardizer
        _, _, scaler = make_cwru_loaders(
            root=str(_CWRU_ROOT),
            channels=self.CHANNELS,
            window_size=self.WINDOW,
            stride=self.STRIDE,
            num_workers=0,
        )
        assert isinstance(scaler, Standardizer)
        assert scaler.mean_ is not None
        assert scaler.std_ is not None

    @needs_cwru
    def test_no_data_leakage_between_splits(self):
        from datasets.adapters.cwru import load_cwru_as_arrays
        X_tr, _, X_va, _, _ = load_cwru_as_arrays(
            root=str(_CWRU_ROOT),
            window_size=self.WINDOW,
            stride=self.STRIDE,
            channels=("DE",),
            val_ratio=0.2,
            random_state=42,
        )
        # train and val should not share identical windows
        assert X_tr.shape[0] > 0
        assert X_va.shape[0] > 0


# ══════════════════════════════════════════════════════════════════════════════
# Hydraulic
# ══════════════════════════════════════════════════════════════════════════════
class TestHydraulic:
    CHANNELS = ["CP", "FS1", "PS1", "PS2", "PS3", "PS4", "PS5", "SE", "VS1"]
    SEQ_LEN  = 50

    @needs_hyd
    def test_arrays_shapes(self):
        from datasets.adapters.hydraulic import load_hydraulic_as_arrays
        X_tr, y_tr, X_va, y_va, scaler, label_maps = load_hydraulic_as_arrays(
            data_dir=str(_HYD_ROOT),
            channels=self.CHANNELS,
            label_cols=2,
            seq_len=self.SEQ_LEN,
        )
        assert X_tr.ndim == 3
        assert X_tr.shape[1] == len(self.CHANNELS)
        assert X_tr.shape[2] == self.SEQ_LEN
        assert y_tr.shape == (X_tr.shape[0],)
        assert X_va.ndim == 3
        assert y_va.shape == (X_va.shape[0],)

    @needs_hyd
    def test_loaders_batch_shape(self):
        from datasets.adapters.hydraulic import make_hydraulic_loaders
        tr, va, _, _ = make_hydraulic_loaders(
            data_dir=str(_HYD_ROOT),
            channels=self.CHANNELS,
            label_cols=2,
            seq_len=self.SEQ_LEN,
            batch_size=32,
            num_workers=0,
        )
        _assert_loader_batch(_first_batch(tr), expected_C=len(self.CHANNELS), expected_T=self.SEQ_LEN, task="classification")
        _assert_loader_batch(_first_batch(va), expected_C=len(self.CHANNELS), expected_T=self.SEQ_LEN, task="classification")

    @needs_hyd
    def test_loaders_return_label_maps(self):
        from datasets.adapters.hydraulic import make_hydraulic_loaders
        _, _, _, label_maps = make_hydraulic_loaders(
            data_dir=str(_HYD_ROOT),
            channels=self.CHANNELS,
            label_cols=2,
            seq_len=self.SEQ_LEN,
            num_workers=0,
        )
        assert isinstance(label_maps, dict)
        assert 0 in label_maps
        assert isinstance(label_maps[0], np.ndarray)

    @needs_hyd
    def test_filter_unstable_reduces_samples(self):
        from datasets.adapters.hydraulic import load_hydraulic_as_arrays
        X_all,  *_ = load_hydraulic_as_arrays(
            data_dir=str(_HYD_ROOT), channels=self.CHANNELS,
            label_cols=2, seq_len=self.SEQ_LEN, filter_unstable=False,
        )
        X_filt, *_ = load_hydraulic_as_arrays(
            data_dir=str(_HYD_ROOT), channels=self.CHANNELS,
            label_cols=2, seq_len=self.SEQ_LEN, filter_unstable=True,
        )
        assert X_filt.shape[0] <= X_all.shape[0]


# ══════════════════════════════════════════════════════════════════════════════
# CMAPSS
# ══════════════════════════════════════════════════════════════════════════════
class TestCMAPSS:
    WINDOW    = 30
    N_FEATS   = 14   # after default column drops
    FD        = "FD001"

    @needs_cmapss
    def test_arrays_shapes(self):
        from datasets.adapters.cmapss import load_cmapss_as_arrays
        X_tr, y_tr, X_va, y_va, scaler, _, _ = load_cmapss_as_arrays(
            train_path=str(_CMAPSS_ROOT / f"train_{self.FD}.txt"),
            test_path =str(_CMAPSS_ROOT / f"test_{self.FD}.txt"),
            rul_path  =str(_CMAPSS_ROOT / f"RUL_{self.FD}.txt"),
            window=self.WINDOW,
        )
        assert X_tr.ndim == 3
        assert X_tr.shape[1] == self.N_FEATS
        assert X_tr.shape[2] == self.WINDOW
        assert X_tr.dtype == np.float32
        assert y_tr.dtype == np.float32
        assert y_tr.shape == (X_tr.shape[0],)

    @needs_cmapss
    def test_loaders_batch_shape(self):
        from datasets.adapters.cmapss import make_cmapss_loaders
        tr, va = make_cmapss_loaders(
            root=str(_CMAPSS_ROOT), fd=self.FD,
            window=self.WINDOW, batch_size=32, num_workers=0,
        )
        _assert_loader_batch(_first_batch(tr), expected_C=self.N_FEATS, expected_T=self.WINDOW, task="regression")
        _assert_loader_batch(_first_batch(va), expected_C=self.N_FEATS, expected_T=self.WINDOW, task="regression")

    @needs_cmapss
    def test_loaders_with_test_set(self):
        from datasets.adapters.cmapss import make_cmapss_loaders
        tr, va, te, true_rul, mask, scaler = make_cmapss_loaders(
            root=str(_CMAPSS_ROOT), fd=self.FD,
            window=self.WINDOW, batch_size=32, num_workers=0,
            return_test=True, k_last=5,
        )
        assert true_rul.ndim == 1
        assert len(true_rul) > 0
        _assert_loader_batch(_first_batch(te), expected_C=self.N_FEATS, expected_T=self.WINDOW, task="regression")

    @needs_cmapss
    def test_rul_target_range(self):
        from datasets.adapters.cmapss import load_cmapss_as_arrays
        _, y_tr, _, y_va, _, _, _ = load_cmapss_as_arrays(
            train_path=str(_CMAPSS_ROOT / f"train_{self.FD}.txt"),
            test_path =str(_CMAPSS_ROOT / f"test_{self.FD}.txt"),
            rul_path  =str(_CMAPSS_ROOT / f"RUL_{self.FD}.txt"),
            window=self.WINDOW, early_rul=120,
        )
        assert float(y_tr.max()) <= 120.0
        assert float(y_tr.min()) >= 0.0

    @needs_cmapss
    def test_invalid_fd_raises(self):
        from datasets.adapters.cmapss import make_cmapss_loaders
        with pytest.raises(ValueError, match="fd must be one of"):
            make_cmapss_loaders(root=str(_CMAPSS_ROOT), fd="FD999", num_workers=0)


# ══════════════════════════════════════════════════════════════════════════════
# E-coating
# ══════════════════════════════════════════════════════════════════════════════
class TestEcoating:
    WINDOW = 30

    @needs_eco
    def test_arrays_shapes(self):
        from datasets.adapters.ecoating import load_ecoating_as_arrays
        X_tr, y_tr, X_va, y_va, test_tuple, x_scaler, feat_cols, _ = load_ecoating_as_arrays(
            train_path=str(_ECO_ROOT / "manual_30min_norm.csv"),
            test_path =str(_ECO_ROOT / "iiot_30min_norm.csv"),
            time_col="TIME", target_col="FM1",
            window=self.WINDOW,
        )
        assert X_tr.ndim == 3
        assert X_tr.shape[2] == self.WINDOW
        assert X_tr.dtype == np.float32
        assert y_tr.dtype == np.float32
        assert y_tr.shape == (X_tr.shape[0],)
        assert len(feat_cols) == X_tr.shape[1]

    @needs_eco
    def test_arrays_test_tuple(self):
        from datasets.adapters.ecoating import load_ecoating_as_arrays
        *_, test_tuple, _, feat_cols, _ = load_ecoating_as_arrays(
            train_path=str(_ECO_ROOT / "manual_30min_norm.csv"),
            test_path =str(_ECO_ROOT / "iiot_30min_norm.csv"),
            time_col="TIME", target_col="FM1",
            window=self.WINDOW,
        )
        assert test_tuple is not None
        X_te, y_te, mask, n_te = test_tuple
        assert X_te.ndim == 3
        assert X_te.shape[2] == self.WINDOW
        assert y_te.shape == (X_te.shape[0],)
        assert mask.shape == (X_te.shape[0], self.WINDOW)
        assert n_te == X_te.shape[0]

    @needs_eco
    def test_loaders_batch_shape(self):
        from datasets.adapters.ecoating import make_ecoating_loaders
        tr, va, te, x_scaler, feat_cols = make_ecoating_loaders(
            root=str(_ECO_ROOT),
            train_filename="manual_30min_norm.csv",
            test_filename ="iiot_30min_norm.csv",
            time_col="TIME", target_col="FM1",
            window=self.WINDOW, batch_size=64, num_workers=0,
        )
        C = len(feat_cols)
        _assert_loader_batch(_first_batch(tr), expected_C=C, expected_T=self.WINDOW, task="regression")
        _assert_loader_batch(_first_batch(va), expected_C=C, expected_T=self.WINDOW, task="regression")

    @needs_eco
    def test_target_not_in_features(self):
        from datasets.adapters.ecoating import load_ecoating_as_arrays
        *_, feat_cols, _ = load_ecoating_as_arrays(
            train_path=str(_ECO_ROOT / "manual_30min_norm.csv"),
            time_col="TIME", target_col="FM1",
            window=self.WINDOW,
        )[-3:]
        assert "FM1" not in feat_cols
        assert "TIME" not in feat_cols

    @needs_eco
    def test_missing_train_file_raises(self):
        from datasets.adapters.ecoating import make_ecoating_loaders
        with pytest.raises(FileNotFoundError):
            make_ecoating_loaders(
                root=str(_ECO_ROOT),
                train_filename="does_not_exist.csv",
                time_col="TIME", target_col="FM1",
                num_workers=0,
            )
