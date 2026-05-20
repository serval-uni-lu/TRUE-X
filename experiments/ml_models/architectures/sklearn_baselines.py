from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    RandomForestRegressor, ExtraTreesRegressor,
)

try:
    from xgboost import XGBRegressor, XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBRegressor = None
    XGBClassifier = None

try:
    from lightgbm import LGBMRegressor, LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover
    LGBMRegressor = None
    LGBMClassifier = None


class TimeSeriesFeaturizer(BaseEstimator, TransformerMixin):
    """
    (N,C,T) -> tabular (N, F) using cheap but strong per-channel stats:
      mean, std, min, max, median, IQR, energy, zero-cross rate,
      lag-1 autocorr, peak rFFT magnitude (exclude DC).
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.ndim != 3:
            raise ValueError(f"Expected (N,C,T), got {X.shape}")
        N, C, T = X.shape
        x = X
        mean = x.mean(axis=2)
        std = x.std(axis=2) + self.eps
        vmin = x.min(axis=2)
        vmax = x.max(axis=2)
        med = np.median(x, axis=2)
        q75 = np.percentile(x, 75, axis=2)
        q25 = np.percentile(x, 25, axis=2)
        iqr = q75 - q25
        energy = (x ** 2).sum(axis=2)
        zcr = ((x[:, :, 1:] * x[:, :, :-1]) < 0).sum(axis=2) / (T - 1 + self.eps)
        xc = x - mean[:, :, None]
        var = (xc ** 2).mean(axis=2) + self.eps
        acf1 = (xc[:, :, :-1] * xc[:, :, 1:]).mean(axis=2) / var
        fft = np.fft.rfft(x, axis=2)
        mag = np.abs(fft)
        peak = mag[:, :, 1:].max(axis=2)  # skip DC
        feats = np.concatenate([mean, std, vmin, vmax, med, iqr, energy, zcr, acf1, peak], axis=1).astype(np.float32)
        return feats


def make_rf_classifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1):
    pipe = Pipeline([
        ("feat", TimeSeriesFeaturizer()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=n_jobs, class_weight=None
        )),
    ])
    pipe.expects_3d = True
    return pipe


def make_et_classifier(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1):
    pipe = Pipeline([
        ("feat", TimeSeriesFeaturizer()),
        ("clf", ExtraTreesClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=n_jobs, class_weight=None
        )),
    ])
    pipe.expects_3d = True
    return pipe


def make_rf_regressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1):
    pipe = Pipeline([
        ("feat", TimeSeriesFeaturizer()),
        ("reg", RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=n_jobs
        )),
    ])
    pipe.expects_3d = True
    return pipe


def make_et_regressor(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1):
    pipe = Pipeline([
        ("feat", TimeSeriesFeaturizer()),
        ("reg", ExtraTreesRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=n_jobs
        )),
    ])
    pipe.expects_3d = True
    return pipe


def make_xgb_regressor(**kwargs):
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. pip install xgboost")
    defaults = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    defaults.update(kwargs)
    return XGBRegressor(**defaults)


def make_lgbm_regressor(**kwargs):
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed. pip install lightgbm")
    defaults = dict(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=0.0,
        random_state=42, n_jobs=-1, boosting_type="gbdt",
    )
    defaults.update(kwargs)
    return LGBMRegressor(**defaults)


def make_xgb_classifier(**kwargs):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. pip install xgboost")
    defaults = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
        objective="multi:softprob",
    )
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def make_lgbm_classifier(**kwargs):
    if LGBMClassifier is None:
        raise ImportError("lightgbm is not installed. pip install lightgbm")
    defaults = dict(
        n_estimators=300, learning_rate=0.1, num_leaves=63, max_depth=-1,
        min_child_samples=10, min_split_gain=0.0,
        subsample=0.9, colsample_bytree=0.9,
        random_state=42, n_jobs=-1, boosting_type="gbdt",
        objective="multiclass", verbose=-1,
    )
    defaults.update(kwargs)
    return LGBMClassifier(**defaults)
