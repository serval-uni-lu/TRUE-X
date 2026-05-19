# ml_models/experiment_model.py
"""
ExperimentModel: a thin compatibility layer that trains/evaluates both
PyTorch (N,C,T) time-series models and sklearn baselines (optionally fed
with (N,C,T) and auto-featurized). Duplicates removed, consistent helpers,
and safe optional dependencies for XGBoost/LightGBM.

- Classification uses CrossEntropyLoss (expects logits).
- Regression uses MSELoss (raw continuous outputs).
- VAE models are handled via (out, mu, logvar) tuple + KL term.

Supports:
  * Torch models registered in ModelRegistry
  * Sklearn pipelines via factories (RF/ET/XGB/LGBM)
  * Simple averaging ensemble for explainability
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Any

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.feature_selection import VarianceThreshold

from ml_models.model_registry import Model, ModelRegistry

from ml_models.architectures import (
    MLP_LSTM_Attention,
    VAE_RUL,
    EncoderTSC,
    Classifier_RESNET,
    Classifier_FCN,
    Classifier_MCDCNN,
    Classifier_TIMECNN,
    InceptionTime,
    TST,
    Classifier_LSTM,
    Classifier_GRU,
    CNN_LSTM_Regressor,
    AttentionLSTMRegressor,
    TSTRegressor,
    TCNRegressor,
    LSTM_FCN_Regressor,
    TFT_Regressor,
    LSTM_Regressor,
    BiLSTM_Regressor,
    LinearRegressor,
    LogisticClassifier,
    NAMTS_Classifier, NAMTS_Regressor,
    SoftDecisionTreeClassifier, SoftDecisionTreeRegressor,
    AttnPoolClassifier, AttnPoolRegressor,
    make_rf_classifier,
    make_et_classifier,
    make_rf_regressor,
    make_et_regressor,
    make_xgb_classifier,
    make_xgb_regressor,
    make_lgbm_classifier,
    make_lgbm_regressor,
)
import logging
logger = logging.getLogger("experiment")

if torch.backends.mps.is_available() and torch.backends.mps.is_built():  # Apple Silicon
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class ExperimentModel:
    """
    Unifies training/prediction across:
      - PyTorch models (nn.Module)
      - sklearn models (optionally with a 3D featurizer that accepts (N,C,T))
    """

    def __init__(self, model_type: Model, input_shape: Tuple[int, int], nb_classes: Optional[int] = None):
        """
        Args:
            model_type: Model enum from ModelRegistry
            input_shape: (T, C)
            nb_classes: required for classification models that need output dim
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.nb_classes = nb_classes

        # Write metadata once per run (idempotent append)
        ModelRegistry.write_model_to_csv(model_type)

        # Infer task from registry
        info = ModelRegistry.get_info(model_type)
        self.task: str = info.get("task", "Unknown")
        self._is_regression: bool = str(self.task).lower().startswith("regress")

        # sklearn flags
        self.is_sklearn: bool = False
        self.expects_3d: bool = False   # sklearn pipeline that wants (N,C,T) directly
        self.vt_: Optional[VarianceThreshold] = None  # used for 2D path

        # Build the underlying model
        self.model: Any = self._build_model(model_type, input_shape, nb_classes)

        # mirror expects_3d flag if set by the sklearn pipeline
        if self.is_sklearn and hasattr(self.model, "expects_3d"):
            self.expects_3d = bool(getattr(self.model, "expects_3d"))

    def _build_model(self, model_type: Model, input_shape: Tuple[int, int], nb_classes: Optional[int]):

        if model_type == Model.RF_CLASSIFIER:
            self.is_sklearn = True
            return make_rf_classifier()
        if model_type == Model.ET_CLASSIFIER:
            self.is_sklearn = True
            return make_et_classifier()
        if model_type == Model.RF_REGRESSOR:
            self.is_sklearn = True
            return make_rf_regressor()
        if model_type == Model.ET_REGRESSOR:
            self.is_sklearn = True
            return make_et_regressor()
        if model_type == Model.XGB_CLASSIFIER:
            self.is_sklearn = True
            return make_xgb_classifier()
        if model_type == Model.XGB_REGRESSOR:
            self.is_sklearn = True
            return make_xgb_regressor()
        if model_type == Model.LGBM_CLASSIFIER:
            self.is_sklearn = True
            return make_lgbm_classifier()
        if model_type == Model.LGBM_REGRESSOR:
            self.is_sklearn = True
            return make_lgbm_regressor()


        T, C = input_shape

        if model_type == Model.VI_ENSEMBLE:
            return [MLP_LSTM_Attention(input_dim=C, output_dim=1).to(DEVICE) for _ in range(2)]
        if model_type == Model.VI_DROPOUT:
            return MLP_LSTM_Attention(input_dim=C, output_dim=1, use_dropout=True).to(DEVICE)
        if model_type == Model.VAE_RUL:
            return VAE_RUL(timesteps=T, input_dim=C, intermediate_dim=32, latent_dim=8).to(DEVICE)

        # classification (nb_classes required)
        if model_type == Model.ENCODER:
            return EncoderTSC(input_shape, nb_classes).to(DEVICE)
        if model_type == Model.RESNET:
            return Classifier_RESNET(input_shape, nb_classes).to(DEVICE)
        if model_type == Model.FCN:
            return Classifier_FCN(input_shape, nb_classes).to(DEVICE)
        if model_type == Model.MCD_CNN:
            return Classifier_MCDCNN(input_shape, nb_classes).to(DEVICE)
        if model_type == Model.TIME_CNN:
            return Classifier_TIMECNN(input_shape, nb_classes).to(DEVICE)
        if model_type == Model.INCEPTIONTIME:
            # Default architecture; if a checkpoint with different hyperparams is loaded later,
            # experiment.py will rebuild using inferred/saved config.
            return InceptionTime(input_shape=input_shape, nb_classes=nb_classes,
                                 n_residual_blocks=6, nb_filters=16, kernel_sizes=[9, 19, 39],
                                 bottleneck=True).to(DEVICE)
        if model_type == Model.TST:
            return TST(input_shape=input_shape, nb_classes=nb_classes,
                       d_model=128, n_heads=8, num_layers=4, d_ff=256,
                       dropout=0.1, patch_len=16, stride=8, use_cls_token=True).to(DEVICE)
        if model_type == Model.LSTM:
            return Classifier_LSTM(input_shape=input_shape, nb_classes=nb_classes,
                                   hidden_size=128, num_layers=2, bidirectional=True,
                                   dropout=0.1, temporal_pool="last").to(DEVICE)
        if model_type == Model.GRU:
            return Classifier_GRU(input_shape=input_shape, nb_classes=nb_classes,
                                  hidden_size=128, num_layers=2, bidirectional=True,
                                  dropout=0.1, temporal_pool="last").to(DEVICE)

        # regression
        if model_type == Model.CNN_LSTM_REGRESSOR:
            return CNN_LSTM_Regressor(input_shape=input_shape,
                                      conv_channels=(64, 128),
                                      conv_kernels=(7, 5),
                                      lstm_hidden=128,
                                      lstm_layers=2,
                                      bidirectional=True,
                                      lstm_dropout=0.1,
                                      head_hidden=None,
                                      head_dropout=0.1).to(DEVICE)
        if model_type == Model.ATTENTION_LSTM_REGRESSOR:
            return AttentionLSTMRegressor(input_shape=input_shape,
                                          hidden_size=128, num_layers=2,
                                          bidirectional=True, lstm_dropout=0.1,
                                          attn_hidden=64, head_hidden=None,
                                          head_dropout=0.1).to(DEVICE)
        if model_type == Model.TST_REGRESSOR:
            return TSTRegressor(input_shape=input_shape, d_model=128, n_heads=8, num_layers=4,
                                d_ff=256, dropout=0.1, patch_len=16, stride=8,
                                use_cls_token=True, emb_dropout=0.1,
                                head_hidden=None, head_dropout=0.1).to(DEVICE)
        if model_type == Model.TCN_REGRESSOR:
            return TCNRegressor(input_shape=input_shape, channels=(64, 64, 128),
                                kernel_size=3, dropout=0.1,
                                head_hidden=64, head_dropout=0.1).to(DEVICE)
        if model_type == Model.LSTM_FCN_REGRESSOR:
            return LSTM_FCN_Regressor(input_shape=input_shape,
                                      lstm_hidden=128, lstm_layers=1, bidirectional=True,
                                      lstm_dropout=0.1, fcn_channels=(128, 256, 128),
                                      kernels=(9, 5, 3), head_hidden=None, head_dropout=0.1).to(DEVICE)
        if model_type == Model.TFT_REGRESSOR:
            return TFT_Regressor(input_shape=input_shape, d_model=128, n_heads=4, n_attn_layers=2,
                                 d_ff=256, lstm_hidden=128, lstm_layers=1,
                                 dropout=0.1, causal_attention=True, head_hidden=None).to(DEVICE)
        if model_type == Model.LSTM_REGRESSOR:
            return LSTM_Regressor(input_shape=input_shape, hidden_size=128,
                                  num_layers=2, bidirectional=True, dropout=0.1,
                                  temporal_pool="last", head_hidden=None).to(DEVICE)
        if model_type == Model.BI_LSTM_REGRESSOR:
            return BiLSTM_Regressor(input_shape=input_shape, hidden_size=128,
                                    num_layers=2, dropout=0.1,
                                    temporal_pool="last", head_hidden=None).to(DEVICE)
        if model_type == Model.LINEAR_REGRESSOR:
            return LinearRegressor(input_shape=input_shape).to(DEVICE)

        # logistic regression baseline (classification)
        if model_type == Model.LOGISTIC_CLASSIFIER:
            return LogisticClassifier(input_shape=input_shape, nb_classes=nb_classes).to(DEVICE)

        if model_type == Model.NAMTS_CLASSIFIER:
            return NAMTS_Classifier(
                input_shape=input_shape,
                nb_classes=nb_classes,
                d_hidden=32,
                dropout=0.1,
            ).to(DEVICE)

        if model_type == Model.SOFT_DT_CLASSIFIER:
            # differentiable binary tree over flattened (C*T)
            return SoftDecisionTreeClassifier(
                input_shape=input_shape,
                nb_classes=nb_classes,
                depth=3,
                tau=2.0,
                use_bias=True,
            ).to(DEVICE)

        if model_type == Model.ATTNPOOL_CLASSIFIER:
            # per-channel temporal attention pooled -> linear head
            return AttnPoolClassifier(
                input_shape=input_shape,
                nb_classes=nb_classes,
                kernel_size=9,
                dropout=0.0,
                init_tau=2.0,
            ).to(DEVICE)

        if model_type == Model.NAMTS_REGRESSOR:
            return NAMTS_Regressor(
                input_shape=input_shape,
                d_hidden=32,
                dropout=0.1,
            ).to(DEVICE)

        if model_type == Model.SOFT_DT_REGRESSOR:
            return SoftDecisionTreeRegressor(
                input_shape=input_shape,
                depth=3,
                tau=2.0,
                use_bias=True,
            ).to(DEVICE)

        if model_type == Model.ATTNPOOL_REGRESSOR:
            return AttnPoolRegressor(
                input_shape=input_shape,
                kernel_size=9,
                dropout=0.0,
                init_tau=2.0,
            ).to(DEVICE)

        raise ValueError(f"Unknown model_type: {model_type}")

    # sklearn helpers
    @staticmethod
    def _flatten_3d_to_2d(X3d: np.ndarray) -> np.ndarray:
        """(N, C, T) -> (N, C*T)"""
        N, C, T = X3d.shape
        return X3d.reshape(N, C * T)

    @staticmethod
    def _loader_to_numpy(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for batch in loader:
            xb = batch["sequence"].cpu().numpy()  # (B,C,T)
            yb = batch.get("label")
            yb = yb.cpu().numpy() if torch.is_tensor(yb) else np.asarray(yb)
            Xs.append(xb)
            ys.append(yb)
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    def fit(
        self,
        X: DataLoader | np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 1e-3,
        val_loader: Optional[DataLoader] = None,
        beta_kl: float = 1e-3,
        early_stopping: Optional[dict] = None,
    ) -> None:
        """
        Train on either:
          - DataLoader that yields dict(sequence, label)
          - numpy arrays X (N,C,T) and y
        """


        if self.is_sklearn:
            if isinstance(X, DataLoader):
                Xtr_3d, ytr = self._loader_to_numpy(X)
                if self.expects_3d:
                    self.model.fit(Xtr_3d, ytr)
                    return
                Xtr_2d = self._flatten_3d_to_2d(Xtr_3d)
                self.vt_ = VarianceThreshold(threshold=0.0)
                Xtr_2d = self.vt_.fit_transform(Xtr_2d)
                self.model.fit(Xtr_2d, ytr)
                return

            # arrays
            Xtr_3d = X if X.ndim == 3 else X.reshape(X.shape[0], -1, 1)
            if self.expects_3d:
                self.model.fit(Xtr_3d, y)
                return
            Xtr_2d = self._flatten_3d_to_2d(Xtr_3d)
            self.vt_ = VarianceThreshold(threshold=0.0)
            Xtr_2d = self.vt_.fit_transform(Xtr_2d)
            self.model.fit(Xtr_2d, y)
            return


        if isinstance(X, DataLoader):
            self._train_with_loader(self.model, X, val_loader, epochs, lr, beta_kl, early_stopping)
        else:
            self._train_with_arrays(self.model, X, y, epochs, batch_size, lr, beta_kl)

    def predict(self, X: DataLoader | np.ndarray) -> np.ndarray:
        """
        Predict on either:
          - DataLoader (returns concatenated outputs)
          - numpy arrays (N,C,T)
        Returns np.ndarray: shape depends on model/task:
            * classification: logits or proba (sklearn may return proba)
            * regression: (N,)
        """


        if self.is_sklearn:
            if isinstance(X, DataLoader):
                X_3d, _ = self._loader_to_numpy(X)
                if self.expects_3d:
                    if self.task.lower() == "classification" and hasattr(self.model, "predict_proba"):
                        return self.model.predict_proba(X_3d)
                    return self.model.predict(X_3d)
                X_2d = self._flatten_3d_to_2d(X_3d)
                if self.vt_ is not None:
                    X_2d = self.vt_.transform(X_2d)
                if self.task.lower() == "classification" and hasattr(self.model, "predict_proba"):
                    return self.model.predict_proba(X_2d)
                return self.model.predict(X_2d)

            # arrays
            X_3d = X if X.ndim == 3 else X.reshape(X.shape[0], -1, 1)
            if self.expects_3d:
                if self.task.lower() == "classification" and hasattr(self.model, "predict_proba"):
                    return self.model.predict_proba(X_3d)
                return self.model.predict(X_3d)
            X_2d = self._flatten_3d_to_2d(X_3d)
            if self.vt_ is not None:
                X_2d = self.vt_.transform(X_2d)
            if self.task.lower() == "classification" and hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X_2d)
            return self.model.predict(X_2d)


        if isinstance(X, DataLoader):
            return self._predict_loader(self.model, X)
        # numpy arrays
        if self.model_type == Model.VI_ENSEMBLE:
            preds = np.array([self._predict_arrays(net, X) for net in self.model])
            return preds.mean(axis=0)
        return self._predict_arrays(self.model, X)

    def _criterion(self):
        return nn.MSELoss() if self._is_regression else nn.CrossEntropyLoss()

    @staticmethod
    def _apply_vae_kl(mu: torch.Tensor, logvar: torch.Tensor, base_loss: torch.Tensor, beta_kl: float):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return base_loss + beta_kl * kl

    def _train_with_arrays(
        self,
        model: nn.Module | List[nn.Module],
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        beta_kl: float,
    ):
        # Support ensemble list
        if isinstance(model, list):
            for net in model:
                self._train_with_arrays(net, X, y, epochs, batch_size, lr, beta_kl)
            return

        device = next(model.parameters()).device
        model.to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = self._criterion()

        X_tensor = torch.from_numpy(X).float().to(device)
        if self._is_regression:
            y_tensor = torch.from_numpy(y).float().to(device)
        else:
            y_tensor = torch.from_numpy(y)
            if y_tensor.ndim > 1:
                y_tensor = torch.argmax(y_tensor, dim=1)
            y_tensor = y_tensor.long().to(device)

        for _ in tqdm(range(epochs), desc="Epochs"):
            perm = torch.randperm(X_tensor.size(0), device=device)
            for i in range(0, X_tensor.size(0), batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_tensor[idx], y_tensor[idx]

                opt.zero_grad(set_to_none=True)
                out = model(xb)

                if self._is_regression:
                    if isinstance(out, tuple):
                        pred, mu, logvar = out
                        pred = pred.squeeze()
                        loss = crit(pred, yb)
                        loss = self._apply_vae_kl(mu, logvar, loss, beta_kl)
                    else:
                        loss = crit(out.squeeze(), yb)
                else:
                    loss = crit(out, yb)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

    def _train_with_loader(
        self,
        model: nn.Module | List[nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        lr: float,
        beta_kl: float,
        early_stopping: Optional[dict] = None,
    ):
        # Support ensemble list
        if isinstance(model, list):
            for net in model:
                self._train_with_loader(net, train_loader, val_loader, epochs, lr, beta_kl, early_stopping)
            return

        es_cfg = {
            "enabled": False,
            "patience": 5,
            "min_delta": 0.0,
            "monitor": "auto",  # "auto" -> loss for regression, accuracy for classification
            "warmup": 0,
        }
        if isinstance(early_stopping, dict):
            es_cfg.update({k: early_stopping.get(k, v) for k, v in es_cfg.items()})

        # Decide monitor & direction
        monitor = str(es_cfg["monitor"]).lower()
        if monitor == "auto":
            mode = "min" if self._is_regression else "max"
            monitor = "loss" if self._is_regression else "accuracy"
        elif monitor in ("accuracy", "acc"):
            mode = "max"
        else:
            mode = "min"

        best_metric = float("inf") if mode == "min" else -float("inf")
        best_state = None
        bad_epochs = 0

        device = next(model.parameters()).device
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = self._criterion()

        for epoch in tqdm(range(epochs), desc="Epochs"):
            model.train()
            for batch in train_loader:
                xb = batch["sequence"].to(device).float()
                yb = batch.get("label")

                opt.zero_grad(set_to_none=True)
                out = model(xb)

                if self._is_regression:
                    yb = yb.to(device).float()
                    if isinstance(out, tuple):
                        pred, mu, logvar = out
                        pred = pred.squeeze()
                        loss = crit(pred, yb)
                        loss = self._apply_vae_kl(mu, logvar, loss, beta_kl)
                    else:
                        loss = crit(out.squeeze(), yb)
                else:
                    yb = yb.to(device).long()
                    loss = crit(out, yb)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            if es_cfg["enabled"] and val_loader is not None:
                metric, _direction = self._eval_on_val(model, val_loader, crit, monitor)

                # Warmup
                if epoch < es_cfg["warmup"]:
                    logger.info(
                        f"[val] epoch={epoch+1}/{epochs} warmup "
                        f"monitor={monitor} metric={metric:.6f}"
                    )
                    best_metric = metric
                    bad_epochs = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    continue

                improved = (
                    (metric < best_metric - es_cfg["min_delta"]) if mode == "min"
                    else (metric > best_metric + es_cfg["min_delta"])
                )

                if improved:
                    best_metric = metric
                    bad_epochs = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    logger.info(
                        f"[val] epoch={epoch+1}/{epochs} monitor={monitor} "
                        f"metric={metric:.6f} best*"
                    )
                else:
                    bad_epochs += 1
                    logger.info(
                        f"[val] epoch={epoch+1}/{epochs} monitor={monitor} "
                        f"metric={metric:.6f} no_improve ({bad_epochs}/{es_cfg['patience']})"
                    )

                if bad_epochs >= es_cfg["patience"]:
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    logger.info(
                        f"[val] early-stop at epoch {epoch+1} "
                        f"(best {monitor}={best_metric:.6f})"
                    )
                    break


    def _predict_arrays(self, model: nn.Module, X: np.ndarray, dropout: bool = False) -> np.ndarray:
        device = next(model.parameters()).device
        model.train() if dropout else model.eval()
        X_tensor = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            y = model(X_tensor)
            if isinstance(y, tuple):
                y = y[0]
        y = y.detach().cpu().numpy()
        # IMPORTANT: only squeeze for regression; keep (B,K) for classification
        if self._is_regression:
            return np.squeeze(y)
        return y  # shape (B,K)

    def _predict_loader(self, model: nn.Module, loader: DataLoader, dropout: bool = False) -> np.ndarray:
        device = next(model.parameters()).device
        model.train() if dropout else model.eval()
        outs = []
        with torch.no_grad():
            for batch in loader:
                xb = batch["sequence"].to(device).float()
                y = model(xb)
                if isinstance(y, tuple):
                    y = y[0]
                outs.append(y.detach().cpu().numpy())
        y = np.concatenate(outs, axis=0)
        if self._is_regression:
            return np.squeeze(y)
        return y  # keep (B,K)


    def _eval_on_val(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        crit: nn.Module,
        monitor: str,
    ) -> tuple[float, str]:
        """Returns (metric_value, direction). 'direction' is 'min' for loss or 'max' for accuracy."""
        monitor = str(monitor).lower()
        use_acc = (monitor in ("accuracy", "acc"))

        device = next(model.parameters()).device
        model.eval()

        if use_acc and not self._is_regression:
            # classification accuracy
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    xb = batch["sequence"].to(device).float()
                    yb = batch["label"].to(device).long()
                    logits = model(xb)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    pred = logits.argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    total += yb.numel()
            acc = (correct / max(total, 1)) if total > 0 else 0.0
            return acc, "max"

        # default: validation loss
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                xb = batch["sequence"].to(device).float()
                yb = batch["label"]
                out = model(xb)
                if self._is_regression:
                    yb = yb.to(device).float()
                    if isinstance(out, tuple):
                        pred, mu, logvar = out
                        pred = pred.squeeze()
                        loss = crit(pred, yb)
                        loss = self._apply_vae_kl(mu, logvar, loss, 0.0)
                    else:
                        loss = crit(out.squeeze(), yb)
                else:
                    yb = yb.to(device).long()
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = crit(out, yb)

                total_loss += loss.item()
                count += 1

        val_loss = total_loss / max(count, 1)
        return val_loss, "min"

    class _AveragingEnsemble(nn.Module):
        """Simple avg wrapper over a list of nets for explainability/prediction."""
        def __init__(self, nets: List[nn.Module]):
            super().__init__()
            self.nets = nn.ModuleList(nets)

        def forward(self, x):
            outs = []
            for net in self.nets:
                y = net(x)
                if isinstance(y, tuple):
                    y = y[0]
                outs.append(y)
            return torch.stack(outs, dim=0).mean(dim=0)

    def get_explainable_model(self, xai_method: Optional[str] = None) -> Optional[nn.Module]:
        """
        Returns an nn.Module suitable for attribution:
          - For torch lists (ensembles), wraps with averaging Module.
          - For single torch Module, returns as-is.
          - For sklearn models, returns None.
        """
        m = self.model
        if isinstance(m, list) and len(m) > 0 and isinstance(m[0], nn.Module):
            wrapper = ExperimentModel._AveragingEnsemble(m).to(next(m[0].parameters()).device)
            wrapper.eval()
            return wrapper
        if isinstance(m, nn.Module):
            return m
        return None

    def _infer_device(self):
        """Return a torch.device for the underlying torch model (or CPU for sklearn)."""
        if self.is_sklearn:
            return torch.device("cpu")
        m = self.model[0] if isinstance(self.model, list) else self.model
        return next(m.parameters()).device

    def _to_device_tensor(self, x):
        """
        Accept np.ndarray or torch.Tensor shaped (B, C, T),
        cast to float32 torch tensor on the correct device.
        """
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            t = x
        else:
            raise TypeError(f"Unsupported input type: {type(x)} (expected np.ndarray or torch.Tensor)")
        if t.dim() != 3:
            raise ValueError(f"Expected (B,C,T), got {tuple(t.shape)}")
        return t.to(self._infer_device()).float()

    def predict_numpy(self, x_bct):
        """
        Accepts (B,C,T) numpy or torch; returns:
          - regression: (B,)
          - classification: (B,K)
        Works for both torch and sklearn backends.
        """
        if self.is_sklearn:
            if isinstance(x_bct, torch.Tensor):
                x_bct = x_bct.detach().cpu().numpy()
            return self.predict(x_bct)

        self.model.eval()
        with torch.no_grad():
            X = self._to_device_tensor(x_bct)
            mdl = self.model
            if isinstance(mdl, list):  # simple avg inference
                outs = []
                for net in mdl:
                    y = net(X)
                    if isinstance(y, (list, tuple)):
                        y = y[0]
                    outs.append(y)
                y = torch.stack(outs, dim=0).mean(dim=0)
            else:
                y = mdl(X)
                if isinstance(y, (list, tuple)):
                    y = y[0]
            y = y.detach().cpu().numpy()
            if self._is_regression and y.ndim == 2 and y.shape[1] == 1:
                y = y[:, 0]
            return y

    def __call__(self, x_bct):
        """Allow direct callable use in metrics, e.g., model(x)."""
        return self.predict_numpy(x_bct)
