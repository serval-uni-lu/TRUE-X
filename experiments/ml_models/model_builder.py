# ml_models/model_builder.py
"""
Shared model factory used by both the tuner (run_tuner.py) and the
training script (train.py).

  build_model_with_params  — instantiate any registered model from a params dict
  load_arch_params         — read best_params JSON from saved_models/
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

from torch import nn

from ml_models.model_registry import Model
from ml_models.architectures import (
    MLP_LSTM_Attention, VAE_RUL,
    CNN_LSTM_Regressor, AttentionLSTMRegressor, TSTRegressor, TCNRegressor,
    LSTM_FCN_Regressor, TFT_Regressor, LSTM_Regressor, BiLSTM_Regressor,
    LinearRegressor,
    InceptionTime, Classifier_RESNET, Classifier_FCN, Classifier_MCDCNN,
    Classifier_TIMECNN, EncoderTSC, TST, Classifier_LSTM, Classifier_GRU,
    LogisticClassifier,
    make_rf_classifier, make_et_classifier,
    make_rf_regressor, make_et_regressor,
    make_xgb_classifier, make_xgb_regressor,
    make_lgbm_classifier, make_lgbm_regressor,
)


def _filter(d: Dict[str, Any], allowed: list) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k in allowed}


def _normalize_model_specific_params(model_enum: Model, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fix illegal param combos (e.g. TST: d_model must be divisible by n_heads)."""
    name = model_enum.name
    out = dict(params)
    if name in ("TST_REGRESSOR", "TST"):
        n_heads = int(out.get("n_heads", 1))
        if n_heads < 1:
            n_heads = 1
        if "d_model" in out:
            d_model = int(out["d_model"])
            if d_model % n_heads != 0:
                d_model = (d_model + n_heads - 1) // n_heads * n_heads
                out["d_model"] = d_model
    return out


def build_model_with_params(model_enum: Model,
                            input_shape: Tuple[int, int],
                            nb_classes: Optional[int],
                            params: Dict[str, Any],
                            task: str):
    """
    Instantiate a model using params dict.  Unknown keys are silently ignored
    via _filter — training params (epochs, lr, batch_size) pass through safely.
    Returns nn.Module, list[nn.Module] (ensemble), or sklearn Pipeline.
    """
    params = _normalize_model_specific_params(model_enum, params)

    if model_enum == Model.VI_ENSEMBLE:
        cfg = _filter(params, ["hidden_dim", "n_hidden_layers", "lstm_hidden_dim", "use_dropout"])
        return [MLP_LSTM_Attention(
            input_dim=input_shape[1], output_dim=1,
            hidden_dim=cfg.get("hidden_dim", 32),
            n_hidden_layers=cfg.get("n_hidden_layers", 2),
            lstm_hidden_dim=cfg.get("lstm_hidden_dim", 32),
            use_dropout=cfg.get("use_dropout", False),
        ) for _ in range(2)]

    if model_enum == Model.VI_DROPOUT:
        cfg = _filter(params, ["hidden_dim", "n_hidden_layers", "lstm_hidden_dim"])
        return MLP_LSTM_Attention(
            input_dim=input_shape[1], output_dim=1,
            hidden_dim=cfg.get("hidden_dim", 32),
            n_hidden_layers=cfg.get("n_hidden_layers", 2),
            lstm_hidden_dim=cfg.get("lstm_hidden_dim", 32),
            use_dropout=True,
        )

    if model_enum == Model.VAE_RUL:
        cfg = _filter(params, ["intermediate_dim", "latent_dim"])
        return VAE_RUL(
            timesteps=input_shape[0], input_dim=input_shape[1],
            intermediate_dim=cfg.get("intermediate_dim", 32),
            latent_dim=cfg.get("latent_dim", 8),
        )

    if model_enum == Model.CNN_LSTM_REGRESSOR:
        cfg = _filter(params, ["conv_channels", "conv_kernels", "lstm_hidden", "lstm_layers",
                               "bidirectional", "lstm_dropout", "head_hidden", "head_dropout"])
        return CNN_LSTM_Regressor(
            input_shape=input_shape,
            conv_channels=cfg.get("conv_channels", (64, 128)),
            conv_kernels=cfg.get("conv_kernels", (7, 5)),
            lstm_hidden=cfg.get("lstm_hidden", 128),
            lstm_layers=cfg.get("lstm_layers", 2),
            bidirectional=cfg.get("bidirectional", True),
            lstm_dropout=cfg.get("lstm_dropout", 0.1),
            head_hidden=cfg.get("head_hidden", None),
            head_dropout=cfg.get("head_dropout", 0.1),
        )

    if model_enum == Model.ATTENTION_LSTM_REGRESSOR:
        cfg = _filter(params, ["hidden_size", "num_layers", "bidirectional", "lstm_dropout",
                               "attn_hidden", "head_hidden", "head_dropout"])
        return AttentionLSTMRegressor(
            input_shape=input_shape,
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            bidirectional=cfg.get("bidirectional", True),
            lstm_dropout=cfg.get("lstm_dropout", 0.1),
            attn_hidden=cfg.get("attn_hidden", 64),
            head_hidden=cfg.get("head_hidden", None),
            head_dropout=cfg.get("head_dropout", 0.1),
        )

    if model_enum == Model.TST_REGRESSOR:
        cfg = _filter(params, ["d_model", "n_heads", "num_layers", "d_ff", "dropout",
                               "patch_len", "stride", "use_cls_token", "emb_dropout",
                               "head_hidden", "head_dropout"])
        return TSTRegressor(
            input_shape=input_shape,
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 8),
            num_layers=cfg.get("num_layers", 4),
            d_ff=cfg.get("d_ff", 256),
            dropout=cfg.get("dropout", 0.1),
            patch_len=cfg.get("patch_len", 16),
            stride=cfg.get("stride", 8),
            use_cls_token=cfg.get("use_cls_token", False),
            emb_dropout=cfg.get("emb_dropout", 0.1),
            head_hidden=cfg.get("head_hidden", None),
            head_dropout=cfg.get("head_dropout", 0.1),
        )

    if model_enum == Model.TCN_REGRESSOR:
        cfg = _filter(params, ["channels", "kernel_size", "dropout", "head_hidden", "head_dropout"])
        return TCNRegressor(
            input_shape=input_shape,
            channels=cfg.get("channels", (64, 64, 128)),
            kernel_size=cfg.get("kernel_size", 3),
            dropout=cfg.get("dropout", 0.1),
            head_hidden=cfg.get("head_hidden", 64),
            head_dropout=cfg.get("head_dropout", 0.1),
        )

    if model_enum == Model.LSTM_FCN_REGRESSOR:
        cfg = _filter(params, ["lstm_hidden", "lstm_layers", "bidirectional", "lstm_dropout",
                               "fcn_channels", "kernels", "head_hidden", "head_dropout"])
        return LSTM_FCN_Regressor(
            input_shape=input_shape,
            lstm_hidden=cfg.get("lstm_hidden", 128),
            lstm_layers=cfg.get("lstm_layers", 1),
            bidirectional=cfg.get("bidirectional", True),
            lstm_dropout=cfg.get("lstm_dropout", 0.1),
            fcn_channels=tuple(cfg.get("fcn_channels", (128, 256, 128))),
            kernels=tuple(cfg.get("kernels", (9, 5, 3))),
            head_hidden=cfg.get("head_hidden", None),
            head_dropout=cfg.get("head_dropout", 0.1),
        )

    if model_enum == Model.TFT_REGRESSOR:
        cfg = _filter(params, ["d_model", "n_heads", "n_attn_layers", "d_ff",
                               "lstm_hidden", "lstm_layers", "dropout", "head_hidden"])
        return TFT_Regressor(
            input_shape=input_shape,
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 4),
            n_attn_layers=cfg.get("n_attn_layers", 2),
            d_ff=cfg.get("d_ff", 256),
            lstm_hidden=cfg.get("lstm_hidden", 128),
            lstm_layers=cfg.get("lstm_layers", 1),
            dropout=cfg.get("dropout", 0.1),
            causal_attention=True,
            head_hidden=cfg.get("head_hidden", None),
        )

    if model_enum == Model.LSTM_REGRESSOR:
        cfg = _filter(params, ["hidden_size", "num_layers", "bidirectional", "dropout",
                               "temporal_pool", "head_hidden"])
        return LSTM_Regressor(
            input_shape=input_shape,
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            bidirectional=cfg.get("bidirectional", True),
            dropout=cfg.get("dropout", 0.1),
            temporal_pool=cfg.get("temporal_pool", "last"),
            head_hidden=cfg.get("head_hidden", None),
        )

    if model_enum == Model.BI_LSTM_REGRESSOR:
        cfg = _filter(params, ["hidden_size", "num_layers", "dropout", "temporal_pool", "head_hidden"])
        return BiLSTM_Regressor(
            input_shape=input_shape,
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.1),
            temporal_pool=cfg.get("temporal_pool", "last"),
            head_hidden=cfg.get("head_hidden", None),
        )

    if model_enum == Model.LINEAR_REGRESSOR:
        cfg = _filter(params, ["use_bias"])
        return LinearRegressor(input_shape=input_shape, use_bias=cfg.get("use_bias", True))

    if model_enum == Model.INCEPTIONTIME:
        cfg = _filter(params, ["n_residual_blocks", "nb_filters", "kernel_sizes", "bottleneck"])
        return InceptionTime(
            input_shape=input_shape,
            nb_classes=nb_classes,
            n_residual_blocks=cfg.get("n_residual_blocks", 3),
            nb_filters=cfg.get("nb_filters", 32),
            kernel_sizes=cfg.get("kernel_sizes", [9, 19, 39]),
            bottleneck=cfg.get("bottleneck", True),
        )

    if model_enum == Model.RESNET:
        cfg = _filter(params, ["block_channels", "kernel_sizes"])
        return Classifier_RESNET(
            input_shape=input_shape,
            nb_classes=nb_classes,
            block_channels=tuple(cfg.get("block_channels", (64, 128, 128))),
            kernel_sizes=tuple(cfg.get("kernel_sizes", (9, 5, 3))),
        )

    if model_enum == Model.FCN:
        cfg = _filter(params, ["channels", "kernels"])
        return Classifier_FCN(
            input_shape=input_shape,
            nb_classes=nb_classes,
            channels=tuple(cfg.get("channels", (128, 256, 128))),
            kernels=tuple(cfg.get("kernels", (9, 5, 3))),
        )

    if model_enum == Model.MCD_CNN:
        cfg = _filter(params, ["branch_filters", "branch_kernels", "pool",
                               "dense_units", "dropout", "use_bn"])
        return Classifier_MCDCNN(
            input_shape=input_shape,
            nb_classes=nb_classes,
            branch_filters=tuple(cfg.get("branch_filters", (8, 8))),
            branch_kernels=tuple(cfg.get("branch_kernels", (5, 5))),
            pool=cfg.get("pool", 2),
            dense_units=cfg.get("dense_units", 732),
            dropout=cfg.get("dropout", 0.0),
            use_bn=cfg.get("use_bn", False),
        )

    if model_enum == Model.TIME_CNN:
        cfg = _filter(params, ["filters", "kernel_size", "pool_size", "dense_units", "dropout"])
        return Classifier_TIMECNN(
            input_shape=input_shape,
            nb_classes=nb_classes,
            filters=tuple(cfg.get("filters", (6, 12))),
            kernel_size=cfg.get("kernel_size", 7),
            pool_size=cfg.get("pool_size", 3),
            dense_units=cfg.get("dense_units", 128),
            dropout=cfg.get("dropout", 0.0),
        )

    if model_enum == Model.ENCODER:
        cfg = _filter(params, ["filters", "kernels", "dropout", "use_instancenorm"])
        return EncoderTSC(
            input_shape=input_shape,
            nb_classes=nb_classes,
            filters=tuple(cfg.get("filters", (128, 256, 512))),
            kernels=tuple(cfg.get("kernels", (5, 11, 21))),
            dropout=cfg.get("dropout", 0.2),
            use_instancenorm=cfg.get("use_instancenorm", True),
        )

    if model_enum == Model.TST:
        cfg = _filter(params, ["d_model", "n_heads", "num_layers", "d_ff", "dropout",
                               "patch_len", "stride", "use_cls_token", "emb_dropout"])
        return TST(
            input_shape=input_shape,
            nb_classes=nb_classes,
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 8),
            num_layers=cfg.get("num_layers", 4),
            d_ff=cfg.get("d_ff", 256),
            dropout=cfg.get("dropout", 0.1),
            patch_len=cfg.get("patch_len", 16),
            stride=cfg.get("stride", 8),
            use_cls_token=cfg.get("use_cls_token", True),
            emb_dropout=cfg.get("emb_dropout", 0.1),
        )

    if model_enum == Model.LSTM:
        cfg = _filter(params, ["hidden_size", "num_layers", "bidirectional", "dropout", "temporal_pool"])
        return Classifier_LSTM(
            input_shape=input_shape,
            nb_classes=nb_classes,
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            bidirectional=cfg.get("bidirectional", True),
            dropout=cfg.get("dropout", 0.1),
            temporal_pool=cfg.get("temporal_pool", "last"),
        )

    if model_enum == Model.GRU:
        cfg = _filter(params, ["hidden_size", "num_layers", "bidirectional", "dropout", "temporal_pool"])
        return Classifier_GRU(
            input_shape=input_shape,
            nb_classes=nb_classes,
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            bidirectional=cfg.get("bidirectional", True),
            dropout=cfg.get("dropout", 0.1),
            temporal_pool=cfg.get("temporal_pool", "last"),
        )

    if model_enum == Model.LOGISTIC_CLASSIFIER:
        cfg = _filter(params, ["use_bias"])
        return LogisticClassifier(
            input_shape=input_shape,
            nb_classes=nb_classes,
            use_bias=cfg.get("use_bias", True),
        )

    if model_enum == Model.RF_CLASSIFIER:
        cfg = _filter(params, ["n_estimators", "max_depth"])
        return make_rf_classifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", None),
        )

    if model_enum == Model.ET_CLASSIFIER:
        cfg = _filter(params, ["n_estimators", "max_depth"])
        return make_et_classifier(
            n_estimators=cfg.get("n_estimators", 500),
            max_depth=cfg.get("max_depth", None),
        )

    if model_enum == Model.RF_REGRESSOR:
        cfg = _filter(params, ["n_estimators", "max_depth"])
        return make_rf_regressor(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", None),
        )

    if model_enum == Model.ET_REGRESSOR:
        cfg = _filter(params, ["n_estimators", "max_depth"])
        return make_et_regressor(
            n_estimators=cfg.get("n_estimators", 500),
            max_depth=cfg.get("max_depth", None),
        )

    if model_enum == Model.XGB_CLASSIFIER:
        cfg = _filter(params, ["n_estimators", "learning_rate", "max_depth",
                               "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"])
        return make_xgb_classifier(**cfg)

    if model_enum == Model.XGB_REGRESSOR:
        cfg = _filter(params, ["n_estimators", "learning_rate", "max_depth",
                               "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"])
        return make_xgb_regressor(**cfg)

    if model_enum == Model.LGBM_CLASSIFIER:
        cfg = _filter(params, ["n_estimators", "learning_rate", "num_leaves",
                               "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"])
        return make_lgbm_classifier(**cfg)

    if model_enum == Model.LGBM_REGRESSOR:
        cfg = _filter(params, ["n_estimators", "learning_rate", "num_leaves",
                               "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"])
        return make_lgbm_regressor(**cfg)

    raise NotImplementedError(f"build_model_with_params not implemented for {model_enum.name}")


def load_arch_params(params_root: str, dataset: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Load best_params JSON for a (dataset, model) pair.
    Returns the params dict, or None if the file doesn't exist.
    """
    path = os.path.join(params_root, f"{dataset}_{model_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("params", {})
