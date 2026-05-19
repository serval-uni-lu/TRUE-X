import json
import os
from typing import Any, Dict, Optional, Tuple

from torch import nn

from ml_models.model_registry import Model
from ml_models.architectures import (
    InceptionTime, Classifier_RESNET,
    TST, Classifier_LSTM,
    LogisticClassifier,
)


def _filter(d: Dict[str, Any], allowed: list) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k in allowed}


def _normalize_model_specific_params(model_enum: Model, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fix illegal param combos (e.g. TST: d_model must be divisible by n_heads)."""
    out = dict(params)
    if model_enum == Model.TST:
        n_heads = int(out.get("n_heads", 1))
        if n_heads < 1:
            n_heads = 1
        if "d_model" in out:
            d_model = int(out["d_model"])
            if d_model % n_heads != 0:
                d_model = (d_model + n_heads - 1) // n_heads * n_heads
                out["d_model"] = d_model
    return out


def build_model_with_params(
    model_enum: Model,
    input_shape: Tuple[int, int],
    nb_classes: Optional[int],
    params: Dict[str, Any],
    task: str,
):
    """
    Instantiate a model from a params dict.
    Unknown keys are silently ignored via _filter.
    Returns an nn.Module.
    """
    params = _normalize_model_specific_params(model_enum, params)

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

    if model_enum == Model.LOGISTIC_CLASSIFIER:
        cfg = _filter(params, ["use_bias"])
        return LogisticClassifier(
            input_shape=input_shape,
            nb_classes=nb_classes,
            use_bias=cfg.get("use_bias", True),
        )

    raise NotImplementedError(f"build_model_with_params not implemented for {model_enum.name}")


def load_arch_params(params_root: str, dataset: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Load best_params JSON for a (dataset, model) pair, or None if not found."""
    path = os.path.join(params_root, f"{dataset}_{model_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("params", {})
