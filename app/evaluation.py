"""
Live evaluation backend for TRUE-X Entry B.

Adds XURL to sys.path and delegates to XURL's existing infrastructure:
ExperimentModel, ExplainerFactory, and ExpliTest metrics.
"""
import os
import re
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("truex.evaluation")

_APP_DIR  = Path(__file__).resolve().parent   # TRUE-X/app/
_ROOT     = _APP_DIR.parent                   # TRUE-X/ (repo root)
XURL_PATH      = str(_ROOT / "experiments")
EXPLITEST_PATH = str(_ROOT / "explitest")
DATASETS_CFG   = str(_ROOT / "experiments" / "configs" / "datasets_config.yaml")


def _ensure_xurl_on_path():
    for p in (XURL_PATH, EXPLITEST_PATH):
        if p not in sys.path:
            sys.path.insert(0, p)


def _normalize_metric(name: str) -> str:
    """Map XURL metric names to Entry A display names.

    Examples:
        'Average Sensitivity (Reg)'   -> 'Average Sensitivity'
        'Sparseness (Reg, Elem)'       -> 'Sparseness (Elem)'
        'Complexity (Classif, Chan)'   -> 'Complexity (Chan)'
    """
    # 'Foo (Reg, Bar)' -> 'Foo (Bar)'
    name = re.sub(r"\((Reg|Classif),\s*", "(", name)
    # 'Foo (Reg)' -> 'Foo'
    name = re.sub(r"\s*\((Reg|Classif)\)", "", name)
    return name.strip()


def _load_json_meta(weight_path: str) -> dict:
    p = Path(weight_path).with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------
# Performance helpers
# ---------------------------------------------------------------

def _compute_performance(exp, X_exp: np.ndarray, Y_exp, task: str):
    """
    Returns (metric_name, value) for the model on the explicand set.
    - Classification → ("Accuracy", accuracy)   — higher is better, no sign change
    - Regression     → ("RMSE",    -rmse)        — negated so higher = better (plots.py requirement)
    Returns (None, None) if labels are unavailable or prediction fails.
    """
    if Y_exp is None or len(Y_exp) == 0:
        return None, None
    try:
        preds = exp.predict_numpy(X_exp)
        if task.lower().startswith("classif"):
            pred_labels = preds.argmax(axis=1) if preds.ndim == 2 else preds.astype(int)
            correct = (pred_labels.astype(int) == Y_exp.astype(int)).mean()
            return "Accuracy", float(correct)
        else:
            rmse = float(np.sqrt(np.mean((Y_exp.squeeze() - preds.squeeze()) ** 2)))
            return "RMSE", -rmse  # negated: higher is better in plots.py
    except Exception as e:
        logger.warning(f"Performance computation failed: {e}")
        return None, None


# ---------------------------------------------------------------
# B1 — benchmark datasets + pre-trained models
# ---------------------------------------------------------------

def _run_b1(
    dataset_info: dict,
    window_info: dict,
    model_info: dict,
    explainer_info: dict,
    max_explain: int = 200,
) -> pd.DataFrame:
    _ensure_xurl_on_path()

    import yaml
    from run_xai_evaluation import (
        get_loaders_from_config,
        build_exp_model_with_params,
        load_weights_into_exp,
        compute_attributions_for_method,
        run_xai_metrics_for_batch,
        build_train_background,
        _loader_to_numpy,
        _loader_has_labels,
        infer_input_shape_from_loader,
    )
    from ml_models.model_registry import ModelRegistry

    with open(DATASETS_CFG) as f:
        datasets_cfg = yaml.safe_load(f)

    # Resolve relative data paths → absolute so adapters work regardless of CWD,
    # and disable DataLoader multiprocessing (incompatible with Streamlit).
    for entry in datasets_cfg.values():
        args = entry.get("args") or {}
        for key in ("root", "data_dir"):
            if key in args and not os.path.isabs(str(args[key])):
                args[key] = str((Path(XURL_PATH) / args[key]).resolve())
        if "num_workers" in args:
            args["num_workers"] = 0

    dataset_name = dataset_info["dataset_name"]
    methods      = explainer_info["methods"]

    loaders_entry = get_loaders_from_config(dataset_name, datasets_cfg)

    train_loader  = loaders_entry["train"]
    val_loader    = loaders_entry["val"]
    test_loader   = loaders_entry.get("test")
    nb_classes    = loaders_entry.get("nb_classes")
    task          = loaders_entry["task"]  # lowercase from YAML

    input_shape = infer_input_shape_from_loader(train_loader)  # (T, C)

    explicand_loader = (
        test_loader
        if test_loader is not None and _loader_has_labels(test_loader)
        else val_loader
    )
    X_exp, Y_exp = _loader_to_numpy(explicand_loader, n_max=max_explain)
    bg_small, baseline = build_train_background(train_loader, max_pool=2000, kmeans_k=50)

    rows = []
    for model_entry in model_info["models"]:
        architecture = model_entry["architecture"]
        weight_path  = model_entry["weight_path"]

        if weight_path is None:
            logger.warning(f"No weight file for {architecture} — skipped.")
            continue

        try:
            meta       = _load_json_meta(weight_path)
            params     = meta.get("params", {})
            model_enum = ModelRegistry.get_enum_by_name(architecture)
            exp        = build_exp_model_with_params(model_enum, input_shape, nb_classes, params, task)
            load_weights_into_exp(exp, weight_path)
            logger.info(f"Loaded {architecture} from {Path(weight_path).name}")
        except Exception as e:
            logger.error(f"Could not load {architecture}: {e}")
            continue

        # ---- Performance score (model-level, repeated per explainer) ----
        perf_metric, perf_value = _compute_performance(exp, X_exp, Y_exp, task)

        for method in methods:
            try:
                A       = compute_attributions_for_method(exp, method, X_exp, bg_small, baseline)
                metrics = run_xai_metrics_for_batch(exp, X_exp, A, train_loader=train_loader)
                for raw_name, arr in metrics.items():
                    rows.append({
                        "Dataset":   dataset_name,
                        "Model":     architecture,
                        "Explainer": method,
                        "Metric":    _normalize_metric(raw_name),
                        "Value":     float(np.nanmean(arr)),
                    })
                # Repeat performance row for this (model, explainer) pair
                if perf_metric is not None:
                    rows.append({
                        "Dataset":   dataset_name,
                        "Model":     architecture,
                        "Explainer": method,
                        "Metric":    perf_metric,
                        "Value":     perf_value,
                    })
            except Exception as e:
                logger.error(f"Failed {architecture}/{method}: {e}")
                continue

    return pd.DataFrame(rows, columns=["Dataset", "Model", "Explainer", "Metric", "Value"])


# ---------------------------------------------------------------
# B2 — custom dataset (stub)
# ---------------------------------------------------------------

def _run_b2(*_args, **_kwargs) -> pd.DataFrame:
    raise NotImplementedError(
        "Custom dataset evaluation (B2) is not yet implemented. "
        "Please use a benchmark dataset for now."
    )


# ---------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------

def run_evaluation(
    dataset_info: dict,
    window_info: dict,
    model_info: dict,
    explainer_info: dict,
    max_explain: int = 200,
) -> pd.DataFrame:
    if dataset_info["mode"] == "benchmark":
        return _run_b1(dataset_info, window_info, model_info, explainer_info, max_explain)
    return _run_b2(dataset_info, window_info, model_info, explainer_info)
