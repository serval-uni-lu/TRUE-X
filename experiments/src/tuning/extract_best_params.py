# tuning/extract_best_params.py
"""
Writes tuning/saved_models/best_params_{dataset}_{model}.json for every
experiment in tuning_config.yaml using one of two sources:

  1. Optuna CSV (tuning_results/) — uses the best COMPLETE trial's params.
  2. Hardcoded defaults (from build_model_with_params) — used when no CSV
     exists yet (e.g. CWRU_12k before its first tuning run, or the
     probabilistic models that have never been tuned).

Usage (run from src/):
    python -m tuning.extract_best_params
"""

import ast
import csv
import json
import os
import yaml

TUNING_DIR  = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(TUNING_DIR, "tuning_results")
SAVED_DIR   = os.path.join(TUNING_DIR, "../..", "saved_models")
CONFIG_PATH = os.path.join(TUNING_DIR, "tuning_config.yaml")

# Hardcoded defaults mirror the .get("param", default) calls in
# build_model_with_params (run_tuner.py). Keep in sync if those change.
_MODEL_DEFAULTS = {
    "VI_ENSEMBLE":              {"hidden_dim": 32, "n_hidden_layers": 2, "lstm_hidden_dim": 32, "use_dropout": False},
    "VI_DROPOUT":               {"hidden_dim": 32, "n_hidden_layers": 2, "lstm_hidden_dim": 32},
    "VAE_RUL":                  {"intermediate_dim": 32, "latent_dim": 8},
    "CNN_LSTM_REGRESSOR":       {"conv_channels": [64, 128], "conv_kernels": [7, 5], "lstm_hidden": 128, "lstm_layers": 2, "bidirectional": True, "lstm_dropout": 0.1, "head_hidden": None, "head_dropout": 0.1},
    "ATTENTION_LSTM_REGRESSOR": {"hidden_size": 128, "num_layers": 2, "bidirectional": True, "lstm_dropout": 0.1, "attn_hidden": 64, "head_hidden": None, "head_dropout": 0.1},
    "TST_REGRESSOR":            {"d_model": 128, "n_heads": 8, "num_layers": 4, "d_ff": 256, "dropout": 0.1, "patch_len": 16, "stride": 8, "use_cls_token": False, "emb_dropout": 0.1, "head_hidden": None, "head_dropout": 0.1},
    "TCN_REGRESSOR":            {"channels": [64, 64, 128], "kernel_size": 3, "dropout": 0.1, "head_hidden": 64, "head_dropout": 0.1},
    "LSTM_FCN_REGRESSOR":       {"lstm_hidden": 128, "lstm_layers": 1, "bidirectional": True, "lstm_dropout": 0.1, "fcn_channels": [128, 256, 128], "kernels": [9, 5, 3], "head_hidden": None, "head_dropout": 0.1},
    "TFT_REGRESSOR":            {"d_model": 128, "n_heads": 4, "n_attn_layers": 2, "d_ff": 256, "lstm_hidden": 128, "lstm_layers": 1, "dropout": 0.1, "head_hidden": None},
    "LSTM_REGRESSOR":           {"hidden_size": 128, "num_layers": 2, "bidirectional": True, "dropout": 0.1, "temporal_pool": "last", "head_hidden": None},
    "BI_LSTM_REGRESSOR":        {"hidden_size": 128, "num_layers": 2, "dropout": 0.1, "temporal_pool": "last", "head_hidden": None},
    "LINEAR_REGRESSOR":         {"use_bias": True},
    "INCEPTIONTIME":            {"n_residual_blocks": 3, "nb_filters": 32, "kernel_sizes": [9, 19, 39], "bottleneck": True},
    "RESNET":                   {"block_channels": [64, 128, 128], "kernel_sizes": [9, 5, 3]},
    "FCN":                      {"channels": [128, 256, 128], "kernels": [9, 5, 3]},
    "MCD_CNN":                  {"branch_filters": [8, 8], "branch_kernels": [5, 5], "pool": 2, "dense_units": 732, "dropout": 0.0, "use_bn": False},
    "TIME_CNN":                 {"filters": [6, 12], "kernel_size": 7, "pool_size": 3, "dense_units": 128, "dropout": 0.0},
    "ENCODER":                  {"filters": [128, 256, 512], "kernels": [5, 11, 21], "dropout": 0.2, "use_instancenorm": True},
    "TST":                      {"d_model": 128, "n_heads": 8, "num_layers": 4, "d_ff": 256, "dropout": 0.1, "patch_len": 16, "stride": 8, "use_cls_token": True, "emb_dropout": 0.1},
    "LSTM":                     {"hidden_size": 128, "num_layers": 2, "bidirectional": True, "dropout": 0.1, "temporal_pool": "last"},
    "GRU":                      {"hidden_size": 128, "num_layers": 2, "bidirectional": True, "dropout": 0.1, "temporal_pool": "last"},
    "LOGISTIC_CLASSIFIER":      {"use_bias": True},
    "RF_CLASSIFIER":            {"n_estimators": 300, "max_depth": None},
    "ET_CLASSIFIER":            {"n_estimators": 500, "max_depth": None},
    "XGB_CLASSIFIER":           {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "LGBM_CLASSIFIER":          {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 63, "subsample": 0.9, "colsample_bytree": 0.9},
    "RF_REGRESSOR":             {"n_estimators": 300, "max_depth": None},
    "ET_REGRESSOR":             {"n_estimators": 500, "max_depth": None},
    "XGB_REGRESSOR":            {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
    "LGBM_REGRESSOR":           {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 63, "subsample": 0.9, "colsample_bytree": 0.9},
}


def _cast(value: str):
    if value == "True":
        return True
    if value == "False":
        return False
    if value in ("", "None"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith("(") and value.endswith(")")) or \
       (value.startswith("[") and value.endswith("]")):
        try:
            parsed = ast.literal_eval(value)
            return list(parsed) if isinstance(parsed, tuple) else parsed
        except Exception:
            pass
    return value


def best_params_from_csv(csv_path: str, direction: str) -> dict | None:
    complete = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("state") == "TrialState.COMPLETE":
                complete.append(row)
    if not complete:
        return None
    pick = min if direction == "minimize" else max
    best = pick(complete, key=lambda r: float(r["value"]))
    skip = {"trial_number", "value", "state"}
    return {k: _cast(v) for k, v in best.items() if k not in skip}


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    direction_map = {
        (exp["dataset"], exp["model"]): exp.get("direction", "minimize")
        for exp in cfg["experiments"]
    }

    os.makedirs(SAVED_DIR, exist_ok=True)

    from_csv      = []
    from_defaults = []
    no_defaults   = []
    no_trials     = []

    for (dataset, model), direction in sorted(direction_map.items()):
        csv_name = f"optuna_{dataset}_{model}_tuning_results.csv"
        csv_path = os.path.join(RESULTS_DIR, csv_name)

        params = None
        source = None

        if os.path.exists(csv_path):
            params = best_params_from_csv(csv_path, direction)
            if params is None:
                no_trials.append(f"{dataset}/{model}")
                continue
            source = "csv"
        else:
            defaults = _MODEL_DEFAULTS.get(model)
            if defaults is None:
                no_defaults.append(f"{dataset}/{model}")
                continue
            params = dict(defaults)
            source = "defaults"

        out = {
            "dataset":   dataset,
            "model":     model,
            "direction": direction,
            "source":    source,
            "params":    params,
        }
        out_path = os.path.join(SAVED_DIR, f"best_params_{dataset}_{model}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        if source == "csv":
            from_csv.append(f"{dataset}/{model}")
        else:
            from_defaults.append(f"{dataset}/{model}")

    print(f"Written {len(from_csv) + len(from_defaults)} JSON files total.")
    print(f"  From Optuna CSV:       {len(from_csv)}")
    print(f"  From hardcoded defaults: {len(from_defaults)}")
    if no_trials:
        print(f"  No COMPLETE trials (skipped): {no_trials}")
    if no_defaults:
        print(f"  No CSV and no defaults entry (skipped): {no_defaults}")


if __name__ == "__main__":
    main()
