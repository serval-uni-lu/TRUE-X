# tuning/run_tuner.py
"""
Optuna tuner for XRUL models with:
- Per-dataset / per-model search spaces (from tuning_config.yaml)
- Optional early stopping (requested via YAML; falls back if unsupported)
- Robust trial error handling (OOM / NaNs / adapter issues) via TrialPruned
- Clean logging & CSV logging of all trials
- Retrain best params and save model artifact
- CLI overrides for trials / epochs (for dry runs / smoke tests)
"""

import os
import gc
import csv
import json
import math
import yaml
import joblib
import logging
import argparse
import optuna
import numpy as np
from typing import Any, Dict, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, accuracy_score
from optuna.exceptions import TrialPruned
from optuna.trial import TrialState

from ml_models.model_registry import Model, ModelRegistry
from ml_models.experiment_model import ExperimentModel
from ml_models.model_builder import build_model_with_params, _normalize_model_specific_params

logger = logging.getLogger("xrul.tuner")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

def load_yaml_file(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def import_callable(path: str):
    """'package.module:func_name' -> callable"""
    import importlib
    module_path, func_name = path.split(":")
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)

def standardize_adapter_return(ret):
    """
    Normalize various adapter returns to:
      (train_loader, val_loader, test_loader_or_None, nb_classes_or_None)
    """
    tl = vl = te = None
    nb_classes = None

    if isinstance(ret, dict):
        tl = ret.get("train_loader")
        vl = ret.get("val_loader")
        te = ret.get("test_loader")
        nb_classes = ret.get("nb_classes")

    elif isinstance(ret, (tuple, list)):
        dls = [x for x in ret if isinstance(x, DataLoader)]
        if len(dls) >= 1:
            tl = dls[0]
        if len(dls) >= 2:
            vl = dls[1]
        if len(dls) >= 3:
            te = dls[2]
        for x in ret:
            if isinstance(x, (int, np.integer)):
                nb_classes = int(x)
                break

    elif isinstance(ret, DataLoader):
        tl = ret
        vl = None

    if tl is None or vl is None:
        raise ValueError("Adapter did not return at least (train_loader, val_loader).")

    return tl, vl, te, nb_classes

def collect_labels(loader) -> np.ndarray:
    ys = []
    for batch in loader:
        y = batch["label"]
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        if isinstance(y, np.ndarray) and y.ndim > 1:  # multitask -> take first
            y = y[:, 0]
        ys.append(y)
    return np.concatenate(ys, axis=0)

def infer_input_shape_from_loader(loader) -> Tuple[int, int]:
    """Return (T, C) from a batch shaped (B, C, T)."""
    batch = next(iter(loader))
    x = batch["sequence"]
    if x.dim() != 3:
        raise ValueError(f"expected (B,C,T), got {tuple(x.shape)}")
    _, C, T = x.shape
    return (T, C)

def make_loaders(dataset_name: str,
                 datasets_cfg: dict,
                 arg_overrides: Optional[Dict[str, Any]] = None):
    if dataset_name not in datasets_cfg:
        raise ValueError(f"Dataset '{dataset_name}' not found in configs/datasets_config.yaml")
    entry = datasets_cfg[dataset_name]
    adapter_path = entry["adapter"]
    task = entry["task"]
    args = dict(entry.get("args", {}) or {})
    if arg_overrides:
        args.update(arg_overrides)

    adapter = import_callable(adapter_path)
    ret = adapter(**args)
    train_loader, val_loader, test_loader, nb_classes = standardize_adapter_return(ret)

    # infer classes if missing (classification)
    if task.lower() == "classification" and nb_classes is None:
        y = collect_labels(train_loader)
        nb_classes = int(np.max(y)) + 1
    return train_loader, val_loader, test_loader, nb_classes, task

def _to_immutable(x):
    """Convert lists to tuples recursively so Optuna categorical choices are immutable."""
    if isinstance(x, list):
        return tuple(_to_immutable(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_immutable(v) for k, v in x.items()}
    return x

def get_search_params(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    for k, spec in (space or {}).items():
        t = spec["type"]
        if t == "loguniform":
            params[k] = trial.suggest_float(k, float(spec["low"]), float(spec["high"]), log=True)
        elif t == "uniform":
            params[k] = trial.suggest_float(k, float(spec["low"]), float(spec["high"]))
        elif t == "int":
            # support step if provided
            step = int(spec.get("step", 1))
            params[k] = trial.suggest_int(k, int(spec["low"]), int(spec["high"]), step=step)
        elif t == "categorical":
            choices = [_to_immutable(c) for c in spec["choices"]]
            params[k] = trial.suggest_categorical(k, choices)
        else:
            raise ValueError(f"Unknown search space type: {t}")
    return params

def save_model_object(model_obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # torch nn.Module or list[nn.Module]
    if isinstance(model_obj, list) and all(isinstance(m, nn.Module) for m in model_obj):
        for idx, net in enumerate(model_obj):
            torch.save(net.state_dict(), f"{path}_ensemble{idx}.pt")
        return
    if isinstance(model_obj, nn.Module):
        torch.save(model_obj.state_dict(), path + ".pt")
        return
    # sklearn
    joblib.dump(model_obj, path + ".pkl")

def _rebuild_loader(loader: Optional[DataLoader], batch_size: int, *, shuffle: bool) -> Optional[DataLoader]:
    """Recreate a DataLoader with a different batch_size but same dataset/collate/worker settings."""
    if loader is None:
        return None
    try:
        return DataLoader(
            loader.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=getattr(loader, "num_workers", 0),
            pin_memory=getattr(loader, "pin_memory", False),
            drop_last=getattr(loader, "drop_last", False),
            collate_fn=getattr(loader, "collate_fn", None),
        )
    except Exception as e:
        logger.warning(f"Couldn't rebuild DataLoader (batch_size={batch_size}): {e}")
        return loader

def _fit_with_optional_es(exp_model, train_loader, val_loader, *, epochs: int, lr: float, es_cfg: dict):
    """
    Call ExperimentModel.fit with 'early_stopping' dict. Fallback if signature
    doesn't accept it (older build).
    """
    try:
        return exp_model.fit(
            train_loader,
            epochs=epochs,
            lr=lr,
            val_loader=val_loader,
            early_stopping=es_cfg,
        )
    except TypeError:
        return exp_model.fit(train_loader, epochs=epochs, lr=lr, val_loader=val_loader)

def _validate_config(cfg: dict) -> None:
    valid_types = {"loguniform", "uniform", "int", "categorical"}
    valid_directions = {"minimize", "maximize"}
    errors = []
    for exp in cfg.get("experiments", []):
        label = f"{exp.get('dataset', '?')}/{exp.get('model', '?')}"
        model_name = exp.get("model", "")
        try:
            ModelRegistry.get_enum_by_name(model_name)
        except ValueError:
            errors.append(f"{label}: unknown model '{model_name}'")
        direction = exp.get("direction", "minimize").lower()
        if direction not in valid_directions:
            errors.append(f"{label}: invalid direction '{direction}'")
        for param, spec in (exp.get("search_space") or {}).items():
            t = spec.get("type", "")
            if t not in valid_types:
                errors.append(f"{label}: param '{param}' has unknown type '{t}'")
    if errors:
        raise ValueError(
            "tuning_config.yaml validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


class Tuner:
    def __init__(self, config_path: str = "tuning/tuning_config.yaml",
                 datasets_cfg_path: str = "configs/datasets_config.yaml",
                 override_trials: Optional[int] = None,
                 override_epochs: Optional[int] = None,
                 filter_dataset: Optional[str] = None,
                 filter_model: Optional[str] = None):
        self.cfg = load_yaml_file(config_path)
        _validate_config(self.cfg)
        self.datasets_cfg = load_yaml_file(datasets_cfg_path)
        self.override_trials = override_trials
        self.override_epochs = override_epochs
        self.filter_dataset = filter_dataset
        self.filter_model = filter_model
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else
                  "cpu")
        logger.info(f"[tuner init] device={device} | torch={torch.__version__} | optuna={optuna.__version__}")

    def _prepare_data(self, dataset: str, dataset_arg_overrides: Optional[Dict[str, Any]]):
        return make_loaders(dataset, self.datasets_cfg, dataset_arg_overrides)

    def _build_exp_model(self, model_enum: Model, input_shape, nb_classes, params, task):
        # Create base ExperimentModel to get flags & training utilities
        exp_model = ExperimentModel(model_type=model_enum, input_shape=input_shape, nb_classes=nb_classes)
        try:
            # normalize illegal combos first
            params = _normalize_model_specific_params(model_enum, params)
            model = build_model_with_params(model_enum, input_shape, nb_classes, params, task)
            # swap in our parameterized model
            exp_model.model = model
            # if sklearn, mark flags accordingly
            if not isinstance(model, nn.Module):
                exp_model.is_sklearn = True
                exp_model.expects_3d = bool(getattr(model, "expects_3d", False))
        except NotImplementedError:
            # fall back to default model from ExperimentModel factory
            pass
        return exp_model

    def _score(self, task: str, y_true: np.ndarray, y_pred: np.ndarray, direction: str) -> float:
        task = task.lower()
        if task == "regression":
            pred = np.asarray(y_pred).squeeze()
            rmse = math.sqrt(mean_squared_error(np.asarray(y_true).squeeze(), pred))
            return rmse  # minimize
        else:
            # classification
            yp = y_pred
            if torch.is_tensor(yp):
                yp = yp.detach().cpu().numpy()
            yp = np.asarray(yp)

            if yp.ndim > 1:
                pred_lbl = np.argmax(yp, axis=1)
            else:
                # 1-D output may already be hard labels from sklearn
                if np.issubdtype(yp.dtype, np.integer):
                    pred_lbl = yp.astype(int)
                else:
                    pred_lbl = (yp > 0.5).astype(int)
            acc = accuracy_score(np.asarray(y_true).astype(int), pred_lbl.astype(int))
            return acc  # maximize

    def _objective(self, trial: optuna.Trial, exp: dict) -> float:
        dataset = exp["dataset"]
        model_name = exp["model"]
        direction = exp.get("direction", "minimize").lower()
        dataset_overrides = exp.get("dataset_arg_overrides", {})

        # params for model + trainer
        params = get_search_params(trial, exp.get("search_space", {}))
        epochs = int(params.get("epochs", 10))
        lr = float(params.get("learning_rate", 1e-3))
        if self.override_epochs is not None:
            epochs = int(self.override_epochs)

        # early stopping config (prefer per-experiment, else global YAML, else default)
        es_cfg = exp.get(
            "early_stopping",
            self.cfg.get("early_stopping", {"enabled": True, "patience": 3, "min_delta": 0.0, "monitor": "auto", "warmup": 0, "mode": "auto", "restore_best_weights": True})
        )

        logger.info(f"[trial {trial.number}] {dataset} / {model_name} | params={params}")

        exp_model = None
        try:
            train_loader, val_loader, _test_loader, nb_classes, task = self._prepare_data(dataset, dataset_overrides)
            input_shape = infer_input_shape_from_loader(train_loader)
            logger.info(f"[trial {trial.number}] input_shape={input_shape}, nb_classes={nb_classes}, task={task}")

            # apply tuned batch size (if present)
            tuned_bs = int(params.get("batch_size", getattr(train_loader, "batch_size", 32)))
            train_loader = _rebuild_loader(train_loader, tuned_bs, shuffle=True)
            val_loader   = _rebuild_loader(val_loader,   tuned_bs, shuffle=False)

            model_enum = ModelRegistry.get_enum_by_name(model_name)
            exp_model = self._build_exp_model(model_enum, input_shape, nb_classes, params, task)

            _fit_with_optional_es(exp_model, train_loader, val_loader, epochs=epochs, lr=lr, es_cfg=es_cfg)

            y_true = collect_labels(val_loader)
            y_pred = exp_model.predict(val_loader)

            if y_pred is None:
                raise RuntimeError("predict() returned None")

            # safety checks
            if isinstance(y_pred, np.ndarray) and (not np.all(np.isfinite(y_pred))):
                raise ValueError("Non-finite values in predictions")

            score = self._score(task, y_true, y_pred, direction)
            if not np.isfinite(score):
                raise ValueError("Non-finite score")

            logger.info(f"[trial {trial.number}] score={score:.6f} ({'minimize' if direction=='minimize' else 'maximize'})")
            return score

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"[trial {trial.number}] CUDA OOM: {e} -> pruning")
            trial.set_user_attr("oom", True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            raise TrialPruned("CUDA OOM")

        except MemoryError as e:
            logger.warning(f"[trial {trial.number}] MemoryError: {e} -> pruning")
            trial.set_user_attr("oom", True)
            raise TrialPruned("Host MemoryError")

        except TrialPruned:
            raise

        except Exception as e:
            logger.exception(f"[trial {trial.number}] crashed: {e} -> pruning")
            trial.set_user_attr("error", repr(e))
            raise TrialPruned(str(e))

        finally:
            # Always clean up to avoid memory creeping between trials
            try:
                del exp_model
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            # (best-effort) MPS cache clear
            try:
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
            except Exception:
                pass
            gc.collect()

    def run(self):
        for exp in self.cfg["experiments"]:
            dataset = exp["dataset"]
            model = exp["model"]
            if self.filter_dataset and dataset != self.filter_dataset:
                continue
            if self.filter_model and model != self.filter_model:
                continue
            n_trials_yaml = int(exp.get("n_trials", 20))
            n_trials = int(self.override_trials or n_trials_yaml)
            direction = exp.get("direction", "minimize").lower()
            study_name = f"{dataset}_{model}_tuning"
            storage = f"sqlite:///{study_name}.db"

            logger.info(f"\n=== Running study: {study_name} (direction={direction}, trials={n_trials}) ===")
            study = optuna.create_study(
                direction=("maximize" if direction == "maximize" else "minimize"),
                sampler=optuna.samplers.TPESampler(),
                pruner=optuna.pruners.MedianPruner(),
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
            )
            # run optimization
            study.optimize(lambda tr: self._objective(tr, exp), n_trials=n_trials, show_progress_bar=False)

            # report
            n_complete = sum(t.state == TrialState.COMPLETE for t in study.trials)
            n_pruned   = sum(t.state == TrialState.PRUNED   for t in study.trials)
            n_fail     = sum(t.state == TrialState.FAIL     for t in study.trials)
            logger.info(f"Trials summary -> COMPLETE: {n_complete} | PRUNED: {n_pruned} | FAIL: {n_fail}")

            # CSV of all trials (even if none complete)
            csv_name = f"tuning/tuning_results/optuna_{study_name}_results.csv"
            os.makedirs(os.path.dirname(csv_name), exist_ok=True)
            header_params = sorted({k for t in study.trials for k in t.params.keys()})
            with open(csv_name, "w", newline="") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(["trial_number"] + header_params + ["value", "state"])
                for t in study.trials:
                    row = [t.number] + [t.params.get(k, "") for k in header_params] + [t.value, str(t.state)]
                    writer.writerow(row)
            logger.info(f"Wrote trial log -> {csv_name}")

            # If nothing completed, skip best reporting & retrain gracefully
            if n_complete == 0:
                logger.warning(f"No COMPLETE trials for {study_name}; skipping best-value print and retraining.")
                continue

            # best report
            logger.info(f"\n=== {study_name} ===")
            logger.info(f"Best value: {study.best_value}")
            logger.info(f"Best params: {study.best_params}")

            # retrain best & save
            logger.info("Retraining best model on train set and saving...")
            dataset_overrides = exp.get("dataset_arg_overrides", {})
            train_loader, val_loader, _test_loader, nb_classes, task = self._prepare_data(dataset, dataset_overrides)
            input_shape = infer_input_shape_from_loader(train_loader)

            best_params = dict(study.best_params)
            epochs = int(best_params.get("epochs", 10))
            lr = float(best_params.get("learning_rate", 1e-3))
            if self.override_epochs is not None:
                epochs = int(self.override_epochs)

            # apply tuned batch size during retrain
            tuned_bs = int(best_params.get("batch_size", getattr(train_loader, "batch_size", 32)))
            train_loader = _rebuild_loader(train_loader, tuned_bs, shuffle=True)
            val_loader   = _rebuild_loader(val_loader,   tuned_bs, shuffle=False)

            model_enum = ModelRegistry.get_enum_by_name(model)
            # normalize again in case best params violate model constraints
            best_params = _normalize_model_specific_params(model_enum, best_params)
            exp_model = self._build_exp_model(model_enum, input_shape, nb_classes, best_params, task)

            # early stopping config (prefer per-experiment, else global YAML, else default)
            es_cfg = exp.get(
                "early_stopping",
                self.cfg.get("early_stopping", {"enabled": True, "patience": 5, "min_delta": 0.0, "monitor": "auto", "warmup": 0, "mode": "auto", "restore_best_weights": True})
            )
            
            _fit_with_optional_es(exp_model, train_loader, val_loader, epochs=epochs, lr=lr, es_cfg=es_cfg)

            save_path_base = f"saved_models/{dataset}_{model}"
            save_model_object(exp_model.model, save_path_base)
            logger.info(f"Saved model to {save_path_base}.[pt|pkl]")

            params_path = f"saved_models/{dataset}_{model}.json"
            with open(params_path, "w") as _fp:
                json.dump({
                    "dataset":   dataset,
                    "model":     model,
                    "direction": direction,
                    "params":    dict(study.best_params),
                }, _fp, indent=2)
            logger.info(f"Saved best params to {params_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tuning/tuning_config.yaml", help="Path to tuning YAML.")
    parser.add_argument("--datasets", default="configs/datasets_config.yaml", help="Path to datasets config YAML.")
    parser.add_argument("--trials", type=int, default=None, help="Override n_trials for all experiments.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all experiments (tuning + retrain).")
    parser.add_argument("--dataset", default=None, help="Run only experiments for this dataset name.")
    parser.add_argument("--model", default=None, help="Run only experiments for this model name.")
    args = parser.parse_args()

    Tuner(
        config_path=args.config,
        datasets_cfg_path=args.datasets,
        override_trials=args.trials,
        override_epochs=args.epochs,
        filter_dataset=args.dataset,
        filter_model=args.model,
    ).run()
