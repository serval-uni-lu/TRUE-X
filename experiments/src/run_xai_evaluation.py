# run_xai_evaluation.py
import os
#os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gc
import csv
import json
import glob
import math
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# ==== model imports ====
from ml_models.model_registry import Model, ModelRegistry
from ml_models.experiment_model import ExperimentModel
from ml_models.model_builder import build_model_with_params

# metrics XAI metrics
from metrics.metrics.feature_attribution.timeseries_multivariate.faithfulness.faithfulness_correlation import FAMVFaithfulnessCorrelation
from metrics.metrics.feature_attribution.timeseries_multivariate.faithfulness.pixel_flipping import FAMVPixelFlipping
from metrics.metrics.feature_attribution.timeseries_multivariate.robustness.avg_sensitivity import FAMVAvgSensitivity
from metrics.metrics.feature_attribution.timeseries_multivariate.robustness.continuity import FAMVContinuity
from metrics.metrics.feature_attribution.timeseries_multivariate.complexity.sparseness import FAMVSparsenessElement, FAMVSparsenessChannel
from metrics.metrics.feature_attribution.timeseries_multivariate.complexity.complexity_entropy import FAMVComplexityEntropyElement, FAMVComplexityEntropyChannel

# Optional XAI factory
try:
    from xai_methods import ExplainerFactory
except Exception:
    ExplainerFactory = None  # fallback to grad*input

logger = logging.getLogger("truex.xai_eval")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------
# Device / seed / logging
# ------------------------
def device_str() -> str:
    # if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     return "mps"
    # if torch.cuda.is_available():
    #     return "cuda"
    return "cpu"

def set_seed(seed: int = 42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_verbosity(verbosity: int):
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.getLogger().setLevel(level)

# ------------------------
# YAML / adapter utilities
# ------------------------
def load_yaml_file(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def import_callable(path: str):
    module_path, func_name = path.split(":")
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)

def _loader_has_labels(loader: DataLoader) -> bool:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return False
    return isinstance(batch, dict) and ("label" in batch)

def standardize_adapter_return(ret):
    """
    Accept dict OR tuple/list OR plain DataLoader.
    Return (train_loader, val_loader, test_loader_or_None, nb_classes_or_None)
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
        if len(dls) >= 1: tl = dls[0]
        if len(dls) >= 2: vl = dls[1]
        if len(dls) >= 3: te = dls[2]
        for x in ret:
            if isinstance(x, (int, np.integer)):
                nb_classes = int(x); break
    elif isinstance(ret, DataLoader):
        tl = ret

    if tl is None or vl is None:
        raise ValueError("Adapter must return at least train & val loaders.")
    return tl, vl, te, nb_classes

def make_loaders(dataset_name: str, datasets_cfg: dict, overrides: Optional[Dict[str, Any]] = None):
    entry = datasets_cfg[dataset_name]
    adapter_path = entry["adapter"]
    task = entry["task"]
    args = dict(entry.get("args", {}) or {})
    if overrides:
        args.update(overrides)
    # Ask CMAPSS adapter for test if supported
    if "datasets.adapters.cmapss:make_cmapss_loaders" in adapter_path and "return_test" not in args:
        args["return_test"] = True
    adapter = import_callable(adapter_path)
    ret = adapter(**args)
    train_loader, val_loader, test_loader, nb_classes = standardize_adapter_return(ret)

    # infer nb_classes if needed
    if task.lower().startswith("classif") and nb_classes is None:
        ys = []
        for batch in train_loader:
            y = batch["label"]
            y = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
            ys.append(y)
        y = np.concatenate(ys, axis=0)
        if y.ndim > 1: y = y[:, 0]
        nb_classes = int(np.max(y)) + 1

    return train_loader, val_loader, test_loader, nb_classes, task

def infer_input_shape_from_loader(loader) -> Tuple[int, int]:
    batch = next(iter(loader))
    x = batch["sequence"]  # (B, C, T)
    if x.dim() != 3:
        raise ValueError(f"expected (B,C,T), got {tuple(x.shape)}")
    _, C, T = x.shape
    return (T, C)

def _channel_mean_from_loader(train_loader) -> np.ndarray:
    """
    Compute per-channel mean across all training windows: shape (C,).
    Used as the baseline for faithfulness metrics (masking strategy).
    Expects batches with 'sequence' key of shape (B, C, T).
    """
    channel_sum, n_windows = None, 0
    for batch in train_loader:
        x = batch["sequence"]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        channel_sum = x.sum(axis=(0, 2)) if channel_sum is None else channel_sum + x.sum(axis=(0, 2))
        n_windows += x.shape[0] * x.shape[2]
    return (channel_sum / max(n_windows, 1)).astype(np.float32)

def _build_xai_metrics(kind: str) -> Dict[str, Any]:
    """
    Instantiate the full set of XAI evaluation metrics from metrics.
    kind: 'regression' or 'classification'
    seed=42 is fixed for reproducibility across all stochastic metrics.
    """
    suffix = "Reg" if kind == "regression" else "Classif"
    return {
        f"Average Sensitivity ({suffix})":       FAMVAvgSensitivity(kind=kind, sigma=0.01, tau=0.05, n_perturbations=5, seed=42),
        f"Continuity ({suffix})":                FAMVContinuity(kind=kind, sigma=0.01, tau=0.05, n_perturbations=5, seed=42),
        f"Faithfulness Correlation ({suffix})":  FAMVFaithfulnessCorrelation(kind=kind, subset_size=10, n_runs=100, seed=42),
        f"Pixel Flipping ({suffix})":            FAMVPixelFlipping(kind=kind, features_in_step=1),
        f"Sparseness ({suffix}, Elem)":          FAMVSparsenessElement(),
        f"Complexity ({suffix}, Elem)":          FAMVComplexityEntropyElement(),
        f"Sparseness ({suffix}, Chan)":          FAMVSparsenessChannel(),
        f"Complexity ({suffix}, Chan)":          FAMVComplexityEntropyChannel(),
    }

# ------------------------
# Tuning + model builders
# ------------------------
def read_best_params_from_csv(dataset: str, model: str, direction: str) -> Optional[Dict[str, Any]]:
    csv_path = os.path.join("tuning", "tuning_results", f"optuna_{dataset}_{model}_tuning_results.csv")
    if not os.path.isfile(csv_path):
        return None
    trials = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = str(row.get("state", ""))
            if "COMPLETE" not in state:
                continue
            val = row.get("value")
            if val in (None, ""):
                continue
            try:
                value = float(val)
            except Exception:
                continue
            params = {k: row[k] for k in row.keys() if k not in ("trial_number", "value", "state")}
            casted = {}
            for k, v in params.items():
                s = str(v)
                if s == "" or s.lower() == "null":
                    casted[k] = None
                elif s.lower() in ("true", "false"):
                    casted[k] = (s.lower() == "true")
                else:
                    try:
                        if "." in s or "e" in s.lower():
                            casted[k] = float(s)
                            if float(casted[k]).is_integer():
                                casted[k] = int(casted[k])
                        else:
                            casted[k] = int(s)
                    except Exception:
                        if s.startswith("[") and s.endswith("]"):
                            try:
                                casted[k] = tuple(eval(s, {"__builtins__": {}}))
                            except Exception:
                                casted[k] = s
                        else:
                            casted[k] = s
            trials.append((value, casted))
    if not trials:
        return None
    if direction.lower().startswith("max"):
        best = max(trials, key=lambda t: t[0])
    else:
        best = min(trials, key=lambda t: t[0])
    return best[1]

def load_tuned_artifact_or_none(dataset: str, model: str) -> Optional[str]:
    base = os.path.join("../saved_models", f"best_{dataset}_{model}")
    if os.path.isfile(base + ".pt"):  return base + ".pt"
    if os.path.isfile(base + ".pkl"): return base + ".pkl"
    return None

def find_latest_cwru_ckpt(model: str) -> Optional[str]:
    root = os.path.join("../saved_models", "CWRU_12k", model)
    if not os.path.isdir(root): return None
    subdirs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
    if not subdirs: return None
    for d in reversed(subdirs):
        pt = os.path.join(d, "model_state.pt")
        jb = os.path.join(d, "model.joblib")
        if os.path.isfile(pt): return pt
        if os.path.isfile(jb): return jb
    return None

def build_exp_model_with_params(model_enum: Model, input_shape, nb_classes, params, task) -> ExperimentModel:
    exp = ExperimentModel(model_type=model_enum, input_shape=input_shape, nb_classes=nb_classes)
    try:
        model_obj = build_model_with_params(model_enum, input_shape, nb_classes, params or {}, task)
        exp.model = model_obj
        if not isinstance(model_obj, nn.Module):
            exp.is_sklearn = True
            exp.expects_3d = bool(getattr(model_obj, "expects_3d", False))
    except NotImplementedError:
        pass
    return exp

def load_weights_into_exp(exp: ExperimentModel, artifact_path: str):
    import joblib
    if artifact_path.endswith(".pt"):
        sd = torch.load(artifact_path, map_location="cpu")
        if isinstance(sd, dict):
            for k in ("model_state", "state_dict", "model", "net", "model_state_dict"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]; break
        if not isinstance(exp.model, nn.Module):
            raise TypeError("Torch .pt provided but ExperimentModel.model is not nn.Module.")
        exp.model.load_state_dict(sd, strict=True)
        exp.model.eval()
    elif artifact_path.endswith((".pkl", ".joblib")):
        exp.model = joblib.load(artifact_path)
        exp.is_sklearn = True
        exp.expects_3d = bool(getattr(exp.model, "expects_3d", False))
    else:
        raise ValueError(f"Unknown artifact format: {artifact_path}")

# ------------------------
# Data helpers
# ------------------------
def _gather_from_loader(loader: DataLoader, n_max: int) -> np.ndarray:
    Xs, total = [], 0
    for b in loader:
        x = b["sequence"]
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        Xs.append(x)
        total += x.shape[0]
        if total >= n_max:
            break
    X = np.concatenate(Xs, axis=0)
    return X[:n_max]

def _loader_to_numpy(loader: DataLoader, n_max: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Loads samples (and optional labels) from a DataLoader.

    Prints progress like: "[gather] selected 32/128 samples"
    where the denominator is min(n_max, len(dataset)) if available,
    else n_max if set, else len(dataset) if available.
    """
    # Figure out the denominator for progress printing
    denom = None
    try:
        ds_len = len(loader.dataset)
        denom = ds_len if n_max is None else min(n_max, ds_len)
    except Exception:
        # dataset length unknown: fall back to n_max if provided
        denom = n_max

    # Print every ~10% (at least every 1)
    log_every = 1
    if isinstance(denom, int) and denom > 10:
        log_every = max(1, denom // 10)

    Xs, Ys = [], []
    selected = 0

    for batch in loader:
        x = batch["sequence"]
        y = batch.get("label")

        x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        if y is not None:
            y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

        if n_max is None:
            # take everything
            Xs.append(x)
            if y is not None:
                Ys.append(y)
            selected += x.shape[0]
        else:
            # take up to the remaining budget
            remaining = n_max - selected
            if remaining <= 0:
                break
            take = min(remaining, x.shape[0])
            Xs.append(x[:take])
            if y is not None:
                Ys.append(y[:take])
            selected += take

        # progress log
        if denom is not None and (selected % log_every == 0 or selected >= denom):
            logger.info(f"[gather] selected {min(selected, denom)}/{denom} samples")

        # stop early if we've reached the cap
        if n_max is not None and selected >= n_max:
            break

    if not Xs:
        return np.empty((0,)), None

    X = np.concatenate(Xs, axis=0)
    Y = None
    if Ys:
        Y = np.concatenate(Ys, axis=0)
        if Y.ndim > 1:
            Y = Y[:, 0]
    return X, Y


def build_train_background(train_loader: DataLoader, max_pool: int = 2000, kmeans_k: int = 50) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    X_pool = _gather_from_loader(train_loader, n_max=max_pool)  # (N,C,T)
    if X_pool.size == 0:
        return None, None
    # background (small) via k-means (or fallback)
    try:
        from sklearn.cluster import KMeans
        B, C, T = X_pool.shape
        X2d = X_pool.reshape(B, C * T)
        k = int(min(max(1, kmeans_k), B))
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X2d)
        centers = []
        for i in range(k):
            idx = np.where(km.labels_ == i)[0]
            if len(idx) == 0: continue
            centers.append(X_pool[idx].mean(axis=0))
        background_small = np.stack(centers, axis=0)
    except Exception:
        k = int(min(max(1, kmeans_k), X_pool.shape[0]))
        rng = np.random.RandomState(42)
        background_small = X_pool[rng.permutation(X_pool.shape[0])[:k]]
    baseline_mean = X_pool.mean(axis=0)  # (C,T)
    return background_small, baseline_mean

# ------------------------
# Fallback explainer (grad*input)
# ------------------------
class _FallbackExplainer:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.to(device).eval()
        self.device = device

    def explain(self, x_val: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_val).float().to(self.device).requires_grad_(True)
        out = self.model(x)
        if isinstance(out, (list, tuple)): out = out[0]
        if out.ndim == 2 and out.shape[1] > 1:
            idx = out.argmax(dim=1)
            out = out[torch.arange(out.size(0)), idx].sum()
        else:
            out = out.squeeze().sum()
        out.backward()
        attr = (x.grad * x).detach().cpu().numpy()
        x.grad = None
        return attr

# ------------------------
# Explain + Metrics
# ------------------------
def compute_attributions_for_method(
    exp_model: ExperimentModel,
    method: str,
    x_explain: np.ndarray,
    background_small: Optional[np.ndarray],
    baseline_mean: Optional[np.ndarray],
) -> np.ndarray:
    xai_model = exp_model.get_explainable_model()
    if not isinstance(xai_model, torch.nn.Module):
        logger.info(f"[XAI] Non-torch model; using |x| attributions for method={method} (smoke).")
        return np.abs(x_explain)

    if ExplainerFactory is None:
        logger.info(f"[XAI] ExplainerFactory missing; fallback gradient*input for {method}.")
        expl = _FallbackExplainer(xai_model, next(xai_model.parameters()).device)
        return expl.explain(x_explain)

    dev = next(xai_model.parameters()).device
    params = {"method": method, "device": str(dev)}
    if method == "integrated_gradients":
        params.update({"steps": 50})
    elif method == "gradientshap":
        params.update({"smooth_samples": 20, "noise_std": 0.1})
    elif method == "expected_gradients":
        params.update({"steps": 50})
    elif method == "occlusion":
        T = x_explain.shape[-1]
        time_win = max(1, T // 5)
        params.update({"mode": "time", "time_window": time_win, "time_stride": time_win, "perturbations_per_eval": 32})

    explainer = ExplainerFactory.get(method, xai_model)

    kwargs = dict(x_val=x_explain, task=exp_model.task, **params)
    if method in {"deepliftshap", "gradientshap", "expected_gradients", "lime_tabular", "shapley_sampling"}:
        if background_small is not None:
            kwargs["background_data"] = background_small
    if method in {"integrated_gradients", "deeplift", "deepliftshap", "expected_gradients"} and baseline_mean is not None:
        kwargs["baseline"] = baseline_mean

    try:
        attr = explainer.explain(**kwargs)
    except TypeError:
        kwargs.pop("task", None)
        kwargs.pop("background_data", None)
        kwargs.pop("baseline", None)
        attr = explainer.explain(**kwargs)

    if attr.ndim == 3 and attr.shape[1] != x_explain.shape[1] and attr.shape[1] == x_explain.shape[2]:
        attr = np.transpose(attr, (0, 2, 1))
    return attr

def run_xai_metrics_for_batch(
    exp_model: ExperimentModel,
    x_bct: np.ndarray,
    attr_bct: np.ndarray,
    train_loader: Optional[DataLoader] = None,
) -> Dict[str, np.ndarray]:
    """
    Return per-sample arrays for each metric. If a metric still returns a scalar,
    we broadcast it to (B,) to avoid crashing.
    """
    task = exp_model.task.lower()
    baseline_c = None
    if train_loader is not None:
        try:
            baseline_c = _channel_mean_from_loader(train_loader)  # (C,)
        except Exception as e:
            logger.info(f"[baseline] channel mean fallback: {e}")
            baseline_c = x_bct.mean(axis=(0, 2))

    kind = "regression" if task.startswith("regress") else "classification"
    metric_dict = _build_xai_metrics(kind)
    results: Dict[str, np.ndarray] = {}

    def _attach_explain_to_robust():
        xai_model = exp_model.get_explainable_model()
        if not isinstance(xai_model, torch.nn.Module):
            return None
        if ExplainerFactory is None:
            fallback = _FallbackExplainer(xai_model, next(xai_model.parameters()).device)
            return lambda xb: fallback.explain(xb)
        ex = ExplainerFactory.get("saliency", xai_model)
        def _fn(xb: np.ndarray) -> np.ndarray:
            try:
                A = ex.explain(x_val=xb, task=exp_model.task, method="saliency", device=str(device_str()))
            except TypeError:
                A = ex.explain(x_val=xb)
            if A.ndim == 3 and A.shape[1] != xb.shape[1] and A.shape[1] == xb.shape[2]:
                A = np.transpose(A, (0, 2, 1))
            return A
        return _fn

    robust_expl = _attach_explain_to_robust()

    for name, metric in metric_dict.items():
        if "Average Sensitivity" in name or "Continuity" in name:
            if robust_expl is None:
                results[name] = np.full(x_bct.shape[0], np.nan, dtype=float)
                continue
            arr = metric.evaluate_attributions(model=exp_model, x=x_bct, attributions=attr_bct, explain_fn=robust_expl)
        elif "Faithfulness Correlation" in name or "Pixel Flipping" in name:
            arr = metric.evaluate_attributions(model=exp_model, x=x_bct, attributions=attr_bct, baseline_channels=baseline_c)
        else:
            arr = metric.evaluate_attributions(model=exp_model, x=x_bct, attributions=attr_bct)

        arr = np.asarray(arr)
        # Broadcast scalar to (B,) if needed
        if arr.ndim == 0 or arr.size == 1:
            arr = np.full(x_bct.shape[0], float(arr), dtype=float)
        else:
            arr = arr.reshape(-1)

        if arr.shape[0] != x_bct.shape[0]:
            raise ValueError(f"Metric '{name}' returned shape {arr.shape}, expected ({x_bct.shape[0]},)")
        results[name] = arr

    return results

# ------------------------
# Saved model discovery (general + CWRU)
# ------------------------
def find_saved_model_path(saved_root: Path, dataset: str, model_name: str) -> Optional[Path]:
    """
    Try common patterns:
      - tuning/saved_models/best_{Dataset}_{Model}.pt
      - tuning/saved_models/{Dataset}/best_{Dataset}_{Model}.pt
      - saved_models/{Dataset}/{MODEL}/**/model_state.pt (CWRU-style)
      - *.pkl / *.joblib for sklearn models
    """
    pats = [
        f"best_{dataset}_{model_name}.*",
        f"{dataset}/best_{dataset}_{model_name}.*",
        f"{dataset}/{model_name}/**/model_state.pt",
        f"{dataset}/{model_name}/**/model_state.pth",
        f"{dataset}/{model_name}/**/*.joblib",
        f"{dataset}/{model_name}/**/*.pkl",
    ]
    for pat in pats:
        hits = list(saved_root.glob(pat))
        if hits:
            hits.sort(key=lambda p: len(p.as_posix()))
            return hits[-1]
    hits = list(saved_root.rglob(f"*{dataset}*{model_name}*.*"))
    return hits[-1] if hits else None

# ------------------------
# CSV helpers (per-dataset files)
# ------------------------
def dataset_csv_path(out_dir: Path, dataset: str) -> Path:
    return out_dir / f"{dataset}.csv"

def ensure_dataset_csv_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        header = ["timestamp","Dataset","Model","Explainer","Split","Index","Metric","Value","ModelPath","N_explained"]
        with open(path, "w", newline="") as f:
            f.write(",".join(header) + "\n")

def log_sample_rows(path: Path,
                    dataset: str,
                    model: str,
                    method: str,
                    split: str,
                    model_path: str,
                    metric_name: str,
                    values: np.ndarray):
    ensure_dataset_csv_header(path)
    ts = datetime.now().isoformat()
    n = int(values.shape[0])
    with open(path, "a", newline="") as f:
        for i, v in enumerate(values):
            row = [ts, dataset, model, method, split, i, metric_name, float(v), model_path, n]
            f.write(",".join(map(str, row)) + "\n")

# ------------------------
# Per-dataset/model runner
# ------------------------
def run_for_dataset_model(
    dataset_name: str,
    model_name: str,
    loaders_entry: dict,
    *,
    saved_root: Path,
    tuning_csv_root: Path,
    max_explain: Optional[int],
    out_dir: Path,
    out_attrs_dir: Optional[Path],
    methods: List[str],
) -> None:
    # unpack loaders/metadata
    train_loader: DataLoader = loaders_entry["train"]
    val_loader: DataLoader   = loaders_entry["val"]
    test_loader: Optional[DataLoader] = loaders_entry.get("test")
    nb_classes: Optional[int] = loaders_entry.get("nb_classes")
    task: str = loaders_entry["task"]

    input_shape = infer_input_shape_from_loader(train_loader)
    model_enum = ModelRegistry.get_enum_by_name(model_name)
    exp = ExperimentModel(model_type=model_enum, input_shape=input_shape, nb_classes=nb_classes)

    # locate and load saved model
    model_path = find_saved_model_path(saved_root, dataset_name, model_name)
    if model_path is None:
        # fallback to generic saved_models root
        model_path = find_saved_model_path(Path("../saved_models"), dataset_name, model_name)
    if model_path is None:
        logger.warning(f"[skip] No saved model found for {dataset_name}/{model_name}")
        return

    # For non-CWRU, use best_* tuning artifact; for CWRU, we might be pointing to saved_models/CWRU_12k/...
    ok = False
    if dataset_name == "CWRU_12k" and model_path.suffix.lower() in (".pt", ".pth") and "best_" not in model_path.name:
        # Follow your previous logic — load latest snapshot
        try:
            artifact = find_latest_cwru_ckpt(model_name)
            if artifact is None:
                logger.warning(f"[skip] CWRU artifact missing for {model_name}")
                return
            exp = ExperimentModel(model_type=model_enum, input_shape=input_shape, nb_classes=nb_classes)
            load_weights_into_exp(exp, artifact)
            logger.info(f"[load] {dataset_name}/{model_name} <- {artifact}")
            ok = True
        except Exception as e:
            logger.error(f"[load] CWRU latest failed: {e}")
            ok = False
    else:
        # tuned path: rebuild arch with best Optuna params and load state
        try:
            direction = "maximize" if (task.lower() == "classification") else "minimize"
            best_params = read_best_params_from_csv(dataset_name, model_name, direction) or {}
            exp = build_exp_model_with_params(model_enum, input_shape, nb_classes, best_params, task)
            load_weights_into_exp(exp, str(model_path))
            logger.info(f"[load] {dataset_name}/{model_name} <- {model_path.name}")
            ok = True
        except Exception as e:
            logger.error(f"[load] tuned load failed for {dataset_name}/{model_name}: {e}")
            ok = False

    if not ok:
        logger.error(f"[skip] {dataset_name}/{model_name}: could not load a compatible checkpoint.")
        return

    # explicand set = test if available with labels; else val
    explicand_loader = test_loader if (test_loader is not None and _loader_has_labels(test_loader)) else val_loader
    split_name = "test" if explicand_loader is test_loader else "val"

    # gather explicand windows
    X_exp, _ = _loader_to_numpy(explicand_loader, n_max=max_explain)
    if X_exp.size == 0:
        logger.warning(f"[skip] {dataset_name}/{model_name}: no explicand samples.")
        return

    # background & baseline strictly from TRAIN
    bg_small, baseline = build_train_background(train_loader, max_pool=2000, kmeans_k=50)

    # run explainers → metrics (+ optional save attrs)
    for method in methods:
        logger.info(f"[XAI] {dataset_name}/{model_name} [{split_name}] → {method}")
        try:
            A = compute_attributions_for_method(exp, method, X_exp, bg_small, baseline)
            # shape checks & normalize to (N,C,T)
            if not isinstance(A, np.ndarray) or A.ndim != 3:
                raise ValueError(f"Attributions must be (N,C,T); got {getattr(A,'shape',None)}")
            if A.shape[1:] != X_exp.shape[1:]:
                # attempt simple transpose (N,T,C) -> (N,C,T)
                if A.shape[2:] == X_exp.shape[1:]:
                    A = np.transpose(A, (0, 2, 1))
                else:
                    raise ValueError(f"Attribution shape mismatch {A.shape} vs input {X_exp.shape}")

            X_bct = np.asarray(X_exp)  # already (N, C, T) — channels-first throughout
            metrics = run_xai_metrics_for_batch(exp, X_bct, A, train_loader=train_loader)

            # === per-dataset CSV logging, one row per sample ===
            ds_csv = dataset_csv_path(out_dir, dataset_name)
            for metric_name, arr in metrics.items():
                log_sample_rows(
                    ds_csv,
                    dataset=dataset_name,
                    model=model_name,
                    method=method,
                    split=split_name,
                    model_path=model_path.as_posix(),
                    metric_name=metric_name,
                    values=arr,
                )

            # === structured dumps per (dataset/model/method/split) ===
            dump_root = out_dir / dataset_name / model_name / method / split_name
            (dump_root / "metrics").mkdir(parents=True, exist_ok=True)

            # Save one .npy per metric (per-sample arrays)
            for metric_name, arr in metrics.items():
                safe = metric_name.replace("/", "_").replace(" ", "_")
                np.save(dump_root / "metrics" / f"{safe}.npy", np.asarray(arr))

            # Small JSON summary with shapes and means (just for quick scan)
            summary = {
                "dataset": dataset_name,
                "model": model_name,
                "explainer": method,
                "split": split_name,
                "n_explained": int(X_exp.shape[0]),
                "metric_means": {k: float(np.nanmean(v)) for k, v in metrics.items()},
                "metric_stds":  {k: float(np.nanstd(v))  for k, v in metrics.items()},
            }
            with open(dump_root / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            # optionally persist arrays of inputs/attrs
            if out_attrs_dir is not None:
                od = out_attrs_dir / dataset_name / model_name / method
                od.mkdir(parents=True, exist_ok=True)
                np.save(od / "x_exp.npy", X_exp)
                np.save(od / "attr.npy", A)
                with open(od / "summary.json", "w") as f:
                    json.dump(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "method": method,
                            "split": split_name,
                            "x_shape": list(X_exp.shape),
                            "attr_shape": list(A.shape),
                            "attr_stats": {
                                "min": float(np.nanmin(A)),
                                "max": float(np.nanmax(A)),
                                "mean": float(np.nanmean(A)),
                                "std": float(np.nanstd(A)),
                            },
                        }, f, indent=2
                    )

        except Exception as e:
            logger.error(f"[XAI] Failed for {dataset_name}/{model_name}/{method}: {e}")
            continue

    # cleanup
    try: del exp
    except Exception: pass
    if torch.cuda.is_available():
        try: torch.cuda.empty_cache()
        except Exception: pass
    gc.collect()

# ------------------------
# Dataset factory (from datasets_config.yaml)
# ------------------------
def get_loaders_from_config(dataset_name: str, datasets_cfg: dict):
    entry = datasets_cfg[dataset_name]
    adapter_path = entry["adapter"]
    task = entry["task"]
    args = entry.get("args", {}) or {}

    if "datasets.adapters.cmapss:make_cmapss_loaders" in adapter_path and "return_test" not in args:
        args = dict(args)
        args["return_test"] = True

    adapter = import_callable(adapter_path)
    ret = adapter(**args)
    train_loader, val_loader, test_loader, nb_classes = standardize_adapter_return(ret)

    if task.lower().startswith("classif") and nb_classes is None:
        y_all = []
        for b in train_loader:
            y = b["label"]
            y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
            y_all.append(y)
        y_all = np.concatenate(y_all, axis=0)
        if y_all.ndim > 1: y_all = y_all[:, 0]
        nb_classes = int(y_all.max()) + 1

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "nb_classes": nb_classes,
        "task": task,
    }

# ------------------------
# CLI
# ------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run explainers + XAI metrics on saved models; store per-sample metric arrays."
    )
    parser.add_argument("--datasets-config", type=str, default="configs/datasets_config.yaml")
    parser.add_argument("--model-dataset-map", type=str, default="configs/model_dataset_map.yaml")
    parser.add_argument("--saved-root", type=str, default="../saved_models",
                        help="Directory with best_{Dataset}_{Model}.* or per-dataset folders.")
    parser.add_argument("--tuning-csv-root", type=str, default="tuning/tuning_results",
                        help="Directory with optuna_{Dataset}_{Model}_tuning_results.csv")
    parser.add_argument("--out-dir", type=str, default="../results",
                        help="Directory to store one CSV per dataset and per-metric .npy dumps.")
    parser.add_argument("--out-attrs-dir", type=str, default=None,
                        help="If set, saves x_exp.npy/attr.npy/summary.json under this directory.")
    parser.add_argument("--methods", type=str,
                        default="expected_gradients,lime_tabular")
                        #default="deepliftshap,smooth_gradient,occlusion,feature_ablation,gradientshap,expected_gradients,lime_tabular")
                        #default="saliency,gradient_x_input,guided_back_prop,integrated_gradients,deeplift,deepliftshap,smooth_gradient,occlusion,feature_ablation,gradientshap,expected_gradients,lime_tabular")
    parser.add_argument("--max-explain", type=int, default=None,
                        help="Cap the number of explained samples per model (from test/val).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args()

    set_verbosity(args.verbose or 1)
    set_seed(args.seed)

    dev = device_str()
    logger.info(f"[device] Using: {dev}")
    torch.set_grad_enabled(False)
    try:
        torch.backends.cudnn.benchmark = True  # CUDA only
    except Exception:
        pass

    datasets_cfg = load_yaml_file(args.datasets_config)
    model_map = load_yaml_file(args.model_dataset_map)

    saved_root = Path(args.saved_root)
    tuning_csv_root = Path(args.tuning_csv_root)
    out_dir = Path(args.out_dir)
    out_attrs_dir = Path(args.out_attrs_dir) if args.out_attrs_dir else None
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    for dataset_name, model_names in model_map.items():
        if dataset_name not in datasets_cfg:
            logger.warning(f"[skip] Dataset '{dataset_name}' not in {args.datasets_config}")
            continue
        if not model_names:
            continue

        logger.info(f"\n=== Dataset: {dataset_name} ===")
        loaders_entry = get_loaders_from_config(dataset_name, datasets_cfg)

        for model_name in model_names:
            try:
                run_for_dataset_model(
                    dataset_name,
                    model_name,
                    loaders_entry,
                    saved_root=saved_root,
                    tuning_csv_root=tuning_csv_root,
                    max_explain=args.max_explain,
                    out_dir=out_dir,
                    out_attrs_dir=out_attrs_dir,
                    methods=methods,
                )
            except Exception as e:
                logger.error(f"[ERROR] {dataset_name}/{model_name}: {e}")
                continue

    logger.info("All XAI evaluations completed.")

if __name__ == "__main__":
    main()
