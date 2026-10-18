# common.py
"""
Shared setup of the explainer-setting sensitivity analysis (Appendix C.1).

Hydraulic, first 30 evaluation samples, models FCN, LSTM and TST, CPU only.
Attribution maps are cached as .npy files under <results-dir>/attributions.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

_EXPERIMENTS = Path(__file__).resolve().parents[2]
_ROOT = _EXPERIMENTS.parent
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

import run_xai_evaluation as rxe  # noqa: E402
from ml_models.model_registry import ModelRegistry  # noqa: E402
from ExpliTest.utils.ts_helpers import spearman_rank_corr  # noqa: E402

RESULTS_DIR = _ROOT / "results" / "r2_6_parameter_sensitivity"
SAVED_MODELS = _ROOT / "saved_models"

MODELS = ["FCN", "LSTM", "TST"]
SEEDS = [42, 43, 44, 45, 46]  # seeds of the stochastic grids


def load_hydraulic_30(results_dir: Path):
    X = np.load(results_dir / "hydraulic_first30_X.npy")
    y = np.load(results_dir / "hydraulic_first30_y.npy")
    return X, y


def load_model_cpu(model_name: str):
    """Hydraulic model from saved_models/ (architecture from the JSON sidecar), CPU, eval mode."""
    datasets_cfg = yaml.safe_load(open(_EXPERIMENTS / "configs" / "datasets_config.yaml"))
    loaders_entry = rxe.get_loaders_from_config("Hydraulic", datasets_cfg)
    train_loader = loaders_entry["train"]
    nb_classes, task = loaders_entry["nb_classes"], loaders_entry["task"]
    input_shape = rxe.infer_input_shape_from_loader(train_loader)

    stem = SAVED_MODELS / f"Hydraulic_{model_name}"
    params = json.load(open(stem.with_suffix(".json")))["params"]
    exp_model = rxe.build_exp_model_with_params(
        ModelRegistry.get_enum_by_name(model_name), input_shape, nb_classes, params, task
    )
    exp_model.model.load_state_dict(torch.load(stem.with_suffix(".pt"), map_location="cpu"), strict=True)
    exp_model.model.to("cpu").eval()
    xai_model = exp_model.get_explainable_model()
    xai_model.to("cpu").eval()
    return exp_model, xai_model, task, train_loader


def build_background(train_loader, kmeans_k: int):
    bg_small, baseline_mean = rxe.build_train_background(train_loader, max_pool=2000, kmeans_k=kmeans_k)
    return bg_small, baseline_mean


def get_or_compute(path: Path, compute_fn) -> np.ndarray:
    if path.exists():
        return np.load(path)
    A = np.asarray(compute_fn(), dtype=np.float64)
    np.save(path, A)
    print(f"  computed {path.name}", flush=True)
    return A


# Diagnostics (attribution-level)

def normalized_l1(a_k: np.ndarray, a_ref: np.ndarray, eps: float = 1e-12) -> float:
    num = np.sum(np.abs(np.abs(a_k) - np.abs(a_ref)))
    den = np.sum(np.abs(a_ref)) + eps
    return float(num / den)


def spearman_abs(a_k: np.ndarray, a_ref: np.ndarray) -> float:
    return float(spearman_rank_corr(np.abs(a_k).ravel(), np.abs(a_ref).ravel()))


def top10_overlap(a_k: np.ndarray, a_ref: np.ndarray) -> float:
    flat_k = np.abs(a_k).ravel()
    flat_ref = np.abs(a_ref).ravel()
    D = flat_k.size
    k = max(1, int(np.ceil(0.10 * D)))
    top_k_idx = set(np.argsort(-flat_k)[:k].tolist())
    top_ref_idx = set(np.argsort(-flat_ref)[:k].tolist())
    return float(len(top_k_idx & top_ref_idx) / k)


def per_sample_diagnostics(A_k: np.ndarray, A_ref: np.ndarray) -> Dict[str, np.ndarray]:
    """A_k, A_ref: (N, C, T). Returns per-sample arrays (N,) for each diagnostic."""
    N = A_k.shape[0]
    spearman = np.zeros(N)
    l1 = np.zeros(N)
    overlap = np.zeros(N)
    for i in range(N):
        spearman[i] = spearman_abs(A_k[i], A_ref[i])
        l1[i] = normalized_l1(A_k[i], A_ref[i])
        overlap[i] = top10_overlap(A_k[i], A_ref[i])
    return {"spearman": spearman, "normalized_l1": l1, "top10_overlap": overlap}


def pairwise_seed_spearman(attrs_by_seed: List[np.ndarray]) -> np.ndarray:
    """attrs_by_seed: list of (N,C,T) arrays, one per seed. Returns (N,) median
    pairwise Spearman across all seed pairs, per sample."""
    n_seeds = len(attrs_by_seed)
    N = attrs_by_seed[0].shape[0]
    out = np.zeros(N)
    for i in range(N):
        vals = []
        for a in range(n_seeds):
            for b in range(a + 1, n_seeds):
                vals.append(spearman_abs(attrs_by_seed[a][i], attrs_by_seed[b][i]))
        out[i] = np.median(vals)
    return out
