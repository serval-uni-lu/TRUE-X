# run_semantic.py
"""
Semantic settings: SmoothGrad noise level, Occlusion width, and zero vs time-step-wise
training-mean baseline for Occlusion and Feature Ablation. Each map is compared with
the map of the value used in the experiments (noise 0.1, width 10, zero baseline).

Writes r2_6_raw_semantic_attribution_diagnostics.csv. Run from experiments/:
    python appendix/explainer_settings/run_semantic.py
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import common as C
from xai_methods import ExplainerFactory

NOISE_GRID = [0.05, 0.10, 0.20]
WINDOW_GRID = [5, 10, 15]
REFERENCE_NOISE = 0.10
REFERENCE_WINDOW = 10
SEMANTIC_SEED = 42

ATTR_FIELDS = ["Model", "Explainer", "Variant", "SampleID", "DiagType", "DiagValue"]


def add_rows(rows, model, explainer, variant, A, A_ref):
    diag = C.per_sample_diagnostics(A, A_ref)
    for i in range(A.shape[0]):
        rows.append([model, explainer, variant, i, "spearman_vs_historical", diag["spearman"][i]])
        rows.append([model, explainer, variant, i, "normalized_l1_vs_historical", diag["normalized_l1"][i]])
        rows.append([model, explainer, variant, i, "top10_overlap_vs_historical", diag["top10_overlap"][i]])


def run_model(model: str, results_dir: Path, rows: list):
    print(f"=== {model} ===", flush=True)
    attr_dir = results_dir / "attributions"
    exp_model, xai_model, task, train_loader = C.load_model_cpu(model)
    X, y = C.load_hydraulic_30(results_dir)
    _, baseline_mean_profile = C.build_background(train_loader, kmeans_k=50)

    sg = ExplainerFactory.get("smooth_gradient", xai_model)
    occ = ExplainerFactory.get("occlusion", xai_model)
    fabl = ExplainerFactory.get("feature_ablation", xai_model)

    noise_attrs = {}
    for noise in NOISE_GRID:
        def _compute(n=noise):
            torch.manual_seed(SEMANTIC_SEED)
            return sg.explain(x_val=X, task=task, device="cpu", smooth_samples=20, noise_std=n)
        noise_attrs[noise] = C.get_or_compute(attr_dir / f"{model}__smoothgrad_noise__noise{noise}.npy", _compute)
    for noise in NOISE_GRID:
        add_rows(rows, model, "smooth_gradient", f"noise={noise}", noise_attrs[noise], noise_attrs[REFERENCE_NOISE])

    window_attrs = {}
    for w in WINDOW_GRID:
        window_attrs[w] = C.get_or_compute(
            attr_dir / f"{model}__occlusion_window__w{w}.npy",
            lambda w=w: occ.explain(x_val=X, task=task, device="cpu", mode="time",
                                     time_window=w, time_stride=w, perturbations_per_eval=32, baseline=None),
        )
    for w in WINDOW_GRID:
        add_rows(rows, model, "occlusion", f"window={w}", window_attrs[w], window_attrs[REFERENCE_WINDOW])

    occ_trainmean = C.get_or_compute(
        attr_dir / f"{model}__occlusion_baseline__trainmean.npy",
        lambda: occ.explain(x_val=X, task=task, device="cpu", mode="time",
                             time_window=REFERENCE_WINDOW, time_stride=REFERENCE_WINDOW,
                             perturbations_per_eval=32, baseline=baseline_mean_profile),
    )
    add_rows(rows, model, "occlusion", "baseline=trainmean_vs_zero", occ_trainmean, window_attrs[REFERENCE_WINDOW])

    fabl_zero = C.get_or_compute(
        attr_dir / f"{model}__featureablation_baseline__zero.npy",
        lambda: fabl.explain(x_val=X, task=task, device="cpu", baseline=None, group_mode="none"),
    )
    fabl_trainmean = C.get_or_compute(
        attr_dir / f"{model}__featureablation_baseline__trainmean.npy",
        lambda: fabl.explain(x_val=X, task=task, device="cpu", baseline=baseline_mean_profile, group_mode="none"),
    )
    add_rows(rows, model, "feature_ablation", "baseline=trainmean_vs_zero", fabl_trainmean, fabl_zero)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=C.RESULTS_DIR)
    args = parser.parse_args()
    (args.results_dir / "attributions").mkdir(parents=True, exist_ok=True)
    rows = []
    for model in C.MODELS:
        run_model(model, args.results_dir, rows)
    out = args.results_dir / "r2_6_raw_semantic_attribution_diagnostics.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ATTR_FIELDS)
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")
