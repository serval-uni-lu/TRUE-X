# build_numerical_diagnostics.py
"""
Diagnostics of the numerical grids from the cached attribution maps: per sample,
Spearman, normalized L1 and top-10% overlap with the map of the largest tested value,
and for the stochastic grids the median pairwise Spearman between the five seeds.

Writes r2_6_raw_numerical_diagnostics.csv, long format:
["Model","Explainer","Parameter","Value","Seed","SampleID","DiagType","DiagValue"]

Usage (from experiments/):
    python appendix/explainer_settings/build_numerical_diagnostics.py
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C

FIELDS = ["Model", "Explainer", "Parameter", "Value", "Seed", "SampleID", "DiagType", "DiagValue"]

IG_STEPS_GRID = [10, 25, 50, 100, 200]
SG_N_GRID = [5, 10, 20, 50, 100]
GS_N_GRID = [5, 10, 20, 50, 100]
EG_STEPS_GRID = [10, 25, 50, 100, 200]
M_GRID = [10, 25, 50, 100, 200]
MAX_BUDGET = 200
MAX_N = 100
MAX_M = 200


def emit_vs_max(rows, model, explainer, parameter, value, seed, A_k, A_ref):
    diag = C.per_sample_diagnostics(A_k, A_ref)
    for i in range(A_k.shape[0]):
        rows.append([model, explainer, parameter, value, seed, i, "spearman_vs_max", diag["spearman"][i]])
        rows.append([model, explainer, parameter, value, seed, i, "normalized_l1_vs_max", diag["normalized_l1"][i]])
        rows.append([model, explainer, parameter, value, seed, i, "top10_overlap_vs_max", diag["top10_overlap"][i]])


def build(results_dir: Path):
    attr_dir = results_dir / "attributions"
    rows = []
    for model in C.MODELS:
        # 1. IG n_steps (deterministic)
        paths = {s: attr_dir / f"{model}__ig__steps{s}.npy" for s in IG_STEPS_GRID}
        if all(p.exists() for p in paths.values()):
            attrs = {s: np.load(p) for s, p in paths.items()}
            ref = attrs[MAX_BUDGET]
            for s in IG_STEPS_GRID:
                emit_vs_max(rows, model, "integrated_gradients", "ig_steps", s, "", attrs[s], ref)

        # 4. EG steps (deterministic, bg fixed at 50)
        paths = {s: attr_dir / f"{model}__eg_steps__steps{s}.npy" for s in EG_STEPS_GRID}
        if all(p.exists() for p in paths.values()):
            attrs = {s: np.load(p) for s, p in paths.items()}
            ref = attrs[MAX_BUDGET]
            for s in EG_STEPS_GRID:
                emit_vs_max(rows, model, "expected_gradients", "eg_steps", s, "", attrs[s], ref)

        # 5. EG background M (deterministic, steps fixed at 50)
        paths = {m: attr_dir / f"{model}__eg_M__M{m}.npy" for m in M_GRID}
        if all(p.exists() for p in paths.values()):
            attrs = {m: np.load(p) for m, p in paths.items()}
            ref = attrs[MAX_M]
            for m in M_GRID:
                emit_vs_max(rows, model, "expected_gradients", "eg_M", m, "", attrs[m], ref)

        # 6. DeepSHAP background M (deterministic)
        paths = {m: attr_dir / f"{model}__deepshap_M__M{m}.npy" for m in M_GRID}
        if all(p.exists() for p in paths.values()):
            attrs = {m: np.load(p) for m, p in paths.items()}
            ref = attrs[MAX_M]
            for m in M_GRID:
                emit_vs_max(rows, model, "deepliftshap", "deepshap_M", m, "", attrs[m], ref)

        # 2. SmoothGrad n_samples x 5 seeds (stochastic)
        seed_complete = {}
        for seed in C.SEEDS:
            paths = {n: attr_dir / f"{model}__smoothgrad__n{n}__seed{seed}.npy" for n in SG_N_GRID}
            if all(p.exists() for p in paths.values()):
                seed_complete[seed] = {n: np.load(p) for n, p in paths.items()}
        for seed, attrs in seed_complete.items():
            ref = attrs[MAX_N]
            for n in SG_N_GRID:
                emit_vs_max(rows, model, "smooth_gradient", "smoothgrad_n", n, seed, attrs[n], ref)
        if len(seed_complete) == len(C.SEEDS):
            for n in SG_N_GRID:
                attrs_by_seed = [seed_complete[seed][n] for seed in C.SEEDS]
                bsv = C.pairwise_seed_spearman(attrs_by_seed)
                for i in range(len(bsv)):
                    rows.append([model, "smooth_gradient", "smoothgrad_n", n, "between", i,
                                 "between_seed_spearman", bsv[i]])

        # 3. GradientSHAP n_samples x 5 seeds (stochastic, bg fixed at 50)
        seed_complete = {}
        for seed in C.SEEDS:
            paths = {n: attr_dir / f"{model}__gradientshap__n{n}__seed{seed}.npy" for n in GS_N_GRID}
            if all(p.exists() for p in paths.values()):
                seed_complete[seed] = {n: np.load(p) for n, p in paths.items()}
        for seed, attrs in seed_complete.items():
            ref = attrs[MAX_N]
            for n in GS_N_GRID:
                emit_vs_max(rows, model, "gradientshap", "gradientshap_n", n, seed, attrs[n], ref)
        if len(seed_complete) == len(C.SEEDS):
            for n in GS_N_GRID:
                attrs_by_seed = [seed_complete[seed][n] for seed in C.SEEDS]
                bsv = C.pairwise_seed_spearman(attrs_by_seed)
                for i in range(len(bsv)):
                    rows.append([model, "gradientshap", "gradientshap_n", n, "between", i,
                                 "between_seed_spearman", bsv[i]])

    out = results_dir / "r2_6_raw_numerical_diagnostics.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=C.RESULTS_DIR)
    build(parser.parse_args().results_dir)
