# composites.py
"""
F/R/C pillar composites per {dataset, model, explainer}: each metric's block medians
are min-max normalized within the dataset (constant column -> 0.5), then the
normalized metrics of a pillar are averaged with equal weights.

Usage (from the repository root):
    python experiments/analysis/composites.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = _ROOT / "results" / "main_benchmark_14_promethee_input" / "promethee_input_14.csv"
OUT_DIR = _ROOT / "results" / "main_benchmark_14_tradeoffs"

METRIC_ALIAS = {
    "Faithfulness Correlation (Reg)": "FC",
    "Faithfulness Correlation (Classif)": "FC",
    "Pixel Flipping (Reg)": "PF",
    "Pixel Flipping (Classif)": "PF",
    "Average Sensitivity (Reg)": "AvgSen",
    "Average Sensitivity (Classif)": "AvgSen",
    "Continuity (Reg)": "Cont",
    "Continuity (Classif)": "Cont",
    "Sparseness (Reg, Elem)": "Spars-E",
    "Sparseness (Classif, Elem)": "Spars-E",
    "Sparseness (Reg, Chan)": "Spars-C",
    "Sparseness (Classif, Chan)": "Spars-C",
    "Complexity (Reg, Elem)": "Compl-E",
    "Complexity (Classif, Elem)": "Compl-E",
    "Complexity (Reg, Chan)": "Compl-C",
    "Complexity (Classif, Chan)": "Compl-C",
}

PILLAR_METRICS = {
    "F": ["PF", "FC"],
    "R": ["Cont", "AvgSen"],
    "C": ["Spars-E", "Spars-C", "Compl-E", "Compl-C"],
}


def minmax_by_dataset_metric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def _mm(g):
        v = g["median_dir"].values.astype(float)
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            g["value_norm01"] = 0.5  # neutral mid if no variation
        else:
            g["value_norm01"] = (v - vmin) / (vmax - vmin)
        return g
    return (df
            .groupby(["Dataset","metric_code"], group_keys=False)
            .apply(_mm))


def pillar_mean(row, metric_list):
    vals = [row[m] for m in metric_list if m in row and pd.notna(row[m])]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    # block medians are already direction-aligned; no direction is applied here
    blk = pd.read_csv(args.input, float_precision="round_trip")
    blk = blk[["Dataset","Model","Explainer","Metric","median_dir"]].copy()
    blk["metric_code"] = blk["Metric"].map(METRIC_ALIAS).fillna(blk["Metric"])
    blk = blk.replace([np.inf, -np.inf], np.nan)
    blk = blk.dropna(subset=["Dataset","Model","Explainer","metric_code","median_dir"])

    norm = minmax_by_dataset_metric(blk)
    pivot = (norm
             .pivot_table(index=["Dataset","Model","Explainer"],
                          columns="metric_code",
                          values="value_norm01",
                          aggfunc="mean")
             .reset_index())
    for m in sorted({m for lst in PILLAR_METRICS.values() for m in lst}):
        if m not in pivot.columns:
            pivot[m] = np.nan

    pivot["F_comp"] = pivot.apply(lambda r: pillar_mean(r, PILLAR_METRICS["F"]), axis=1)
    pivot["R_comp"] = pivot.apply(lambda r: pillar_mean(r, PILLAR_METRICS["R"]), axis=1)
    pivot["C_comp"] = pivot.apply(lambda r: pillar_mean(r, PILLAR_METRICS["C"]), axis=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "pillar_composites_norm01.csv"
    pivot.to_csv(out, index=False)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
