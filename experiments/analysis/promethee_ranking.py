# promethee_ranking.py
"""
PROMETHEE II ranking of the explainers, one table per dataset.

Input : promethee_input.csv  (columns: Dataset, Model, Explainer, Metric, median_dir)
Output: one CSV per dataset, each row an explainer with its per-metric mean of
        block medians and its PROMETHEE net flow ("promethee_net_flow"), sorted
        descending (best first).

Weighting: rank_dataset() takes pillar-level weights, e.g.
{"Faithfulness": 1/3, "Robustness": 1/3, "Complexity": 1/3} (the default), and
splits each pillar's weight evenly among its metrics.

Usage (from the repository root):
    python experiments/analysis/promethee_ranking.py
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = _ROOT / "results" / "main_benchmark_14_promethee" / "promethee_input.csv"
OUTPUT_DIR = _ROOT / "results" / "main_benchmark_14_promethee" / "output"

# metric abbreviations (8 columns, in this exact order)
ABBR = {
    "Faithfulness Correlation (Reg)": "FC",
    "Faithfulness Correlation (Classif)": "FC",
    "Pixel Flipping (Reg)": "PF",
    "Pixel Flipping (Classif)": "PF",
    "Average Sensitivity (Reg)": "AvgSen",
    "Average Sensitivity (Classif)": "AvgSen",
    "Continuity (Reg)": "Cont",
    "Continuity (Classif)": "Cont",
    "Sparseness (Reg, Elem)": "Spars-E",
    "Sparseness (Reg, Chan)": "Spars-C",
    "Sparseness (Classif, Elem)": "Spars-E",
    "Sparseness (Classif, Chan)": "Spars-C",
    "Complexity (Reg, Elem)": "Compl-E",
    "Complexity (Reg, Chan)": "Compl-C",
    "Complexity (Classif, Elem)": "Compl-E",
    "Complexity (Classif, Chan)": "Compl-C",
}
COLS = ["FC", "PF", "AvgSen", "Cont", "Spars-E", "Spars-C", "Compl-E", "Compl-C"]
PILLAR_METRICS = {
    "Faithfulness": ["FC", "PF"],
    "Robustness": ["AvgSen", "Cont"],
    "Complexity": ["Spars-E", "Spars-C", "Compl-E", "Compl-C"],
}
DEFAULT_PILLAR_WEIGHTS = {"Faithfulness": 1.0 / 3.0, "Robustness": 1.0 / 3.0, "Complexity": 1.0 / 3.0}


def pillar_weights_to_metric_weights(pillar_weights: dict, cols: list) -> np.ndarray:
    """
    Convert 3 pillar-level weights (Faithfulness, Robustness, Complexity) into
    the 8 metric-level weights aligned with `cols`, by splitting each pillar's
    weight equally among the metrics belonging to that pillar (PILLAR_METRICS),
    then renormalizing the full vector to sum to 1.

    Validates: all three pillar keys are present, all three weights are
    non-negative, and they sum to 1 within normal floating-point tolerance.
    """
    required = ["Faithfulness", "Robustness", "Complexity"]
    missing = [p for p in required if p not in pillar_weights]
    if missing:
        raise ValueError(f"missing pillar weight(s): {missing}")

    values = [pillar_weights[p] for p in required]
    if any(v < 0 for v in values):
        raise ValueError(f"pillar weights must be non-negative, got {pillar_weights}")

    total = sum(values)
    if not np.isclose(total, 1.0):
        raise ValueError(f"pillar weights must sum to 1 (within tolerance), got sum={total}")

    w = {c: 0.0 for c in cols}
    for pillar, metrics in PILLAR_METRICS.items():
        present = [c for c in metrics if c in cols]
        k = len(present)
        if k > 0:
            w_per = pillar_weights[pillar] / k
            for c in present:
                w[c] = w_per

    s = sum(w.values())
    if s > 0:
        for c in w:
            w[c] /= s
    return np.array([w.get(c, 0.0) for c in cols], dtype=float)


def promethee_net_flow(M: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    PROMETHEE II net flow with the "usual" (step) preference function.

    M: Explainer x Metric matrix of direction-aligned scores (higher = better).
       1) Column-wise min-max normalize to [0,1]. A constant or all-NaN column
          becomes all 0.5 (uninformative -> contributes 0 preference either way).
       2) Remaining NaNs are imputed with the column mean (post-normalization).
       3) "Usual" preference (strict >), maximizing, weighted per column.

    weights: array aligned with M.columns; pass a vector that already sums to 1
             (e.g. from pillar_equal_weights(), or an alternative scheme).

    Returns phi (net flow) per explainer, higher = better.
    """
    A = M.index.to_list()
    X = M.to_numpy(dtype=float)

    col_min = np.nanmin(X, axis=0)
    col_max = np.nanmax(X, axis=0)
    span = col_max - col_min

    X_norm = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        if not np.isfinite(span[j]) or span[j] <= 0:
            X_norm[:, j] = 0.5
        else:
            X_norm[:, j] = (X[:, j] - col_min[j]) / span[j]

    col_mean = np.nanmean(X_norm, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.5, col_mean)
    r, c = np.where(np.isnan(X_norm))
    if len(r):
        X_norm[r, c] = col_mean[c]

    n, m = X_norm.shape
    if n <= 1 or m == 0:
        return pd.Series(np.zeros(n), index=A)

    w = np.asarray(weights, dtype=float)
    if w.shape[0] != m:
        raise ValueError(f"weights length {w.shape[0]} != number of metric columns {m}")

    P = np.zeros((n, n), dtype=float)
    for j in range(m):
        xj = X_norm[:, j]
        P += w[j] * (xj[:, None] > xj[None, :]).astype(float)

    np.fill_diagonal(P, 0.0)
    phi_plus = P.sum(axis=1) / (n - 1)
    phi_minus = P.sum(axis=0) / (n - 1)
    return pd.Series(phi_plus - phi_minus, index=A)


def build_metric_matrix(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Explainer x Metric matrix for one dataset: filter to dataset, map raw
    Metric name -> abbreviation, drop unmapped rows, average median_dir
    across every Model row for each (Explainer, Metric).
    """
    d = df[df["Dataset"].astype(str).str.strip() == str(dataset)].copy()
    d["Col"] = d["Metric"].map(ABBR)
    d = d.dropna(subset=["Col"])

    M = (d.groupby(["Explainer", "Col"], dropna=False)
           .agg(raw_score=("median_dir", "mean"))
           .reset_index()
           .pivot(index="Explainer", columns="Col", values="raw_score")
           .reindex(columns=COLS))
    return M


def rank_dataset(df: pd.DataFrame, dataset: str, pillar_weights: dict = None) -> pd.DataFrame:
    """
    Full per-dataset ranking: build the metric matrix, convert pillar-level
    weights into the 8 metric-level weights, compute PROMETHEE net flow, and
    return a table sorted by descending phi, ties broken by explainer name
    (alphabetical) only.

    pillar_weights: dict with keys "Faithfulness", "Robustness", "Complexity",
                     non-negative values summing to 1. Defaults to the
                     manuscript scheme (1/3 each) via DEFAULT_PILLAR_WEIGHTS.
    """
    if pillar_weights is None:
        pillar_weights = dict(DEFAULT_PILLAR_WEIGHTS)

    M = build_metric_matrix(df, dataset)
    metric_weights = pillar_weights_to_metric_weights(pillar_weights, list(M.columns))
    phi = promethee_net_flow(M, metric_weights)

    out = M.reset_index().rename(columns={"index": "Explainer"})
    out["promethee_net_flow"] = out["Explainer"].map(phi)
    out = out.sort_values(["promethee_net_flow", "Explainer"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "Dataset", dataset)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    for c in ["Dataset", "Model", "Explainer", "Metric"]:
        df[c] = df[c].astype(str).str.strip()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted(df["Dataset"].unique()):
        table = rank_dataset(df, dataset)
        out_path = args.out_dir / f"promethee_ranking_{dataset}.csv"
        table.to_csv(out_path, index=False)
        top = table.iloc[0]
        print(f"[{dataset}] wrote {out_path}  (top explainer: {top['Explainer']}, "
              f"phi={top['promethee_net_flow']:.6f})")


if __name__ == "__main__":
    main()
