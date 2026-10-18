# build_promethee_input.py
"""
PROMETHEE input of the benchmark from the sample-level master table.

Each per-sample metric value is multiplied by the direction of its metric
(+1 if higher is better, -1 otherwise), then the median over the samples of each
{dataset, model, explainer, metric} block is taken (pandas median, NaN skipped).

Usage (from the repository root):
    python experiments/analysis/build_promethee_input.py --master final_main_benchmark_primary_scope.csv
"""
import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = _ROOT / "results" / "main_benchmark_14_promethee_input"

KEY = ["Dataset", "Model", "Explainer", "Metric"]
BLOCK = ["Dataset", "Split", "Model", "Explainer", "Metric"]
TASK_TAG = {"Hydraulic": "Classif", "CWRU_12k": "Classif", "FD001": "Reg", "Ecoating": "Reg"}

# master column: (metric label, direction, column name in promethee_ranking.py)
METRICS = {
    "FC_abs": ("Faithfulness Correlation ({t})", +1, "FC"),
    "PF_AUC": ("Pixel Flipping ({t})", -1, "PF"),
    "AvgSen": ("Average Sensitivity ({t})", -1, "AvgSen"),
    "Continuity": ("Continuity ({t})", -1, "Cont"),
    "Sparseness_E": ("Sparseness ({t}, Elem)", +1, "Spars-E"),
    "Sparseness_C": ("Sparseness ({t}, Chan)", +1, "Spars-C"),
    "Complexity_E": ("Complexity ({t}, Elem)", -1, "Compl-E"),
    "Complexity_C": ("Complexity ({t}, Chan)", -1, "Compl-C"),
}


def block_medians(master: pd.DataFrame) -> pd.DataFrame:
    """Direction-aligned median per block, with the number of samples and of non-NaN values."""
    L = master.melt(id_vars=["dataset", "split", "model", "explainer", "sample_index"],
                    value_vars=list(METRICS), var_name="final_metric", value_name="Value")
    L = L.rename(columns={"dataset": "Dataset", "split": "Split", "model": "Model",
                          "explainer": "Explainer", "sample_index": "Index"})
    L["Metric"] = [METRICS[m][0].format(t=TASK_TAG[d]) for m, d in zip(L.final_metric, L.Dataset)]
    L["dir_sign"] = L.final_metric.map({k: v[1] for k, v in METRICS.items()})
    L["score_dir"] = L.dir_sign * L.Value
    blocks = (L.groupby(BLOCK, dropna=False)
                .agg(n_total=("Index", "size"), n_valid=("score_dir", lambda s: s.notna().sum()),
                     median_dir=("score_dir", "median"))
                .reset_index())
    per_metric = L.drop_duplicates("Metric").set_index("Metric")
    blocks["final_metric"] = blocks.Metric.map(per_metric.final_metric)
    blocks["dir_sign"] = blocks.Metric.map(per_metric.dir_sign)
    blocks["clean_script_column"] = blocks.final_metric.map({k: v[2] for k, v in METRICS.items()})
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True,
                        help="final_main_benchmark_primary_scope.csv (sample-level master table)")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    master = pd.read_csv(args.master, low_memory=False, float_precision="round_trip")
    blocks = block_medians(master)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blocks[KEY + ["median_dir"]].to_csv(args.out_dir / "promethee_input_14.csv", index=False)
    blocks.to_csv(args.out_dir / "promethee_input_14_blocks.csv", index=False)
    print(f"wrote {len(blocks)} blocks to {args.out_dir}")


if __name__ == "__main__":
    main()
