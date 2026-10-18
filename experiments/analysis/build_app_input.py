# build_app_input.py
"""
Input table of the Streamlit app: one row per {dataset, model, explainer, metric}
with the direction-aligned block median of each metric, plus the predictive
performance of the model (Accuracy, or RMSE negated so that higher is better).

Usage (from the repository root):
    python experiments/analysis/build_app_input.py
"""
import argparse
import re
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
BLOCKS = _ROOT / "results" / "main_benchmark_14_promethee_input" / "promethee_input_14_blocks.csv"
PERFORMANCE = _ROOT / "results" / "main_benchmark_14_tradeoffs" / "predictive_performance_36.csv"
OUTPUT = _ROOT / "results" / "main_benchmark_14_with_performance.csv"
TASK = {"Classif": "Classification", "Reg": "Regression"}


def app_table(blocks: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    # "Sparseness (Classif, Elem)" -> "Sparseness (Elem)", "Continuity (Reg)" -> "Continuity"
    tag = blocks.Metric.str.extract(r"\((Classif|Reg)")[0]
    metric = blocks.Metric.map(lambda m: re.sub(r" \((?:Classif|Reg)(?:, (\w+))?\)", lambda g: f" ({g.group(1)})" if g.group(1) else "", m))
    rows = pd.DataFrame({"Task": tag.map(TASK), "Dataset": blocks.Dataset, "Model": blocks.Model,
                         "Explainer": blocks.Explainer, "Metric": metric, "Value": blocks.median_dir})
    p = perf.set_index(["dataset", "model"])
    sign = p.higher_is_better.map({True: 1.0, False: -1.0})
    configs = rows[["Task", "Dataset", "Model", "Explainer"]].drop_duplicates()
    keys = list(zip(configs.Dataset, configs.Model))
    perf_rows = configs.assign(Metric=[p.loc[k, "performance_metric"] for k in keys],
                               Value=[sign.loc[k] * p.loc[k, "performance_value"] for k in keys])
    out = pd.concat([rows, perf_rows], ignore_index=True)
    return out.sort_values(["Task", "Dataset", "Model", "Explainer", "Metric"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=Path, default=BLOCKS)
    parser.add_argument("--performance", type=Path, default=PERFORMANCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = app_table(pd.read_csv(args.blocks, float_precision="round_trip"), pd.read_csv(args.performance))
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
