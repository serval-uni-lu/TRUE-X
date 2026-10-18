# variance_attribution.py
"""
Variance attribution (Type-II ANOVA) of the trustworthiness metrics.

Per metric and task: OLS  y ~ D + M + E + D:M + D:E + M:E  on the direction-aligned
configuration medians, Type-II sums of squares, each term (and the residual) divided by
the sum of all of them, x 100.

Usage (from the repository root):
    python experiments/analysis/variance_attribution.py
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = _ROOT / "results" / "main_benchmark_14_promethee_input" / "promethee_input_14.csv"
OUT_DIR = _ROOT / "results" / "main_benchmark_14_anova"

SHORT = {"Faithfulness Correlation":"FC","Pixel Flipping":"PF","Average Sensitivity":"AvgSen","Continuity":"Cont",
         "Sparseness|Elem":"Spars-E","Sparseness|Chan":"Spars-C","Complexity|Elem":"Compl-E","Complexity|Chan":"Compl-C"}
ORDER = ["FC","PF","AvgSen","Cont","Spars-E","Spars-C","Compl-E","Compl-C"]
TERMS = [("C(D)","Dataset"),("C(M)","Model"),("C(E)","Explainer"),("C(D):C(M)","Dataset×Model"),
         ("C(D):C(E)","Dataset×Explainer"),("C(M):C(E)","Model×Explainer"),("Residual","Residual")]

def parse(metric):
    base, rest = metric.split(" (",1)
    task = "classification" if rest.startswith("Classif") else "regression"
    key = base + ("|Elem" if "Elem" in rest else "|Chan" if "Chan" in rest else "")
    return task, SHORT[key]

def variance_attribution(df, value_col):
    df = df.copy()
    df[["task","metric"]] = df["Metric"].apply(lambda m: pd.Series(parse(m)))
    out = {}
    for task in ["classification","regression"]:
        rows = []
        for met in ORDER:
            d = df[(df.task==task)&(df.metric==met)].rename(columns={"Dataset":"D","Model":"M","Explainer":"E",value_col:"y"})
            d = d[np.isfinite(d.y)]
            fit = smf.ols("y ~ C(D) + C(M) + C(E) + C(D):C(M) + C(D):C(E) + C(M):C(E)", data=d).fit()
            a = anova_lm(fit, typ=2)
            ss = a["sum_sq"]
            tot = ss.sum()
            rows.append([met] + [100*ss[t]/tot for t,_ in TERMS] + [len(d)])
        out[task] = pd.DataFrame(rows, columns=["metric"]+[n for _,n in TERMS]+["n"]).set_index("metric")
    return out

def write_dat(tab, path):
    t = tab.drop(columns="n")
    with open(path,"w") as f:
        f.write("\t".join(["metric"]+list(t.columns))+"\n")
        for met,r in t.iterrows():
            f.write("\t".join([met]+[f"{v:.6f}" for v in r.values])+"\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    tables = variance_attribution(pd.read_csv(args.input), "median_dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for task, tab in tables.items():
        write_dat(tab, args.out_dir / f"variance_attribution_{task}.dat")
        print(task, "configurations per metric:", sorted(set(tab["n"])))

if __name__ == "__main__":
    main()
