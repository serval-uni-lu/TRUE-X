# make_table.py
"""
Values of Table C.3 from summary_all_sources.csv, rankings_per_world.csv and
A_training_seed_detail.csv. Writes table_C3_values.json.

Usage (from the repository root):
    python experiments/appendix/ranking_uncertainty/make_table.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "r1_9_uncertainty_20261016" / "outputs"
DS = ["Hydraulic", "CWRU_12k", "FD001", "Ecoating"]


def f2(x): return f"{x:.2f}"
def rng(lo, hi): return f2(lo) if f2(lo) == f2(hi) else f"{f2(lo)}--{f2(hi)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    O = parser.parse_args().out_dir
    s = pd.read_csv(O / "summary_all_sources.csv"); s = s[s.ranking == "overall"]
    a = pd.read_csv(O / "A_training_seed_detail.csv"); a = a[a.ranking == "overall"].set_index("dataset")
    out = {}
    for ds in DS:
        d = {}
        d["A_tau"] = rng(a.loc[ds, "tau_between_seeds_min"], a.loc[ds, "tau_between_seeds_max"])
        L = json.loads(a.loc[ds, "leaders"]); d["A_leaders"] = L
        d["A_prod_tau"] = rng(a.loc[ds, "tau_prod_vs_seed_min"], a.loc[ds, "tau_prod_vs_seed_max"])
        b = s[(s.source == "B_explainer_rng") & (s.dataset == ds)].iloc[0]
        mx = 2 * b.kendall_tau_b_vs_reference_mean - b.kendall_tau_b_vs_reference_min
        d["B_tau"] = rng(b.kendall_tau_b_vs_reference_min, mx); d["B_leader"] = f"{int(round(b.P_leader_retained * 2))}/2"; d["B_top3"] = f"{int(round(b.P_top3_set_unchanged * 2))}/2"
        for src, lab in [("C1_FC_subsets_M100", "C1"), ("C2_robust_draws_M5", "C2")]:
            x = s[(s.source == src) & (s.dataset == ds)]  # two rows: r1 (= reference for C1) and r2
            t = x.kendall_tau_b_vs_reference_min.iloc[1]; d[lab + "_tau"] = f2(t); d[lab + "_leader"] = bool(x.P_leader_retained.iloc[1] == 1)
            d[lab + "_tau_r1"] = f2(x.kendall_tau_b_vs_reference_min.iloc[0])
        dd = s[(s.source == "D_sample_composition") & (s.dataset == ds)].iloc[0]
        d["D_tau"] = f"{f2(dd.kendall_tau_b_vs_reference_mean)} ({f2(dd.kendall_tau_b_vs_reference_min)})"; d["D_leader"] = dd.P_leader_retained; d["D_n"] = int(dd.n_worlds); d["D_top3"] = dd.P_top3_set_unchanged
        for src, lab in [("F1_q", "F1"), ("F2_sigma", "F2")]:
            x = s[(s.source.str.startswith(src)) & (s.dataset == ds)]
            d[lab + "_tau"] = rng(x.kendall_tau_b_vs_reference_min.min(), x.kendall_tau_b_vs_reference_min.max()); d[lab + "_leader"] = f"{int(x.P_leader_retained.sum())}/{len(x)}"
        out[ds] = d
    json.dump(out, open(O / "table_C3_values.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
