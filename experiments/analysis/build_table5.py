# build_table5.py
"""
Table 5: per dataset, the mean rank of each explainer on each of the eight metrics and
its PROMETHEE II net flow ("Overall"), read from the ranking CSVs (not recomputed).

Metric columns: inside each (Dataset, Split, Model, Metric) block, the explainers are
ranked by median_dir (direction-aligned, higher = better, average ranks), and the ranks
are averaged over models. Shading marks the three best and three worst mean ranks per
column. Rows are ordered by net flow, then mean of the eight mean ranks, then name.

Usage (from the repository root):
    python experiments/analysis/build_table5.py
"""
import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
BLOCKS = _ROOT / "results" / "main_benchmark_14_promethee_input" / "promethee_input_14_blocks.csv"
RUN = _ROOT / "results" / "main_benchmark_14_promethee"
COLS = ["FC", "PF", "AvgSen", "Cont", "Spars-E", "Spars-C", "Compl-E", "Compl-C"]
ABBR = {"Faithfulness Correlation": "FC", "Pixel Flipping": "PF", "Average Sensitivity": "AvgSen", "Continuity": "Cont",
        "Sparseness Elem": "Spars-E", "Sparseness Chan": "Spars-C", "Complexity Elem": "Compl-E", "Complexity Chan": "Compl-C"}
FINAL_OF = {"FC": "FC_abs", "PF": "PF_AUC", "AvgSen": "AvgSen", "Cont": "Continuity", "Spars-E": "Sparseness_E", "Spars-C": "Sparseness_C",
            "Compl-E": "Complexity_E", "Compl-C": "Complexity_C"}
ACR = {"saliency": "Saliency", "gradient_x_input": "GradX", "guided_back_prop": "GuidedBP", "smooth_gradient": "SmoothGrad",
       "integrated_gradients": "IntegGrad", "expected_gradients": "ExpectGrad", "deeplift": "DeepLift", "deepliftshap": "DeepSHAP",
       "gradientshap": "GradSHAP", "occlusion": "Occlusion", "feature_ablation": "FeatAbl", "lime_tabular": "LIME",
       "shap_permutation": "PermutationSHAP", "kernelshap": "KernelSHAP"}
TEX_ORDER = [("Ecoating", "Ecoating dataset -- Regression task", "Tab:ExplainerRanking:Ecoating"),
             ("FD001", "C-MAPSS dataset -- Regression task", "Tab:ExplainerRanking:CMAPPS"),
             ("CWRU_12k", "CWRU dataset -- Classification  task", "Tab:ExplainerRanking:CWRU"),
             ("Hydraulic", "Hydraulic dataset -- Classification task", "Tab:ExplainerRanking:Hydraulic")]


def col_of(metric: str) -> str:
    fam = metric.split(" (")[0]
    if fam in ("Sparseness", "Complexity"):
        fam += " Elem" if "Elem" in metric else " Chan"
    return ABBR[fam]


def ranks_per_block(block: pd.DataFrame) -> pd.DataFrame:
    def rank_within(g):
        g = g.copy()
        g["rank"] = g["median_dir"].rank(method="average", ascending=False) if g["median_dir"].notna().any() else np.nan
        return g
    return (block.groupby(["Dataset", "Split", "Model", "Metric"], group_keys=True).apply(rank_within).reset_index(drop=True))


def mean_rank_ds(rb: pd.DataFrame) -> pd.DataFrame:
    return (rb.groupby(["Dataset", "Metric", "Explainer"], dropna=False)
              .agg(mean_rank_ds=("rank", "mean"), sd_rank_ds=("rank", "std"), n_blocks=("rank", "count")).reset_index())


def shading(M: pd.DataFrame) -> pd.DataFrame:
    """Per column, rank of the mean ranks (average), Top1-3 / Bottom1-3 on distinct values."""
    S = pd.DataFrame("", index=M.index, columns=M.columns)
    for c in M.columns:
        R = M[c].rank(ascending=True, method="average")
        u = np.sort(R.dropna().unique())
        d = u[::-1]
        for e, r in R.items():
            if pd.isna(r):
                continue
            if u.size >= 1 and np.isclose(r, u[0]): S.loc[e, c] = "Top1"
            elif u.size >= 2 and np.isclose(r, u[1]): S.loc[e, c] = "Top2"
            elif u.size >= 3 and np.isclose(r, u[2]): S.loc[e, c] = "Top3"
            elif d.size >= 1 and np.isclose(r, d[0]): S.loc[e, c] = "Bottom1"
            elif d.size >= 2 and np.isclose(r, d[1]): S.loc[e, c] = "Bottom2"
            elif d.size >= 3 and np.isclose(r, d[2]): S.loc[e, c] = "Bottom3"
    return S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=Path, default=BLOCKS)
    parser.add_argument("--rankings", type=Path, default=RUN / "output")
    parser.add_argument("--out-dir", type=Path, default=RUN)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    blk = pd.read_csv(args.blocks, float_precision="round_trip")
    block = blk[["Dataset", "Split", "Model", "Explainer", "Metric", "median_dir"]].copy()
    rb = ranks_per_block(block)
    rb["Col"] = rb.Metric.map(col_of)
    rb["final_metric"] = rb.Col.map(FINAL_OF)
    mr = mean_rank_ds(rb)
    mr["Col"] = mr.Metric.map(col_of)
    mr["final_metric"] = mr.Col.map(FINAL_OF)
    # exact rational mean ranks (ranks are multiples of 1/2) to detect genuine ties
    rb["rank2"] = (rb["rank"] * 2).round().astype(int)
    assert np.allclose(rb.rank2 / 2, rb["rank"])
    frac = rb.groupby(["Dataset", "Metric", "Explainer"]).apply(lambda g: Fraction(int(g.rank2.sum()), 2 * len(g))).rename("mean_rank_exact").reset_index()
    mr = mr.merge(frac, on=["Dataset", "Metric", "Explainer"])
    rb.drop(columns="rank2").to_csv(args.out_dir / "table5_metric_ranks_per_model.csv", index=False)

    rows = []
    checks = {}
    for ds in ["Hydraulic", "CWRU_12k", "FD001", "Ecoating"]:
        out = pd.read_csv(args.rankings / f"promethee_ranking_{ds}.csv", float_precision="round_trip").set_index("Explainer")
        n = len(out)
        den = 12 * (n - 1)
        num = np.rint(out.promethee_net_flow.to_numpy() * den).astype(int)
        phi_exact = pd.Series([Fraction(int(k), den) for k in num], index=out.index)
        d = mr[mr.Dataset == ds]
        M = d.pivot_table(index="Explainer", columns="Col", values="mean_rank_ds", aggfunc="mean").reindex(columns=COLS)
        Mx = d.pivot(index="Explainer", columns="Col", values="mean_rank_exact").reindex(columns=COLS)
        NB = d.pivot(index="Explainer", columns="Col", values="n_blocks").reindex(columns=COLS)
        S = shading(M)
        exact_tie_groups = {c: [sorted(ACR[e] for e in g.index) for _, g in Mx[c].groupby(Mx[c]) if len(g) > 1] for c in COLS}
        # row order on the exact net flow, so that genuine ties fall through to the mean rank and the name
        key = pd.DataFrame({"phi": phi_exact, "mean": M.mean(axis=1, skipna=True), "name": out.index}, index=out.index)
        order = list(key.sort_values(["phi", "mean", "name"], ascending=[False, True, True]).index)
        overall_rank = pd.Series([float(x) for x in phi_exact], index=phi_exact.index).rank(ascending=False, method="average")
        checks[ds] = {
            "overall_ties": [sorted(ACR[e] for e in g.index) for _, g in phi_exact.groupby(phi_exact) if len(g) > 1],
            "metric_mean_rank_exact_ties": {c: v for c, v in exact_tie_groups.items() if v},
        }
        for pos, e in enumerate(order, 1):
            r = {"dataset": ds, "display_position": pos, "explainer": e, "explainer_display": ACR[e]}
            for c in COLS:
                r[f"{c}_mean_rank"] = M.loc[e, c]
                r[f"{c}_mean_rank_exact"] = str(Mx.loc[e, c])
                r[f"{c}_display"] = "--" if pd.isna(M.loc[e, c]) else f"{M.loc[e, c]:.1f}"
                r[f"{c}_shade"] = S.loc[e, c]
                r[f"{c}_n_models"] = int(NB.loc[e, c])
            r.update({"mean_of_metric_mean_ranks": key.loc[e, "mean"], "overall_net_flow": float(out.loc[e, "promethee_net_flow"]),
                      "overall_net_flow_exact": f"{int(num[list(out.index).index(e)])}/{den}",   # unreduced k/(12(n-1))
                      "overall_display": f"{float(phi_exact[e]):.3f}", "overall_rank_avg_ties": float(overall_rank[e]),
                      "audited_csv_position": int(list(out.index).index(e) + 1)})
            rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "table5_revised_14_explainers.csv", index=False)
    mr.assign(mean_rank_exact=mr.mean_rank_exact.map(str)).to_csv(args.out_dir / "table5_metric_mean_ranks.csv", index=False)

    def tex_rows(T):
        L = []
        for r in T.itertuples():
            cells = []
            for c in COLS:
                sh = getattr(r, f"{c.replace('-', '_')}_shade")
                v = getattr(r, f"{c.replace('-', '_')}_display")
                cells.append((f"\\cellcolor{{{sh}}} {v}" if sh else f" {v}"))
            L.append(f"{r.explainer_display:<15} & " + " & ".join(cells) + f" & {float(Fraction(r.overall_net_flow_exact)): .3f} \\\\")
        return L
    T2 = table.rename(columns={f"{c}_{s}": f"{c.replace('-', '_')}_{s}" for c in COLS for s in ("shade", "display")})
    hdr = [r"\toprule", r"\multicolumn{1}{c}{} & \multicolumn{2}{c}{\textbf{F}} & \multicolumn{2}{c}{\textbf{R}} &",
           r"\multicolumn{4}{c}{\textbf{C}} & \\", r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-9}",
           r"\textbf{XAI} & \rotatebox{90}{FC} & \rotatebox{90}{PF} &", r"\rotatebox{90}{AvgSen} & \rotatebox{90}{Cont} &",
           r"\rotatebox{90}{Spars-E} & \rotatebox{90}{Spars-C} &", r"\rotatebox{90}{Compl-E} & \rotatebox{90}{Compl-C} &",
           r"\rotatebox{90}{Overall} \\", r"\midrule"]
    tex = ["% Revised Table 5 (Tab:ExplainerRanking): final 14-explainer benchmark, 9-model primary scope.",
           "% Generated by experiments/analysis/build_table5.py - do not edit by hand.",
           "% Metric columns: mean over models of within-model ranks of block medians (FC=FC_abs, PF=PF_AUC; direction-aligned).",
           "% Overall: audited PROMETHEE II net flow read from output/promethee_ranking_<Dataset>.csv (not recomputed).",
           r"\begin{table*}[t]", r"\centering\footnotesize", ""]
    for i, (ds, subcap, lab) in enumerate(TEX_ORDER):
        tex += [f"\\subfigure[{subcap}\\label{{{lab}}}]{{", r"  \begin{minipage}{0.48\textwidth}", r"  \centering",
                r"  \begin{tabular}{lrrrrrrrrr}", *hdr, *tex_rows(T2[T2.dataset == ds]), r"\bottomrule", r"\end{tabular}", r"  \end{minipage}",
                *({0: ["}\\hfill"], 1: ["}", "\\hfill"], 2: ["}", "\\hfill"], 3: ["}"]}[i]), ""]
    tex += [r'\caption{Trustworthiness ranking of explainers per dataset. Explainers are ranked (column ``Overall") using Promethee~II over the '
            r"eight F, R, C trustworthiness metrics. Metric-specific ranks correspond to mean ranks computed from median configuration-level "
            r"scores and averaged across models (lower values $=$ better performance). Green- and red-highlighted cells denote the top and "
            r"bottom three explainers per metric, respectively.}", r"\label{Tab:ExplainerRanking}", r"\end{table*}"]
    (args.out_dir / "table5_revised_14_explainers.tex").write_text("\n".join(tex) + "\n")

    mark = {"Top1": "+1", "Top2": "+2", "Top3": "+3", "Bottom1": "-1", "Bottom2": "-2", "Bottom3": "-3", "": "  "}
    R = ["REVISED TABLE 5 - final 14-explainer benchmark (9-model primary scope)",
         "Metric columns: mean rank across models (lower = better); FC = FC_abs, PF = PF_AUC. Marker +k = Top-k shading, -k = Bottom-k.",
         "Overall: audited PROMETHEE II net flow (higher = better), read from output/promethee_ranking_<Dataset>.csv; column exact = k/156; models = models averaged for that explainer.", ""]
    for ds in ["Hydraulic", "CWRU_12k", "FD001", "Ecoating"]:
        T = table[table.dataset == ds]
        R.append(f"=== {ds} ===")
        R.append(f"{'#':>2}  {'XAI':<16}" + "".join(f"{c:>10}" for c in COLS) + f"{'Overall':>10}  {'exact':>9}  models")
        for rd in T.to_dict("records"):
            R.append(f"{rd['display_position']:>2}  {rd['explainer_display']:<16}" +
                     "".join(f"{rd[f'{c}_display']:>7} {mark[rd[f'{c}_shade']]}" for c in COLS) +
                     f"{rd['overall_display']:>10}  {rd['overall_net_flow_exact']:>9}  {rd['FC_n_models']}")
        ch = checks[ds]
        R.append(f"Overall ties: {ch['overall_ties'] or 'none'}; exact ties in metric mean ranks: "
                 + ("; ".join(f"{c}: {v}" for c, v in ch["metric_mean_rank_exact_ties"].items()) or "none"))
        R.append("")
    (args.out_dir / "table5_revised_14_explainers_readable.txt").write_text("\n".join(R) + "\n")


if __name__ == "__main__":
    main()
