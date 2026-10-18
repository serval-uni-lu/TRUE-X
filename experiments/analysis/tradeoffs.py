# tradeoffs.py
"""
Pairwise trade-offs between the F, R and C pillar composites, per dataset.

For each pair: weak-dominance Pareto front, Spearman correlation with a bootstrap CI
(2000 resamples), and the share of configurations in the top quartile of both pillars
(DualQ). One PDF per (dataset, pair) with the models of performance rank <= 2
highlighted, a legend PDF, the statistics CSVs, and Table 6 (CSV and LaTeX).

Usage (from the repository root):
    python experiments/analysis/tradeoffs.py
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
from scipy.stats import spearmanr
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction


# -------- PDF / vector-friendly defaults --------
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.transparent"] = False

# -------- Bigger fonts (global) --------
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# ---------------- I/O ----------------
_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = _ROOT / "results" / "main_benchmark_14_tradeoffs"
IN_COMPOSITES = OUT_ROOT / "pillar_composites_norm01.csv"
IN_PERFORMANCE = OUT_ROOT / "predictive_performance_36.csv"

# Which datasets to render as figures
MAIN_DATASETS = ["CWRU_12k", "Hydraulic", "Ecoating", "FD001"]

# Headroom so points/polyline near top-right aren’t cramped
AX_MAX = 1.2

# Canonical IDs are kept in all data and file names; display names are used only for human-facing text.
DISPLAY_NAME = {"CWRU_12k": "CWRU", "Hydraulic": "Hydraulic", "Ecoating": "Ecoating", "FD001": "C-MAPSS"}
TABLE6_ORDER = ["CWRU_12k", "Ecoating", "FD001", "Hydraulic"]   # row order of Table 6

# ---------------- Highlighted models: performance rank <= 2, ties kept ----------------
# Loaded in main() from predictive_performance_36.csv; {dataset: [model, ...]} ordered by (rank, model).
HIGHLIGHT_BY_DATASET = {}

# ---------------- Highlight colors (NO GRAY, NO REUSE, HIGH CONTRAST) ----------------
_TAB10 = plt.get_cmap("tab10")
_TAB20 = plt.get_cmap("tab20")

# Fixed colours per (dataset, model). Any other highlighted model takes the next spare colour,
# which is not gray (tab10(7) excluded).
HISTORICAL_HILITE_COLOR = {
    ("CWRU_12k", "FCN"): _TAB10(0), ("CWRU_12k", "RESNET"): _TAB10(1),                    # blue / orange
    ("Hydraulic", "RESNET"): _TAB10(2), ("Hydraulic", "LSTM"): _TAB10(3),                  # green / red
    ("Ecoating", "TST_REGRESSOR"): _TAB10(4), ("Ecoating", "CNN_LSTM_REGRESSOR"): _TAB10(5),  # purple / brown
    ("FD001", "TCN_REGRESSOR"): _TAB10(9), ("FD001", "TFT_REGRESSOR"): _TAB10(6),          # cyan / pink
}
SPARE_HILITE_COLORS = [_TAB10(8)]   # olive

def hilite_colors(dataset: str) -> dict:
    """{model: rgba} for the highlighted models of a dataset."""
    out, spare = {}, list(SPARE_HILITE_COLORS)
    for md in HIGHLIGHT_BY_DATASET.get(dataset, []):
        if (dataset, md) in HISTORICAL_HILITE_COLOR:
            out[md] = HISTORICAL_HILITE_COLOR[(dataset, md)]
        elif spare:
            out[md] = spare.pop(0)
        else:
            raise SystemExit(f"not enough spare highlight colours for {dataset}")
    return out

# ---------------- Explicit explainer -> marker mapping ----------------
EXPLAINER_MARKERS = {
    "deeplift": "o", "deepliftshap": "s", "expected_gradients": "^", "feature_ablation": "D",
    "gradient_x_input": "P", "gradientshap": "v", "guided_back_prop": "X", "integrated_gradients": "<",
    "lime_tabular": ">", "occlusion": "h", "saliency": "8", "smooth_gradient": "*",
    "kernelshap": "p",
    "shap_permutation": "d",
}
LEGEND_ORDER = sorted(EXPLAINER_MARKERS)

# ---------------- Display rounding policy ----------------
# Decimal ROUND_HALF_UP = round half away from zero. Percentages are rounded from the exact rational
# 100*k/n; rho and the CI bounds from the shortest repr of the float. A value that rounds to zero is
# shown without a sign.
def round_half_up_str(x, nd: int) -> str:
    d = Decimal(x.numerator) / Decimal(x.denominator) if isinstance(x, Fraction) else Decimal(repr(float(x)))
    q = d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)
    return f"{q:.{nd}f}"

def pct_fraction(k: int, n: int) -> Fraction:
    return Fraction(100 * k, n)

# ---------------- Point styling ----------------
BASE_S_DOM = 42
BASE_S_FR = 95
HILITE_SCALE = 1.35

BLACK_RGBA = mcolors.to_rgba("black")

# ---- OPACITY CONTROLS FOR BLACK POINTS (THIS IS WHERE YOU CONTROL "LIGHTNESS") ----
ALPHA_DOM_BASE = 0.20   # dominated non-highlight black (lower => lighter)
ALPHA_FR_BASE  = 0.30   # frontier non-highlight black (lower => lighter)

# highlighted points (colored) more opaque
ALPHA_DOM_HI = 0.92
ALPHA_FR_HI  = 0.98

EDGE_DOM = "white"
EDGE_FR = "black"
LW_DOM = 0.6
LW_FR = 1.2

# ---------------- Utils ----------------
def pareto_front_flags(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    is_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            is_front[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            if not np.isfinite(x[j]) or not np.isfinite(y[j]):
                continue
            if (x[j] >= x[i] and y[j] >= y[i]) and (x[j] > x[i] or y[j] > y[i]):
                is_front[i] = False
                break
    return is_front

def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 7):
    rng = np.random.default_rng(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan, (np.nan, np.nan)
    rho, _ = spearmanr(x, y)
    if not np.isfinite(rho):
        rho = np.nan
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(x[idx], y[idx])
        if np.isfinite(r):
            boots.append(r)
    if not boots:
        return rho, (np.nan, np.nan)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, (lo, hi)

# ---------------- Plot one pair ----------------
def plot_pair(ax, sub_df, xcol, ycol, expl_markers):
    vals = sub_df[[xcol, ycol]].to_numpy()
    front_mask = pareto_front_flags(vals[:, 0], vals[:, 1])

    dfp = sub_df.copy()
    dfp["on_front"] = front_mask

    dataset = str(dfp["Dataset"].iloc[0]) if not dfp.empty else ""
    hi_colors = {m: mcolors.to_rgba(c) for m, c in hilite_colors(dataset).items()}

    def _scatter_group(g, expl, on_front: bool):
        base_s = BASE_S_FR if on_front else BASE_S_DOM
        base_edge = EDGE_FR if on_front else EDGE_DOM
        base_lw = LW_FR if on_front else LW_DOM

        alpha_base = ALPHA_FR_BASE if on_front else ALPHA_DOM_BASE
        alpha_hi = ALPHA_FR_HI if on_front else ALPHA_DOM_HI

        models = g["Model"].astype(str).to_numpy()
        is_hi = np.isin(models, list(hi_colors))
        sizes = np.where(is_hi, base_s * HILITE_SCALE, base_s)

        # start all as black, override highlights with colored RGBA
        facecolors = np.tile(BLACK_RGBA, (len(g), 1))
        for md, rgba in hi_colors.items():
            facecolors[models == md, :] = rgba

        # per-point alpha: lighter for non-highlight; more opaque for highlights
        alphas = np.where(is_hi, alpha_hi, alpha_base)

        ax.scatter(
            g[xcol], g[ycol],
            s=sizes,
            marker=expl_markers[expl],
            c=facecolors,
            alpha=alphas,
            edgecolors=base_edge,
            linewidths=base_lw,
            zorder=12 if on_front else 8
        )

    dom = dfp[~dfp["on_front"]]
    for expl, g in dom.groupby("Explainer", sort=False):
        _scatter_group(g, expl, on_front=False)

    fr = dfp[dfp["on_front"]]
    for expl, g in fr.groupby("Explainer", sort=False):
        _scatter_group(g, expl, on_front=True)

    if len(fr) >= 2:
        pts = fr[[xcol, ycol]].sort_values(by=[xcol, ycol]).to_numpy()
        ax.plot(pts[:, 0], pts[:, 1], color="black", lw=1.6, alpha=0.9, zorder=20)

    ax.set_xlim(0, AX_MAX)
    ax.set_ylim(0, AX_MAX)
    ax.set_xlabel(xcol.replace("_comp", ""))
    ax.set_ylabel(ycol.replace("_comp", ""))

    # Return frontier points for CSV (keep original dataset string)
    front_points = dfp[dfp["on_front"]][["Dataset", "Model", "Explainer", xcol, ycol]].rename(
        columns={xcol: "x", ycol: "y"}
    )
    return front_points

def save_legend_pdf(all_explainers, expl_markers, out_pdf: Path, ncol: int = 6):
    """
    Legend PDF containing ONLY explainer markers (black), no Top-1/Top-2 boxes.
    """
    fig = plt.figure(figsize=(10.8, 0.95))
    ax = fig.add_subplot(111)
    ax.axis("off")

    # Explainer markers (black) - slightly larger for readability
    expl_handles = [
        Line2D(
            [0], [0],
            marker=expl_markers[e],
            color="none",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=10,
            linewidth=0,
            label=e,
        )
        for e in all_explainers
    ]

    fig.legend(
        handles=expl_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=ncol,
        frameon=False,
        fontsize=13,         # bigger legend font
        handletextpad=0.7,
        columnspacing=1.6,
        borderaxespad=0.0,
    )

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[ok] wrote {out_pdf}")

# ---------------- Table 6 (CSV and LaTeX from one dataframe) ----------------
def build_table6(stats_df: pd.DataFrame) -> pd.DataFrame:
    t = stats_df.copy()
    t["Dataset"] = t["Dataset_id"].map(DISPLAY_NAME)
    t["rho_display"] = [round_half_up_str(v, 2) for v in t["rho_spearman"]]
    t["pareto_pct_display"] = [round_half_up_str(pct_fraction(k, n), 1) for k, n in zip(t["n_on_front"], t["n_points"])]
    t["dualq_pct_display"] = [round_half_up_str(pct_fraction(k, n), 1) for k, n in zip(t["n_dualq"], t["n_points"])]
    t["_o"] = t["Dataset_id"].map(TABLE6_ORDER.index)
    t["_p"] = t["pair"].map(["F vs R", "F vs C", "R vs C"].index)
    t = t.sort_values(["_o", "_p"]).drop(columns=["_o", "_p"]).reset_index(drop=True)
    return t.rename(columns={"pair": "Pair", "n_points": "N configurations", "rho_spearman": "rho",
                             "front_share_pct": "Pareto %", "dual_top_quartile_pct": "DualQ %"})[
        ["Dataset", "Dataset_id", "Pair", "N configurations", "rho", "Pareto %", "DualQ %",
         "n_on_front", "n_dualq", "rho_display", "pareto_pct_display", "dualq_pct_display"]]


def table6_latex(t: pd.DataFrame) -> str:
    def num(s):   # math-mode minus, as in the submitted table
        return f"${s}$"
    L = [
        "% Generated by experiments/analysis/tradeoffs.py -- do not edit by hand.",
        "% Display rounding: decimal ROUND_HALF_UP (half away from zero); rho to 2 dp from the float's shortest repr,",
        "% percentages to 1 dp from the exact rational 100*k/N. Full precision: table6_revised_14_explainers.csv.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5.5pt}",
        r"  \caption{Summary statistics for trustworthiness trade-offs (complementing \figurename~\ref{fig:tradeoffs_top2_grid}). "
        r"For each dataset and trustworthiness-pillar pair, we report the number of supported \{model, explainer\} configurations "
        r"($N$), the Spearman correlation, the percentage of Pareto-efficient configurations, and the percentage in the top quartile "
        r"of both pillar scores (DualQ).}",
        r"  \label{tab:rq2_stats}",
        r"  \begin{tabular}{l l r r r r}",
        r"    \toprule",
        r"    \textbf{Dataset} & \textbf{Pair} & \textbf{$N$} & \textbf{$\rho$} & \textbf{Pareto \%} & \textbf{DualQ \%} \\",
        r"    \midrule",
    ]
    datasets = list(dict.fromkeys(t["Dataset"]))
    for i, ds in enumerate(datasets):
        rows = t[t["Dataset"] == ds]
        for j, (_, r) in enumerate(rows.iterrows()):
            if j == 0:
                L.append(rf"    \multirow{{{len(rows)}}}{{*}}{{\textsc{{{ds}}}}}")
            L.append(f"      & {r['Pair']} & {int(r['N configurations'])} & {num(r['rho_display'])} & "
                     f"{num(r['pareto_pct_display'])} & {num(r['dualq_pct_display'])} \\\\")
        if i < len(datasets) - 1:
            L.append(r"    \midrule")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composites", type=Path, default=IN_COMPOSITES)
    parser.add_argument("--performance", type=Path, default=IN_PERFORMANCE)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.composites, float_precision="round_trip")

    need = {"Dataset", "Model", "Explainer", "F_comp", "R_comp", "C_comp"}
    if not need.issubset(df.columns):
        raise SystemExit(f"Composites file must contain {need}; got {df.columns.tolist()}")

    perf = pd.read_csv(args.performance)
    for ds, g in perf[perf["highlighted"]].groupby("dataset"):
        HIGHLIGHT_BY_DATASET[ds] = g.sort_values(["performance_rank", "model"])["model"].tolist()

    all_explainers = list(pd.unique(df["Explainer"]))
    missing = sorted(set(all_explainers) - set(EXPLAINER_MARKERS))
    if missing or len(set(EXPLAINER_MARKERS.values())) != len(EXPLAINER_MARKERS):
        raise SystemExit(f"marker mapping incomplete or not unique: missing {missing}")
    expl_markers = EXPLAINER_MARKERS

    save_legend_pdf(
        LEGEND_ORDER,
        expl_markers,
        fig_dir / "tradeoff_legend.pdf",
        ncol=7,
    )

    PAIRS = [("F_comp", "R_comp"), ("F_comp", "C_comp"), ("R_comp", "C_comp")]

    stats_rows = []
    point_rows = []

    for ds, g in df.groupby("Dataset", dropna=False):
        make_plot = ds in MAIN_DATASETS

        for (xa, ya) in PAIRS:
            sub = g[["Dataset", "Model", "Explainer", xa, ya]].dropna()
            if sub.empty:
                continue

            rho, (lo, hi) = spearman_ci(sub[xa].to_numpy(), sub[ya].to_numpy(), n_boot=2000, seed=13)

            flags = pareto_front_flags(sub[xa].to_numpy(), sub[ya].to_numpy())
            n = len(sub)
            n_front = int(flags.sum())
            share_pct = 100.0 * n_front / n if n else np.nan
            qx, qy = np.nanquantile(sub[xa], 0.75), np.nanquantile(sub[ya], 0.75)
            dual_mask = ((sub[xa] >= qx) & (sub[ya] >= qy)).to_numpy()
            dual_rate = 100.0 * dual_mask.sum() / n if n else np.nan

            pair_name = f"{xa.replace('_comp','')} vs {ya.replace('_comp','')}"
            annotation = (f"ρ={round_half_up_str(rho, 2)} [{round_half_up_str(lo, 2)},{round_half_up_str(hi, 2)}]\n"
                          f"Pareto={round_half_up_str(pct_fraction(n_front, n), 0)}%\n"
                          f"DualQ={round_half_up_str(pct_fraction(int(dual_mask.sum()), n), 0)}%")
            stats_rows.append({
                "Dataset": ds,
                "Dataset_id": ds,
                "display_name": DISPLAY_NAME.get(ds, ds),
                "pair": pair_name,
                "n_points": n,
                "rho_spearman": rho,
                "rho_ci_low": lo,
                "rho_ci_high": hi,
                "n_on_front": n_front,
                "front_share_pct": share_pct,
                "n_dualq": int(dual_mask.sum()),
                "dual_top_quartile_pct": dual_rate,
                "q75_x": qx,
                "q75_y": qy,
                "annotation": annotation,
            })
            for (md, ex, xv, yv), fr, dq in zip(sub[["Model", "Explainer", xa, ya]].itertuples(index=False, name=None),
                                               flags, dual_mask):
                point_rows.append(dict(Dataset=ds, pair=pair_name, Model=md, Explainer=ex, x=xv, y=yv,
                                       on_front=bool(fr), dualq=bool(dq),
                                       highlighted=md in HIGHLIGHT_BY_DATASET.get(ds, [])))

            if make_plot:
                fig, ax = plt.subplots(1, 1, figsize=(5.3, 4.8))

                plot_pair(ax, sub, xa, ya, expl_markers)

                ax.text(
                    0.02, 0.98,
                    annotation,
                    transform=ax.transAxes,
                    va="top", ha="left",
                    fontsize=11,  # bigger annotation font
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
                )

                ax.set_title("")
                plt.tight_layout()

                out_pdf = fig_dir / f"tradeoff_{ds}_{xa.replace('_comp','')}_vs_{ya.replace('_comp','')}.pdf"
                fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
                plt.close(fig)
                print(f"[ok] wrote {out_pdf}")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.drop(columns=["Dataset_id"]).to_csv(args.out_dir / "pairwise_tradeoff_stats.csv", index=False, float_format="%.17g")
    print(f"[ok] wrote {args.out_dir / 'pairwise_tradeoff_stats.csv'}")
    pd.DataFrame(point_rows).to_csv(args.out_dir / "tradeoff_points_long.csv", index=False, float_format="%.17g")
    print(f"[ok] wrote {args.out_dir / 'tradeoff_points_long.csv'}")

    # Table 6 from the same statistics
    t6 = build_table6(stats_df)
    t6.to_csv(args.out_dir / "table6_revised_14_explainers.csv", index=False, float_format="%.17g")
    (args.out_dir / "table6_revised_14_explainers.tex").write_text(table6_latex(t6))
    print(f"[ok] wrote {args.out_dir / 'table6_revised_14_explainers.csv'} and .tex")

if __name__ == "__main__":
    main()
