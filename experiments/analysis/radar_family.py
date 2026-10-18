# radar_family.py
"""
Radar of the explainer families (Gra, Pat, Per, Sur) for each (dataset, model).

For each metric, the block medians of the explainers of a family are averaged, the
families are ranked (average ranks, higher is better), and the rank is mapped to
[0, 1] (best family = 1). One PDF per (dataset, model), a legend PDF, and
radar_values_long.csv with the family means, ranks and r01 values.

Usage (from the repository root):
    python experiments/analysis/radar_family.py
"""
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

# --------------------- Vector-friendly defaults ---------------------
mpl.rcParams["ps.fonttype"] = 42     # embed TrueType in EPS
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["text.usetex"] = False
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["font.size"] = 10       # base font size (we override some below)

# --------------------- Tunables (spacing / highlight) ---------------------
HIGHLIGHT_BANDS = []  # e.g. [(0.20, 0.40), (0.60, 0.80)]
HIGHLIGHT_COLOR = "#e6e6e6"
GRID_COLOR = "#cfcfcf"

FILL_ALPHA = 0.18

SINGLE_XTICK_FS = 12

# Start distance (in POINTS) for labels measured outward from the radar circle.
# Auto-adjust will push farther only if needed.
SINGLE_LABEL_OFFSET_PT = 18

# Clearance between label bbox and circle, in PIXELS.
LABEL_CLEARANCE_PX = 6

# Offset growth in POINTS per iteration when a label still intersects the circle.
LABEL_STEP_PT = 2

# Safety cap (points)
LABEL_MAX_OFFSET_PT = 60

# Number of points sampled on the circle arc near the spoke to test intersections.
# Higher = more robust.
CIRCLE_SAMPLE_N = 11

# --------------------- Paths ---------------------
_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = _ROOT / "results" / "main_benchmark_14_promethee_input" / "promethee_input_14.csv"
OUT_ROOT = _ROOT / "results" / "main_benchmark_14_radars"

# --------------------- XAI families ---------------------
FAMILY_MAP = {
    # Gradient-based (Gra)
    "saliency": "Gra",
    "gradient_x_input": "Gra",
    "guided_back_prop": "Gra",
    "smooth_gradient": "Gra",

    # Path- & decomposition-based (Pat)
    "integrated_gradients": "Pat",
    "expected_gradients": "Pat",
    "deeplift": "Pat",
    "deepliftshap": "Pat",
    "gradientshap": "Pat",

    # Perturbation-based (Per)
    "occlusion": "Per",
    "feature_ablation": "Per",

    # Surrogate / game-theoretic (Sur)
    "lime_tabular": "Sur",
    "shap_permutation": "Sur",
    "kernelshap": "Sur",
}

FAMILIES_ORDER = ["Gra", "Pat", "Per", "Sur"]

# --------------------- Metric abbreviations & order ---------------------
METRIC_ABBR = {
    "Average Sensitivity (Reg)": "AvgSen",
    "Average Sensitivity (Classif)": "AvgSen",
    "Continuity (Reg)": "Cont",
    "Continuity (Classif)": "Cont",
    "Faithfulness Correlation (Reg)": "FC",
    "Faithfulness Correlation (Classif)": "FC",
    "Pixel Flipping (Reg)": "PF",
    "Pixel Flipping (Classif)": "PF",
    "Sparseness (Reg, Elem)": "Spars-E",
    "Sparseness (Reg, Chan)": "Spars-C",
    "Sparseness (Classif, Elem)": "Spars-E",
    "Sparseness (Classif, Chan)": "Spars-C",
    "Complexity (Reg, Elem)": "Compl-E",
    "Complexity (Reg, Chan)": "Compl-C",
    "Complexity (Classif, Elem)": "Compl-E",
    "Complexity (Classif, Chan)": "Compl-C",
}
ABBR_ORDER = ["AvgSen","Cont","FC","PF","Spars-E","Spars-C","Compl-E","Compl-C"]

# --------------------- Exclusions ---------------------
EXCLUDE = { "Hydraulic": [r"(?i)\bmcd[_\s]*cnn\b", r"(?i)\bmcdc?cnn\b"] }
def _is_excluded(dataset: str, model: str) -> bool:
    pats = EXCLUDE.get(dataset, [])
    return any(re.search(p, model) for p in pats)

# --------------------- Color mapping (per FAMILY, stable across figs) ---------------------
tab10 = get_cmap("tab10")
COLOR_OF_FAMILY = {fam: tab10(i % 10) for i, fam in enumerate(FAMILIES_ORDER)}


# --------------------- Rank01 within a (Dataset, Model) block at FAMILY level ---------------------
def rank01_family_block(df_block: pd.DataFrame, return_all: bool = False):
    # return_all=True also returns the intermediates (M, ranks, k) for export
    df = df_block.copy()
    df["MetricAbbr"] = df["Metric"].map(METRIC_ABBR)
    df["Family"] = df["Explainer"].astype(str).str.lower().map(FAMILY_MAP)

    df = df.dropna(subset=["MetricAbbr", "Family"])
    if df.empty:
        return pd.DataFrame()

    M = df.pivot_table(
        index="MetricAbbr",
        columns="Family",
        values="median_dir",
        aggfunc="mean",
    )

    ranks = M.rank(axis=1, ascending=False, method="average")
    k = ranks.notna().sum(axis=1)
    denom = (k - 1).replace(0, np.nan)
    r01 = 1 - (ranks.sub(1)).div(denom, axis=0)

    present = [abbr for abbr in ABBR_ORDER if abbr in r01.index]

    if return_all:
        return r01.loc[present], M.loc[present], ranks.loc[present], k.loc[present]
    return r01.loc[present]

# --------------------- Polar helpers ---------------------
def _draw_inside_bands(ax, bands=HIGHLIGHT_BANDS, color=HIGHLIGHT_COLOR, zorder=3):
    theta = np.linspace(0, 2*np.pi, 720)
    for r1, r2 in bands:
        ax.fill_between(theta, r1, r2, color=color, zorder=zorder,
                        linewidth=0, clip_on=True)

def _padded_bbox(bbox, pad_px):
    return Bbox.from_extents(bbox.x0 - pad_px, bbox.y0 - pad_px, bbox.x1 + pad_px, bbox.y1 + pad_px)

def _circle_points_display(ax, angles, r=1.0):
    pts = [ax.transData.transform((a, r)) for a in angles]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    return xs, ys

def _place_radar_labels_centered_autopush(
    ax,
    angles,
    labels,
    offset_pt_base,
    fontsize,
    rotate=False,
):
    """
    Labels are ALWAYS centered on their spoke (ha='center', va='center').
    They are positioned by an offset (in points) from the radar circle (r=1),
    along the outward radial direction, and auto-increased until the label
    bbox no longer intersects the circle near that spoke.
    """
    ax.set_xticks(angles)
    ax.set_xticklabels([])

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    m = len(labels)
    sector = 2*np.pi / max(m, 1)
    # Sample an arc around each spoke to ensure long labels don't clip the circle.
    half_arc = 0.40 * sector

    texts = []
    for a, lab in zip(angles, labels):
        a = float(a) % (2*np.pi)

        # Outward radial direction in DISPLAY coords:
        p1 = ax.transData.transform((a, 1.00))
        p2 = ax.transData.transform((a, 1.02))
        v = p2 - p1
        vn = v / (np.linalg.norm(v) + 1e-9)

        # We will annotate at the circle (r=1) and push outward in offset points.
        offset_pt = float(offset_pt_base)

        rot = (np.degrees(a) - 90.0) if rotate else 0.0

        ann = ax.annotate(
            lab,
            xy=(a, 1.0),
            xycoords="data",
            xytext=(vn[0] * offset_pt, vn[1] * offset_pt),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=fontsize,
            rotation=rot,
            rotation_mode="anchor",
            clip_on=False,
            zorder=20,
        )

        # Iteratively push outward if the label intersects the circle near this spoke.
        for _ in range(300):
            fig.canvas.draw()
            bbox = ann.get_window_extent(renderer=renderer)
            pb = _padded_bbox(bbox, LABEL_CLEARANCE_PX)

            # sample points on the circle around this spoke
            samp_angles = np.linspace(a - half_arc, a + half_arc, CIRCLE_SAMPLE_N)
            xs, ys = _circle_points_display(ax, samp_angles, r=1.0)

            intersects = np.any((xs >= pb.x0) & (xs <= pb.x1) & (ys >= pb.y0) & (ys <= pb.y1))
            if not intersects:
                break

            offset_pt += LABEL_STEP_PT
            if offset_pt > LABEL_MAX_OFFSET_PT:
                offset_pt = LABEL_MAX_OFFSET_PT
                ann.set_position((vn[0] * offset_pt, vn[1] * offset_pt))
                break

            ann.set_position((vn[0] * offset_pt, vn[1] * offset_pt))

        texts.append(ann)

    return texts

def _style_polar(
    ax,
    metric_labels,
    show_rticks=False,
    xtick_fs=7,
    label_offset_pt=12,
    rotate_labels=False,
):
    m = len(metric_labels)
    angles = np.linspace(0, 2*np.pi, m, endpoint=False)

    ax.set_facecolor("white")
    ax.set_theta_offset(np.pi/2)   # start at top
    ax.set_theta_direction(-1)     # clockwise
    ax.set_ylim(0, 1)

    _place_radar_labels_centered_autopush(
        ax,
        angles=angles,
        labels=metric_labels,
        offset_pt_base=label_offset_pt,
        fontsize=xtick_fs,
        rotate=rotate_labels,
    )

    if show_rticks:
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8], angle=90, fontsize=8)
    else:
        ax.set_yticklabels([])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])

    ax.grid(True, linestyle='-', linewidth=0.4, color=GRID_COLOR, zorder=5)

def _plot_radar(ax, values_per_label, metric_labels, color_per_label,
                line_w=1.6, fill_alpha=0.0):
    m = len(metric_labels)
    angles = np.linspace(0, 2*np.pi, m, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    for lab, vals in values_per_label.items():
        v = np.asarray(vals, dtype=float)
        v = np.nan_to_num(v, nan=0.0)
        v_closed = np.concatenate([v, v[:1]])

        if fill_alpha and fill_alpha > 0:
            ax.fill(angles_closed, v_closed,
                    color=color_per_label[lab],
                    alpha=fill_alpha, linewidth=0, zorder=8, clip_on=True)

        ax.plot(angles_closed, v_closed,
                color=color_per_label[lab],
                linewidth=line_w, zorder=10)

# --------------------- Single model (PDF) ---------------------
def save_single_model_radar(figs_dir, dataset, model, r01_block, families, metric_labels,
                            fig_w=4.8, fig_h=3.2):
    families = [f for f in FAMILIES_ORDER if f in families and f in r01_block.columns]

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_polar = fig.add_subplot(111, polar=True)

    _style_polar(
        ax_polar,
        metric_labels,
        show_rticks=False,
        xtick_fs=SINGLE_XTICK_FS,
        label_offset_pt=SINGLE_LABEL_OFFSET_PT,
        rotate_labels=False,
    )
    _draw_inside_bands(ax_polar)

    values = {f: r01_block.loc[metric_labels, f].values for f in families}
    _plot_radar(
        ax_polar, values, metric_labels,
        {f: COLOR_OF_FAMILY[f] for f in families},
        line_w=1.7, fill_alpha=FILL_ALPHA,
    )

    out_base = figs_dir / f"radar_{dataset}_{model}"
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
    print("Saved:", out_base.with_suffix(".pdf"))

# --------------------- Separate legend figure ---------------------
def save_family_legend(out_root):
    fam_used = [f for f in FAMILIES_ORDER]

    fig, ax = plt.subplots(figsize=(2.4, 3.2))
    ax.axis('off')

    handles = [Line2D([0], [0], color=COLOR_OF_FAMILY[f], lw=2.5) for f in fam_used]
    labels = fam_used

    ax.legend(
        handles, labels,
        loc='center',
        fontsize=10,
        frameon=False,
        handlelength=2.4,
        title="XAI family",
        title_fontsize=10,
        ncol=1,
        labelspacing=0.8,
        borderaxespad=0.0,
    )

    out_base = out_root / "radar_family_legend"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("Saved:", out_base.with_suffix(".pdf"))

# --------------------- Driver ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    figs_dir = args.out_dir / "radars"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # block medians are already direction-aligned; no direction is applied here
    blk = pd.read_csv(args.input, float_precision="round_trip")
    need = {'Dataset','Model','Explainer','Metric','median_dir'}
    missing = need - set(blk.columns)
    if missing:
        raise ValueError(f"Aggregates missing columns: {missing}")
    blk = blk[['Dataset','Model','Explainer','Metric','median_dir']].copy()

    rows = []

    # 1) Singles (radar only, no legend)
    for (ds, md), g in blk.groupby(["Dataset","Model"]):
        if _is_excluded(ds, md):
            print(f"[skip single] excluded: {ds}/{md}")
            continue
        r01_fam, M, ranks, k = rank01_family_block(g, return_all=True)
        if r01_fam.empty or r01_fam.shape[0] < 3:
            continue
        metric_labels = [a for a in ABBR_ORDER if a in r01_fam.index]
        families_present = [f for f in FAMILIES_ORDER if f in r01_fam.columns]
        if not families_present:
            continue
        # a present family must have a value on every axis; _plot_radar would draw NaN as 0
        if r01_fam[families_present].isna().any().any():
            raise ValueError(f"NaN r01 for a present family in {ds}/{md}")

        save_single_model_radar(
            figs_dir, ds, md, r01_fam, families_present, metric_labels,
            fig_w=4.8, fig_h=3.2,
        )

        for i, met in enumerate(metric_labels):
            for fam in families_present:
                rows.append(dict(dataset=ds, model=md, metric=met, axis_index=i, family=fam,
                                 family_mean=M.loc[met, fam], family_rank=ranks.loc[met, fam],
                                 k_families=int(k.loc[met]), r01=r01_fam.loc[met, fam]))

    pd.DataFrame(rows).to_csv(args.out_dir / "radar_values_long.csv", index=False, float_format="%.17g")

    # 2) Separate legend figure (PDF)
    save_family_legend(args.out_dir)

if __name__ == "__main__":
    main()
