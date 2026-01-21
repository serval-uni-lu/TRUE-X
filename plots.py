from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ============================================================
# Expected input dataframe columns
# ============================================================
# df must contain:
#   Dataset (optional), Model, Explainer, Metric, Value
#
# - Value must be direction-aligned already:
#     higher = better for ALL metrics (including performance)
#   e.g. for RMSE you should pass -RMSE as Value
#
# - We infer Metric -> Category internally.
# - Performance is included as Metric(s) and treated like other categories:
#     normalize per metric, then mean within Performance category.


FAITHFULNESS_METRICS = {
    "Faithfulness Correlation",
    "Pixel Flipping",
}

ROBUSTNESS_METRICS = {
    "Average Sensitivity",
    "Continuity",
}

COMPLEXITY_METRICS = {
    "Sparseness (Elem)",
    "Sparseness (Chan)",
    "Complexity (Chan)",
    "Complexity (Elem)",
}

PERFORMANCE_METRICS = {
    "Accuracy",
    "RMSE",
    # "F1",
    # "AUC",
    # "MAE",
}

METRIC_TO_CATEGORY: Dict[str, str] = {}
METRIC_TO_CATEGORY.update({m: "Faithfulness" for m in FAITHFULNESS_METRICS})
METRIC_TO_CATEGORY.update({m: "Robustness" for m in ROBUSTNESS_METRICS})
METRIC_TO_CATEGORY.update({m: "Complexity" for m in COMPLEXITY_METRICS})
METRIC_TO_CATEGORY.update({m: "Performance" for m in PERFORMANCE_METRICS})


def _infer_metric_category(metric: Any) -> Optional[str]:
    if metric is None:
        return None
    return METRIC_TO_CATEGORY.get(str(metric))


def _minmax_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if not s.notna().any():
        return pd.Series(np.nan, index=s.index)
    mn = np.nanmin(s.values)
    mx = np.nanmax(s.values)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return pd.Series(np.nan, index=s.index)
    return (s - mn) / (mx - mn)


def compute_weighted_scores(
    df: pd.DataFrame,
    coeffs: Dict[str, float],
    *,
    dataset_col: Optional[str] = "Dataset",
    model_col: str = "Model",
    explainer_col: str = "Explainer",
    metric_col: str = "Metric",
    value_col: str = "Value",
    strict_metric_mapping: bool = True,
    normalize_per_dataset: bool = True,
) -> pd.DataFrame:
    """
    Produces one row per (Model, Explainer) (or per (Dataset,Model,Explainer) if Dataset exists)
    with:
      - Faithfulness / Robustness / Complexity / Performance:
          mean of per-metric min-max normalized values within that category
      - final_score:
          T*(F*Faith + R*Rob + C*Comp) + P*Perf

    Normalization:
      - Value is min-max normalized PER Metric.
      - If dataset_col exists and normalize_per_dataset=True:
            min-max is done per (Dataset, Metric)
        else:
            per Metric over the entire df.

    IMPORTANT:
      - "Performance" is treated like any other category:
        it can have multiple performance metrics (Accuracy, -RMSE, etc.)
        and we take the mean of their normalized values.
    """
    base_required = {model_col, explainer_col, metric_col, value_col}
    missing = base_required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    has_dataset = dataset_col is not None and dataset_col in df.columns
    group_keys: List[str] = [model_col, explainer_col]
    if has_dataset:
        group_keys = [dataset_col] + group_keys

    work_cols = group_keys + [metric_col, value_col]
    work = df[work_cols].copy()

    work["_metric_category"] = work[metric_col].map(_infer_metric_category)
    unknown = work["_metric_category"].isna()
    if unknown.any():
        unknown_metrics = sorted(set(work.loc[unknown, metric_col].astype(str).tolist()))
        msg = f"Unknown metric(s) with no category mapping: {unknown_metrics}"
        if strict_metric_mapping:
            raise ValueError(msg)
        work = work.loc[~unknown].copy()

    if has_dataset and normalize_per_dataset:
        work["metric_norm"] = work.groupby([dataset_col, metric_col], dropna=False)[value_col].transform(_minmax_series)
    else:
        work["metric_norm"] = work.groupby(metric_col, dropna=False)[value_col].transform(_minmax_series)

    cat_means = (
        work.groupby(group_keys + ["_metric_category"], dropna=False)
        .agg(cat_mean=("metric_norm", "mean"))
        .reset_index()
    )

    out = (
        cat_means.pivot_table(
            index=group_keys,
            columns="_metric_category",
            values="cat_mean",
            aggfunc="first",
        )
        .reset_index()
    )

    for col in ["Faithfulness", "Robustness", "Complexity", "Performance"]:
        if col not in out.columns:
            out[col] = np.nan

    F = float(coeffs.get("F", 0.0))
    R = float(coeffs.get("R", 0.0))
    C = float(coeffs.get("C", 0.0))
    P = float(coeffs.get("P", 0.0))

    if "T" in coeffs:
        T = float(coeffs["T"])
    else:
        T = 1.0 if (F != 0.0 or R != 0.0 or C != 0.0) else 0.0

    faith = out["Faithfulness"].fillna(0.0)
    rob = out["Robustness"].fillna(0.0)
    comp = out["Complexity"].fillna(0.0)
    perf = out["Performance"].fillna(0.0)

    out["xai_component"] = (F * faith) + (R * rob) + (C * comp)
    out["xai_gated"] = T * out["xai_component"]
    out["perf_component"] = P * perf
    out["final_score"] = out["xai_gated"] + out["perf_component"]

    if (F == 0.0) and (R == 0.0) and (C == 0.0) and (P == 0.0):
        out["final_score"] = np.nan
        out["note"] = "No coefficients selected (F,R,C,P all zero)."
    else:
        out["note"] = ""

    sort_cols = ["final_score"] + group_keys
    out = out.sort_values(
        sort_cols,
        ascending=[False] + [True] * len(group_keys),
        na_position="last",
    ).reset_index(drop=True)
    return out


def rank_and_select_top_k(
    df: pd.DataFrame,
    coeffs: Dict[str, float],
    k: int,
    *,
    dataset_col: Optional[str] = "Dataset",
    model_col: str = "Model",
    explainer_col: str = "Explainer",
    metric_col: str = "Metric",
    value_col: str = "Value",
    strict_metric_mapping: bool = True,
    normalize_per_dataset: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str]]]:
    """
    Returns:
      ranking_all  : one row per group with final_score and rank
      ranking_topk : top-k subset
      topk_pairs   : list[(Model, Explainer)] (dataset not included on purpose)
    """
    if k is None or int(k) < 0:
        raise ValueError("k must be a non-negative integer.")
    k = int(k)

    ranking_all = compute_weighted_scores(
        df, coeffs,
        dataset_col=dataset_col,
        model_col=model_col,
        explainer_col=explainer_col,
        metric_col=metric_col,
        value_col=value_col,
        strict_metric_mapping=strict_metric_mapping,
        normalize_per_dataset=normalize_per_dataset,
    ).copy()

    if ranking_all["final_score"].notna().any():
        ranking_all["rank"] = ranking_all["final_score"].rank(method="average", ascending=False)
    else:
        ranking_all["rank"] = np.nan

    eligible = ranking_all[ranking_all["final_score"].notna()].copy()
    ranking_topk = eligible.head(min(k, len(eligible))).reset_index(drop=True)

    topk_pairs = list(zip(ranking_topk[model_col].astype(str), ranking_topk[explainer_col].astype(str)))
    return ranking_all, ranking_topk, topk_pairs


DIM_TO_AXIS = {
    "F": ("Faithfulness", "Faithfulness"),
    "R": ("Robustness", "Robustness"),
    "C": ("Complexity", "Complexity"),
    "P": ("Performance", "Performance"),
}


def _active_dims_FRCP(coeffs: Dict[str, float]) -> List[str]:
    return [d for d in ["F", "R", "C", "P"] if float(coeffs.get(d, 0.0)) != 0.0]


def generate_topk_plots_auto(
    df: pd.DataFrame,
    coeffs: Dict[str, float],
    k: int,
    **kwargs,
) -> List[go.Figure]:
    """
    Routes based on how many active coeffs among F,R,C,P:
      - 3/4 -> radar
      - 2   -> two-axis
      - 1   -> one-bar
      - 0   -> []
    """
    active_dims = _active_dims_FRCP(coeffs)
    if len(active_dims) == 0:
        return []
    if len(active_dims) in (3, 4):
        return generate_topk_radar_plots(df, coeffs, k, **kwargs)
    if len(active_dims) == 2:
        return generate_topk_two_axis_plots(df, coeffs, k, **kwargs)
    if len(active_dims) == 1:
        return generate_topk_one_bar_plots(df, coeffs, k, **kwargs)
    return []


def generate_topk_radar_plots(
    df: pd.DataFrame,
    coeffs: Dict[str, float],
    k: int,
    *,
    dataset_col: Optional[str] = "Dataset",
    model_col: str = "Model",
    explainer_col: str = "Explainer",
    metric_col: str = "Metric",
    value_col: str = "Value",
    strict_metric_mapping: bool = True,
    normalize_per_dataset: bool = True,
) -> List[go.Figure]:
    active_dims = _active_dims_FRCP(coeffs)
    if len(active_dims) not in (3, 4):
        raise ValueError("Radar requires 3 or 4 active coeffs among F,R,C,P.")

    _, topk, _ = rank_and_select_top_k(
        df, coeffs, k,
        dataset_col=dataset_col,
        model_col=model_col,
        explainer_col=explainer_col,
        metric_col=metric_col,
        value_col=value_col,
        strict_metric_mapping=strict_metric_mapping,
        normalize_per_dataset=normalize_per_dataset,
    )

    axes = [DIM_TO_AXIS[d][0] for d in active_dims]
    cols = [DIM_TO_AXIS[d][1] for d in active_dims]

    figs: List[go.Figure] = []
    for _, row in topk.iterrows():
        model = str(row[model_col])
        expl = str(row[explainer_col])

        values = [float(row[c]) if pd.notna(row[c]) else 0.0 for c in cols]
        theta = axes + [axes[0]]
        rvals = values + [values[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=rvals, theta=theta, fill="toself"))
        fig.update_layout(
            title=f"{model} / {expl}",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
        )
        figs.append(fig)
    return figs


def generate_topk_two_axis_plots(
    df: pd.DataFrame,
    coeffs: Dict[str, float],
    k: int,
    *,
    dataset_col: Optional[str] = "Dataset",
    model_col: str = "Model",
    explainer_col: str = "Explainer",
    metric_col: str = "Metric",
    value_col: str = "Value",
    strict_metric_mapping: bool = True,
    normalize_per_dataset: bool = True,
    marker_size: int = 16,
    marker_line_width: int = 2,
) -> List[go.Figure]:
    active_dims = _active_dims_FRCP(coeffs)
    if len(active_dims) != 2:
        raise ValueError("2-axis plot requires exactly 2 active coeffs among F,R,C,P.")

    _, topk, _ = rank_and_select_top_k(
        df, coeffs, k,
        dataset_col=dataset_col,
        model_col=model_col,
        explainer_col=explainer_col,
        metric_col=metric_col,
        value_col=value_col,
        strict_metric_mapping=strict_metric_mapping,
        normalize_per_dataset=normalize_per_dataset,
    )

    d1, d2 = active_dims
    x_label, x_col = DIM_TO_AXIS[d1]
    y_label, y_col = DIM_TO_AXIS[d2]

    figs: List[go.Figure] = []
    for _, row in topk.iterrows():
        model = str(row[model_col])
        expl = str(row[explainer_col])

        x = float(row[x_col]) if pd.notna(row[x_col]) else 0.0
        y = float(row[y_col]) if pd.notna(row[y_col]) else 0.0

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                text=[f"{model}/{expl}"],
                textposition="top center",
                showlegend=False,
                cliponaxis=False,
                marker=dict(
                    size=marker_size,
                    line=dict(width=marker_line_width),
                ),
            )
        )
        fig.update_layout(
            title=f"{model} / {expl}",
            xaxis=dict(title=x_label, range=[-0.03, 1.0]),
            yaxis=dict(title=y_label, range=[-0.03, 1.0]),
        )
        figs.append(fig)
    return figs


def generate_topk_one_bar_plots(
    df: pd.DataFrame,
    coeffs: Dict[str, float],
    k: int,
    *,
    dataset_col: Optional[str] = "Dataset",
    model_col: str = "Model",
    explainer_col: str = "Explainer",
    metric_col: str = "Metric",
    value_col: str = "Value",
    strict_metric_mapping: bool = True,
    normalize_per_dataset: bool = True,
) -> List[go.Figure]:
    active_dims = _active_dims_FRCP(coeffs)
    if len(active_dims) != 1:
        raise ValueError("1-bar plot requires exactly 1 active coeff among F,R,C,P.")

    _, topk, _ = rank_and_select_top_k(
        df, coeffs, k,
        dataset_col=dataset_col,
        model_col=model_col,
        explainer_col=explainer_col,
        metric_col=metric_col,
        value_col=value_col,
        strict_metric_mapping=strict_metric_mapping,
        normalize_per_dataset=normalize_per_dataset,
    )

    d = active_dims[0]
    axis_label, axis_col = DIM_TO_AXIS[d]

    figs: List[go.Figure] = []
    for _, row in topk.iterrows():
        model = str(row[model_col])
        expl = str(row[explainer_col])
        val = float(row[axis_col]) if pd.notna(row[axis_col]) else 0.0

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[axis_label], y=[val], showlegend=False))
        fig.update_layout(
            title=f"{model} / {expl}",
            xaxis=dict(title="Dimension"),
            yaxis=dict(title="Normalized value", range=[0, 1]),
        )
        figs.append(fig)
    return figs


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


def make_explainer_style_plotly(explainers: List[str]):
    colors = px.colors.qualitative.Safe + px.colors.qualitative.Set3 + px.colors.qualitative.Bold
    symbols_cycle = [
        "circle", "square", "triangle-up", "diamond", "cross", "triangle-down",
        "x", "triangle-left", "triangle-right", "hexagon", "octagon", "star", "pentagon"
    ]
    color_map = {e: colors[i % len(colors)] for i, e in enumerate(explainers)}
    symbol_map = {e: symbols_cycle[i % len(symbols_cycle)] for i, e in enumerate(explainers)}
    return color_map, symbol_map


def _active_dims_FRC(coeffs: Dict[str, float]) -> List[str]:
    return [d for d in ["F", "R", "C"] if float(coeffs.get(d, 0.0)) != 0.0]


def _pairs_from_active_frc(active: List[str]) -> List[Tuple[str, str]]:
    dim_to_cat = {"F": "Faithfulness", "R": "Robustness", "C": "Complexity"}
    if len(active) == 3:
        return [
            (dim_to_cat["F"], dim_to_cat["R"]),
            (dim_to_cat["F"], dim_to_cat["C"]),
            (dim_to_cat["R"], dim_to_cat["C"]),
        ]
    if len(active) == 2:
        a, b = active
        return [(dim_to_cat[a], dim_to_cat[b])]
    return []


def _coef_for_category(cat: str, coeffs: Dict[str, float]) -> float:
    if cat == "Faithfulness":
        return float(coeffs.get("F", 0.0))
    if cat == "Robustness":
        return float(coeffs.get("R", 0.0))
    if cat == "Complexity":
        return float(coeffs.get("C", 0.0))
    return 1.0


def generate_tradeoff_figures_with_pareto(
    df: pd.DataFrame,
    coeffs: Dict[str, float],
    *,
    dataset_col: Optional[str] = "Dataset",
    model_col: str = "Model",
    explainer_col: str = "Explainer",
    metric_col: str = "Metric",
    value_col: str = "Value",
    axis_max: float = 1.2,
    axis_min_pad: float = 0.03,
    pareto_use_coeffs: bool = True,
    pareto_use_abs_coeffs: bool = True,
    pareto_fallback_unweighted_if_zero: bool = True,
    strict_metric_mapping: bool = True,
    normalize_per_dataset: bool = True,
    
    dominated_marker_size: int = 12,
    dominated_marker_opacity: float = 0.80,
    dominated_marker_border_width: int = 2,
) -> List[go.Figure]:
    """
    Coeff cases (F,R,C only):
      - 3 active -> returns [ONE fig] with 3 subplots: F vs R, F vs C, R vs C
      - 2 active -> returns [ONE fig] with 1 subplot for that pair
      - 1 or 0 active -> returns []
    """
    active_frc = _active_dims_FRC(coeffs)
    pairs = _pairs_from_active_frc(active_frc)
    if not pairs:
        return []

    scored = compute_weighted_scores(
        df, coeffs,
        dataset_col=dataset_col,
        model_col=model_col,
        explainer_col=explainer_col,
        metric_col=metric_col,
        value_col=value_col,
        strict_metric_mapping=strict_metric_mapping,
        normalize_per_dataset=normalize_per_dataset,
    ).copy()

    explainers = list(pd.unique(scored[explainer_col].astype(str)))
    color_map, symbol_map = make_explainer_style_plotly(explainers)

    fig = make_subplots(
        rows=1,
        cols=len(pairs),
        subplot_titles=[f"{x} vs {y}" for (x, y) in pairs],
        horizontal_spacing=0.08 if len(pairs) > 1 else 0.03,
    )

    legend_added_for: set[str] = set()

    for col_idx, (xcol, ycol) in enumerate(pairs, start=1):
        sub = scored[[model_col, explainer_col, xcol, ycol]].dropna().copy()
        if sub.empty:
            continue

        cx = _coef_for_category(xcol, coeffs) if pareto_use_coeffs else 1.0
        cy = _coef_for_category(ycol, coeffs) if pareto_use_coeffs else 1.0
        if pareto_use_abs_coeffs:
            cx, cy = abs(cx), abs(cy)
        if pareto_fallback_unweighted_if_zero and pareto_use_coeffs and (cx == 0.0 or cy == 0.0):
            cx, cy = 1.0, 1.0

        sub["_x_pareto"] = sub[xcol] * cx
        sub["_y_pareto"] = sub[ycol] * cy
        sub["on_front"] = pareto_front_flags(sub["_x_pareto"].to_numpy(), sub["_y_pareto"].to_numpy())

        dom = sub[~sub["on_front"]]
        fr = sub[sub["on_front"]]

        for expl, g in dom.groupby(explainer_col, sort=False):
            expl = str(expl)
            show_leg = expl not in legend_added_for
            if show_leg:
                legend_added_for.add(expl)

            fig.add_trace(
                go.Scatter(
                    x=g[xcol], y=g[ycol],
                    mode="markers",
                    name=expl,
                    legendgroup=expl,
                    showlegend=show_leg,
                    marker=dict(
                        size=dominated_marker_size,
                        color=color_map.get(expl),
                        symbol=symbol_map.get(expl, "circle"),
                        opacity=dominated_marker_opacity,
                        line=dict(color="white", width=dominated_marker_border_width),
                    ),
                    cliponaxis=False,
                ),
                row=1, col=col_idx,
            )

        for expl, g in fr.groupby(explainer_col, sort=False):
            expl = str(expl)
            fig.add_trace(
                go.Scatter(
                    x=g[xcol], y=g[ycol],
                    mode="markers",
                    name=expl,
                    legendgroup=expl,
                    showlegend=False,
                    marker=dict(
                        size=14,
                        color=color_map.get(expl),
                        symbol=symbol_map.get(expl, "circle"),
                        opacity=0.95,
                        line=dict(color="black", width=2),
                    ),
                    cliponaxis=False,
                ),
                row=1, col=col_idx,
            )

        if len(fr) >= 2:
            pts = fr[[xcol, ycol]].sort_values(by=[xcol, ycol]).to_numpy()
            fig.add_trace(
                go.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=col_idx,
            )

        fig.update_xaxes(title_text=xcol, range=[-axis_min_pad, axis_max], row=1, col=col_idx)
        fig.update_yaxes(title_text=ycol, range=[-axis_min_pad, axis_max], row=1, col=col_idx)

    fig.update_layout(
        title="Tradeoffs (F/R/C) with Pareto front",
        margin=dict(l=60, r=40, t=80, b=110),
        legend=dict(
            title=explainer_col,
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
    )

    return [fig]


def fig_to_plotly_dict(fig: go.Figure) -> Dict[str, Any]:
    """JSON-serializable figure dict (useful in backend responses)."""
    return fig.to_dict()

'''
{"Dataset": ds, "Model": model, "Explainer": expl, "Metric": "Faithfulness Correlation", "Value": f1},
{"Dataset": ds, "Model": model, "Explainer": expl, "Metric": "Pixel Flipping",            "Value": f2},
{"Dataset": ds, "Model": model, "Explainer": expl, "Metric": "Average Sensitivity",       "Value": r1},
{"Dataset": ds, "Model": model, "Explainer": expl, "Metric": "Continuity",                "Value": r2},
{"Dataset": ds, "Model": model, "Explainer": expl, "Metric": "Sparseness (Elem)",         "Value": c1},
{"Dataset": ds, "Model": model, "Explainer": expl, "Metric": "Complexity (Chan)",         "Value": c2},
{"Dataset": ds, "Model": model, "Explainer": expl, "Metric": "Accuracy",                  "Value": acc},
{"Dataset": ds, "Model": model, "Explainer": expl, "Metric": "RMSE",                      "Value": -rmse},  # direction-aligned


coeffs = {"T": 1.0, "F": 0.3, "R": 0.3, "C": 0.2, "P": 0.2}
figs = generate_topk_plots_auto(
    dummy, coeffs, k=2,
    dataset_col="Dataset",
    normalize_per_dataset=True,
    strict_metric_mapping=True,
)


coef_tradeoff = {"F": 1.0, "R": 1.0, "C": 1.0}
figs = generate_tradeoff_figures_with_pareto(
    dummy, coef_tradeoff,
    dataset_col="Dataset",
    normalize_per_dataset=True,
    strict_metric_mapping=True,
)
'''
