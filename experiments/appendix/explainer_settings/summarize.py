# summarize.py
"""
Summaries of the explainer-setting diagnostics (Table C.1 and Figure C.1): per model
and tested value, the median over samples of each diagnostic.

Usage (from experiments/):
    python appendix/explainer_settings/summarize.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "r2_6_parameter_sensitivity"

MODELS = ["FCN", "LSTM", "TST"]
PARAM_EXPLAINER = {
    "ig_steps": "integrated_gradients", "smoothgrad_n": "smooth_gradient",
    "gradientshap_n": "gradientshap", "eg_steps": "expected_gradients",
    "eg_M": "expected_gradients", "deepshap_M": "deepliftshap",
}
PARAM_GRID = {
    "ig_steps": [10, 25, 50, 100, 200], "smoothgrad_n": [5, 10, 20, 50, 100],
    "gradientshap_n": [5, 10, 20, 50, 100], "eg_steps": [10, 25, 50, 100, 200],
    "eg_M": [10, 25, 50, 100, 200], "deepshap_M": [10, 25, 50, 100, 200],
}
HISTORICAL_VALUE = {
    "ig_steps": 50, "smoothgrad_n": 20, "gradientshap_n": 20,
    "eg_steps": 50, "eg_M": 50, "deepshap_M": 50,
}
STOCHASTIC_PARAMS = {"smoothgrad_n", "gradientshap_n"}

PLATEAU_SPEARMAN = 0.98  # reported alongside the continuous curve, never as the sole verdict
PLATEAU_L1 = 0.05


def numerical_summary(num: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for param, explainer in PARAM_EXPLAINER.items():
        sub = num[num.Parameter == param]
        is_stochastic = param in STOCHASTIC_PARAMS
        vs_max = sub[sub.DiagType != "between_seed_spearman"]
        between = sub[sub.DiagType == "between_seed_spearman"]
        for model in MODELS:
            m_vs_max = vs_max[vs_max.Model == model]
            m_between = between[between.Model == model]
            for value in PARAM_GRID[param]:
                v_vs_max = m_vs_max[m_vs_max.Value == value]
                spear = v_vs_max[v_vs_max.DiagType == "spearman_vs_max"].DiagValue
                l1 = v_vs_max[v_vs_max.DiagType == "normalized_l1_vs_max"].DiagValue
                top10 = v_vs_max[v_vs_max.DiagType == "top10_overlap_vs_max"].DiagValue
                v_between = m_between[m_between.Value == value].DiagValue

                rows.append({
                    "Model": model, "Explainer": explainer, "Parameter": param, "Value": value,
                    "IsHistorical": value == HISTORICAL_VALUE[param],
                    "N_ValueObs": len(spear),
                    "Median_Spearman_vs_Max": float(spear.median()),
                    "IQR_Spearman_vs_Max": float(spear.quantile(.75) - spear.quantile(.25)),
                    "Median_NormL1_vs_Max": float(l1.median()),
                    "IQR_NormL1_vs_Max": float(l1.quantile(.75) - l1.quantile(.25)),
                    "Median_Top10Overlap_vs_Max": float(top10.median()),
                    "IsStochastic": is_stochastic,
                    "Median_BetweenSeedSpearman": float(v_between.median()) if is_stochastic else np.nan,
                    "MeetsPlateauThreshold_Spearman0.98_L1_0.05": bool(
                        spear.median() >= PLATEAU_SPEARMAN and l1.median() <= PLATEAU_L1
                    ),
                })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "r2_6_numerical_summary.csv", index=False)
    return out


def semantic_summary(sem_attr: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []

    def attr_stats(explainer, variant):
        sub = sem_attr[(sem_attr.Explainer == explainer) & (sem_attr.Variant == variant)]
        out = {}
        for dt, key in [("spearman_vs_historical", "Spearman"), ("normalized_l1_vs_historical", "NormL1"),
                        ("top10_overlap_vs_historical", "Top10Overlap")]:
            v = sub[sub.DiagType == dt].DiagValue
            out[f"Attr_Median_{key}"] = float(v.median())
            out[f"Attr_ModelToModel_Std"] = out.get(f"Attr_ModelToModel_Std", np.nan)
        # model-to-model variation on Spearman specifically
        per_model_median = sub[sub.DiagType == "spearman_vs_historical"].groupby("Model").DiagValue.median()
        out["Attr_ModelToModel_Std_Spearman"] = float(per_model_median.std())
        return out

    experiments = [
        ("A_SmoothGrad_noise", "smooth_gradient", ["noise=0.05", "noise=0.1", "noise=0.2"], "noise=0.1"),
        ("B_Occlusion_window", "occlusion", ["window=5", "window=10", "window=15"], "window=10"),
        ("C_Occlusion_baseline", "occlusion", ["baseline=trainmean", "baseline=zero_historical"], "baseline=zero_historical"),
        ("D_FeatureAblation_baseline", "feature_ablation", ["baseline=trainmean", "baseline=zero_historical"], "baseline=zero_historical"),
    ]
    for group, explainer, variants, ref_variant in experiments:
        for variant in variants:
            row = {"Group": group, "Explainer": explainer, "Variant": variant,
                   "IsHistoricalReference": variant == ref_variant}
            # attribution-level: for baseline checks, the stored diag variant name differs (single combined row)
            attr_variant = "baseline=trainmean_vs_zero" if "baseline=" in variant else variant
            row.update(attr_stats(explainer, attr_variant))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "r2_6_semantic_summary.csv", index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    out_dir = args.results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    numerical_summary(pd.read_csv(args.results_dir / "r2_6_raw_numerical_diagnostics.csv"), out_dir)
    semantic_summary(pd.read_csv(args.results_dir / "r2_6_raw_semantic_attribution_diagnostics.csv"), out_dir)
    print("Wrote r2_6_numerical_summary.csv and r2_6_semantic_summary.csv to", out_dir)
