import streamlit as st
import pandas as pd
from pathlib import Path

_APP_DIR  = Path(__file__).resolve().parent
_ROOT     = _APP_DIR.parent
RANKS_CSV = str(_ROOT / "results" / "ranks_per_block_with_performance.csv")

try:
    from PipelineProfiler import get_pipeline_profiler_html
    _PIPELINE_PROFILER_OK = True
except ImportError:
    _PIPELINE_PROFILER_OK = False
from export_profiler import create_pipelines_from_csv
from plots import generate_tradeoff_figures_with_pareto, generate_topk_plots_auto
from entry_b import (
    render_dataset_upload,
    render_windowing_config,
    render_model_selection,
    render_explainer_selection,
    render_evaluation_launch,
    render_results_visualisation,
)
import streamlit.components.v1 as components
from sklearn import set_config
import numpy as np

st.set_page_config(layout="wide", page_title="TRUE-X")

# ---------------------------------------------------------------
# Sidebar — mode selector
# ---------------------------------------------------------------
st.sidebar.title("TRUE-X")
st.sidebar.markdown("*Decision Support for XAI Trustworthiness*")
st.sidebar.divider()

MODE_BROWSE    = "Browse Benchmark"
MODE_EVALUATE  = "Evaluate My Data"
MODE_TRAIN     = "Train From Scratch"

mode = st.sidebar.radio(
    "Select mode",
    [MODE_BROWSE, MODE_EVALUATE, MODE_TRAIN],
)

st.sidebar.divider()

# ---------------------------------------------------------------
# ENTRY A — Browse pre-computed benchmark results
# ---------------------------------------------------------------
if mode == MODE_BROWSE:

    df_rank = pd.read_csv(RANKS_CSV)

    dataset = st.sidebar.selectbox("Select a dataset", list(df_rank["Dataset"].unique()))
    df_rank = df_rank[df_rank["Dataset"] == dataset]

    default_models      = list(df_rank["Model"].unique())
    default_xai_methods = list(df_rank["Explainer"].unique())
    default_perf_metrics = [m for m in df_rank["Metric"].unique() if m in ["Accuracy", "RMSE"]]

    metric_groups = {
        "Faithfulness": ["Pixel Flipping", "Faithfulness Correlation"],
        "Robustness":   ["Average Sensitivity", "Continuity"],
        "Complexity":   ["Complexity (Chan)", "Complexity (Elem)", "Sparseness (Chan)", "Sparseness (Elem)"],
    }

    st.header("Model Selection")
    selected_models = st.multiselect("Select ML models", default_models, default=default_models)
    if not selected_models:
        st.warning("Please select at least one model.")

    st.header("XAI Method Selection")
    selected_xai = st.multiselect("Select XAI methods", default_xai_methods, default=default_xai_methods)
    if not selected_xai:
        st.warning("Please select at least one XAI method.")

    st.header("Performance Metrics")
    selected_performance = st.multiselect("Select performance metrics", default_perf_metrics, default=default_perf_metrics)
    if not selected_performance:
        st.warning("Please select at least one performance metric.")

    st.header("Trustworthiness Metrics")
    selected_metrics = {}
    for group_name, metrics in metric_groups.items():
        st.subheader(group_name)
        selected_metrics[group_name] = st.multiselect(
            f"Select metrics for {group_name}", metrics, default=metrics
        )
        if not selected_metrics[group_name]:
            st.warning(f"Please select at least one {group_name} metric.")

    if selected_performance and selected_models and selected_xai and all(selected_metrics.values()):

        all_selected_metrics = [m for g in selected_metrics.values() for m in g] + selected_performance

        pipelines, manual_primitive_types = create_pipelines_from_csv(
            RANKS_CSV, "Metric", ["Model", "Explainer"], ["Value"],
            dataset=dataset,
            models=selected_models,
            explainers=selected_xai,
            selected_metrics=all_selected_metrics,
        )

        df_rank_filtered = df_rank[
            df_rank["Model"].isin(selected_models) &
            df_rank["Explainer"].isin(selected_xai) &
            df_rank["Metric"].isin(all_selected_metrics)
        ]

        # ---- AHP weighting ----
        st.header("AHP-Based Metric Importance (pairwise comparisons)")

        SAATY_OPTIONS = [
            ("Equal importance (1)", 1),
            ("Between equal & moderate (2)", 2),
            ("Moderate importance (3)", 3),
            ("Between moderate & strong (4)", 4),
            ("Strong importance (5)", 5),
            ("Between strong & very strong (6)", 6),
            ("Very strong importance (7)", 7),
            ("Between very strong & extreme (8)", 8),
            ("Extreme importance (9)", 9),
        ]

        def select_slider_with_labels(label, default_numeric=1):
            labels      = [o[0] for o in SAATY_OPTIONS]
            numeric_map = {o[0]: o[1] for o in SAATY_OPTIONS}
            default_label = next((l for l, v in SAATY_OPTIONS if v == default_numeric), labels[0])
            return numeric_map[st.select_slider(label, options=labels, value=default_label)]

        st.subheader("Performance vs Trustworthiness")
        perf_trust = select_slider_with_labels("Performance over Trustworthiness", default_numeric=1)

        col_sum = np.array([[1, perf_trust], [1/perf_trust, 1]]).sum(axis=0)
        normalized = np.array([[1, perf_trust], [1/perf_trust, 1]]) / col_sum
        weights_pt = normalized.mean(axis=1)
        weights_pt_pct = (weights_pt / weights_pt.sum()) * 100

        st.write("##### Computed Importance Weights (Performance vs Trustworthiness)")
        for crit, w in zip(["Performance", "Trustworthiness"], weights_pt_pct):
            st.write(f"{crit}: {w:.2f}%")

        coef = {"P": weights_pt_pct[0], "T": weights_pt_pct[1]}

        st.subheader("Trustworthiness Criteria Comparisons")
        col1, col2, col3 = st.columns(3)
        with col1:
            f_r = select_slider_with_labels("Faithfulness over Robustness", default_numeric=1)
        with col2:
            f_c = select_slider_with_labels("Faithfulness over Complexity", default_numeric=1)
        with col3:
            r_c = select_slider_with_labels("Robustness over Complexity", default_numeric=1)

        matrix = np.array([[1, f_r, f_c], [1/f_r, 1, r_c], [1/f_c, 1/r_c, 1]])
        col_sum = matrix.sum(axis=0)
        normalized = matrix / col_sum
        weights = normalized.mean(axis=1)
        weights_pct = (weights / weights.sum()) * 100

        coef["F"] = weights_pct[0]
        coef["R"] = weights_pct[1]
        coef["C"] = weights_pct[2]

        lambda_max = np.max(col_sum * weights)
        CI = (lambda_max - 3) / 2
        CR = CI / 0.58

        st.write(f"Consistency Ratio: {CR:.3f}")
        if CR < 0.1:
            st.success("Consistency is acceptable.")
            st.write("##### Computed Importance Weights for Trustworthiness Criteria")
            for crit, w in zip(["Faithfulness", "Robustness", "Complexity"], weights_pct):
                st.write(f"{crit}: {w:.2f}%")

            if st.button("Launch Analysis"):
                st.info("Running analysis with selected configuration...")
                tab1, tab2, tab3 = st.tabs([
                    "Combination Comparison",
                    "Trustworthiness Trade-offs",
                    "Performance and Trustworthiness",
                ])

                with tab1:
                    st.write("### Combination comparison")
                    if not _PIPELINE_PROFILER_OK:
                        st.error(
                            "PipelineProfiler is not installed. "
                            "Build the PipelineVis submodule first (see README)."
                        )
                    else:
                        html_text = get_pipeline_profiler_html(
                            list(pipelines.values()), coef,
                            manual_primitive_types=manual_primitive_types,
                        )
                        set_config(display="html")
                        components.html(html_text, width=1600, height=2000, scrolling=True)

                with tab2:
                    st.write("### Trustworthiness Trade-offs")
                    for fig in generate_tradeoff_figures_with_pareto(df_rank_filtered, coef):
                        st.plotly_chart(fig)

                with tab3:
                    st.write("### Performance and Trustworthiness")
                    @st.fragment
                    def show_topk_plots():
                        k = st.slider("Select top K experiments", min_value=1, max_value=10, value=3)
                        for fig in generate_topk_plots_auto(df_rank_filtered, coef, k):
                            st.plotly_chart(fig)
                    show_topk_plots()
        else:
            st.warning("Consistency is poor. Please review your comparisons.")

# ---------------------------------------------------------------
# ENTRY B — Upload dataset + pre-trained model → live evaluation
# ---------------------------------------------------------------
elif mode == MODE_EVALUATE:
    st.header("Evaluate My Data")
    st.markdown(
        "Upload your multivariate time-series dataset, pick a model and explainers, "
        "and TRUE-X will run the trustworthiness evaluation live."
    )
    st.divider()
    dataset_info = render_dataset_upload()

    if dataset_info is not None:
        st.divider()
        window_info = render_windowing_config(dataset_info)

        if window_info is not None:
            st.divider()
            model_info = render_model_selection(dataset_info, window_info)

            if model_info is not None:
                st.divider()
                explainer_info = render_explainer_selection()

                if explainer_info is not None:
                    st.divider()
                    live_results = render_evaluation_launch(
                        dataset_info, window_info, model_info, explainer_info
                    )

                    if live_results is not None:
                        st.divider()
                        render_results_visualisation(live_results)

# ---------------------------------------------------------------
# ENTRY C — Upload dataset → train from scratch → evaluate
# ---------------------------------------------------------------
elif mode == MODE_TRAIN:
    st.header("Train From Scratch")
    st.info(
        "Upload your multivariate time-series dataset, choose a model architecture, "
        "and TRUE-X will train the model, run the explainers, compute trustworthiness "
        "metrics, and visualise the results end-to-end."
    )
    st.warning("Entry C — coming soon. Under construction.")
