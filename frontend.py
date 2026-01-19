import streamlit as st
import pandas as pd
from PipelineProfiler import get_pipeline_profiler_html
from export_profiler import create_pipelines_from_csv
import streamlit.components.v1 as components
from sklearn import set_config

import numpy as np


ranks = "ranks_per_block_with_performance.csv"
df_rank = pd.read_csv(ranks)

# -------------------------------
# Sidebar: Dataset Upload
# -------------------------------
#st.sidebar.header("Upload Dataset")
uploaded_file = None #st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

dataset = st.sidebar.selectbox('Select a dataset', list(df_rank['Dataset'].unique()))
df_rank = df_rank[df_rank['Dataset'] == dataset]

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Dataset")
    st.dataframe(df.head())

# -------------------------------
# Prefilled lists
# -------------------------------
default_models = list(df_rank['Model'].unique())
default_xai_methods =list(df_rank['Explainer'].unique())
default_performance_metrics = [m for m in df_rank['Metric'].unique() if m in ["Accuracy", "RMSE"]]

metric_groups = {
    "Faithfulness": ["Pixel Flipping", "Faithfulness Correlation"],
    "Robustness": ["Average Sensitivity", "Continuity"],
    "Complexity": ["Complexity (Chan)", "Complexity (Elem)", "Sparseness (Chan)", "Sparseness (Elem)"]
}



# -------------------------------
# Model Selection
# -------------------------------
st.set_page_config(layout="wide")
st.header("Model Selection")
selected_models = st.multiselect("Select ML models", default_models, default=default_models)
if not selected_models:
    st.warning("Please select at least one model.")



# Option to upload pre-trained models
# uploaded_models = st.file_uploader("Upload pre-trained models (Pickle)", type=["pkl"], accept_multiple_files=True)
# if uploaded_models:
#     st.success(f"{len(uploaded_models)} pre-trained models uploaded.")
#     selected_models += [f"Uploaded Model {i + 1}" for i in range(len(uploaded_models))]

# -------------------------------
# XAI Method Selection
# -------------------------------
st.header("XAI Method Selection")
selected_xai = st.multiselect("Select XAI methods", default_xai_methods, default=default_xai_methods)
if not selected_xai:
    st.warning("Please select at least one XAI method.")



# -------------------------------
# Metric Groups with Importance
# -------------------------------

st.header("Performance Metrics")
selected_performance = st.multiselect("Select performance metrics", default_performance_metrics, default=default_performance_metrics)
if not selected_performance:
    st.warning("Please select at least one performance metric.")



st.header("Trustworthiness Metrics")
selected_metrics = {}

for group_name, metrics in metric_groups.items():
    st.subheader(group_name)
    selected_metrics[group_name] = st.multiselect(f"Select metrics for {group_name}", metrics, default=metrics)
    if not selected_metrics[group_name]:
        st.warning("Please select at least one %s metric." % group_name)
    



if selected_performance and selected_models and selected_xai and all(selected_metrics.values()):

    pipelines, manual_primitive_types = create_pipelines_from_csv(ranks, 'Metric', ['Model', 'Explainer'], ['Value'],
                                                        dataset=dataset,
                                                        models=selected_models,
                                                        explainers=selected_xai,
                                                        selected_metrics=[m for group_metrics in selected_metrics.values() for m in group_metrics] + selected_performance)


    html_text = get_pipeline_profiler_html(list(pipelines.values()), manual_primitive_types=manual_primitive_types)


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
        """Return the numeric value picked from labeled Saaty scale options."""
        labels = [opt[0] for opt in SAATY_OPTIONS]
        numeric_map = {opt[0]: opt[1] for opt in SAATY_OPTIONS}
        # Find the default label corresponding to the numeric default
        default_label = next((l for l, v in SAATY_OPTIONS if v == default_numeric), labels[0])
        chosen_label = st.select_slider(label, options=labels, value=default_label)
        return numeric_map[chosen_label]

    st.subheader("Performance vs Trustworthiness comparison")
    perf_trust = select_slider_with_labels("Performance over Trustworthiness", default_numeric=1)

    matrix = np.array([
        [1, perf_trust],
        [1/perf_trust, 1]
    ])

    # Normalize and compute weights
    col_sum = matrix.sum(axis=0)
    normalized = matrix / col_sum
    weights_perf_trust = normalized.mean(axis=1)
    weights_percent = (weights_perf_trust / weights_perf_trust.sum()) * 100

    criteria = ["Performance", "Thrustworthiness"]

    st.write("##### Computed Importance Weights (Performance vs Trustworthiness)")
    for crit, w in zip(criteria, weights_percent):
        st.write(f"{crit}: {w:.2f}%")


    st.subheader("Trustworthiness Criteria Comparisons")

    col1, col2, col3 = st.columns(3)
    with col1:
        f_r = select_slider_with_labels("Faithfulness over Robustness", default_numeric=1)
    with col2:
        f_c = select_slider_with_labels("Faithfulness over Complexity", default_numeric=1)
    with col3:
        r_c = select_slider_with_labels("Robustness over Complexity", default_numeric=1)


    # Build AHP matrix
    matrix = np.array([
        [1, f_r, f_c],
        [1/f_r, 1, r_c],
        [1/f_c, 1/r_c, 1]
    ])

    # Normalize and compute weights
    col_sum = matrix.sum(axis=0)
    normalized = matrix / col_sum
    weights = normalized.mean(axis=1)
    weights_percent = (weights / weights.sum()) * 100

    criteria = ["Faithfulness", "Robustness", "Complexity"]

    # Consistency check (optional)
    lambda_max = np.max(col_sum * weights)
    CI = (lambda_max - len(criteria)) / (len(criteria) - 1)
    RI = 0.58  # Random Index for n=3
    CR = CI / RI
    st.write(f"Consistency Ratio: {CR:.3f}")
    if CR < 0.1:
        st.success("Consistency is acceptable.")
        # Display results
        st.write("##### Computed Importance Weights for Trustworthiness Criteria")
        for crit, w in zip(criteria, weights_percent):
            st.write(f"{crit}: {w:.2f}%")
        # -------------------------------
        # Launch Button
        # -------------------------------

        if st.button("Launch Analysis"):
            st.info("Running analysis with selected configuration...")
            tab1, tab2, tab3 = st.tabs(["Combination Comparison", "Alternative View 1", "Alternative View 2"])
            
            with tab1:
                st.write("### Combination comparison")
                set_config(display="html")
                components.html(html_text, width=1600, height=2000, scrolling=True)
            
            with tab2:
                st.write("### Alternative Visualization 1")
                # Add your alternative visualization here
                st.info("Alternative view to be implemented")
            
            with tab3:
                st.write("### Alternative Visualization 2")
                # Add your alternative visualization here
                st.info("Alternative view to be implemented")
    else:
        st.warning("Consistency is poor. Please review your comparisons.")



