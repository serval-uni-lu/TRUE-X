import streamlit as st
import pandas as pd
from PipelineProfiler import get_pipeline_profiler_html
from export_profiler import create_pipelines_from_csv
import streamlit.components.v1 as components
from sklearn import set_config

import numpy as np

ranks = "ranks_per_block.csv"
df_rank = pd.read_csv(ranks)

# -------------------------------
# Sidebar: Dataset Upload
# -------------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

dataset = st.sidebar.selectbox('Or Select a dataset', list(df_rank['Dataset'].unique()))

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Dataset")
    st.dataframe(df.head())


features = ['Model', 'Explainer']
problem = 'Metric'
metrics = ['median_dir']



# -------------------------------
# Prefilled lists
# -------------------------------
default_models = list(df_rank['Model'].unique())
default_xai_methods =list(df_rank['Explainer'].unique())

metric_groups = {
    "Faithfulness": ["Pixel Flipping", "Faithfulness Correlation"],
    "Robustness": ["Average Sensitivity", "Continuity"],
    "Complexity": ["Complexity", "Sparseness"]
}



# -------------------------------
# Model Selection
# -------------------------------
st.set_page_config(layout="wide")
st.header("Model Selection")
selected_models = st.multiselect("Select ML models", default_models, default=default_models)

# Option to upload pre-trained models
uploaded_models = st.file_uploader("Upload pre-trained models (Pickle)", type=["pkl"], accept_multiple_files=True)
if uploaded_models:
    st.success(f"{len(uploaded_models)} pre-trained models uploaded.")
    selected_models += [f"Uploaded Model {i + 1}" for i in range(len(uploaded_models))]

# -------------------------------
# XAI Method Selection
# -------------------------------
st.header("XAI Method Selection")
selected_xai = st.multiselect("Select XAI methods", default_xai_methods, default=default_xai_methods)


pipelines, manual_primitive_types = create_pipelines_from_csv(ranks, problem, features, metrics,
                                                              dataset="CWRU_12k",
                                                              models=selected_models,
                                                              explainers=selected_xai)

html_text = get_pipeline_profiler_html(list(pipelines.values()), manual_primitive_types=manual_primitive_types)
# -------------------------------
# Metric Groups with Importance
# -------------------------------
st.header("Metrics")
importance_values = {}
total_importance = 0

selected_metrics = {}

for group_name, metrics in metric_groups.items():
    st.subheader(group_name)
    selected_metrics[group_name] = st.multiselect(f"Select metrics for {group_name}", metrics, default=metrics)
    #importance = st.slider(f"Importance for {group_name} (%)", min_value=0, max_value=100, value=33)
    #importance_values[group_name] = importance
    #total_importance += importance

st.subheader("AHP-Based Metric Importance (pairwise comparisons)")

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
    st.write("### Computed Importance Weights")
    for crit, w in zip(criteria, weights_percent):
        st.write(f"{crit}: {w:.2f}%")
    if st.button("Launch Analysis"):
        st.info("Running analysis with selected configuration...")
        st.write("### Combination comparison")
        set_config(display="html")
        components.html(html_text, width=1600, height=2000, scrolling=True)
else:
    st.warning("Consistency is poor. Please review your comparisons.")

# -------------------------------
# Launch Button
# -------------------------------

