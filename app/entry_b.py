import streamlit as st
import pandas as pd
from pathlib import Path

MIN_SAMPLES = 20   # minimum windows required to proceed


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

_DROP_HINTS = {"time", "date", "index", "epoch", "timestamp", "id", "datetime"}
_TARGET_HINTS = {"target", "label", "y", "class", "output", "fault"}


def _guess_drop_cols(df: pd.DataFrame) -> list[str]:
    """Return columns that look like time/index columns."""
    auto_drop = []
    for col in df.columns:
        if any(hint in col.lower() for hint in _DROP_HINTS):
            auto_drop.append(col)
            continue
        if df[col].dtype == object:
            try:
                pd.to_datetime(df[col].iloc[:5], infer_datetime_format=True)
                auto_drop.append(col)
            except Exception:
                pass
    return auto_drop


def _guess_target(df: pd.DataFrame, exclude: list[str]) -> str | None:
    """Return the most likely target column."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    if not numeric:
        return None
    for col in numeric:
        if col.lower() in _TARGET_HINTS:
            return col
    return numeric[-1]


# ---------------------------------------------------------------
# Public: dataset upload + column role UI
# ---------------------------------------------------------------

def render_dataset_upload() -> dict | None:
    """
    Renders Step 1 (and Step 2 for custom data) of Entry B.

    Returns a dict with keys:
        mode          : "benchmark" | "custom"
        dataset_name  : str — benchmark name (benchmark) or uploaded filename (custom)
        df            : DataFrame | None — None for benchmark (data lives on disk)
        feature_cols  : list[str]
        target_col    : str
    or None if the step is not yet complete.
    """

    st.subheader("Step 1 — Dataset")

    data_source = st.radio(
        "Data source",
        options=["Use a benchmark dataset", "Upload your own dataset"],
        horizontal=True,
        key="entry_b_data_source",
    )

    st.divider()

    # ================================================================
    # B1 — benchmark dataset
    # ================================================================
    if data_source == "Use a benchmark dataset":

        st.caption(
            "Select one of the four datasets used in the paper. "
            "TRUE-X will use the pre-defined windowing and column configuration — "
            "no upload needed."
        )

        dataset_name = st.selectbox(
            "Benchmark dataset",
            options=list(_BENCHMARK_SPECS.keys()),
            key="entry_b_benchmark_name",
        )

        spec = _BENCHMARK_SPECS[dataset_name]

        # Info card
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Task",     spec["task"])
        col_b.metric("Channels", spec["n_channels"])
        col_c.metric("Window",   spec["window"])

        st.info(
            f"**{dataset_name}** — {spec['description']}\n\n"
            f"**Features:** {', '.join(spec['feature_cols'])}\n\n"
            f"**Target:** {spec['target_col']}"
            + (f"\n\n**Classes:** {spec['n_classes']}" if spec["n_classes"] else "")
        )

        st.success(
            f"✅ Configuration locked — windowing and column roles are "
            f"pre-defined for {dataset_name}."
        )

        return {
            "mode":         "benchmark",
            "dataset_name": dataset_name,
            "df":           None,
            "feature_cols": spec["feature_cols"],
            "target_col":   spec["target_col"],
        }

    # ================================================================
    # B2 — custom dataset
    # ================================================================
    else:

        st.caption(
            "Upload a CSV file where each **row is one timestep** "
            "and each **column is a sensor / feature channel** plus a target column."
        )

        uploaded = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            key="entry_b_upload",
        )

        if uploaded is None:
            st.info("Upload a CSV file to continue.")
            return None

        # ---- Parse ----
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            return None

        if df.empty:
            st.error("The uploaded file is empty.")
            return None

        if len(df.columns) < 2:
            st.error("The file must have at least two columns (one feature + one target).")
            return None

        # ---- Preview ----
        st.write(f"**Detected:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        with st.expander("Preview — first 5 rows", expanded=True):
            st.dataframe(df.head(), use_container_width=True)

        # ---- Step 2: column roles ----
        st.subheader("Step 2 — Confirm column roles")
        st.caption("TRUE-X will auto-detect roles. Correct anything that looks wrong.")

        auto_drop      = _guess_drop_cols(df)
        guessed_target = _guess_target(df, exclude=auto_drop)
        all_cols       = df.columns.tolist()

        col_left, col_right = st.columns(2)

        with col_left:
            target_index = all_cols.index(guessed_target) if guessed_target in all_cols else 0
            target_col = st.selectbox(
                "🎯 Target column",
                options=all_cols,
                index=target_index,
                help="The column the model will predict.",
                key="entry_b_target",
            )

        with col_right:
            drop_options = [c for c in all_cols if c != target_col]
            drop_default = [c for c in auto_drop if c != target_col]
            cols_to_drop = st.multiselect(
                "🗑 Columns to drop",
                options=drop_options,
                default=drop_default,
                help="Time index, IDs, or any column that should not be used as a feature.",
                key="entry_b_drop",
            )

        # ---- Derive feature list ----
        feature_cols = [
            c for c in all_cols
            if c != target_col and c not in cols_to_drop
        ]

        non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            st.warning(
                f"These feature columns are non-numeric and will be ignored: "
                f"{', '.join(non_numeric)}"
            )
            feature_cols = [c for c in feature_cols if c not in non_numeric]

        if not feature_cols:
            st.error("No feature columns remaining. Adjust the target or drop selections above.")
            return None

        st.success(f"✅ **{len(feature_cols)} feature column(s):** {', '.join(feature_cols)}")

        return {
            "mode":         "custom",
            "dataset_name": uploaded.name,
            "df":           df,
            "feature_cols": feature_cols,
            "target_col":   target_col,
        }


# ---------------------------------------------------------------
# Internal helpers for Step 3
# ---------------------------------------------------------------

def _guess_task(df: pd.DataFrame, target_col: str) -> str:
    """Return 'Classification' if target looks like discrete labels, else 'Regression'."""
    series = df[target_col].dropna()
    is_int_like = pd.api.types.is_integer_dtype(series) or (
        pd.api.types.is_float_dtype(series) and (series == series.astype(int)).all()
    )
    if (is_int_like or series.dtype == object) and series.nunique() <= 20:
        return "Classification"
    return "Regression"


def _build_label_encoding(series: pd.Series) -> dict:
    """Return {original_value: class_index} mapping, sorted for consistency."""
    unique_vals = sorted(series.dropna().unique(), key=lambda x: str(x))
    return {val: idx for idx, val in enumerate(unique_vals)}


def _count_windows(n_timesteps: int, window: int, shift: int) -> int:
    if n_timesteps < window:
        return 0
    return (n_timesteps - window) // shift + 1


# ---------------------------------------------------------------
# Public: windowing + task configuration
# ---------------------------------------------------------------

def render_windowing_config(dataset_info: dict) -> dict | None:
    """
    Renders Step 3: windowing parameters, task type, and train/test split.

    For B1 (mode="benchmark"): shows a locked, read-only configuration derived from
    _BENCHMARK_SPECS — no df is required.
    For B2 (mode="custom"): shows editable widgets, validates window counts.

    Returns a dict with keys:
        window      : int   — window length L
        shift       : int   — hop size H
        task        : str   — 'Classification' or 'Regression'
        test_ratio  : float
        n_samples   : int | None  — None for benchmark (computed at runtime)
        n_train     : int | None
        n_test      : int | None
        label_encoding : dict | None
        n_classes   : int | None
    or None if the configuration is invalid.
    """
    mode         = dataset_info["mode"]
    target_col   = dataset_info["target_col"]
    feature_cols = dataset_info["feature_cols"]

    st.subheader("Step 3 — Windowing & task configuration")

    # ================================================================
    # B1 — benchmark: configuration is fixed, show read-only display
    # ================================================================
    if mode == "benchmark":
        dataset_name = dataset_info["dataset_name"]
        spec = _BENCHMARK_SPECS[dataset_name]

        task   = spec["task"]
        window = spec["window"]
        shift  = spec["shift"]

        st.caption(
            f"Configuration is pre-defined for **{dataset_name}** — "
            "values are locked to match the benchmark training setup."
        )

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Task",       task)
        col_b.metric("Window L",   window)
        col_c.metric("Hop size H", shift)
        col_d.metric("Channels",   spec["n_channels"])

        if task == "Classification":
            n_classes = spec["n_classes"]
            label_encoding = None   # actual encoding built at runtime from dataset files
            st.info(
                f"**Target:** `{spec['target_col']}` — {n_classes} classes "
                "(encoding is applied at runtime from the dataset files)."
            )
        else:
            n_classes = None
            label_encoding = None
            st.info(
                f"**Target:** `{spec['target_col']}` — continuous regression target."
            )

        st.caption(
            f"Each sample shape: ({window} timesteps × {len(feature_cols)} channels)"
        )
        st.success("✅ Windowing configuration locked.")

        return {
            "window":         window,
            "shift":          shift,
            "task":           task,
            "test_ratio":     0.2,  # benchmark uses fixed 80/20 split
            "n_samples":      None,
            "n_train":        None,
            "n_test":         None,
            "label_encoding": label_encoding,
            "n_classes":      n_classes,
        }

    # ================================================================
    # B2 — custom: configurable windowing
    # ================================================================
    df          = dataset_info["df"]
    n_timesteps = len(df)

    st.caption(
        "TRUE-X slices your time series into fixed-length windows. "
        "Each window becomes one sample fed to the model."
    )

    # ---- Task type ----
    guessed_task = _guess_task(df, target_col)

    task = st.radio(
        "Task type",
        options=["Classification", "Regression"],
        index=0 if guessed_task == "Classification" else 1,
        horizontal=True,
        help=(
            "Classification: target is a discrete class label. "
            "Regression: target is a continuous value."
        ),
        key="entry_b_task",
    )

    # ---- Target preview ----
    target_series = df[target_col].dropna()

    if task == "Classification":
        label_encoding = _build_label_encoding(target_series)
        n_classes = len(label_encoding)

        st.write(f"**Target '{target_col}'** — {n_classes} class(es) detected, "
                 f"will be encoded as:")
        enc_df = pd.DataFrame(
            list(label_encoding.items()),
            columns=["Original value", "Encoded as (class index)"]
        )
        st.dataframe(enc_df, use_container_width=False, hide_index=True)

        if n_classes < 2:
            st.error("Classification requires at least 2 classes in the target column.")
            return None
    else:
        label_encoding = None
        n_classes = None
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Min",  f"{target_series.min():.4f}")
        col_b.metric("Mean", f"{target_series.mean():.4f}")
        col_c.metric("Max",  f"{target_series.max():.4f}")

    st.divider()

    # ---- Window length and hop size ----
    col_l, col_h = st.columns(2)

    max_window = max(2, n_timesteps // 2)
    default_window = min(30, max_window)

    with col_l:
        window = st.number_input(
            "Window length L (timesteps per sample)",
            min_value=2,
            max_value=max_window,
            value=default_window,
            step=1,
            help=f"How many consecutive timesteps form one input sample. Max = {max_window} (half your dataset length).",
            key="entry_b_window",
        )

    with col_h:
        shift = st.number_input(
            "Hop size H (step between windows)",
            min_value=1,
            max_value=int(window),
            value=1,
            step=1,
            help=(
                "1 = fully overlapping windows (most samples). "
                f"{int(window)} = non-overlapping windows (fewest samples)."
            ),
            key="entry_b_shift",
        )

    # ---- Train / test split ----
    test_ratio = st.slider(
        "Test set size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        format="%.0f%%",
        help="Fraction of windows held out for evaluation.",
        key="entry_b_split",
    )

    # ---- Live summary ----
    n_samples = _count_windows(n_timesteps, int(window), int(shift))
    n_test    = max(1, int(n_samples * test_ratio))
    n_train   = n_samples - n_test

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total windows",    f"{n_samples:,}")
    c2.metric("Training windows", f"{n_train:,}")
    c3.metric("Test windows",     f"{n_test:,}")

    st.caption(
        f"Each sample shape: ({int(window)} timesteps × {len(feature_cols)} channels)"
    )

    # ---- Validation ----
    if n_samples < MIN_SAMPLES:
        st.error(
            f"Only {n_samples} window(s) produced — need at least {MIN_SAMPLES}. "
            "Reduce the window length or hop size."
        )
        return None

    if n_train < 10:
        st.error(
            f"Training set too small ({n_train} windows). "
            "Reduce the test ratio or adjust the window settings."
        )
        return None

    st.success("✅ Windowing configuration is valid.")

    return {
        "window":         int(window),
        "shift":          int(shift),
        "task":           task,
        "test_ratio":     test_ratio,
        "n_samples":      n_samples,
        "n_train":        n_train,
        "n_test":         n_test,
        "label_encoding": label_encoding,
        "n_classes":      n_classes,
    }


# ---------------------------------------------------------------
# Constants — benchmark dataset specs (shared by Steps 1, 3, 4)
# ---------------------------------------------------------------

# All values derived from xurl/configs/datasets_config.yaml and the dataset adapters.
# feature_cols = numeric sensor columns after dropping time/id columns and the target.
# shift for Hydraulic is equal to window because each cycle is already one independent sample.
_BENCHMARK_SPECS = {
    "CWRU_12k": {
        "task":         "Classification",
        "n_channels":   2,
        "window":       2048,
        "shift":        512,
        "n_classes":    4,
        "feature_cols": ["DE", "FE"],
        "target_col":   "fault_type",
        "description":  "Bearing fault detection — 4 classes: Normal, Ball, Inner Race, Outer Race",
    },
    "Ecoating": {
        "task":         "Regression",
        "n_channels":   5,
        "window":       30,
        "shift":        1,
        "n_classes":    None,
        "feature_cols": ["PE1", "PE2", "PE3", "PE4", "TP1"],
        "target_col":   "FM1",
        "description":  "E-coating process quality — predict coating thickness (FM1)",
    },
    "FD001": {
        "task":         "Regression",
        "n_channels":   14,
        "window":       30,
        "shift":        1,
        "n_classes":    None,
        "feature_cols": ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"],
        "target_col":   "RUL",
        "description":  "Turbofan engine remaining useful life — CMAPSS FD001",
    },
    "Hydraulic": {
        "task":         "Classification",
        "n_channels":   9,
        "window":       50,
        "shift":        50,
        "n_classes":    4,
        "feature_cols": ["CP","FS1","PS1","PS2","PS3","PS4","PS5","SE","VS1"],
        "target_col":   "pump_condition",
        "description":  "Hydraulic system condition — 4 pump states",
    },
}

# Model architectures available per task (names match Model enum in XURL)
_ARCHITECTURES_CLASSIFICATION = [
    "ENCODER", "RESNET", "FCN", "MCD_CNN", "TIME_CNN",
    "INCEPTIONTIME", "TST", "LSTM", "GRU",
    "LOGISTIC_CLASSIFIER", "RF_CLASSIFIER", "ET_CLASSIFIER",
    "XGB_CLASSIFIER", "LGBM_CLASSIFIER",
]
_ARCHITECTURES_REGRESSION = [
    "LSTM_REGRESSOR", "BI_LSTM_REGRESSOR", "ATTENTION_LSTM_REGRESSOR",
    "CNN_LSTM_REGRESSOR", "LSTM_FCN_REGRESSOR", "TCN_REGRESSOR",
    "TST_REGRESSOR", "TFT_REGRESSOR",
    "LINEAR_REGRESSOR", "RF_REGRESSOR", "ET_REGRESSOR",
    "XGB_REGRESSOR", "LGBM_REGRESSOR",
]


# ---------------------------------------------------------------
# Internal helpers for Step 4
# ---------------------------------------------------------------

def _scan_pretrained_models(saved_models_dir: str, task: str) -> list[dict]:
    """
    Scan saved_models_dir for .json files and return metadata for models
    whose benchmark task matches the requested task.
    """
    import json
    from pathlib import Path

    results = []
    path = Path(saved_models_dir)
    if not path.exists():
        return results

    for json_file in sorted(path.glob("*.json")):
        try:
            meta = json.loads(json_file.read_text())
        except Exception:
            continue

        dataset    = meta.get("dataset", "")
        model_name = meta.get("model", "")
        spec       = _BENCHMARK_SPECS.get(dataset, {})

        if spec.get("task", "") != task:
            continue

        # only PyTorch .pt weights are supported (all explainers require it)
        pt_path = json_file.with_suffix(".pt")
        if not pt_path.exists():
            continue

        results.append({
            "key":        json_file.stem,          # e.g. "CWRU_12k_LSTM"
            "dataset":    dataset,
            "model":      model_name,
            "n_channels": spec.get("n_channels"),
            "window":     spec.get("window"),
            "n_classes":  spec.get("n_classes"),
        })

    return results


def _shape_compatible(model_meta: dict, window_info: dict, dataset_info: dict) -> bool:
    return (
        model_meta["n_channels"] == len(dataset_info["feature_cols"])
        and model_meta["window"] == window_info["window"]
    )


# ---------------------------------------------------------------
# Public: model selection
# ---------------------------------------------------------------

_APP_DIR = Path(__file__).resolve().parent   # TRUE-X/app/
_ROOT    = _APP_DIR.parent                   # TRUE-X/ (repo root)
SAVED_MODELS_DIR = str(_ROOT / "saved_models")


def render_model_selection(dataset_info: dict, window_info: dict) -> dict | None:
    """
    Renders Step 4: choose one or more models to evaluate.

    B1 (benchmark mode): multiselect from pre-trained models for the selected dataset.
    B2 (custom mode):    upload one or more model files, each with its own architecture.

    Returns a dict with key:
        models : list[dict]  — each entry has:
            source        : "pretrained" | "upload"
            model_key     : str
            dataset       : str
            architecture  : str
            weight_path   : str | None  (pretrained only)
            uploaded_bytes: bytes | None (upload only)
            file_ext      : str | None   (upload only)
    or None if the step is incomplete.
    """
    from pathlib import Path

    mode       = dataset_info["mode"]
    task       = window_info["task"]
    n_channels = len(dataset_info["feature_cols"])
    window     = window_info["window"]
    n_classes  = window_info["n_classes"]

    st.subheader("Step 4 — Model selection")

    # ================================================================
    # B1 — benchmark: multiselect from pre-trained models for THIS dataset
    # ================================================================
    if mode == "benchmark":
        dataset_name = dataset_info["dataset_name"]

        st.caption(
            f"Select one or more pre-trained **{dataset_name}** models to evaluate. "
            "All listed models were trained on this dataset and are shape-compatible."
        )

        available = _scan_pretrained_models(SAVED_MODELS_DIR, task)
        dataset_models = [m for m in available if m["dataset"] == dataset_name]

        if not dataset_models:
            st.error(
                f"No pre-trained models found for **{dataset_name}** in `{SAVED_MODELS_DIR}`. "
                "Make sure the `.json` metadata files and the matching `.pt` or `.pkl` weight "
                "files are both present."
            )
            return None

        rows = [{
            "Model key":     m["key"],
            "Architecture":  m["model"],
            "Trained shape": f"{m['window']} steps × {m['n_channels']} channels",
        } for m in dataset_models]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        all_keys = [m["key"] for m in dataset_models]
        selected_keys = st.multiselect(
            "Select model(s) to evaluate",
            options=all_keys,
            default=all_keys[:1],
            key="entry_b_benchmark_models",
        )

        if not selected_keys:
            st.warning("Select at least one model to continue.")
            return None

        selected_models = []
        for key in selected_keys:
            meta = next(m for m in dataset_models if m["key"] == key)
            base = Path(SAVED_MODELS_DIR) / key
            pt = base.with_suffix(".pt")
            weight_path = str(pt) if pt.exists() else None

            selected_models.append({
                "source":         "pretrained",
                "model_key":      key,
                "dataset":        dataset_name,
                "architecture":   meta["model"],
                "weight_path":    weight_path,
                "uploaded_bytes": None,
                "file_ext":       None,
            })

        st.success(f"✅ {len(selected_models)} model(s) selected.")
        return {"models": selected_models}

    # ================================================================
    # B2 — custom: upload one or more model files
    # ================================================================
    arch_options = (
        _ARCHITECTURES_CLASSIFICATION
        if task == "Classification"
        else _ARCHITECTURES_REGRESSION
    )

    st.caption(
        "Upload one or more pre-trained model files. "
        "For each file, specify the architecture so TRUE-X can reconstruct the network."
    )

    st.caption(
        "Only PyTorch `.pt` weight files are accepted — "
        "all XAI explainers require a differentiable PyTorch model."
    )
    uploaded_files = st.file_uploader(
        "Upload model weight files (.pt)",
        type=["pt"],
        accept_multiple_files=True,
        key="entry_b_model_upload",
    )

    if not uploaded_files:
        st.info("Upload at least one model file to continue.")
        return None

    selected_models = []
    for i, uf in enumerate(uploaded_files):
        file_ext = uf.name.rsplit(".", 1)[-1].lower()
        col_name, col_arch = st.columns([2, 2])

        with col_name:
            st.write(f"**{uf.name}**")

        with col_arch:
            architecture = st.selectbox(
                f"Architecture",
                options=arch_options,
                key=f"entry_b_arch_{i}",
                help="Must match the architecture the model was trained with.",
                label_visibility="collapsed",
            )

        selected_models.append({
            "source":         "upload",
            "model_key":      uf.name,
            "dataset":        "custom",
            "architecture":   architecture,
            "weight_path":    None,
            "uploaded_bytes": uf.read(),
            "file_ext":       file_ext,
        })

    st.caption(
        f"Expected input shape: {window} timesteps × {n_channels} channels"
        + (f" → {n_classes} output classes" if n_classes else " → scalar output")
    )
    st.success(f"✅ {len(selected_models)} model(s) configured.")
    return {"models": selected_models}


# ---------------------------------------------------------------
# XAI method groups (used in Step 5)
# ---------------------------------------------------------------

_XAI_GROUPS = {
    "Gradient-based": [
        "saliency",
        "guided_back_prop",
        "gradient_x_input",
        "smooth_gradient",
        "integrated_gradients",
        "deeplift",
    ],
    "Gradient + reference": [
        "expected_gradients",
        "deepliftshap",
        "gradientshap",
    ],
    "Perturbation-based": [
        "occlusion",
        "feature_ablation",
        "shapley_sampling",
    ],
    "Surrogate models": [
        "lime_tabular",
        "shap",
    ],
}

_ALL_XAI_METHODS = [m for methods in _XAI_GROUPS.values() for m in methods]

_DEFAULT_XAI_METHODS = [
    "saliency",
    "integrated_gradients",
    "expected_gradients",
    "occlusion",
    "lime_tabular",
]


# ---------------------------------------------------------------
# Public: XAI method selection
# ---------------------------------------------------------------

def render_explainer_selection() -> dict | None:
    """
    Renders Step 5: select which XAI explainers to run.

    Returns a dict with key:
        methods : list[str]  — registry keys from ExplainerFactory
    or None if no methods are selected.
    """
    st.subheader("Step 5 — XAI methods")
    st.caption(
        "Select the explainers to run. "
        "Background data and method parameters are handled automatically."
    )

    selected = []
    for group_name, methods in _XAI_GROUPS.items():
        chosen = st.multiselect(
            group_name,
            options=methods,
            default=[m for m in _DEFAULT_XAI_METHODS if m in methods],
            key=f"entry_b_xai_{group_name}",
        )
        selected.extend(chosen)

    if not selected:
        st.warning("Select at least one XAI method to continue.")
        return None

    st.success(f"✅ {len(selected)} method(s) selected: {', '.join(selected)}")
    return {"methods": selected}


# ---------------------------------------------------------------
# Public: evaluation launch
# ---------------------------------------------------------------

def render_evaluation_launch(
    dataset_info: dict,
    window_info: dict,
    model_info: dict,
    explainer_info: dict,
) -> "pd.DataFrame | None":
    """
    Renders Step 6: configuration summary + launch button.

    Runs the evaluation when the button is clicked and stores results in
    st.session_state["live_results"]. Returns the results DataFrame if
    already computed, otherwise None.
    """
    st.subheader("Step 6 — Run evaluation")

    n_models  = len(model_info["models"])
    n_methods = len(explainer_info["methods"])

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Dataset",    dataset_info["dataset_name"])
    col_b.metric("Models",     n_models)
    col_c.metric("Explainers", n_methods)

    st.caption(
        f"TRUE-X will run **{n_methods} XAI method(s)** on "
        f"**{n_models} model(s)** and compute all trustworthiness metrics. "
        "This may take a few minutes."
    )

    if st.button("Launch Evaluation", type="primary", key="entry_b_launch"):
        from evaluation import run_evaluation
        with st.spinner("Running evaluation — please wait..."):
            try:
                results = run_evaluation(dataset_info, window_info, model_info, explainer_info)
                if results.empty:
                    st.error(
                        "Evaluation produced no results. "
                        "Check that the weight files and dataset files are present."
                    )
                else:
                    st.session_state["live_results"] = results
                    st.success(
                        f"✅ Evaluation complete — "
                        f"{results['Model'].nunique()} model(s) × "
                        f"{results['Explainer'].nunique()} explainer(s) × "
                        f"{results['Metric'].nunique()} metrics."
                    )
            except NotImplementedError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.exception(e)

    return st.session_state.get("live_results")


# ---------------------------------------------------------------
# Public: results visualisation (Step 7)
# ---------------------------------------------------------------

def render_results_visualisation(live_results: "pd.DataFrame") -> None:
    """
    Renders Step 7: AHP weighting + three-tab Panel B visualisation,
    mirroring Entry A but driven by live evaluation results.
    """
    import numpy as np
    from PipelineProfiler import get_pipeline_profiler_html
    from export_profiler import create_pipelines_from_csv
    from plots import generate_tradeoff_figures_with_pareto, generate_topk_plots_auto
    import streamlit.components.v1 as components
    from sklearn import set_config
    import tempfile, os

    st.subheader("Step 7 — Results")

    # ---- Step 8: export ----
    st.download_button(
        label="⬇ Download results as CSV",
        data=live_results.to_csv(index=False).encode("utf-8"),
        file_name="truex_live_results.csv",
        mime="text/csv",
        key="b8_download",
    )

    st.divider()

    available_metrics   = sorted(live_results["Metric"].unique().tolist())
    available_models    = sorted(live_results["Model"].unique().tolist())
    available_explainers = sorted(live_results["Explainer"].unique().tolist())

    perf_metrics  = [m for m in available_metrics if m in ("Accuracy", "RMSE")]
    trust_metrics = [m for m in available_metrics if m not in ("Accuracy", "RMSE")]

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

    def _slider(label, default=1, key_suffix=""):
        labels      = [o[0] for o in SAATY_OPTIONS]
        numeric_map = {o[0]: o[1] for o in SAATY_OPTIONS}
        default_label = next((l for l, v in SAATY_OPTIONS if v == default), labels[0])
        return numeric_map[st.select_slider(label, options=labels, value=default_label, key=f"b7_{key_suffix}")]

    st.subheader("Performance vs Trustworthiness")
    perf_trust = _slider("Performance over Trustworthiness", key_suffix="pt")
    mat_pt = np.array([[1, perf_trust], [1/perf_trust, 1]])
    w_pt = (mat_pt / mat_pt.sum(axis=0)).mean(axis=1)
    w_pt_pct = w_pt / w_pt.sum() * 100
    coef = {"P": w_pt_pct[0], "T": w_pt_pct[1]}
    st.write("##### Computed Importance Weights (Performance vs Trustworthiness)")
    for crit, w in zip(["Performance", "Trustworthiness"], w_pt_pct):
        st.write(f"{crit}: {w:.2f}%")

    st.subheader("Trustworthiness Criteria Comparisons")
    col1, col2, col3 = st.columns(3)
    with col1:
        f_r = _slider("Faithfulness over Robustness",  key_suffix="fr")
    with col2:
        f_c = _slider("Faithfulness over Complexity",  key_suffix="fc")
    with col3:
        r_c = _slider("Robustness over Complexity",    key_suffix="rc")

    matrix = np.array([[1, f_r, f_c], [1/f_r, 1, r_c], [1/f_c, 1/r_c, 1]])
    col_sum = matrix.sum(axis=0)
    weights = (matrix / col_sum).mean(axis=1)
    weights_pct = weights / weights.sum() * 100
    coef["F"] = weights_pct[0]
    coef["R"] = weights_pct[1]
    coef["C"] = weights_pct[2]

    lambda_max = np.max(col_sum * weights)
    CR = ((lambda_max - 3) / 2) / 0.58
    st.write(f"Consistency Ratio: {CR:.3f}")

    if CR >= 0.1:
        st.warning("Consistency is poor. Please review your comparisons.")
        return

    st.success("Consistency is acceptable.")
    st.write("##### Computed Importance Weights for Trustworthiness Criteria")
    for crit, w in zip(["Faithfulness", "Robustness", "Complexity"], weights_pct):
        st.write(f"{crit}: {w:.2f}%")

    if not st.button("Launch Analysis", key="b7_launch"):
        return

    st.info("Running analysis...")

    # Write live_results to a temp CSV so create_pipelines_from_csv can read it
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as tmp:
        live_results.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        all_selected_metrics = trust_metrics + perf_metrics

        dataset_name = str(live_results["Dataset"].iloc[0]) if "Dataset" in live_results.columns else ""
        pipelines, manual_primitive_types = create_pipelines_from_csv(
            tmp_path, "Metric", ["Model", "Explainer"], ["Value"],
            dataset=dataset_name,
            models=available_models,
            explainers=available_explainers,
            selected_metrics=all_selected_metrics,
        )
    finally:
        os.unlink(tmp_path)

    tab1, tab2, tab3 = st.tabs([
        "Combination Comparison",
        "Trustworthiness Trade-offs",
        "Performance and Trustworthiness",
    ])

    with tab1:
        st.write("### Combination comparison")
        html_text = get_pipeline_profiler_html(
            list(pipelines.values()), coef,
            manual_primitive_types=manual_primitive_types,
        )
        set_config(display="html")
        components.html(html_text, width=1600, height=2000, scrolling=True)

    with tab2:
        st.write("### Trustworthiness Trade-offs")
        for fig in generate_tradeoff_figures_with_pareto(live_results, coef):
            st.plotly_chart(fig)

    with tab3:
        st.write("### Performance and Trustworthiness")
        @st.fragment
        def show_topk():
            k = st.slider("Select top K experiments", min_value=1, max_value=10, value=3, key="b7_topk")
            for fig in generate_topk_plots_auto(live_results, coef, k):
                st.plotly_chart(fig)
        show_topk()
