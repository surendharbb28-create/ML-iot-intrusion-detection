"""
dashboard.py — Streamlit Cybersecurity SOC Dashboard
======================================================
MLCS-HACK-26 — IoT Intrusion Detection & Risk Analysis

Run:
    streamlit run app/dashboard.py
"""

import os
import sys
import io
import json
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ───────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    load_csv, discover_datasets, inspect_columns,
    clean_dataframe, detect_label_column, map_labels, _load_config,
)
from src.feature_engineering import engineer_features, get_feature_docs
from src.preprocessing import preprocess_pipeline, load_preprocessing
from src.train import (
    train_random_forest,
    save_model, load_model, list_saved_models,
)
from src.evaluation import compute_metrics, save_evaluation_report
from src.risk_scoring import (
    compute_risk_score, risk_level, risk_color,
    batch_risk_scores, risk_distribution,
)
from src.predict import predict_batch, predict_single, prepare_input
from src.explainability import explain_feature_importance, get_shap_values
from src.database import (
    store_prediction, store_batch, get_predictions,
    get_prediction_stats, init_db, clear_history,
)

logging.basicConfig(level=logging.INFO)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MLCS-HACK-26 — IoT IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Dark SOC Theme ──────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #111927 50%, #0d1321 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #0a1628 100%);
        border-right: 1px solid #1e3a5f;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00d4ff !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f1923 0%, #162236 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 24px rgba(0,100,200,0.08);
    }
    div[data-testid="stMetric"] label {
        color: #8899aa !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e0f0ff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Headers */
    h1, h2, h3 { color: #e0f0ff !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #0f1923;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        color: #8899aa;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #003366, #004488) !important;
        border-color: #00aaff !important;
        color: #ffffff !important;
    }

    /* DataFrames */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0055aa, #0077cc);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0077cc, #0099ee);
        box-shadow: 0 4px 16px rgba(0,119,204,0.4);
        transform: translateY(-1px);
    }

    /* Alert badge */
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
    }
    .risk-LOW      { background: #00382010; color: #00c853; border: 1px solid #00c853; }
    .risk-MEDIUM   { background: #33280010; color: #ff9800; border: 1px solid #ff9800; }
    .risk-HIGH     { background: #33100010; color: #ff5722; border: 1px solid #ff5722; }
    .risk-CRITICAL { background: #33000010; color: #d50000; border: 1px solid #d50000; }

    /* SOC Header */
    .soc-header {
        background: linear-gradient(90deg, #001833, #002855, #001833);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 30px;
        margin-bottom: 24px;
        text-align: center;
    }
    .soc-header h1 {
        font-size: 2rem !important;
        letter-spacing: 2px;
        margin: 0 !important;
        background: linear-gradient(90deg, #00aaff, #00ddff, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .soc-header p { color: #6688aa; margin: 4px 0 0 0; font-size: 0.95rem; }

    .demo-badge {
        background: #442200;
        color: #ffaa00;
        border: 1px solid #ff8800;
        border-radius: 6px;
        padding: 6px 16px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────

def _is_demo_data(df: pd.DataFrame) -> bool:
    """Check if the loaded data appears to be synthetic demo data."""
    if df is None:
        return True
    # Check filename heuristic stored in session state
    name = st.session_state.get("loaded_dataset_name", "")
    return "demo" in name.lower() or "synthetic" in name.lower()


def _plotly_dark_layout(fig):
    """Apply consistent dark theme to plotly figures."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,25,35,0.8)",
        font=dict(color="#aabbcc", family="Inter, sans-serif"),
        title_font=dict(color="#e0f0ff", size=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8899aa")),
        xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _risk_badge_html(level: str) -> str:
    return f'<span class="risk-badge risk-{level}">{level}</span>'


# ── Sidebar navigation ──────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ IoT IDS")
    st.markdown("**MLCS-HACK-26**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📊 Data Explorer",
            "🤖 Model Training",
            "🎯 Attack Analysis",
            "⚠️ Risk Analysis",
            "🔮 Prediction",
            "🧠 Explainability",
            "📜 History",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<p style='color:#556677;font-size:0.75rem;text-align:center;'>"
        "Academic Cybersecurity Project<br>Defensive Analysis Only</p>",
        unsafe_allow_html=True,
    )

# ── SOC Header ───────────────────────────────────────────────
st.markdown(
    '<div class="soc-header">'
    "<h1>MLCS-HACK-26</h1>"
    "<p>IoT Intrusion Detection &amp; Risk Analysis</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    # Try to load the most recent evaluation report
    eval_dir = os.path.join(PROJECT_ROOT, "reports", "evaluation")
    metrics = None
    if os.path.isdir(eval_dir):
        files = sorted([f for f in os.listdir(eval_dir) if f.endswith(".json")], reverse=True)
        if files:
            with open(os.path.join(eval_dir, files[0]), "r") as f:
                report = json.load(f)
                metrics = report.get("metrics", {})

    # Try to load dataset stats
    datasets = discover_datasets()
    total = normal = attacks = 0
    if datasets:
        try:
            df_overview = load_csv(datasets[0])
            df_overview = clean_dataframe(df_overview)
            lbl = detect_label_column(df_overview)
            if lbl:
                binary, _ = map_labels(df_overview[lbl])
                total = len(df_overview)
                normal = int((binary == "NORMAL").sum())
                attacks = int((binary == "ATTACK").sum())
        except Exception:
            pass

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{total:,}")
    c2.metric("Normal", f"{normal:,}")
    c3.metric("Attacks", f"{attacks:,}")
    c4.metric("Attack %", f"{(attacks/total*100):.1f}%" if total else "N/A")

    c5, c6, c7 = st.columns(3)
    if metrics:
        c5.metric("Detection Rate", f"{metrics.get('detection_rate', 0):.2%}")
        c6.metric("FPR", f"{metrics.get('false_positive_rate', 0):.2%}")
        c7.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
    else:
        c5.metric("Detection Rate", "—")
        c6.metric("FPR", "—")
        c7.metric("F1 Score", "—")
        st.info("ℹ️ Train a model to see evaluation metrics here.")

    # Demo data warning
    if datasets and "demo" in datasets[0].lower():
        st.markdown(
            '<div class="demo-badge">⚠ DEMO / SYNTHETIC DATA — Results do not represent real-world performance</div>',
            unsafe_allow_html=True,
        )

    # Quick stats charts
    if total > 0:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure(go.Pie(
                labels=["Normal", "Attack"],
                values=[normal, attacks],
                marker=dict(colors=["#00c853", "#d50000"]),
                hole=0.55,
                textinfo="percent+label",
                textfont=dict(color="white"),
            ))
            fig.update_layout(title="Traffic Distribution", showlegend=False)
            _plotly_dark_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # DB prediction stats
            db_stats = get_prediction_stats()
            if db_stats["total"] > 0:
                risk_data = db_stats.get("risk_levels", {})
                risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                risk_colors = ["#00c853", "#ff9800", "#ff5722", "#d50000"]
                vals = [risk_data.get(r, 0) for r in risk_order]
                fig2 = go.Figure(go.Bar(
                    x=risk_order, y=vals,
                    marker_color=risk_colors,
                    text=vals, textposition="auto",
                    textfont=dict(color="white"),
                ))
                fig2.update_layout(title="Prediction Risk Distribution")
                _plotly_dark_layout(fig2)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Run predictions to see risk distribution.")

    # Models status
    st.markdown("### 🤖 Model Status")
    models = list_saved_models()
    if models:
        for m in models:
            st.success(f"✅ {m}")
    else:
        st.warning("No trained models found. Go to **Model Training** to train.")


# ══════════════════════════════════════════════════════════════
#  PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown("## 📊 Data Explorer")

    upload = st.file_uploader("Upload a CSV file", type=["csv", "tsv", "txt"])

    df_explore = None
    source = None

    if upload is not None:
        df_explore = pd.read_csv(upload)
        source = upload.name
        st.session_state.df = df_explore
        st.session_state.uploaded_filename = source
    else:
        datasets = discover_datasets()
        if datasets:
            chosen = st.selectbox("Or select a dataset from data/raw/", datasets)
            if chosen:
                df_explore = load_csv(chosen)
                source = os.path.basename(chosen)

    if df_explore is not None:
        st.session_state["loaded_dataset_name"] = source or ""
        st.markdown(f"**Dataset:** `{source}`")

        info = inspect_columns(df_explore)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{info['shape'][0]:,}")
        c2.metric("Columns", info["shape"][1])
        c3.metric("Duplicates", f"{info['duplicates']:,}")

        tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Columns", "Missing Values", "Statistics"])

        with tab1:
            st.dataframe(df_explore.head(100), use_container_width=True, height=400)

        with tab2:
            col_df = pd.DataFrame({
                "Column": info["columns"],
                "Data Type": [info["dtypes"][c] for c in info["columns"]],
                "Missing": [info["missing"][c] for c in info["columns"]],
                "Missing %": [info["missing_pct"][c] for c in info["columns"]],
            })
            st.dataframe(col_df, use_container_width=True, height=400)

        with tab3:
            missing = {k: v for k, v in info["missing"].items() if v > 0}
            if missing:
                fig = go.Figure(go.Bar(
                    x=list(missing.keys()),
                    y=list(missing.values()),
                    marker_color="#ff5722",
                    text=list(missing.values()),
                    textposition="auto",
                    textfont=dict(color="white"),
                ))
                fig.update_layout(title="Missing Values by Column")
                _plotly_dark_layout(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values!")

        with tab4:
            st.dataframe(df_explore.describe().T, use_container_width=True, height=400)
    else:
        st.info("Upload a CSV or place one in `data/raw/` to explore.")


# ══════════════════════════════════════════════════════════════
#  PAGE: MODEL TRAINING
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("## 🤖 Model Training")

    datasets = discover_datasets()
    if not datasets and "uploaded_filename" not in st.session_state:
        st.warning("No dataset found. Generating synthetic demo data…")
        from generate_demo_data import main as gen_main
        gen_main()
        datasets = discover_datasets()

    options = datasets.copy()
    if "uploaded_filename" in st.session_state:
        # Prepend the uploaded dataset to the options
        if st.session_state.uploaded_filename not in options:
            options.insert(0, st.session_state.uploaded_filename)

    def format_dataset(ds):
        if "uploaded_filename" in st.session_state and ds == st.session_state.uploaded_filename:
            return f"Uploaded: {ds}"
        return os.path.basename(ds)

    chosen_ds = st.selectbox("Select dataset", options, format_func=format_dataset)
    label_override = st.text_input("Label column (leave blank for auto-detect)", value="")

    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Loading and preprocessing data…"):
            try:
                if "uploaded_filename" in st.session_state and chosen_ds == st.session_state.uploaded_filename:
                    df_train = st.session_state.df.copy()
                else:
                    df_train = load_csv(chosen_ds)
                
                df_train = clean_dataframe(df_train)
                label_col = detect_label_column(df_train) if not label_override else label_override

                if label_col is None or label_col not in df_train.columns:
                    st.error(f"Could not detect label column. Available columns: {list(df_train.columns)}")
                    st.stop()

                df_train = engineer_features(df_train)
                config = _load_config()

                result = preprocess_pipeline(
                    df_train, label_col,
                    test_size=config["ml"]["test_size"],
                    random_state=config["ml"]["random_state"],
                )

                st.success(f"✅ Preprocessed: {len(result['X_train'])} train / {len(result['X_test'])} test samples, {len(result['feature_names'])} features")

            except Exception as e:
                st.error(f"Preprocessing error: {e}")
                st.stop()

        # Train
        progress = st.progress(0, text="Training…")
        trained_models = {}

        progress.progress(50, text="Training Random Forest…")
        rf = train_random_forest(result["X_train"], result["y_train"], config)
        save_model(rf, "random_forest")
        trained_models["random_forest"] = rf

        progress.progress(100, text="✅ Training complete!")

        # Evaluate
        st.markdown("### 📊 Evaluation Results")
        for name, mdl in trained_models.items():
            st.markdown(f"#### {name.replace('_', ' ').title()}")
            preds = mdl.predict(result["X_test"])
            m = compute_metrics(result["y_test"], preds, result["class_names"])

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy", f"{m['accuracy']:.4f}")
            mc2.metric("Precision", f"{m['precision']:.4f}")
            mc3.metric("Recall", f"{m['recall']:.4f}")
            mc4.metric("F1 Score", f"{m['f1_score']:.4f}")

            mc5, mc6, mc7 = st.columns(3)
            mc5.metric("Detection Rate", f"{m['detection_rate']:.4f}")
            mc6.metric("FPR", f"{m['false_positive_rate']:.4f}")
            mc7.metric("FNR", f"{m['false_negative_rate']:.4f}")

            # Save report
            fi = None
            if hasattr(mdl, "feature_importances_"):
                fi = mdl.feature_importances_
            elif hasattr(mdl, "coef_"):
                fi = mdl.coef_[0]
            save_evaluation_report(m, name, feature_names=result["feature_names"], feature_importances=fi)

            st.markdown("---")


# ══════════════════════════════════════════════════════════════
#  PAGE: ATTACK ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🎯 Attack Analysis":
    st.markdown("## 🎯 Attack Analysis")

    datasets = discover_datasets()
    if not datasets:
        st.warning("No dataset available.")
        st.stop()

    chosen = st.selectbox("Dataset", datasets, format_func=os.path.basename)
    df_atk = load_csv(chosen)
    df_atk = clean_dataframe(df_atk)
    lbl = detect_label_column(df_atk)

    if lbl is None:
        st.error("Cannot detect label column.")
        st.stop()

    binary, original = map_labels(df_atk[lbl])
    df_atk["_binary"] = binary
    df_atk["_category"] = original

    col1, col2 = st.columns(2)

    with col1:
        # Normal vs Attack pie
        counts = binary.value_counts()
        fig = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            marker=dict(colors=["#d50000", "#00c853"] if counts.index[0] == "ATTACK" else ["#00c853", "#d50000"]),
            hole=0.5,
            textinfo="percent+label",
            textfont=dict(color="white"),
        ))
        fig.update_layout(title="Normal vs Attack")
        _plotly_dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Attack categories
        attack_cats = original[binary == "ATTACK"].value_counts().head(15)
        if not attack_cats.empty:
            fig2 = go.Figure(go.Bar(
                x=attack_cats.values.tolist(),
                y=attack_cats.index.tolist(),
                orientation="h",
                marker_color="#ff5722",
                text=attack_cats.values.tolist(),
                textposition="auto",
                textfont=dict(color="white"),
            ))
            fig2.update_layout(title="Attack Categories")
            _plotly_dark_layout(fig2)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No attack categories to display.")

    # Protocol distribution
    proto_cols = [c for c in df_atk.columns if c.lower() in ("proto", "protocol")]
    if proto_cols:
        st.markdown("### Protocol Distribution")
        proto_counts = df_atk[proto_cols[0]].value_counts()
        fig3 = go.Figure(go.Bar(
            x=proto_counts.index.tolist(),
            y=proto_counts.values.tolist(),
            marker_color="#00aaff",
            text=proto_counts.values.tolist(),
            textposition="auto",
            textfont=dict(color="white"),
        ))
        fig3.update_layout(title="Traffic by Protocol")
        _plotly_dark_layout(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    # Traffic distribution by service
    svc_cols = [c for c in df_atk.columns if c.lower() in ("service",)]
    if svc_cols:
        st.markdown("### Service Distribution")
        svc_counts = df_atk[svc_cols[0]].value_counts().head(10)
        fig4 = go.Figure(go.Bar(
            x=svc_counts.index.tolist(),
            y=svc_counts.values.tolist(),
            marker_color="#7c4dff",
            text=svc_counts.values.tolist(),
            textposition="auto",
            textfont=dict(color="white"),
        ))
        fig4.update_layout(title="Traffic by Service")
        _plotly_dark_layout(fig4)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: RISK ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Analysis":
    st.markdown("## ⚠️ Risk Analysis")

    models = list_saved_models()
    if not models or "random_forest" not in models:
        st.warning("Train a model first.")
        st.stop()

    datasets = discover_datasets()
    if not datasets:
        st.warning("No dataset.")
        st.stop()

    df_risk = load_csv(datasets[0])
    df_risk = clean_dataframe(df_risk)
    df_risk = engineer_features(df_risk)

    try:
        preproc = load_preprocessing()
        model = load_model("random_forest")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    X = prepare_input(df_risk, preproc)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        le = preproc.get("label_encoder")
        attack_idx = 0
        if le:
            classes = list(le.classes_)
            attack_idx = classes.index("ATTACK") if "ATTACK" in classes else min(1, len(classes) - 1)
        risks = batch_risk_scores(probas, attack_class_index=attack_idx)
    else:
        st.warning("Model doesn't support probability estimates.")
        st.stop()

    scores = [r["risk_score"] for r in risks]
    levels = [r["risk_level"] for r in risks]

    # Distribution
    dist = risk_distribution(scores)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🟢 LOW", dist.get("LOW", 0))
    col2.metric("🟡 MEDIUM", dist.get("MEDIUM", 0))
    col3.metric("🟠 HIGH", dist.get("HIGH", 0))
    col4.metric("🔴 CRITICAL", dist.get("CRITICAL", 0))

    # Histogram
    fig = go.Figure(go.Histogram(
        x=scores,
        nbinsx=50,
        marker_color="#00aaff",
        opacity=0.85,
    ))
    fig.update_layout(
        title="Risk Score Distribution (0–100)",
        xaxis_title="Risk Score",
        yaxis_title="Count",
    )
    _plotly_dark_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Risk level pie
    risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_colors = ["#00c853", "#ff9800", "#ff5722", "#d50000"]
    vals = [dist.get(r, 0) for r in risk_order]
    fig2 = go.Figure(go.Pie(
        labels=risk_order,
        values=vals,
        marker=dict(colors=risk_colors),
        hole=0.5,
        textinfo="percent+label",
        textfont=dict(color="white"),
    ))
    fig2.update_layout(title="Risk Level Distribution")
    _plotly_dark_layout(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    # Timeline if timestamp available
    ts_cols = [c for c in df_risk.columns if c.lower() in ("ts", "timestamp", "datetime", "time")]
    if ts_cols:
        try:
            df_risk["_ts"] = pd.to_datetime(df_risk[ts_cols[0]], errors="coerce")
            df_risk["_risk_score"] = scores
            df_risk_ts = df_risk.dropna(subset=["_ts"]).sort_values("_ts")
            if not df_risk_ts.empty:
                st.markdown("### Risk Timeline")
                fig3 = go.Figure(go.Scatter(
                    x=df_risk_ts["_ts"],
                    y=df_risk_ts["_risk_score"],
                    mode="markers",
                    marker=dict(
                        color=df_risk_ts["_risk_score"],
                        colorscale=[[0, "#00c853"], [0.3, "#ff9800"], [0.6, "#ff5722"], [1, "#d50000"]],
                        size=4,
                        colorbar=dict(title="Risk"),
                    ),
                ))
                fig3.update_layout(title="Risk Score Over Time", xaxis_title="Time", yaxis_title="Risk Score")
                _plotly_dark_layout(fig3)
                st.plotly_chart(fig3, use_container_width=True)
        except Exception:
            pass

    st.markdown(
        "> **Note:** The risk score is the application's ML-derived score (`attack_probability × 100`). "
        "It is NOT an industry-standard security score."
    )


# ══════════════════════════════════════════════════════════════
#  PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.markdown("## 🔮 Prediction")

    models = list_saved_models()
    if not models:
        st.warning("No trained models. Train a model first.")
        st.stop()

    if not models or "random_forest" not in models:
        st.warning("Train the model first.")
        st.stop()

    model_name = "random_forest"

    st.markdown("---")
    st.markdown("### Step 1 — Upload CSV")

    # CSV Template Download
    template_cols = [
        "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", 
        "proto", "service", "duration", "orig_bytes", "resp_bytes", 
        "conn_state", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes"
    ]
    template_csv = ",".join(template_cols) + "\n"
    
    st.download_button(
        label="📄 Download CSV Template",
        data=template_csv,
        file_name="iot_traffic_template.csv",
        mime="text/csv"
    )

    upload = st.file_uploader("Upload IoT Network Traffic CSV", type=["csv"])
    
    if upload is not None:
        df_pred = pd.read_csv(upload)
        
        st.markdown(f"**Uploaded Records:** {len(df_pred)}")
        st.markdown(f"**Input Features:** {len(df_pred.columns)}")
        st.dataframe(df_pred.head(5), use_container_width=True)

        st.markdown("---")
        
        if st.button("🔮 Run Predictions", use_container_width=True):
            with st.spinner("Predicting…"):
                try:
                    results = predict_batch(df_pred, model_name=model_name)

                    # Store in DB
                    records = []
                    for _, row in results.iterrows():
                        records.append({
                            "prediction": row["prediction"],
                            "confidence": row["confidence"],
                            "risk_score": row["risk_score"],
                            "risk_level": row["risk_level"],
                            "model_name": model_name,
                            "explanation": row.get("explanation", ""),
                        })
                    store_batch(records)

                    st.markdown("========================================")
                    st.markdown("### 🔮 PREDICTION RESULTS")
                    st.markdown("========================================")

                    total_rec = len(results)
                    normal_rec = int((results["prediction"] == "NORMAL").sum())
                    attack_rec = int((results["prediction"] == "ATTACK").sum())
                    avg_risk = int(results["risk_score"].mean()) if total_rec > 0 else 0
                    attack_pct = (attack_rec / total_rec * 100) if total_rec > 0 else 0

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Records", total_rec)
                    c2.metric("Normal Traffic", normal_rec)
                    c3.metric("Attack Traffic", attack_rec)
                    c4.metric("Average Risk Score", avg_risk)
                    
                    st.metric("Attack Percentage", f"{attack_pct:.1f}%")

                    st.markdown("---")

                    # Select specific columns to match the requested format
                    display_df = results[["prediction", "confidence", "risk_score", "risk_level"]].copy()
                    display_df.index = display_df.index + 1
                    display_df.index.name = "Record"
                    
                    # Format confidence as percentage
                    display_df["confidence"] = (display_df["confidence"] * 100).apply(lambda x: f"{x:.1f}%")

                    st.dataframe(display_df, use_container_width=True, height=400)
                    
                    st.markdown("---")
                    st.success("Prediction results have been saved to History.")

                except Exception as e:
                    st.error(f"Prediction error: {e}")


# ══════════════════════════════════════════════════════════════
#  PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
elif page == "🧠 Explainability":
    st.markdown("## 🧠 Explainable AI")

    models = list_saved_models()
    if not models or "random_forest" not in models:
        st.warning("Train the model first.")
        st.stop()

    model_name = "random_forest"

    try:
        model = load_model(model_name)
        preproc = load_preprocessing()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    feature_names = preproc["feature_names"]
    fi = explain_feature_importance(model, feature_names, top_n=20)

    if fi["importances"]:
        st.markdown("### Feature Importance")
        top = fi["top_features"][:15]
        fig = go.Figure(go.Bar(
            x=[f["importance"] for f in reversed(top)],
            y=[f["feature"] for f in reversed(top)],
            orientation="h",
            marker_color="#00aaff",
            text=[f"{f['importance']:.4f}" for f in reversed(top)],
            textposition="auto",
            textfont=dict(color="white"),
        ))
        fig.update_layout(title=f"Top Features — {model_name}")
        _plotly_dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Summary:** {fi['summary']}")

        # Feature importance table
        st.markdown("### All Features")
        fi_df = pd.DataFrame(fi["importances"])
        st.dataframe(fi_df, use_container_width=True, height=400)

    else:
        st.info("Feature importance not available for this model type.")

    # SHAP (optional)
    st.markdown("### SHAP Analysis")
    if st.button("🧮 Compute SHAP Values (may take time)", use_container_width=True):
        with st.spinner("Computing SHAP values…"):
            datasets = discover_datasets()
            if datasets:
                df_shap = load_csv(datasets[0], nrows=200)
                df_shap = clean_dataframe(df_shap)
                df_shap = engineer_features(df_shap)
                X_shap = prepare_input(df_shap, preproc)

                shap_result = get_shap_values(model, X_shap, feature_names)
                if shap_result is not None:
                    st.success("SHAP values computed!")
                    sv = shap_result["shap_values"]
                    if isinstance(sv, list):
                        sv = sv[-1]  # ATTACK class
                    mean_abs = np.abs(sv).mean(axis=0)
                    shap_df = pd.DataFrame({
                        "Feature": feature_names[:len(mean_abs)],
                        "Mean |SHAP|": mean_abs,
                    }).sort_values("Mean |SHAP|", ascending=False)

                    fig_s = go.Figure(go.Bar(
                        x=shap_df["Mean |SHAP|"].head(15).values[::-1],
                        y=shap_df["Feature"].head(15).values[::-1],
                        orientation="h",
                        marker_color="#7c4dff",
                        textfont=dict(color="white"),
                    ))
                    fig_s.update_layout(title="SHAP Feature Importance")
                    _plotly_dark_layout(fig_s)
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
                    st.warning("SHAP computation was not possible for this model.")
            else:
                st.warning("No dataset available for SHAP analysis.")


# ══════════════════════════════════════════════════════════════
#  PAGE: HISTORY
# ══════════════════════════════════════════════════════════════
elif page == "📜 History":
    st.markdown("## 📜 Prediction History")

    init_db()

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        filt_risk = st.selectbox("Risk Level", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    with fc2:
        filt_pred = st.selectbox("Prediction", ["All", "ATTACK", "NORMAL"])
    with fc3:
        filt_model = st.text_input("Model Name", "")
    with fc4:
        filt_limit = st.number_input("Max Records", min_value=10, max_value=5000, value=200)

    records = get_predictions(
        limit=filt_limit,
        risk_level=filt_risk if filt_risk != "All" else None,
        prediction=filt_pred if filt_pred != "All" else None,
        model_name=filt_model if filt_model else None,
    )

    if records:
        df_hist = pd.DataFrame(records)
        st.dataframe(df_hist, use_container_width=True, height=500)

        # Stats
        stats = get_prediction_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", stats["total"])
        c2.metric("Attacks Detected", stats["attacks"])
        c3.metric("Normal", stats["normals"])

        # Download
        csv_hist = df_hist.to_csv(index=False)
        st.download_button("📥 Download History", csv_hist, "prediction_history.csv", "text/csv")

        if st.button("🗑️ Clear History", use_container_width=True):
            clear_history()
            st.rerun()
    else:
        st.info("No prediction history yet. Run some predictions first.")
