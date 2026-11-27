"""Integrated Digital Twin Dashboard for PEM Green Hydrogen Production.

Run from project root:
streamlit run app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.forecasting.train_models import build_candidate_models
from src.integration.hybrid_digital_twin import (
    CURRENT_COL,
    VOLTAGE_COL,
    FaradayPhysicsCalibrator,
    HybridPEMForecaster,
)
from src.uncertainty.conformal import interval_metrics
from src.uncertainty.cqr import ConformalizedQuantileForecaster
from src.resilience.ood_and_faults import OODDetector
from src.digital_twin.twin_frame import TwinFrame
from src.digital_twin.orchestrator import DigitalTwinOrchestrator
from scripts.run_hybrid_loeo import TARGET, build_no_history_dataset


st.set_page_config(
    page_title="PEM Hydrogen Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #F7FAFC; color: #1E293B; }
    section[data-testid="stSidebar"] { background-color: #EEF5F9; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DCE8F2;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading NLR dataset...")
def load_horizon_data(horizon: int):
    return build_no_history_dataset(horizon)


@st.cache_resource(show_spinner="Training Physics-Residual CQR + OOD Digital Twin...")
def build_live_twin(horizon: int, held_out: str, alpha: float = 0.10):
    X, y, meta = load_horizon_data(horizon)
    groups = meta["Experiment"]
    test_idx = meta.index[groups == held_out]
    train_idx = meta.index[groups != held_out]

    Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
    ytr, yte = y.loc[train_idx], y.loc[test_idx]
    current_col = f"{TARGET}_current"

    base_hgb = build_candidate_models(random_state=42, include_optional=False)["HistGradientBoosting"]

    # 1. Calibrated Faraday physics baseline
    cal = FaradayPhysicsCalibrator().fit(Xtr, Xtr[current_col])
    faraday_pred = cal.predict_faraday_flow(Xte)

    # 2. Adaptive Aleatoric Uncertainty via Physics-Residual CQR
    cqr = ConformalizedQuantileForecaster(
        point_forecaster_factory=lambda: HybridPEMForecaster(
            base_hgb, mode="physics_residual", current_flow_col=current_col
        ),
        current_flow_col=current_col,
        alpha=alpha,
        cal_fraction=0.20,
        random_state=42,
    ).fit(Xtr, ytr, meta.loc[train_idx])
    pred, lower, upper = cqr.predict_interval(Xte)

    # 3. Epistemic / Regime OOD Detector
    ood = OODDetector(contamination=0.05, random_state=42).fit(Xtr)
    is_ood, ood_scores = ood.predict_ood(Xte)

    # 4. Synchronized Sequential Orchestrator
    orchestrator = DigitalTwinOrchestrator(
        conformal_forecaster=cqr,
        ood_detector=ood,
        feature_columns=list(X.columns),
        current_flow_col=current_col,
        horizon_steps=horizon,
        feature_means=Xtr.mean().to_dict(),
        feature_stds=Xtr.std().to_dict(),
    )

    y_actual = yte.to_numpy(dtype=float)
    pi = interval_metrics(y_actual, lower, upper, alpha=alpha)
    mae = float(np.mean(np.abs(y_actual - pred)))

    df_out = pd.DataFrame({
        "Step": np.arange(len(y_actual)),
        "Actual_H2_Flow": y_actual,
        "Hybrid_Forecast": pred,
        "Faraday_Physics": faraday_pred,
        "Lower_90": lower,
        "Upper_90": upper,
        "Is_OOD": is_ood,
        "OOD_Score": ood_scores,
        "Stack_Current_A": Xte[CURRENT_COL].to_numpy(dtype=float),
        "Stack_Voltage_V": Xte[VOLTAGE_COL].to_numpy(dtype=float),
    })

    summary = {
        "k_eff": cal.k_eff_,
        "mae": mae,
        "coverage": pi["Coverage_Percent"],
        "width": pi["Avg_Width_kg_hr"],
        "ood_pct": float(np.mean(is_ood) * 100.0),
        "neg_preds": int((pred < 0).sum()),
    }
    return df_out, summary, orchestrator, Xte.reset_index(drop=True)


def load_csv_if_exists(path: Path):
    return pd.read_csv(path) if path.exists() else None


# ---------------- Sidebar Controls ----------------
st.sidebar.title("PEM Digital Twin Controls")
horizon = st.sidebar.selectbox("Forecast Horizon (steps)", [1, 3, 5, 10], index=0)
_, _, meta_all = load_horizon_data(horizon)
experiments = sorted(meta_all["Experiment"].unique())
selected_exp = st.sidebar.selectbox("Held-Out Wind Experiment (LOEO)", experiments, index=0)
window_size = st.sidebar.slider("Plot Window (rows)", 100, 1500, 400, step=100)

st.title("Uncertainty-Aware Explainable Digital Twin — PEM Electrolysis")
st.caption(
    "Synchronized TwinFrame Pipeline · Physics-Residual Hybrid · Adaptive CQR (Aleatoric) · Isolation Forest OOD (Epistemic)"
)

tab1, tab2, tab3 = st.tabs([
    "1. Live Digital Twin & Tagged Frame Inspector",
    "2. Physics–AI & Uncertainty Benchmarks",
    "3. Explainability (XAI) & Resilience",
])

# ---------------- Tab 1: Live Replay + Tagged Frame Inspector ----------------
with tab1:
    df_live, stats, orchestrator, Xte_reset = build_live_twin(horizon, selected_exp, alpha=0.10)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hybrid MAE", f"{stats['mae']:.3f} kg/hr")
    c2.metric("90% CQR Coverage", f"{stats['coverage']:.1f}%")
    c3.metric("Adaptive PI Width", f"{stats['width']:.2f} kg/hr")
    c4.metric("OOD Flagged", f"{stats['ood_pct']:.1f}%")
    c5.metric("Calibrated k_eff", f"{stats['k_eff']:.2f}")

    if stats["ood_pct"] > 20.0:
        st.warning(
            f"**Epistemic OOD Alert:** {stats['ood_pct']:.1f}% of rows in `{selected_exp}` fall outside "
            "the training distribution. Intervals may under-cover; inspect individual frames below."
        )
    else:
        st.success(
            f"**Normal Operating Regime:** OOD rate ({stats['ood_pct']:.1f}%) is within expected bounds. "
            f"Physically inadmissible (<0) predictions: {stats['neg_preds']}."
        )

    max_start = max(0, len(df_live) - window_size)
    start_row = st.slider("Plot Start Row", 0, max_start, 0, step=50)
    view = df_live.iloc[start_row : start_row + window_size]

    fig, ax = plt.subplots(figsize=(12, 4.3))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFCFE")

    ax.fill_between(
        view["Step"], view["Lower_90"], view["Upper_90"],
        color="#B8D8F8", alpha=0.45, label="90% Adaptive CQR Interval",
    )
    ax.plot(view["Step"], view["Actual_H2_Flow"], color="#1E293B", lw=1.6, label="Actual H₂ Flow")
    ax.plot(view["Step"], view["Hybrid_Forecast"], color="#2B7A78", lw=1.5, ls="--", label="Physics-Residual Hybrid")
    ax.plot(view["Step"], view["Faraday_Physics"], color="#E09F7D", lw=1.2, alpha=0.85, label="Faraday Physics Baseline")

    ood_pts = view[view["Is_OOD"]]
    if not ood_pts.empty:
        ax.scatter(
            ood_pts["Step"], ood_pts["Actual_H2_Flow"],
            color="#D9534F", s=18, zorder=5, label="OOD Flagged",
        )

    ax.set_xlabel("Time Step (row index)")
    ax.set_ylabel("Hydrogen Flow (kg/hr)")
    ax.set_title(f"{selected_exp} — Horizon {horizon} Step(s)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    st.pyplot(fig)

    st.divider()
    st.subheader("Synchronized Single-Frame Inspector (TwinFrame)")
    inspect_step = st.number_input(
        "Select Step Index to Trace Through All 5 Layers",
        min_value=0,
        max_value=len(Xte_reset) - 1,
        value=int(start_row),
        step=1,
    )
    raw_frame = TwinFrame.create(
        step_index=int(inspect_step),
        experiment=selected_exp,
        raw_sensors=Xte_reset.iloc[int(inspect_step)].to_dict(),
    )
    synced = orchestrator.process_frame(raw_frame)

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Frame Tag", synced.frame_id)
    f2.metric("Status", synced.resilience.status)
    f3.metric("Forecast [90% PI]", f"{synced.forecast.predicted_flow_kg_hr:.2f} [{synced.uncertainty.lower_bound_kg_hr:.2f}, {synced.uncertainty.upper_bound_kg_hr:.2f}]")
    f4.metric("Epistemic Score", f"{synced.uncertainty.epistemic_score:+.4f}")

    st.info(f"**Pipeline Trace:** `{' → '.join(synced.pipeline_trace)}`  \n**Local Explanation:** {synced.xai.explanation_summary}")

# ---------------- Tab 2: Benchmarks ----------------
with tab2:
    st.subheader(f"Physics–AI Hybrid vs. Pure AI (12-Fold LOEO, Horizon {horizon})")
    hybrid_df = load_csv_if_exists(Path(f"reports/forecasting/hybrid_loeo_{horizon}_step_summary.csv"))
    if hybrid_df is not None:
        st.dataframe(hybrid_df, use_container_width=True)

    st.subheader("Split Conformal Uncertainty Summary (All Horizons)")
    unc_df = load_csv_if_exists(Path("reports/uncertainty/conformal_loeo_summary.csv"))
    if unc_df is not None:
        st.dataframe(unc_df, use_container_width=True)

# ---------------- Tab 3: XAI & Resilience ----------------
with tab3:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Top Drivers (LOEO Permutation Importance)")
        xai_df = load_csv_if_exists(Path("reports/xai/xai_permutation_loeo_1_step_summary.csv"))
        if xai_df is not None:
            top10 = xai_df.head(10).iloc[::-1]
            fig_xai, ax_xai = plt.subplots(figsize=(6, 4.2))
            ax_xai.barh(top10["Feature"], top10["Mean_MAE_Increase"], color="#8ECAE6")
            ax_xai.set_xlabel("Mean MAE Increase When Shuffled (kg/hr)")
            ax_xai.grid(axis="x", alpha=0.25)
            st.pyplot(fig_xai)

    with col_r:
        st.subheader("OOD Detection Rate by Experiment")
        ood_df = load_csv_if_exists(Path("reports/resilience/ood_loeo_1_step.csv"))
        if ood_df is not None:
            st.dataframe(ood_df, use_container_width=True)

    st.subheader("Sensor-Fault Stress Test (Horizon 1, LOEO Mean)")
    fault_df = load_csv_if_exists(Path("reports/resilience/sensor_faults_loeo_1_step_summary.csv"))
    if fault_df is not None:
        st.dataframe(fault_df, use_container_width=True)