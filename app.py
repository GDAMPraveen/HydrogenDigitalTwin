"""Industrial Green Hydrogen PEM Digital Twin — Real-Time SCADA & AI Dashboard.

Run from project root:
streamlit run app.py
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
from src.plant.plant_simulator import GreenHydrogenPlantSimulator
from src.plant.reporting import generate_period_report
from scripts.run_hybrid_loeo import TARGET, build_no_history_dataset


st.set_page_config(
    page_title="PEM Green Hydrogen Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light pastel industrial theme
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


# ---------------- Cached Data & Model Loaders ----------------
@st.cache_data(show_spinner="Simulating 7-day Wind + Solar PEM Plant...")
def load_plant_simulation(days: int = 7, seed: int = 42) -> pd.DataFrame:
    return GreenHydrogenPlantSimulator().run(
        start_date="2026-11-01", days=days, freq_minutes=15, seed=seed
    )


@st.cache_data(show_spinner="Loading NLR LOEO dataset...")
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

    cal = FaradayPhysicsCalibrator().fit(Xtr, Xtr[current_col])
    faraday_pred = cal.predict_faraday_flow(Xte)

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

    ood = OODDetector(contamination=0.05, random_state=42).fit(Xtr)
    is_ood, ood_scores = ood.predict_ood(Xte)

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
sim_days = st.sidebar.slider("Plant Simulation Horizon (days)", 3, 14, 7)
horizon = st.sidebar.selectbox("AI Forecast Horizon (steps)", [1, 3, 5, 10], index=0)
_, _, meta_all = load_horizon_data(horizon)
experiments = sorted(meta_all["Experiment"].unique())
selected_exp = st.sidebar.selectbox("NLR Wind Experiment (LOEO)", experiments, index=0)

st.title("Green Hydrogen PEM Electrolyzer — Industrial Digital Twin")
st.caption(
    "Wind + Solar PV + BESS → EMS Dispatch → Synchronized TwinFrame (Physics + CQR + OOD + XAI) → H₂ Storage & Reports"
)

tab_plant, tab_reports, tab_twin, tab_bench, tab_xai = st.tabs([
    "1. Live Plant Operations (Real-Time)",
    "2. Date-Range Operational Reports",
    "3. AI Twin & Tagged Frame Inspector",
    "4. Physics–AI & Uncertainty Benchmarks",
    "5. Explainability (XAI) & Resilience",
])

plant_df = load_plant_simulation(days=sim_days, seed=42)


# ---------------- TAB 1: Real-Time Plant Streaming ----------------
with tab_plant:
    if "stream_idx" not in st.session_state:
        st.session_state.stream_idx = 48  # Start 12 hours into Day 1
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False

    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1, 1, 1, 2, 2])
    if ctrl1.button("▶ Play", use_container_width=True):
        st.session_state.is_playing = True
    if ctrl2.button("⏸ Pause", use_container_width=True):
        st.session_state.is_playing = False
    if ctrl3.button("↺ Reset", use_container_width=True):
        st.session_state.is_playing = False
        st.session_state.stream_idx = 12

    step_jump = ctrl4.selectbox("Playback Speed", [1, 4, 12, 24], index=1, format_func=lambda x: f"{x} steps / tick")
    window_pts = ctrl5.slider("Visible Window (steps)", 48, 288, 96, step=24)

    cur_idx = min(st.session_state.stream_idx, len(plant_df) - 1)
    now = plant_df.iloc[cur_idx]
    start_slice = max(0, cur_idx - window_pts + 1)
    live_slice = plant_df.iloc[start_slice : cur_idx + 1]

    st.markdown(
        f"**Live Plant Clock:** `{now['timestamp']}` &nbsp;|&nbsp; "
        f"**Operating Mode:** `{now['operating_mode']}` &nbsp;|&nbsp; "
        f"**Step:** `{cur_idx + 1} / {len(plant_df)}`"
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Wind + Solar", f"{now['total_renewable_kw']:.0f} kW", f"W:{now['wind_kw']:.0f} / S:{now['solar_kw']:.0f}")
    m2.metric("PEM Stack Load", f"{now['stack_power_kw']:.0f} kW", f"{now['stack_current_a']:.0f} A")
    m3.metric("H₂ Production Rate", f"{now['h2_rate_kg_hr']:.2f} kg/hr")
    m4.metric("BESS SoC", f"{now['bess_soc_pct']:.1f}%", f"{now['bess_power_kw']:+.0f} kW")
    m5.metric("H₂ Tank Level", f"{now['tank_fill_pct']:.1f}%", f"{now['tank_pressure_bar']:.0f} bar")
    m6.metric("Cumulative H₂", f"{plant_df.iloc[: cur_idx + 1]['h2_produced_kg'].sum():.1f} kg")

    fig_scada = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Power Supply & PEM Stack Dispatch (kW)", "H₂ Production Rate (kg/hr) & Storage Tank Fill (%)"),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    # Top subplot: Power flows
    fig_scada.add_trace(go.Scatter(
        x=live_slice["timestamp"], y=live_slice["wind_kw"],
        name="Wind (kW)", fill="tozeroy", line=dict(color="#6EC1E4", width=1.5),
    ), row=1, col=1)
    fig_scada.add_trace(go.Scatter(
        x=live_slice["timestamp"], y=live_slice["solar_kw"],
        name="Solar PV (kW)", fill="tozeroy", line=dict(color="#F6C28B", width=1.5),
    ), row=1, col=1)
    fig_scada.add_trace(go.Scatter(
        x=live_slice["timestamp"], y=live_slice["stack_power_kw"],
        name="PEM Stack (kW)", line=dict(color="#1E293B", width=2.2),
    ), row=1, col=1)

    # Bottom subplot: H2 rate & Tank %
    fig_scada.add_trace(go.Scatter(
        x=live_slice["timestamp"], y=live_slice["h2_rate_kg_hr"],
        name="H₂ Rate (kg/hr)", line=dict(color="#2B7A78", width=2.2),
    ), row=2, col=1, secondary_y=False)
    fig_scada.add_trace(go.Scatter(
        x=live_slice["timestamp"], y=live_slice["tank_fill_pct"],
        name="Tank Fill (%)", line=dict(color="#E07A5F", width=2.0, dash="dot"),
    ), row=2, col=1, secondary_y=True)

    fig_scada.update_layout(
        height=540, template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    fig_scada.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig_scada.update_yaxes(title_text="H₂ Rate (kg/hr)", row=2, col=1, secondary_y=False)
    fig_scada.update_yaxes(title_text="Tank Fill (%)", range=[0, 105], row=2, col=1, secondary_y=True)

    st.plotly_chart(fig_scada, use_container_width=True)

    if st.session_state.is_playing:
        if st.session_state.stream_idx < len(plant_df) - 1:
            st.session_state.stream_idx = min(len(plant_df) - 1, st.session_state.stream_idx + step_jump)
            time.sleep(0.35)
            st.rerun()
        else:
            st.session_state.is_playing = False


# ---------------- TAB 2: Date-Range Operational Reports ----------------
with tab_reports:
    st.subheader("Generate & Download Date-Range Operational Report")
    min_d = pd.to_datetime(plant_df["timestamp"]).dt.date.min()
    max_d = pd.to_datetime(plant_df["timestamp"]).dt.date.max()

    d1, d2 = st.columns(2)
    start_d = d1.date_input("From Date", value=min_d, min_value=min_d, max_value=max_d)
    end_d = d2.date_input("To Date", value=max_d, min_value=min_d, max_value=max_d)

    if start_d > end_d:
        st.error("`From Date` must be on or before `To Date`.")
    else:
        report_md, daily_df, kpis = generate_period_report(plant_df, str(start_d), str(end_d))

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total H₂ Produced", f"{kpis['total_h2_kg']:.1f} kg")
        k2.metric("Wind + Solar", f"{kpis['total_wind_mwh'] + kpis['total_solar_mwh']:.1f} MWh")
        k3.metric("Avg SEC", f"{kpis['avg_sec_kwh_kg']:.2f} kWh/kg")
        k4.metric("Water Consumed", f"{kpis['total_water_l']:.0f} L")
        k5.metric("CO₂ Avoided", f"{kpis['total_co2_avoided_kg']:.0f} kg")

        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button(
            "⬇ Download Executive Report (.md)",
            data=report_md,
            file_name=f"pem_plant_report_{start_d}_to_{end_d}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        dl2.download_button(
            "⬇ Download Daily Summary (.csv)",
            data=daily_df.to_csv(index=False),
            file_name=f"pem_daily_summary_{start_d}_to_{end_d}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        ts_mask = (pd.to_datetime(plant_df["timestamp"]).dt.date >= start_d) & (
            pd.to_datetime(plant_df["timestamp"]).dt.date <= end_d
        )
        dl3.download_button(
            "⬇ Download Raw Telemetry (.csv)",
            data=plant_df[ts_mask].to_csv(index=False),
            file_name=f"pem_telemetry_{start_d}_to_{end_d}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("### Day-by-Day Summary")
        st.dataframe(daily_df, use_container_width=True)

        with st.expander("Preview Full Markdown Executive Report"):
            st.markdown(report_md)


# ---------------- TAB 3: Live AI Twin & Tagged Frame Inspector ----------------
with tab_twin:
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

    w_col, s_col = st.columns(2)
    win_size = w_col.slider("Plot Window (rows)", 100, 1500, 400, step=100)
    start_row = s_col.slider("Start Row", 0, max(0, len(df_live) - win_size), 0, step=50)
    view = df_live.iloc[start_row : start_row + win_size]

    fig_ai = go.Figure()
    fig_ai.add_trace(go.Scatter(
        x=pd.concat([view["Step"], view["Step"][::-1]]),
        y=pd.concat([view["Upper_90"], view["Lower_90"][::-1]]),
        fill="toself", fillcolor="rgba(184, 216, 248, 0.45)",
        line=dict(color="rgba(255,255,255,0)"), name="90% Adaptive CQR Interval",
    ))
    fig_ai.add_trace(go.Scatter(
        x=view["Step"], y=view["Actual_H2_Flow"],
        name="Actual H₂ Flow", line=dict(color="#1E293B", width=1.8),
    ))
    fig_ai.add_trace(go.Scatter(
        x=view["Step"], y=view["Hybrid_Forecast"],
        name="Physics-Residual Hybrid", line=dict(color="#2B7A78", width=1.8, dash="dash"),
    ))
    fig_ai.add_trace(go.Scatter(
        x=view["Step"], y=view["Faraday_Physics"],
        name="Faraday Physics", line=dict(color="#E09F7D", width=1.3),
    ))
    ood_pts = view[view["Is_OOD"]]
    if not ood_pts.empty:
        fig_ai.add_trace(go.Scatter(
            x=ood_pts["Step"], y=ood_pts["Actual_H2_Flow"],
            mode="markers", name="OOD Flagged", marker=dict(color="#D9534F", size=6),
        ))
    fig_ai.update_layout(
        height=420, template="plotly_white",
        xaxis_title="Time Step (row index)", yaxis_title="Hydrogen Flow (kg/hr)",
        title=f"{selected_exp} — Horizon {horizon} Step(s)",
        margin=dict(l=40, r=20, t=40, b=30),
    )
    st.plotly_chart(fig_ai, use_container_width=True)

    st.divider()
    st.subheader("Synchronized Single-Frame Inspector (TwinFrame)")
    inspect_step = st.number_input(
        "Select Step Index to Trace Through All 5 Layers",
        min_value=0, max_value=len(Xte_reset) - 1, value=int(start_row), step=1,
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
    f3.metric(
        "Forecast [90% PI]",
        f"{synced.forecast.predicted_flow_kg_hr:.2f} [{synced.uncertainty.lower_bound_kg_hr:.2f}, {synced.uncertainty.upper_bound_kg_hr:.2f}]",
    )
    f4.metric("Epistemic Score", f"{synced.uncertainty.epistemic_score:+.4f}")
    st.info(
        f"**Pipeline Trace:** `{' → '.join(synced.pipeline_trace)}`  \n"
        f"**Local Explanation:** {synced.xai.explanation_summary}"
    )


# ---------------- TAB 4: Benchmarks ----------------
with tab_bench:
    st.subheader(f"Physics–AI Hybrid vs. Pure AI (12-Fold LOEO, Horizon {horizon})")
    hybrid_df = load_csv_if_exists(Path(f"reports/forecasting/hybrid_loeo_{horizon}_step_summary.csv"))
    if hybrid_df is not None:
        st.dataframe(hybrid_df, use_container_width=True)

    st.subheader("Split Conformal Uncertainty Summary (All Horizons)")
    unc_df = load_csv_if_exists(Path("reports/uncertainty/conformal_loeo_summary.csv"))
    if unc_df is not None:
        st.dataframe(unc_df, use_container_width=True)


# ---------------- TAB 5: XAI & Resilience ----------------
with tab_xai:
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Top Drivers (LOEO Permutation Importance)")
        xai_df = load_csv_if_exists(Path("reports/xai/xai_permutation_loeo_1_step_summary.csv"))
        if xai_df is not None:
            top10 = xai_df.head(10).iloc[::-1]
            fig_xai = go.Figure(go.Bar(
                x=top10["Mean_MAE_Increase"], y=top10["Feature"],
                orientation="h", marker_color="#8ECAE6",
            ))
            fig_xai.update_layout(
                height=380, template="plotly_white",
                xaxis_title="Mean MAE Increase When Shuffled (kg/hr)",
                margin=dict(l=20, r=20, t=20, b=30),
            )
            st.plotly_chart(fig_xai, use_container_width=True)

    with col_r:
        st.subheader("OOD Detection Rate by Experiment")
        ood_df = load_csv_if_exists(Path("reports/resilience/ood_loeo_1_step.csv"))
        if ood_df is not None:
            st.dataframe(ood_df, use_container_width=True)

    st.subheader("Sensor-Fault Stress Test (Horizon 1, LOEO Mean)")
    fault_df = load_csv_if_exists(Path("reports/resilience/sensor_faults_loeo_1_step_summary.csv"))
    if fault_df is not None:
        st.dataframe(fault_df, use_container_width=True)