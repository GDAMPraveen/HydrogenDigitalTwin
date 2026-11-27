"""End-to-end unit tests for the synchronized Digital Twin Orchestrator and CQR."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.integration.hybrid_digital_twin import (
    CURRENT_COL,
    CURRENT_CMD_COL,
    VOLTAGE_COL,
    HybridPEMForecaster,
)
from src.uncertainty.cqr import ConformalizedQuantileForecaster
from src.resilience.ood_and_faults import OODDetector
from src.digital_twin.twin_frame import TwinFrame
from src.digital_twin.orchestrator import DigitalTwinOrchestrator


def _make_synthetic_pem_data(n_rows: int = 240):
    rng = np.random.default_rng(42)
    current = rng.uniform(100.0, 500.0, size=n_rows)
    voltage = 40.0 + 0.02 * current + rng.normal(0, 0.2, size=n_rows)
    # Approximate Faraday scaling (~0.0075 kg/hr per A for ~200 cells)
    flow_now = np.clip(0.0075 * current + rng.normal(0, 0.15, size=n_rows), 0.05, None)
    flow_next = np.clip(flow_now + rng.normal(0, 0.10, size=n_rows), 0.05, None)

    X = pd.DataFrame({
        CURRENT_COL: current,
        CURRENT_CMD_COL: current + rng.normal(0, 2.0, size=n_rows),
        VOLTAGE_COL: voltage,
        "IVAL_f_FM011_Flow_current": flow_now,
    })
    y = pd.Series(flow_next, name="target")
    meta = pd.DataFrame({
        "Experiment": ["exp_A"] * (n_rows // 2) + ["exp_B"] * (n_rows // 2)
    })
    return X, y, meta


def _build_orchestrator():
    X, y, meta = _make_synthetic_pem_data()
    current_col = "IVAL_f_FM011_Flow_current"
    base_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("ridge", Ridge())])

    cqr = ConformalizedQuantileForecaster(
        point_forecaster_factory=lambda: HybridPEMForecaster(
            base_model, mode="physics_residual", current_flow_col=current_col
        ),
        current_flow_col=current_col,
        alpha=0.10,
        cal_fraction=0.20,
    ).fit(X, y, meta)

    ood = OODDetector(contamination=0.05, random_state=42).fit(X)

    orchestrator = DigitalTwinOrchestrator(
        conformal_forecaster=cqr,
        ood_detector=ood,
        feature_columns=list(X.columns),
        current_flow_col=current_col,
        horizon_steps=1,
        feature_means=X.mean().to_dict(),
        feature_stds=X.std().to_dict(),
    )
    return orchestrator, X


def test_twin_frame_sequential_synchronization():
    orchestrator, X = _build_orchestrator()
    frame = TwinFrame.create(step_index=0, experiment="exp_test", raw_sensors=X.iloc[0].to_dict())

    processed = orchestrator.process_frame(frame)

    assert processed.pipeline_trace == ["physics", "forecast", "resilience", "uncertainty", "xai"]
    assert processed.forecast.predicted_flow_kg_hr >= 0.0
    assert processed.uncertainty.lower_bound_kg_hr >= 0.0
    assert processed.uncertainty.upper_bound_kg_hr >= processed.uncertainty.lower_bound_kg_hr
    assert processed.frame_id in processed.xai.explanation_summary


def test_missing_current_sensor_triggers_fault_status():
    orchestrator, X = _build_orchestrator()
    bad_sensors = X.iloc[0].to_dict()
    bad_sensors[CURRENT_COL] = np.nan

    frame = TwinFrame.create(step_index=1, experiment="exp_test", raw_sensors=bad_sensors)
    processed = orchestrator.process_frame(frame)

    assert processed.resilience.status == "FAULT_DETECTED"
    assert "MISSING_STACK_CURRENT" in processed.resilience.sensor_faults_detected
    assert "MISSING_STACK_CURRENT" in processed.xai.explanation_summary


def test_stuck_current_sensor_detected_after_10_identical_readings():
    orchestrator, X = _build_orchestrator()
    stuck_row = X.iloc[0].to_dict()

    last_frame = None
    for step in range(12):
        frame = TwinFrame.create(step_index=step, experiment="exp_test", raw_sensors=stuck_row)
        last_frame = orchestrator.process_frame(frame)

    assert last_frame is not None
    assert "STUCK_STACK_CURRENT" in last_frame.resilience.sensor_faults_detected
    assert last_frame.resilience.status == "FAULT_DETECTED"