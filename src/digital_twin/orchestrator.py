"""Sequential Digital Twin Orchestrator enforcing tagged, lockstep layer execution."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.integration.hybrid_digital_twin import (
    CURRENT_COL,
    CURRENT_CMD_COL,
    VOLTAGE_COL,
    HybridPEMForecaster,
)
from src.uncertainty.conformal import SplitConformalForecaster
from src.resilience.ood_and_faults import OODDetector
from src.digital_twin.twin_frame import (
    ForecastSnapshot,
    PhysicsSnapshot,
    ResilienceSnapshot,
    TwinFrame,
    UncertaintySnapshot,
    XAISnapshot,
)


class DigitalTwinOrchestrator:
    """Runs a single tagged TwinFrame sequentially through Physics -> AI -> Uncertainty -> Resilience -> XAI."""

    def __init__(
        self,
        conformal_forecaster: SplitConformalForecaster,
        ood_detector: OODDetector,
        feature_columns: list[str],
        current_flow_col: str,
        horizon_steps: int = 1,
        feature_means: dict[str, float] | None = None,
        feature_stds: dict[str, float] | None = None,
    ):
        self.conformal = conformal_forecaster
        self.ood = ood_detector
        self.feature_columns = list(feature_columns)
        self.current_flow_col = current_flow_col
        self.horizon_steps = horizon_steps
        self.feature_means = feature_means or {}
        self.feature_stds = feature_stds or {}
        self._last_current_a: float | None = None
        self._stuck_count: int = 0

    def _detect_live_sensor_faults(self, row: pd.Series) -> list[str]:
        faults = []
        current_val = row.get(CURRENT_COL, np.nan)
        voltage_val = row.get(VOLTAGE_COL, np.nan)

        if pd.isna(current_val):
            faults.append("MISSING_STACK_CURRENT")
        else:
            if self._last_current_a is not None and abs(float(current_val) - self._last_current_a) < 1e-6:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
            self._last_current_a = float(current_val)
            if self._stuck_count >= 10:
                faults.append("STUCK_STACK_CURRENT")

        if pd.isna(voltage_val):
            faults.append("MISSING_STACK_VOLTAGE")

        return faults

    def _compute_local_z_drivers(self, row: pd.Series, top_k: int = 3) -> dict[str, float]:
        """Identify which sensors deviate most from normal training distribution for this exact frame."""
        z_scores = {}
        for col, mean_val in self.feature_means.items():
            std_val = self.feature_stds.get(col, 0.0)
            val = row.get(col, np.nan)
            if pd.notna(val) and std_val > 1e-6:
                z_scores[col] = round(float((val - mean_val) / std_val), 2)
        ranked = sorted(z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        return dict(ranked)

    def process_frame(self, frame: TwinFrame) -> TwinFrame:
        """Process one tagged frame through all 5 layers in strict sequential order."""
        row_df = pd.DataFrame([frame.raw_sensors])[self.feature_columns]
        hybrid: HybridPEMForecaster = self.conformal.forecaster_
        assert hybrid is not None and hybrid.calibrator_ is not None

        # 1. PHYSICS LAYER
        aug_df = hybrid.calibrator_.augment_features(row_df, self.current_flow_col)
        faraday_flow = float(aug_df["phys_faraday_flow_kg_hr"].iloc[0])
        dc_power_kw = float(aug_df["phys_dc_power_kw"].iloc[0])
        live_res = float(aug_df["phys_flow_residual_kg_hr"].iloc[0])
        cmd_gap = (
            float(aug_df["phys_current_cmd_gap_a"].iloc[0])
            if "phys_current_cmd_gap_a" in aug_df.columns
            else None
        )
        frame.physics = PhysicsSnapshot(
            faraday_flow_kg_hr=faraday_flow,
            dc_power_kw=dc_power_kw,
            live_flow_residual_kg_hr=live_res,
            current_cmd_gap_a=cmd_gap,
        )
        frame.mark_stage("physics")

        # 2. FORECASTING LAYER
        pred_arr, lower_arr, upper_arr = self.conformal.predict_interval(row_df)
        pred_val = float(pred_arr[0])
        frame.forecast = ForecastSnapshot(
            horizon_steps=self.horizon_steps,
            predicted_flow_kg_hr=pred_val,
            faraday_baseline_kg_hr=faraday_flow,
            ai_residual_kg_hr=pred_val - faraday_flow,
        )
        frame.mark_stage("forecast")

        # 3. RESILIENCE & OOD LAYER
        is_ood_arr, score_arr = self.ood.predict_ood(row_df)
        is_ood = bool(is_ood_arr[0])
        anom_score = float(score_arr[0])
        faults = self._detect_live_sensor_faults(row_df.iloc[0])

        if faults:
            status = "FAULT_DETECTED"
        elif is_ood:
            status = "WARNING_OOD"
        else:
            status = "NORMAL"

        frame.resilience = ResilienceSnapshot(
            is_ood=is_ood,
            anomaly_score=anom_score,
            sensor_faults_detected=faults,
            status=status,
        )
        frame.mark_stage("resilience")

        # 4. UNCERTAINTY LAYER (Aleatoric conformal width + Epistemic OOD score)
        lo, hi = float(lower_arr[0]), float(upper_arr[0])
        frame.uncertainty = UncertaintySnapshot(
            lower_bound_kg_hr=lo,
            upper_bound_kg_hr=hi,
            interval_width_kg_hr=hi - lo,
            aleatoric_half_width_kg_hr=float(self.conformal.quantile_ or 0.0),
            epistemic_score=anom_score,
            confidence_level=1.0 - self.conformal.alpha,
        )
        frame.mark_stage("uncertainty")

        # 5. LOCAL XAI LAYER (Synchronized to this exact frame_id)
        top_drivers = self._compute_local_z_drivers(row_df.iloc[0], top_k=3)
        if status == "FAULT_DETECTED":
            summary = f"[{frame.frame_id}] Sensor fault(s): {', '.join(faults)}."
        elif status == "WARNING_OOD":
            driver_str = ", ".join(f"{k} ({v:+.1f}σ)" for k, v in top_drivers.items())
            summary = f"[{frame.frame_id}] OOD regime driven by: {driver_str}."
        else:
            summary = (
                f"[{frame.frame_id}] Normal operation. Faraday={faraday_flow:.2f} kg/hr, "
                f"AI residual={pred_val - faraday_flow:+.2f} kg/hr."
            )
        frame.xai = XAISnapshot(top_drivers=top_drivers, explanation_summary=summary)
        frame.mark_stage("xai")

        frame.assert_synchronized()
        return frame