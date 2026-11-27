"""Physics-informed and residual-hybrid forecasting wrapper for PEM electrolysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.physics.faraday_law import calculate_per_cell_ideal_rate


CURRENT_COL = "H2E_f_PSU_A_Current"
CURRENT_CMD_COL = "H2E_f_NLR_CurrentCmd (A)"
VOLTAGE_COL = "Power Supply Average Voltage (Vdc)"


class FaradayPhysicsCalibrator:
    """Calibrates effective stack factor k_eff = N_cells * eta_F on training data only."""

    def __init__(self, min_current_a: float = 50.0):
        self.min_current_a = min_current_a
        self.k_eff_: float | None = None

    def fit(self, X_train: pd.DataFrame, current_flow_train: pd.Series) -> "FaradayPhysicsCalibrator":
        current = np.clip(X_train[CURRENT_COL].fillna(0.0).to_numpy(dtype=float), 0.0, None)
        flow = np.clip(current_flow_train.fillna(0.0).to_numpy(dtype=float), 0.0, None)

        ideal_per_cell = np.array([calculate_per_cell_ideal_rate(c) for c in current], dtype=float)
        mask = (current >= self.min_current_a) & (ideal_per_cell > 0) & (flow > 0)

        if not np.any(mask):
            raise ValueError("No valid positive-current rows available to calibrate Faraday physics.")

        # Least-squares through the origin: flow ≈ k_eff * ideal_per_cell
        denom = float(np.dot(ideal_per_cell[mask], ideal_per_cell[mask]))
        self.k_eff_ = float(np.dot(ideal_per_cell[mask], flow[mask]) / denom)
        return self

    def predict_faraday_flow(self, X: pd.DataFrame) -> np.ndarray:
        if self.k_eff_ is None:
            raise RuntimeError("Calibrator must be fitted before predicting.")
        current = np.clip(X[CURRENT_COL].fillna(0.0).to_numpy(dtype=float), 0.0, None)
        ideal_per_cell = np.array([calculate_per_cell_ideal_rate(c) for c in current], dtype=float)
        return np.clip(self.k_eff_ * ideal_per_cell, 0.0, None)

    def augment_features(self, X: pd.DataFrame, current_flow_col: str) -> pd.DataFrame:
        out = X.copy()
        faraday_flow = self.predict_faraday_flow(out)
        current_a = np.clip(out[CURRENT_COL].fillna(0.0).to_numpy(dtype=float), 0.0, None)
        voltage_v = np.clip(out[VOLTAGE_COL].fillna(0.0).to_numpy(dtype=float), 0.0, None)
        current_flow = out[current_flow_col].fillna(0.0).to_numpy(dtype=float)

        out["phys_faraday_flow_kg_hr"] = faraday_flow
        out["phys_dc_power_kw"] = (voltage_v * current_a) / 1000.0
        out["phys_flow_residual_kg_hr"] = current_flow - faraday_flow

        if CURRENT_CMD_COL in out.columns:
            cmd_a = out[CURRENT_CMD_COL].fillna(0.0).to_numpy(dtype=float)
            out["phys_current_cmd_gap_a"] = cmd_a - current_a

        return out


class HybridPEMForecaster:
    """Unified wrapper for Pure AI, Physics-Informed AI, and Physics-Residual Hybrid."""

    def __init__(self, base_model, mode: str, current_flow_col: str):
        if mode not in {"pure_ai", "physics_informed", "physics_residual"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.base_model = base_model
        self.mode = mode
        self.current_flow_col = current_flow_col
        self.calibrator_: FaradayPhysicsCalibrator | None = None
        self.model_ = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "HybridPEMForecaster":
        self.model_ = clone(self.base_model)

        if self.mode == "pure_ai":
            self.model_.fit(X_train, y_train)
            return self

        self.calibrator_ = FaradayPhysicsCalibrator().fit(X_train, X_train[self.current_flow_col])
        X_aug = self.calibrator_.augment_features(X_train, self.current_flow_col)

        if self.mode == "physics_informed":
            self.model_.fit(X_aug, y_train)
        elif self.mode == "physics_residual":
            faraday_base = X_aug["phys_faraday_flow_kg_hr"].to_numpy()
            y_residual = y_train.to_numpy(dtype=float) - faraday_base
            self.model_.fit(X_aug, y_residual)

        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predicting.")

        if self.mode == "pure_ai":
            return np.asarray(self.model_.predict(X_test), dtype=float)

        assert self.calibrator_ is not None
        X_aug = self.calibrator_.augment_features(X_test, self.current_flow_col)

        if self.mode == "physics_informed":
            pred = np.asarray(self.model_.predict(X_aug), dtype=float)
            return np.clip(pred, 0.0, None)

        # physics_residual: Faraday baseline + AI residual, physically bounded >= 0
        faraday_base = X_aug["phys_faraday_flow_kg_hr"].to_numpy()
        pred_residual = np.asarray(self.model_.predict(X_aug), dtype=float)
        return np.clip(faraday_base + pred_residual, 0.0, None)