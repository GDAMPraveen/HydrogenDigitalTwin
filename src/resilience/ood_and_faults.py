"""Out-of-distribution (OOD) detection and controlled sensor-fault injection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class OODDetector:
    """Unsupervised Isolation Forest detector for unfamiliar operating regimes."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.pipeline_: Pipeline | None = None

    def fit(self, X_train: pd.DataFrame) -> "OODDetector":
        self.pipeline_ = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "iforest",
                IsolationForest(
                    n_estimators=150,
                    contamination=self.contamination,
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
            ),
        ])
        self.pipeline_.fit(X_train)
        return self

    def predict_ood(self, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (is_ood_bool_array, anomaly_score_array where higher = more anomalous)."""
        if self.pipeline_ is None:
            raise RuntimeError("OODDetector must be fitted before predicting.")
        labels = self.pipeline_.predict(X_test)  # -1 = OOD/anomaly, 1 = normal
        scores = -self.pipeline_.decision_function(X_test)
        return (labels == -1), scores


def inject_sensor_fault(
    X: pd.DataFrame,
    sensor_col: str,
    fault_type: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject a controlled sensor fault into a copy of X without altering other columns."""
    if sensor_col not in X.columns:
        raise ValueError(f"Sensor column not found: {sensor_col}")

    corrupted = X.copy()
    values = corrupted[sensor_col].to_numpy(dtype=float, copy=True)
    std = float(np.nanstd(values))
    if std == 0.0 or np.isnan(std):
        std = 1.0

    n = len(values)

    if fault_type == "clean":
        return corrupted
    elif fault_type == "missing_20pct":
        mask = rng.random(n) < 0.20
        values[mask] = np.nan
    elif fault_type == "gaussian_noise":
        values = values + rng.normal(0.0, 0.15 * std, size=n)
    elif fault_type == "stuck_sensor":
        first_valid = values[np.isfinite(values)][0] if np.any(np.isfinite(values)) else 0.0
        values[:] = first_valid
    elif fault_type == "linear_drift":
        values = values + np.linspace(0.0, 0.30 * std, num=n)
    else:
        raise ValueError(f"Unknown fault_type: {fault_type}")

    corrupted[sensor_col] = values
    return corrupted