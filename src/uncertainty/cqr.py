"""Conformalized Quantile Regression (CQR) for adaptive aleatoric uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.integration.hybrid_digital_twin import HybridPEMForecaster
from src.uncertainty.conformal import chronological_tail_split_by_group


def _make_quantile_pipeline(quantile: float, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "hgb",
            HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                max_iter=200,
                random_state=random_state,
            ),
        ),
    ])


class ConformalizedQuantileForecaster:
    """Adaptive aleatoric uncertainty via Physics-Residual CQR (Romano et al., 2019)."""

    def __init__(
        self,
        point_forecaster_factory,
        current_flow_col: str,
        alpha: float = 0.10,
        cal_fraction: float = 0.20,
        random_state: int = 42,
    ):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.point_forecaster_factory = point_forecaster_factory
        self.current_flow_col = current_flow_col
        self.alpha = alpha
        self.cal_fraction = cal_fraction
        self.random_state = random_state

        self.forecaster_ = None
        self.lower_forecaster_ = None
        self.upper_forecaster_ = None
        self.quantile_: float | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        meta_train: pd.DataFrame,
        group_col: str = "Experiment",
    ) -> "ConformalizedQuantileForecaster":
        fit_idx, cal_idx = chronological_tail_split_by_group(
            meta_train, group_col=group_col, cal_fraction=self.cal_fraction
        )
        X_fit, y_fit = X_train.loc[fit_idx], y_train.loc[fit_idx]
        X_cal, y_cal = X_train.loc[cal_idx], y_train.loc[cal_idx].to_numpy(dtype=float)

        # 1. Point forecaster (Physics-Residual Hybrid)
        self.forecaster_ = self.point_forecaster_factory()
        self.forecaster_.fit(X_fit, y_fit)

        # 2. Lower & Upper Quantile Physics-Residual forecasters
        self.lower_forecaster_ = HybridPEMForecaster(
            _make_quantile_pipeline(self.alpha / 2.0, self.random_state),
            mode="physics_residual",
            current_flow_col=self.current_flow_col,
        ).fit(X_fit, y_fit)

        self.upper_forecaster_ = HybridPEMForecaster(
            _make_quantile_pipeline(1.0 - self.alpha / 2.0, self.random_state),
            mode="physics_residual",
            current_flow_col=self.current_flow_col,
        ).fit(X_fit, y_fit)

        # 3. CQR conformity scores on chronological calibration tail
        q_lo_cal = self.lower_forecaster_.predict(X_cal)
        q_hi_cal = self.upper_forecaster_.predict(X_cal)
        scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)

        n = len(scores)
        level = min(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0)
        self.quantile_ = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_interval(self, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.forecaster_ is None or self.quantile_ is None:
            raise RuntimeError("ConformalizedQuantileForecaster must be fitted before predicting.")

        point_pred = np.asarray(self.forecaster_.predict(X_test), dtype=float)
        q_lo = np.asarray(self.lower_forecaster_.predict(X_test), dtype=float)
        q_hi = np.asarray(self.upper_forecaster_.predict(X_test), dtype=float)

        lower = np.clip(q_lo - self.quantile_, 0.0, None)
        upper = np.maximum(lower, q_hi + self.quantile_)
        return point_pred, lower, upper