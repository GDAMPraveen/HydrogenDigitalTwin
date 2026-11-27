"""Split Conformal Prediction wrapper for physics-informed PEM forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_conformal_quantile(residuals: np.ndarray, alpha: float = 0.10) -> float:
    """Finite-sample conformal quantile for target coverage (1 - alpha)."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    abs_res = np.abs(np.asarray(residuals, dtype=float))
    n = len(abs_res)
    if n == 0:
        raise ValueError("Calibration residuals cannot be empty.")
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(abs_res, level, method="higher"))


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> dict[str, float]:
    """Evaluate empirical coverage, sharpness (width), and Winkler interval score."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)

    covered = (y >= lo) & (y <= hi)
    width = hi - lo

    # Winkler score: penalizes width + (2/alpha) * miss distance outside interval
    below_penalty = (2.0 / alpha) * np.maximum(lo - y, 0.0)
    above_penalty = (2.0 / alpha) * np.maximum(y - hi, 0.0)
    winkler = width + below_penalty + above_penalty

    return {
        "Coverage_Percent": float(np.mean(covered) * 100.0),
        "Avg_Width_kg_hr": float(np.mean(width)),
        "Winkler_Score": float(np.mean(winkler)),
    }


def chronological_tail_split_by_group(
    meta: pd.DataFrame,
    group_col: str = "Experiment",
    cal_fraction: float = 0.20,
) -> tuple[pd.Index, pd.Index]:
    """Split each training experiment into (fit_idx, cal_idx) preserving time order."""
    fit_indices = []
    cal_indices = []
    for _, group_df in meta.groupby(group_col, sort=False):
        n = len(group_df)
        split_at = max(1, int(n * (1.0 - cal_fraction)))
        fit_indices.append(group_df.index[:split_at])
        cal_indices.append(group_df.index[split_at:])
    return fit_indices[0].append(fit_indices[1:]), cal_indices[0].append(cal_indices[1:])


class SplitConformalForecaster:
    """Wraps a forecaster to produce physically bounded conformal prediction intervals."""

    def __init__(self, forecaster_factory, alpha: float = 0.10, cal_fraction: float = 0.20):
        self.forecaster_factory = forecaster_factory
        self.alpha = alpha
        self.cal_fraction = cal_fraction
        self.forecaster_ = None
        self.quantile_: float | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        meta_train: pd.DataFrame,
        group_col: str = "Experiment",
    ) -> "SplitConformalForecaster":
        fit_idx, cal_idx = chronological_tail_split_by_group(
            meta_train, group_col=group_col, cal_fraction=self.cal_fraction
        )

        self.forecaster_ = self.forecaster_factory()
        self.forecaster_.fit(X_train.loc[fit_idx], y_train.loc[fit_idx])

        cal_pred = self.forecaster_.predict(X_train.loc[cal_idx])
        cal_residuals = y_train.loc[cal_idx].to_numpy(dtype=float) - cal_pred
        self.quantile_ = compute_conformal_quantile(cal_residuals, alpha=self.alpha)
        return self

    def predict_interval(self, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.forecaster_ is None or self.quantile_ is None:
            raise RuntimeError("SplitConformalForecaster must be fitted before predicting.")

        point_pred = np.asarray(self.forecaster_.predict(X_test), dtype=float)
        lower = np.clip(point_pred - self.quantile_, 0.0, None)
        upper = point_pred + self.quantile_
        return point_pred, lower, upper