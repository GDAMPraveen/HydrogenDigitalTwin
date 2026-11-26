"""Evaluate existing benchmark model configurations on independent Test2.

Models train on the chronological first 80% of Test4.
Test2 is used only for evaluation, not hyperparameter tuning.
"""

from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from benchmark_models import (
    FEATURES,
    TARGET,
    make_models,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "benchmark" / "test2_model_comparison"
MODEL_DIR = ROOT / "models"


def metric_rows(name, y, pred, current):
    groups = {
        "all": np.ones(len(y), dtype=bool),
        "low_current_below_0.5A": current < 0.5,
        "intermediate_current_0.5_to_3A": (
            (current >= 0.5) & (current < 3.0)
        ),
        "high_current_at_least_3A": current >= 3.0,
    }

    results = []
    for group, mask in groups.items():
        if not mask.any():
            continue

        actual = y[mask]
        predicted = pred[mask]
        results.append({
            "model": name,
            "group": group,
            "rows": int(mask.sum()),
            "MAE_V": mean_absolute_error(actual, predicted),
            "RMSE_V": np.sqrt(mean_squared_error(actual, predicted)),
            "bias_measured_minus_predicted_V": float(
                np.mean(actual - predicted)
            ),
            "R2": (
                r2_score(actual, predicted)
                if len(actual) >= 2 and np.var(actual) > 0
                else np.nan
            ),
        })

    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "test4_subset.csv")
    test_df = pd.read_csv(DATA_DIR / "test2_subset.csv")

    required = FEATURES + [TARGET]
    train_df = train_df.dropna(subset=required).reset_index(drop=True)

    missing = [col for col in required if col not in test_df.columns]
    if missing:
        raise ValueError(f"Test2 is missing required columns: {missing}")

    test_df = test_df.copy()
    test_df.insert(0, "raw_row_index", np.arange(len(test_df)))

    split = int(len(train_df) * 0.8)
    train = train_df.iloc[:split].copy()

    # Every model is evaluated on exactly the same eligible Test2 rows.
    eligible = test_df[required].notna().all(axis=1)
    test = test_df.loc[eligible].copy()

    X_train = train[FEATURES].astype(float)
    y_train = train[TARGET].astype(float)
    X_test = test[FEATURES].astype(float)
    y_test = test[TARGET].astype(float).to_numpy()
    current = test["PS-I-MON"].astype(float).to_numpy()

    print(f"Test4 rows: {len(train_df):,}")
    print(f"Training rows (chronological first 80%): {len(train):,}")
    print(f"Test2 raw rows: {len(test_df):,}")
    print(f"Shared eligible Test2 rows: {len(test):,}")
    print(f"Excluded Test2 rows: {len(test_df) - len(test):,}")
    print("\nTraining benchmark models on Test4; evaluating on Test2...\n")

    results = []
    prediction_frame = pd.DataFrame({
        "raw_row_index": test["raw_row_index"].to_numpy(),
        "current_A": current,
        "measured_voltage_V": y_test,
    })

    models = make_models()

    # Match the original benchmark's reproducible sampling rule for SVR.
    svr_limit = 12000
    svr_indices = np.linspace(
        0, len(X_train) - 1, min(svr_limit, len(X_train)), dtype=int
    )

    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        start = time.perf_counter()

        try:
            if name == "SVR_sampled":
                model.fit(X_train.iloc[svr_indices], y_train.iloc[svr_indices])
            else:
                model.fit(X_train, y_train)

            seconds = time.perf_counter() - start
            pred = np.asarray(model.predict(X_test), dtype=float)

            results.extend(metric_rows(name, y_test, pred, current))
            prediction_frame[f"{name}_predicted_V"] = pred
            prediction_frame[f"{name}_error_V"] = y_test - pred
            prediction_frame[f"{name}_absolute_error_V"] = np.abs(y_test - pred)

            joblib.dump(model, MODEL_DIR / f"test2_eval_{name.lower()}.joblib")
            print(
                f"  Overall MAE={mean_absolute_error(y_test, pred):.6f} V; "
                f"low-current MAE="
                f"{mean_absolute_error(y_test[current < 0.5], pred[current < 0.5]):.6f} V; "
                f"time={seconds:.1f}s",
                flush=True,
            )

        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            results.append({
                "model": name,
                "group": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })

    metrics = pd.DataFrame(results)
    metrics_path = OUT_DIR / "test2_metrics.csv"
    predictions_path = OUT_DIR / "test2_predictions.csv"

    metrics.to_csv(metrics_path, index=False)
    prediction_frame.to_csv(predictions_path, index=False)

    print("\nOverall Test2 ranking by MAE:")
    overall = metrics[metrics["group"] == "all"].dropna(
        subset=["MAE_V"]
    ).sort_values("MAE_V")

    if not overall.empty:
        print(overall[
            ["model", "rows", "MAE_V", "RMSE_V",
             "bias_measured_minus_predicted_V", "R2"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nLow-current Test2 ranking by MAE:")
    low = metrics[metrics["group"] == "low_current_below_0.5A"].dropna(
        subset=["MAE_V"]
    ).sort_values("MAE_V")

    if not low.empty:
        print(low[
            ["model", "rows", "MAE_V", "RMSE_V",
             "bias_measured_minus_predicted_V", "R2"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
