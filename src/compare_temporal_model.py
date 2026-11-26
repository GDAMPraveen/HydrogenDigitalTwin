"""Compare baseline and temporal Gradient Boosting models on Test 4.

Uses the same chronological 80/20 split as the original benchmark.
Keeps Test 2 and Test 3 reserved for independent evaluation.
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "test4_subset.csv"
TEMPORAL_PATH = ROOT / "data" / "test4_temporal_features.csv"
OUT_DIR = ROOT / "data" / "benchmark" / "temporal_experiment"
MODEL_DIR = ROOT / "models"

TARGET = "CV-mean"
BASE_FEATURES = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1"]


def make_model():
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        loss="squared_error",
        random_state=42,
    )


def calculate_metrics(y_true, y_pred, current, label):
    result = {
        "model": label,
        "rows": len(y_true),
        "MAE_V": mean_absolute_error(y_true, y_pred),
        "RMSE_V": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }

    masks = {
        "low_current_below_0_5A": current < 0.5,
        "intermediate_current_0_5_to_3A": (current >= 0.5) & (current < 3),
        "high_current_at_least_3A": current >= 3,
    }

    for name, mask in masks.items():
        result[f"{name}_rows"] = int(mask.sum())
        result[f"{name}_MAE_V"] = (
            mean_absolute_error(y_true[mask], y_pred[mask])
            if mask.any() else np.nan
        )

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_PATH).dropna(
        subset=BASE_FEATURES + [TARGET]
    ).reset_index(drop=True)

    temporal = pd.read_csv(TEMPORAL_PATH)

    # Verify that temporal features align with the corresponding raw rows.
    # The temporal builder removes the first five rows due to lag/rolling NaNs.
    expected = raw.iloc[6:].reset_index(drop=True)

    if len(temporal) != len(expected):
        raise ValueError(
            f"Row mismatch: temporal={len(temporal)}, expected={len(expected)}"
        )

    for col in BASE_FEATURES + [TARGET]:
        if not np.allclose(
            temporal[col].to_numpy(dtype=float),
            expected[col].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise ValueError(f"Temporal/raw row alignment failed for {col}")

    # Preserve the original raw-data split boundary.
    split = int(len(raw) * 0.8)
    temporal_train_end = split - 6

    baseline_train = raw.iloc[:split]
    baseline_val = raw.iloc[split:]

    temporal_train = temporal.iloc[:temporal_train_end]
    temporal_val = temporal.iloc[temporal_train_end:]

    if len(temporal_val) != len(baseline_val):
        raise ValueError(
            "Validation sizes differ; refusing to make an unfair comparison."
        )

    # Ensure both models predict exactly the same target rows.
    if not np.allclose(
        temporal_val[TARGET].to_numpy(dtype=float),
        baseline_val[TARGET].to_numpy(dtype=float),
    ):
        raise ValueError("Validation targets do not match.")

    excluded = {TARGET}
    temporal_features = [
        col for col in temporal.columns if col not in excluded
    ]

    X_base_train = baseline_train[BASE_FEATURES]
    X_base_val = baseline_val[BASE_FEATURES]
    X_temp_train = temporal_train[temporal_features]
    X_temp_val = temporal_val[temporal_features]

    y_train_base = baseline_train[TARGET].astype(float)
    y_val = baseline_val[TARGET].astype(float)
    y_train_temp = temporal_train[TARGET].astype(float)

    print(f"Raw rows: {len(raw):,}")
    print(f"Baseline training rows: {len(X_base_train):,}")
    print(f"Temporal training rows: {len(X_temp_train):,}")
    print(f"Shared validation rows: {len(X_base_val):,}")
    print("Split: chronological; Test 2 and Test 3 remain untouched.")

    results = []
    predictions = pd.DataFrame({
        "raw_row_index": baseline_val.index,
        "current_A": baseline_val["PS-I-MON"].to_numpy(),
        "measured_voltage_V": y_val.to_numpy(),
    })

    for label, X_train, y_train, X_val, features in [
        ("GradientBoosting_baseline", X_base_train, y_train_base,
         X_base_val, BASE_FEATURES),
        ("GradientBoosting_temporal", X_temp_train, y_train_temp,
         X_temp_val, temporal_features),
    ]:
        print(f"\nTraining {label}...", flush=True)
        model = make_model()
        started = time.perf_counter()
        model.fit(X_train, y_train)
        seconds = time.perf_counter() - started
        pred = model.predict(X_val)

        results.append({
            **calculate_metrics(
                y_val.to_numpy(),
                pred,
                baseline_val["PS-I-MON"].to_numpy(),
                label,
            ),
            "training_seconds": round(seconds, 3),
            "training_features": len(features),
            "training_rows": len(X_train),
        })

        predictions[f"{label}_predicted_V"] = pred
        predictions[f"{label}_error_V"] = y_val.to_numpy() - pred

        model_path = MODEL_DIR / f"{label.lower()}.joblib"
        joblib.dump(
            {"model": model, "features": features},
            model_path,
        )
        print(f"MAE: {results[-1]['MAE_V']:.6f} V")
        print(f"RMSE: {results[-1]['RMSE_V']:.6f} V")
        print(f"R²: {results[-1]['R2']:.6f}")
        print(f"Saved model: {model_path}")

    metrics_df = pd.DataFrame(results)
    metrics_path = OUT_DIR / "metrics.csv"
    predictions_path = OUT_DIR / "validation_predictions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("\nComparison summary:")
    print(metrics_df[
        ["model", "MAE_V", "RMSE_V", "R2",
         "low_current_below_0_5A_MAE_V",
         "high_current_at_least_3A_MAE_V"]
    ].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
