"""Compare Test4-trained baseline and temporal models on the same Test2 rows.

Both models are trained on the chronological first 80% of Test4.
Evaluation uses eligible Test2 rows only, with no retraining on Test2.
Temporal features use Test2's own measured history (one-step evaluation).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "benchmark" / "temporal_experiment" / "test2_model_comparison"

FEATURES = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1"]
TARGET = "CV-mean"
REQUIRED = FEATURES + [TARGET]


def make_model():
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        loss="squared_error",
        random_state=42,
    )


def add_temporal_features(df):
    data = df[REQUIRED].copy()

    for col in FEATURES + [TARGET]:
        data[f"{col}_lag1"] = data[col].shift(1)
        data[f"{col}_lag5"] = data[col].shift(5)

    data["current_change_5rows"] = (
        data["PS-I-MON"].shift(1) - data["PS-I-MON"].shift(6)
    )
    data["voltage_change_5rows"] = (
        data[TARGET].shift(1) - data[TARGET].shift(6)
    )
    data["temperature_change_5rows"] = (
        data["T-ELY-CH1"].shift(1) - data["T-ELY-CH1"].shift(6)
    )
    data["voltage_mean_5rows"] = (
        data[TARGET].shift(1).rolling(window=5, min_periods=5).mean()
    )
    data["current_mean_5rows"] = (
        data["PS-I-MON"].rolling(window=5, min_periods=5).mean()
    )

    return data


def report_metrics(name, y, pred, current):
    masks = {
        "all": np.ones(len(y), dtype=bool),
        "low_current_below_0.5A": current < 0.5,
        "intermediate_current_0.5_to_3A": (current >= 0.5) & (current < 3),
        "high_current_at_least_3A": current >= 3,
    }

    rows = []
    for group, mask in masks.items():
        if not mask.any():
            continue
        actual = y[mask]
        predicted = pred[mask]
        rows.append({
            "model": name,
            "group": group,
            "rows": int(mask.sum()),
            "MAE_V": mean_absolute_error(actual, predicted),
            "RMSE_V": np.sqrt(mean_squared_error(actual, predicted)),
            "bias_measured_minus_predicted_V": float(np.mean(actual - predicted)),
            "R2": (
                r2_score(actual, predicted)
                if len(actual) >= 2 and np.var(actual) > 0
                else np.nan
            ),
        })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "test4_subset.csv")
    test2 = pd.read_csv(DATA_DIR / "test2_subset.csv")

    train = train.dropna(subset=REQUIRED).reset_index(drop=True)
    test2 = test2[REQUIRED].copy()
    test2.insert(0, "raw_row_index", np.arange(len(test2)))

    split = int(len(train) * 0.8)
    train_base = train.iloc[:split].copy()

    temporal_train = add_temporal_features(train)
    temporal_train = temporal_train.iloc[:split]
    temporal_train = temporal_train.dropna(
        subset=[c for c in temporal_train.columns if c != TARGET]
        + [TARGET]
    )

    temporal_test = add_temporal_features(test2[REQUIRED])
    temporal_test.insert(0, "raw_row_index", np.arange(len(test2)))

    base_eligible = test2[REQUIRED].notna().all(axis=1)
    temporal_features = [
        c for c in temporal_test.columns
        if c not in [TARGET, "raw_row_index"]
    ]
    temporal_eligible = temporal_test[
        temporal_features + [TARGET]
    ].notna().all(axis=1)

    # Enforce identical eligible rows for both model predictions.
    eligible = base_eligible & temporal_eligible
    test = test2.loc[eligible].copy()
    temporal_eval = temporal_test.loc[eligible].copy()

    if test.empty:
        raise ValueError("No shared eligible Test2 rows.")

    base_model = make_model()
    temp_model = make_model()

    base_model.fit(train_base[FEATURES], train_base[TARGET].astype(float))
    temp_model.fit(
        temporal_train[temporal_features],
        temporal_train[TARGET].astype(float),
    )

    y = test[TARGET].astype(float).to_numpy()
    current = test["PS-I-MON"].astype(float).to_numpy()

    pred_base = base_model.predict(test[FEATURES].astype(float))
    pred_temp = temp_model.predict(
        temporal_eval[temporal_features].astype(float)
    )

    predictions = pd.DataFrame({
        "raw_row_index": test["raw_row_index"].to_numpy(),
        "current_A": current,
        "measured_voltage_V": y,
        "baseline_predicted_V": pred_base,
        "baseline_error_V": y - pred_base,
        "temporal_predicted_V": pred_temp,
        "temporal_error_V": y - pred_temp,
        "baseline_absolute_error_V": np.abs(y - pred_base),
        "temporal_absolute_error_V": np.abs(y - pred_temp),
    })

    all_results = []
    all_results.extend(report_metrics("baseline", y, pred_base, current))
    all_results.extend(report_metrics("temporal", y, pred_temp, current))
    metrics = pd.DataFrame(all_results)

    predictions_path = OUT_DIR / "test2_predictions.csv"
    metrics_path = OUT_DIR / "test2_metrics.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    print(f"Test4 training rows: {len(train):,}")
    print(f"Baseline training rows: {len(train_base):,}")
    print(f"Temporal training rows: {len(temporal_train):,}")
    print(f"Test2 raw rows: {len(test2):,}")
    print(f"Shared eligible Test2 rows: {len(test):,}")
    print(f"Excluded rows: {len(test2) - len(test):,}")
    print("\nTest2 metrics:")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    low = predictions[predictions["current_A"] < 0.5]
    print("\nLow-current absolute-error comparison:")
    print(f"Rows: {len(low):,}")
    print(f"Baseline MAE: {low['baseline_absolute_error_V'].mean():.6f} V")
    print(f"Temporal MAE: {low['temporal_absolute_error_V'].mean():.6f} V")
    baseline_low_mae = low["baseline_absolute_error_V"].mean()
    temporal_low_mae = low["temporal_absolute_error_V"].mean()

    change_pct = 100 * (temporal_low_mae / baseline_low_mae - 1)

    print(f"Temporal MAE change vs baseline: {change_pct:.2f}%")

    print(f"\nSaved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
