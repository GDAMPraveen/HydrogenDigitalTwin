"""Evaluate the Test 4 temporal model independently on Test 2 and Test 3.

Uses each experiment's own historical measurements for lag features.
This is one-step evaluation, not recursive multi-step forecasting.
No interpolation or model retraining is performed.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "gradientboosting_temporal.joblib"
OUT_DIR = (
    ROOT / "data" / "benchmark" / "temporal_experiment" / "ood"
)

FEATURES = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1"]
TARGET = "CV-mean"
REQUIRED_COLUMNS = FEATURES + [TARGET]


def make_temporal_features(df):
    """Build exactly the temporal features used by the Test 4 experiment."""
    data = df[REQUIRED_COLUMNS].copy()

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


def calculate_metrics(y_true, y_pred, current, label):
    result = {
        "experiment": label,
        "evaluated_rows": len(y_true),
        "MAE_V": mean_absolute_error(y_true, y_pred),
        "RMSE_V": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": (
            r2_score(y_true, y_pred)
            if len(y_true) >= 2
            else np.nan
        ),
    }

    ranges = {
        "low_current_below_0_5A": current < 0.5,
        "intermediate_current_0_5_to_3A": (
            (current >= 0.5) & (current < 3)
        ),
        "high_current_at_least_3A": current >= 3,
    }

    for name, mask in ranges.items():
        result[f"{name}_rows"] = int(mask.sum())
        result[f"{name}_MAE_V"] = (
            mean_absolute_error(y_true[mask], y_pred[mask])
            if mask.any()
            else np.nan
        )

    return result


def evaluate_experiment(test_name, model, model_features):
    path = DATA_DIR / f"{test_name.lower()}_subset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = pd.read_csv(path)
    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in raw.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{test_name} is missing required columns: {missing_columns}"
        )

    # Keep the original row index so exclusions and predictions are traceable.
    data = make_temporal_features(raw)
    data.insert(0, "raw_row_index", np.arange(len(raw)))

    # Match the exact feature order stored with the trained model.
    absent_features = [
        col for col in model_features if col not in data.columns
    ]
    if absent_features:
        raise ValueError(
            f"Features required by the saved model are absent: {absent_features}"
        )

    eligible = data[model_features + [TARGET]].notna().all(axis=1)
    evaluation = data.loc[eligible].copy()

    if evaluation.empty:
        raise ValueError(f"No eligible rows to evaluate in {test_name}.")

    X = evaluation[model_features].astype(float)
    y_true = evaluation[TARGET].astype(float).to_numpy()
    current = evaluation["PS-I-MON"].astype(float).to_numpy()
    y_pred = model.predict(X)

    metrics = calculate_metrics(
        y_true, y_pred, current, test_name
    )
    metrics["raw_rows"] = len(raw)
    metrics["excluded_rows"] = int(len(raw) - len(evaluation))
    metrics["first_evaluated_raw_row"] = int(
        evaluation["raw_row_index"].iloc[0]
    )
    metrics["last_evaluated_raw_row"] = int(
        evaluation["raw_row_index"].iloc[-1]
    )

    predictions = pd.DataFrame({
        "experiment": test_name,
        "raw_row_index": evaluation["raw_row_index"].to_numpy(),
        "current_A": current,
        "hydrogen_pressure_bar": evaluation["H-P1"].to_numpy(),
        "oxygen_pressure_bar": evaluation["O-P1"].to_numpy(),
        "temperature_C": evaluation["T-ELY-CH1"].to_numpy(),
        "measured_voltage_V": y_true,
        "predicted_voltage_V": y_pred,
        "error_V": y_true - y_pred,
        "absolute_error_V": np.abs(y_true - y_pred),
    })

    return metrics, predictions


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved temporal model not found: {MODEL_PATH}\n"
            "Run python src/compare_temporal_model.py first."
        )

    saved = joblib.load(MODEL_PATH)

    if not isinstance(saved, dict) or "model" not in saved or "features" not in saved:
        raise ValueError(
            "Expected a saved dictionary containing 'model' and 'features'."
        )

    model = saved["model"]
    model_features = saved["features"]

    if model.n_features_in_ != len(model_features):
        raise ValueError(
            "Saved model feature count does not match saved feature list."
        )

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Model input features: {len(model_features)}")
    print("Evaluation: one-step, measured historical voltage available")
    print("Test 2 and Test 3 will be evaluated separately.")

    all_metrics = []
    all_predictions = []

    for test_name in ["Test2", "Test3"]:
        metrics, predictions = evaluate_experiment(
            test_name, model, model_features
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)

        print(f"\n{test_name}")
        print(f"Raw rows: {metrics['raw_rows']:,}")
        print(f"Evaluated rows: {metrics['evaluated_rows']:,}")
        print(f"Excluded rows: {metrics['excluded_rows']:,}")
        print(f"MAE: {metrics['MAE_V']:.6f} V")
        print(f"RMSE: {metrics['RMSE_V']:.6f} V")
        print(f"R²: {metrics['R2']:.6f}")

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics_path = OUT_DIR / "temporal_ood_metrics.csv"
    predictions_path = OUT_DIR / "temporal_ood_predictions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\nSummary")
    print(
        metrics_df[
            ["experiment", "raw_rows", "evaluated_rows", "excluded_rows",
             "MAE_V", "RMSE_V", "R2"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
