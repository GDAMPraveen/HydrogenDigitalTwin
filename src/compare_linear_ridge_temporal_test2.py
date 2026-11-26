from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

TEST4_PATH = ROOT / "data" / "test4_subset.csv"
TEST2_PATH = ROOT / "data" / "test2_subset.csv"

OUTPUT_DIR = (
    ROOT / "data" / "benchmark" / "linear_ridge_temporal_test2"
)

FEATURES = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1"]
TARGET = "CV-mean"
RANDOM_STATE = 42


def create_temporal_features(df):
    """Create historical features without using future observations."""
    data = df[FEATURES + [TARGET]].copy()
    data["_raw_row"] = np.arange(len(data))

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

    return data.dropna().reset_index(drop=True)


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Bias_measured_minus_predicted": np.mean(y_true - y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test4 = pd.read_csv(TEST4_PATH)
    test2 = pd.read_csv(TEST2_PATH)

    split_index = int(len(test4) * 0.80)

    # Baseline training data: first 80% of Test4.
    train_base = test4.iloc[:split_index][FEATURES + [TARGET]].dropna()

    # Temporal training data: retain only rows with complete history.
    temporal_test4 = create_temporal_features(test4)
    train_temp = temporal_test4[
        temporal_test4["_raw_row"] < split_index
    ].copy()

    # Construct Test2 temporal features using Test2's own past measurements.
    temporal_test2 = create_temporal_features(test2)

    # Use exactly the same Test2 rows for all four models.
    eligible_rows = temporal_test2["_raw_row"].to_numpy()

    test2_base = test2.iloc[eligible_rows][FEATURES + [TARGET]].copy()
    test2_temp = temporal_test2.copy()

    assert np.array_equal(
        test2_base.index.to_numpy(),
        eligible_rows
    )

    print("Test4 rows:", len(test4))
    print("Test4 chronological split:", split_index)
    print("Baseline training rows:", len(train_base))
    print("Temporal training rows:", len(train_temp))
    print("Test2 raw rows:", len(test2))
    print("Common eligible Test2 rows:", len(test2_temp))
    print("Excluded Test2 rows:", len(test2) - len(test2_temp))

    baseline_features = FEATURES
    temporal_features = [
        col for col in temporal_test4.columns
        if col not in [TARGET, "_raw_row"]
    ]

    configurations = [
        ("LinearRegression_operating", LinearRegression(),
         baseline_features, train_base, test2_base),
        ("Ridge_operating", Ridge(alpha=1.0),
         baseline_features, train_base, test2_base),
        ("LinearRegression_temporal", LinearRegression(),
         temporal_features, train_temp, test2_temp),
        ("Ridge_temporal", Ridge(alpha=1.0),
         temporal_features, train_temp, test2_temp),
    ]

    metrics = []
    predictions = []

    for name, estimator, feature_cols, train_df, test_df in configurations:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

        model.fit(train_df[feature_cols], train_df[TARGET])
        y_pred = model.predict(test_df[feature_cols])
        y_true = test_df[TARGET].to_numpy()

        result = {
            "Model": name,
            "Feature_count": len(feature_cols),
            "Training_rows": len(train_df),
            "Test2_rows": len(test_df),
            **evaluate(y_true, y_pred),
        }
        metrics.append(result)

        prediction_df = pd.DataFrame({
            "Model": name,
            "Test2_raw_row": test_df["_raw_row"].to_numpy()
            if "_raw_row" in test_df.columns
            else test_df.index.to_numpy(),
            "Current_A": test_df["PS-I-MON"].to_numpy(),
            "Measured_voltage_V": y_true,
            "Predicted_voltage_V": y_pred,
            "Absolute_error_V": np.abs(y_true - y_pred),
        })

        prediction_df["Current_regime"] = np.select(
            [
                prediction_df["Current_A"] < 0.5,
                prediction_df["Current_A"] < 3.0,
            ],
            ["low (<0.5 A)", "intermediate (0.5–3 A)"],
            default="high (>=3 A)",
        )

        predictions.append(prediction_df)

        print(f"\n{name}")
        print(f"  MAE: {result['MAE']:.6f} V")
        print(f"  RMSE: {result['RMSE']:.6f} V")
        print(f"  Bias: {result['Bias_measured_minus_predicted']:.6f} V")
        print(f"  R2: {result['R2']:.6f}")

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(predictions, ignore_index=True)

    # Also report performance by current regime.
    regime_metrics = []

    for (model_name, regime), group in predictions_df.groupby(
        ["Model", "Current_regime"]
    ):
        result = evaluate(
            group["Measured_voltage_V"],
            group["Predicted_voltage_V"],
        )
        regime_metrics.append({
            "Model": model_name,
            "Current_regime": regime,
            "Rows": len(group),
            **result,
        })

    regime_df = pd.DataFrame(regime_metrics)

    metrics_path = OUTPUT_DIR / "model_metrics.csv"
    predictions_path = OUTPUT_DIR / "predictions.csv"
    regime_path = OUTPUT_DIR / "current_regime_metrics.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    regime_df.to_csv(regime_path, index=False)

    print("\nOVERALL TEST2 COMPARISON")
    print(metrics_df.to_string(index=False))

    print("\nCURRENT-REGIME COMPARISON")
    print(regime_df.to_string(index=False))

    print("\nSaved:")
    print(metrics_path)
    print(predictions_path)
    print(regime_path)


if __name__ == "__main__":
    main()
