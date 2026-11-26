from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

TEST2_PATH = ROOT / "data" / "test2_subset.csv"
PREDICTIONS_PATH = (
    ROOT / "data" / "benchmark" /
    "linear_ridge_temporal_test2" / "predictions.csv"
)
OUTPUT_DIR = ROOT / "data" / "benchmark" / "linear_ridge_temporal_test2"

TARGET = "CV-mean"
CURRENT = "PS-I-MON"


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Bias": np.mean(y_true - y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    test2 = pd.read_csv(TEST2_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    # Same eligible rows used by the temporal models.
    temporal_predictions = predictions[
        predictions["Model"].isin([
            "LinearRegression_temporal",
            "Ridge_temporal",
        ])
    ]

    rows = np.sort(
        temporal_predictions["Test2_raw_row"].unique().astype(int)
    )

    actual = test2.loc[rows, TARGET].to_numpy()
    previous_voltage = test2[TARGET].shift(1).loc[rows].to_numpy()

    current = test2.loc[rows, CURRENT].to_numpy()

    results = []

    persistence = metrics(actual, previous_voltage)
    results.append({
        "Model": "Persistence_previous_voltage",
        "Rows": len(rows),
        **persistence,
    })

    for model_name in [
        "LinearRegression_temporal",
        "Ridge_temporal",
    ]:
        model_data = (
            temporal_predictions[
                temporal_predictions["Model"] == model_name
            ]
            .set_index("Test2_raw_row")
            .loc[rows]
        )

        predicted = model_data["Predicted_voltage_V"].to_numpy()

        results.append({
            "Model": model_name,
            "Rows": len(rows),
            **metrics(actual, predicted),
        })

    result_df = pd.DataFrame(results)

    # Check how closely the previous voltage tracks the current voltage.
    correlation = np.corrcoef(actual, previous_voltage)[0, 1]

    print("\nPERSISTENCE BASELINE VS TEMPORAL MODELS")
    print(result_df.to_string(index=False))
    print(f"\nCorrelation: previous voltage vs current voltage = {correlation:.8f}")

    regime = np.select(
        [current < 0.5, current < 3.0],
        ["low (<0.5 A)", "intermediate (0.5–3 A)"],
        default="high (>=3 A)",
    )

    regime_results = []

    for label in [
        "low (<0.5 A)",
        "intermediate (0.5–3 A)",
        "high (>=3 A)",
    ]:
        mask = regime == label

        for name, predicted in [
            ("Persistence_previous_voltage", previous_voltage),
            ("LinearRegression_temporal",
             temporal_predictions[
                 temporal_predictions["Model"] ==
                 "LinearRegression_temporal"
             ].set_index("Test2_raw_row").loc[rows][
                 "Predicted_voltage_V"
             ].to_numpy()),
            ("Ridge_temporal",
             temporal_predictions[
                 temporal_predictions["Model"] ==
                 "Ridge_temporal"
             ].set_index("Test2_raw_row").loc[rows][
                 "Predicted_voltage_V"
             ].to_numpy()),
        ]:
            regime_results.append({
                "Current_regime": label,
                "Model": name,
                "Rows": int(mask.sum()),
                **metrics(actual[mask], predicted[mask]),
            })

    regime_df = pd.DataFrame(regime_results)

    print("\nPERFORMANCE BY CURRENT REGIME")
    print(regime_df.to_string(index=False))

    result_df.to_csv(
        OUTPUT_DIR / "persistence_comparison.csv", index=False
    )
    regime_df.to_csv(
        OUTPUT_DIR / "persistence_regime_comparison.csv", index=False
    )

    print("\nSaved persistence comparison files.")


if __name__ == "__main__":
    main()
