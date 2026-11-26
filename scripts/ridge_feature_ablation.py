
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.forecasting.data_preparation import prepare_forecasting_data
from src.forecasting.feature_engineering import create_supervised_dataset


DATA = Path("data/processed/nlr/nlr_observation_clean.csv")
TARGET = "IVAL_f_FM011_Flow"
HORIZON = 1
ALPHA = 0.01

OUTPUT_DIR = Path("reports/forecasting")

EXCLUDED = {
    "H2E_f_Elec_CalcProdRate",
    "Efficiency",
    "Unnamed: 31",
    "Original_Row_Index",
    "Flow_Was_Interpolated",
    "H2E_f_TT641_Temp",
}


def calculate_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA)),
    ])


def main():

    print("Loading NLR data...")

    df, feature_cols = prepare_forecasting_data(
        DATA,
        TARGET,
        experiment_column="Experiment",
    )

    feature_cols = [
        c for c in feature_cols
        if c not in EXCLUDED
    ]

    df = df[
        df["Experiment"] != "characterization_200.csv"
    ].copy()

    wind_experiments = sorted(
        df["Experiment"].dropna().unique()
    )

    print(f"Wind experiments found: {len(wind_experiments)}")

    if len(wind_experiments) != 12:
        raise ValueError(
            f"Expected 12 wind experiments, "
            f"found {len(wind_experiments)}"
        )

    print("\nCreating supervised dataset...")

    X, y, metadata = create_supervised_dataset(
        df=df,
        target_column=TARGET,
        feature_columns=feature_cols,
        experiment_column="Experiment",
        horizon_steps=HORIZON,
        lags=(1, 2, 3, 5, 10),
        rolling_windows=(3, 5, 10),
    )

    groups = metadata["Experiment"].to_numpy()

    history_cols = [
        c for c in X.columns
        if c == f"{TARGET}_current"
        or c.startswith(f"{TARGET}_lag_")
        or c.startswith(f"{TARGET}_rolling_")
    ]

    current_flow_col = f"{TARGET}_current"

    if current_flow_col not in X.columns:
        raise ValueError(
            f"Required persistence feature missing: {current_flow_col}"
        )

    experiments = {
        "A_Full": list(X.columns),

        "B_No_History": [
            c for c in X.columns
            if c not in history_cols
            or c == current_flow_col
        ],

        "C_Sensors_Only": [
            c for c in X.columns
            if c not in history_cols
        ],

        "D_History_Only": history_cols,
    }

    print("\nFeature configuration sizes:")

    for name, columns in experiments.items():
        print(f"{name}: {len(columns)} features")

        if not columns:
            raise ValueError(f"{name} has no features.")

    all_results = []
    pooled_results = {
        name: {"actual": [], "predicted": []}
        for name in experiments
    }

    pooled_results["Persistence"] = {
        "actual": [],
        "predicted": [],
    }

    print("\n========== CROSS-EXPERIMENT EVALUATION ==========")

    for fold, held_out in enumerate(wind_experiments, start=1):

        held_mask = groups == held_out

        X_train = X.loc[~held_mask]
        y_train = y.loc[~held_mask]

        X_test = X.loc[held_mask]
        y_test = y.loc[held_mask]

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError(
                f"Invalid fold for {held_out}: "
                f"train={len(X_train)}, test={len(X_test)}"
            )

        if set(metadata.loc[held_mask, "Experiment"]) != {held_out}:
            raise ValueError(f"Incorrect held-out group: {held_out}")

        print(f"\n[{fold}/12] Held out: {held_out}")
        print(f"Training rows: {len(X_train)}")
        print(f"Testing rows: {len(X_test)}")

        actual = y_test.to_numpy()

        # Persistence baseline: next flow equals current measured flow.
        persistence_predictions = X_test[
            current_flow_col
        ].to_numpy()

        persistence_metrics = calculate_metrics(
            actual,
            persistence_predictions,
        )

        all_results.append({
            "Held_Out_Experiment": held_out,
            "Configuration": "Persistence",
            "Features": 1,
            **persistence_metrics,
            "Prediction_Min": persistence_predictions.min(),
            "Prediction_Max": persistence_predictions.max(),
            "Negative_Predictions": int(
                (persistence_predictions < 0).sum()
            ),
            "Test_Rows": len(X_test),
        })

        pooled_results["Persistence"]["actual"].extend(actual)
        pooled_results["Persistence"]["predicted"].extend(
            persistence_predictions
        )

        for name, columns in experiments.items():

            print(f"  Running {name} ({len(columns)} features)...")

            model = build_model()

            model.fit(
                X_train[columns],
                y_train,
            )

            predictions = model.predict(
                X_test[columns]
            )

            metrics = calculate_metrics(
                actual,
                predictions,
            )

            all_results.append({
                "Held_Out_Experiment": held_out,
                "Configuration": name,
                "Features": len(columns),
                **metrics,
                "Prediction_Min": predictions.min(),
                "Prediction_Max": predictions.max(),
                "Negative_Predictions": int(
                    (predictions < 0).sum()
                ),
                "Test_Rows": len(X_test),
            })

            pooled_results[name]["actual"].extend(actual)
            pooled_results[name]["predicted"].extend(
                predictions
            )

    # Per-experiment results
    result_df = pd.DataFrame(all_results)

    # Macro-average and pooled out-of-fold metrics
    summary_results = []

    for name, values in pooled_results.items():

        actual = np.asarray(values["actual"])
        predictions = np.asarray(values["predicted"])

        if len(actual) == 0 or len(actual) != len(predictions):
            raise ValueError(
                f"Invalid pooled predictions for {name}"
            )

        pooled_metrics = calculate_metrics(
            actual,
            predictions,
        )

        per_experiment = result_df[
            result_df["Configuration"] == name
        ]

        summary_results.append({
            "Configuration": name,
            "Experiments": len(per_experiment),
            "Features": (
                int(per_experiment["Features"].iloc[0])
            ),
            "Macro_MAE": per_experiment["MAE"].mean(),
            "Macro_RMSE": per_experiment["RMSE"].mean(),
            "Macro_R2": per_experiment["R2"].mean(),
            "Pooled_MAE": pooled_metrics["MAE"],
            "Pooled_RMSE": pooled_metrics["RMSE"],
            "Pooled_R2": pooled_metrics["R2"],
            "Total_Test_Rows": len(actual),
            "Negative_Predictions": int(
                (predictions < 0).sum()
            ),
            "Prediction_Min": predictions.min(),
            "Prediction_Max": predictions.max(),
        })

    summary_df = pd.DataFrame(summary_results)

    # Ensure each expected output is new; preserve existing reports.
    detailed_output = (
        OUTPUT_DIR / "ridge_feature_ablation_all_experiments.csv"
    )

    summary_output = (
        OUTPUT_DIR / "ridge_feature_ablation_summary.csv"
    )

    for output in [detailed_output, summary_output]:
        if output.exists():
            raise FileExistsError(
                f"Output already exists; refusing to overwrite: {output}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(detailed_output, index=False)
    summary_df.to_csv(summary_output, index=False)

    print("\n========== PER-EXPERIMENT RESULTS ==========")
    print(result_df.to_string(index=False))

    print("\n========== CROSS-EXPERIMENT SUMMARY ==========")
    print(summary_df.to_string(index=False))

    print("\nResults saved:")
    print(detailed_output)
    print(summary_output)

    print("\n========== VALIDATION ==========")
    print(f"Expected folds: 12")
    print(f"Completed folds: {result_df['Held_Out_Experiment'].nunique()}")
    print(f"Expected Ridge evaluations: 48")
    print(
        "Completed Ridge evaluations:",
        len(result_df[result_df["Configuration"] != "Persistence"])
    )
    print(f"Persistence evaluations: {len(wind_experiments)}")


if __name__ == "__main__":
    main()
