from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "test4_subset.csv"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "data" / "benchmark"
FIG_DIR = OUT_DIR / "figures"

FEATURES = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1"]
TARGET = "CV-mean"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH).dropna(
        subset=FEATURES + [TARGET]
    ).reset_index(drop=True)

    split = int(len(df) * 0.8)
    val = df.iloc[split:].copy()
    X_val = val[FEATURES]
    y = val[TARGET].to_numpy()

    model_files = {
        "Gradient Boosting": "benchmark_gradientboosting.joblib",
        "CatBoost": "benchmark_catboost.joblib",
        "Random Forest": "benchmark_randomforest.joblib",
    }

    predictions = pd.DataFrame({
        "row_index": val.index.to_numpy(),
        "current_A": val["PS-I-MON"].to_numpy(),
        "measured_voltage_V": y,
    })

    summary = []

    for name, filename in model_files.items():
        path = MODEL_DIR / filename
        if not path.exists():
            print(f"Skipping missing model: {path}")
            continue

        model = joblib.load(path)
        pred = model.predict(X_val)
        error = y - pred

        predictions[f"{name}_predicted_V"] = pred
        predictions[f"{name}_error_V"] = error

        summary.append({
            "model": name,
            "MAE_V": mean_absolute_error(y, pred),
            "RMSE_V": np.sqrt(mean_squared_error(y, pred)),
            "R2": r2_score(y, pred),
            "low_current_MAE_V": (
                mean_absolute_error(y[val["PS-I-MON"].to_numpy() < 0.5],
                                    pred[val["PS-I-MON"].to_numpy() < 0.5])
                if (val["PS-I-MON"] < 0.5).any() else np.nan
            ),
        })

    predictions.to_csv(
        OUT_DIR / "top_models_validation_predictions.csv", index=False
    )
    pd.DataFrame(summary).to_csv(
        OUT_DIR / "top_models_diagnostic_metrics.csv", index=False
    )

    # Plot the last 20% in chronological order.
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(
        predictions["row_index"],
        predictions["measured_voltage_V"],
        label="Measured",
        linewidth=1.4,
        color="black",
    )

    for name in model_files:
        col = f"{name}_predicted_V"
        if col in predictions:
            axes[0].plot(
                predictions["row_index"],
                predictions[col],
                label=name,
                linewidth=1.0,
            )

    axes[0].set_ylabel("Cell voltage (V)")
    axes[0].set_title("Validation: measured vs predicted voltage")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for name in model_files:
        col = f"{name}_error_V"
        if col in predictions:
            axes[1].plot(
                predictions["row_index"],
                predictions[col],
                label=name,
                linewidth=0.9,
            )

    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Dataset row index (not verified elapsed time)")
    axes[1].set_ylabel("Measured − predicted (V)")
    axes[1].set_title("Validation prediction errors")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "top_models_validation_diagnostics.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nDiagnostic metrics")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nSaved predictions: data/benchmark/top_models_validation_predictions.csv")
    print("Saved metrics: data/benchmark/top_models_diagnostic_metrics.csv")
    print("Saved plot: data/benchmark/figures/top_models_validation_diagnostics.png")


if __name__ == "__main__":
    main()