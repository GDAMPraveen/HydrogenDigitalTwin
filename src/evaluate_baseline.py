from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/test4_subset.csv")
MODEL_PATH = Path("models/voltage_random_forest.joblib")
OUTPUT_DIR = Path("data/figures")

FEATURES = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1"]
TARGET = "CV-mean"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    split_index = int(len(df) * 0.8)
    valid = df.iloc[split_index:].copy()

    model = joblib.load(MODEL_PATH)
    valid["predicted_voltage"] = model.predict(valid[FEATURES])
    valid["error"] = (
        valid["predicted_voltage"] - valid[TARGET]
    )
    valid["absolute_error"] = valid["error"].abs()

    print("\nOverall validation metrics")
    print("--------------------------")
    print(f"MAE:  {mean_absolute_error(valid[TARGET], valid['predicted_voltage']):.6f} V")
    print(f"RMSE: {np.sqrt(mean_squared_error(valid[TARGET], valid['predicted_voltage'])):.6f} V")
    print(f"R²:   {r2_score(valid[TARGET], valid['predicted_voltage']):.6f}")

    # Exploratory categories, not verified operational-state labels.
    current = valid["PS-I-MON"]
    valid["current_group"] = pd.cut(
        current,
        bins=[-np.inf, 0.5, 3.0, np.inf],
        labels=["Low (<0.5 A)", "Intermediate (0.5–3 A)", "High (>=3 A)"],
        right=False,
    )

    print("\nError by current group")
    print("----------------------")
    report = valid.groupby("current_group", observed=False).agg(
        rows=("error", "size"),
        MAE=("absolute_error", "mean"),
        bias=("error", "mean"),
        RMSE=("error", lambda x: np.sqrt(np.mean(x ** 2))),
    )
    print(report.round(5).to_string())

    print("\nFeature importance")
    print("------------------")
    importance = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    print(importance.round(4).to_string())

    # Save time-ordered validation predictions for further analysis.
    predictions_path = Path("models/validation_predictions.csv")
    valid[FEATURES + [TARGET, "predicted_voltage", "error",
                      "absolute_error"]].to_csv(predictions_path, index=False)
    print(f"\nSaved predictions: {predictions_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(valid.index, valid[TARGET], label="Measured", linewidth=0.8)
    axes[0].plot(
        valid.index, valid["predicted_voltage"],
        label="Predicted", linewidth=0.8,
    )
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("Validation: Measured vs Predicted Voltage")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(valid.index, valid["error"], linewidth=0.7)
    axes[1].axhline(0, linestyle="--")
    axes[1].set_ylabel("Prediction error (V)")
    axes[1].set_xlabel("Original row index")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = OUTPUT_DIR / "baseline_error_diagnostics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic plot: {plot_path}")


if __name__ == "__main__":
    main()