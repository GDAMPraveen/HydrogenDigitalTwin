from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/test4_subset.csv")
MODEL_DIR = Path("models")
FIGURE_DIR = Path("data/figures")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1"]
TARGET = "CV-mean"

def main():
    df = pd.read_csv(DATA_PATH)

    # Preserve the original chronological order.
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    split_index = int(len(df) * 0.8)
    train = df.iloc[:split_index]
    valid = df.iloc[split_index:]

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_valid = valid[FEATURES]
    y_valid = valid[TARGET]

    model = RandomForestRegressor(
    n_estimators=150,
    min_samples_leaf=3,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
    )

    print("Training Random Forest baseline...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, predictions)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))
    r2 = r2_score(y_valid, predictions)

    print("\nChronological validation results")
    print("--------------------------------")
    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(valid):,}")
    print(f"MAE:  {mae:.6f} V")
    print(f"RMSE: {rmse:.6f} V")
    print(f"R²:   {r2:.6f}")

    model_path = MODEL_DIR / "voltage_random_forest.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved model: {model_path}")

    metrics = pd.DataFrame([{
    "model": "RandomForestRegressor",
    "training_rows": len(train),
    "validation_rows": len(valid),
    "MAE_V": mae,
    "RMSE_V": rmse,
    "R2": r2,
    }])
    metrics.to_csv(MODEL_DIR / "baseline_metrics.csv", index=False)
    print(f"Saved metrics: {MODEL_DIR / 'baseline_metrics.csv'}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_valid, predictions, s=8, alpha=0.35)

    lower = min(y_valid.min(), predictions.min())
    upper = max(y_valid.max(), predictions.max())
    ax.plot([lower, upper], [lower, upper], linestyle="--")

    ax.set_xlabel("Measured mean cell voltage (V)")
    ax.set_ylabel("Predicted mean cell voltage (V)")
    ax.set_title("Random Forest Baseline: Test 4 Chronological Validation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = FIGURE_DIR / "baseline_prediction_scatter.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved prediction plot: {plot_path}")

if __name__ == "__main__":
    main()