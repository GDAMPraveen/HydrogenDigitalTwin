
"""Benchmark regression models for PEM electrolyzer cell-voltage prediction.

Training/validation: chronological split of Test 4 only.
Test 2 and Test 3 remain reserved for future OOD evaluation.
"""

from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "test4_subset.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "data" / "benchmark"

FEATURES = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1"]
TARGET = "CV-mean"
RANDOM_STATE = 42


def current_group_metrics(y_true, y_pred, current):
    """Return MAE for low, intermediate and high current ranges."""
    groups = {
        "low_current_mae_V": current < 0.5,
        "intermediate_current_mae_V": (current >= 0.5) & (current < 3.0),
        "high_current_mae_V": current >= 3.0,
    }
    output = {}
    for name, mask in groups.items():
        output[name] = (
            float(mean_absolute_error(y_true[mask], y_pred[mask]))
            if mask.any()
            else np.nan
        )
    return output


def make_models():
    """Create models. Scaling is included where it is important."""
    models = {
        "Dummy_mean": DummyRegressor(strategy="mean"),
        "LinearRegression": make_pipeline(
            StandardScaler(), LinearRegression()
        ),
        "Ridge": make_pipeline(
            StandardScaler(), Ridge(alpha=1.0)
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            max_features=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            max_features=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            loss="squared_error",
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=15, weights="distance"),
        ),
        "SVR_sampled": make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=10.0, epsilon=0.02),
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size=512,
                learning_rate_init=0.001,
                max_iter=150,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=12,
                random_state=RANDOM_STATE,
            ),
        ),
    }

    # Optional libraries: include them only if importable.
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=1.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    except ImportError as exc:
        print(f"Skipping XGBoost: {exc}")

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=250,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=1.0,
            verbosity=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    except ImportError as exc:
        print(f"Skipping LightGBM: {exc}")

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=250,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            verbose=False,
            thread_count=-1,
            random_seed=RANDOM_STATE,
        )
    except ImportError as exc:
        print(f"Skipping CatBoost: {exc}")

    return models


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Retain the original chronological order; only remove rows missing
    # a feature or target. Test 4 was previously audited as complete.
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)

    split = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    current_val = X_val["PS-I-MON"].to_numpy()

    print(f"Dataset: {DATA_PATH.name}")
    print(f"Rows after required-column cleanup: {len(df):,}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print("Split: chronological 80/20; Test 2 and Test 3 are not used.")
    print()

    results = []
    models = make_models()

    # SVR training is quadratic or worse in sample count for many settings.
    # Use a reproducible, evenly spaced sample from the training period.
    svr_limit = 12000
    svr_indices = np.linspace(
        0, len(X_train) - 1, min(svr_limit, len(X_train)), dtype=int
    )
    X_svr = X_train.iloc[svr_indices]
    y_svr = y_train.iloc[svr_indices]

    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        start = time.perf_counter()

        try:
            if name == "SVR_sampled":
                model.fit(X_svr, y_svr)
            else:
                model.fit(X_train, y_train)

            train_seconds = time.perf_counter() - start
            predictions = np.asarray(model.predict(X_val), dtype=float)

            metrics = {
                "model": name,
                "validation_rows": len(y_val),
                "training_seconds": round(train_seconds, 3),
                "MAE_V": mean_absolute_error(y_val, predictions),
                "RMSE_V": np.sqrt(mean_squared_error(y_val, predictions)),
                "R2": r2_score(y_val, predictions),
                **current_group_metrics(
                    y_val.to_numpy(), predictions, current_val
                ),
            }

            if name == "SVR_sampled":
                metrics["training_rows_used"] = len(X_svr)
            else:
                metrics["training_rows_used"] = len(X_train)

            results.append(metrics)

            # Save each fitted model separately.
            model_path = MODELS_DIR / f"benchmark_{name.lower()}.joblib"
            joblib.dump(model, model_path)

            print(
                f"  MAE={metrics['MAE_V']:.6f} V | "
                f"RMSE={metrics['RMSE_V']:.6f} V | "
                f"R2={metrics['R2']:.6f} | "
                f"time={train_seconds:.1f}s",
                flush=True,
            )

        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            results.append({
                "model": name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    results_df = pd.DataFrame(results)
    results_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    valid = results_df.dropna(subset=["MAE_V"]).sort_values("MAE_V")
    print("\nBenchmark summary (sorted by validation MAE)")
    print("---------------------------------------------")
    if not valid.empty:
        columns = [
            "model", "MAE_V", "RMSE_V", "R2",
            "low_current_mae_V", "high_current_mae_V",
            "training_seconds",
        ]
        print(valid[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print(f"\nLowest overall validation MAE: {valid.iloc[0]['model']}")
    else:
        print("No models completed successfully.")

    print(f"\nSaved results: {results_path}")
    print(f"Saved fitted models under: {MODELS_DIR}")


if __name__ == "__main__":
    main()