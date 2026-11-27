"""Train/evaluate direct multi-horizon hydrogen-flow forecasting models.

Run from project root:
python -m scripts.run_forecasting_experiments --mode loeo --feature-set no_history --horizons 1 --skip-optional --models Ridge HistGradientBoosting
python -m scripts.run_forecasting_experiments --mode both --feature-set no_history --horizons 1 3 5 10
python -m scripts.run_forecasting_experiments --mode loeo --feature-set full --skip-optional

All horizons are row-based steps, not seconds/minutes.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.forecasting.data_preparation import prepare_forecasting_data
from src.forecasting.feature_engineering import create_supervised_dataset
from src.forecasting.train_models import build_candidate_models
from src.forecasting.forecast_evaluation import (
    calculate_forecast_metrics,
    chronological_train_test_split,
)
from src.forecasting.forecast_engine import ForecastEngine
from src.forecasting.baseline_models import mean_baseline

DATA = "data/processed/nlr/nlr_observation_clean.csv"
TARGET = "IVAL_f_FM011_Flow"

# Unified exclusion list (synced with ridge_feature_ablation.py to prevent leakage)
EXCLUDED = {
    "H2E_f_Elec_CalcProdRate",
    "Efficiency",
    "Original_Row_Index",
    "Flow_Was_Interpolated",
    "Calculated_PSU_Power_W",
    "Power_Difference_W",
    "Power_Relative_Error_Percent",
    "Calculated_PSU_Power_kW",
    "Power_Difference_kW",
    "Power_Relative_Error_kW_Percent",
    "H2E_f_TT641_Temp",
    "Unnamed: 31",
}


def prediction_diagnostics(pred):
    """Track physically inadmissible (negative) hydrogen-flow predictions."""
    pred_arr = np.asarray(pred, dtype=float)
    return {
        "Negative_Predictions": int((pred_arr < 0).sum()),
        "Prediction_Min": float(pred_arr.min()),
        "Prediction_Max": float(pred_arr.max()),
    }


def build_dataset(horizon, feature_set="no_history"):
    df, features = prepare_forecasting_data(DATA, TARGET, experiment_column="Experiment")
    features = [f for f in features if f not in EXCLUDED]
    df = df[df["Experiment"] != "characterization_200.csv"].copy()

    # Keep identical lag/rolling row dropping so sample support matches the ablation study
    X, y, meta = create_supervised_dataset(
        df,
        TARGET,
        horizon_steps=horizon,
        lags=(1, 2, 3, 5, 10),
        rolling_windows=(3, 5, 10),
        experiment_column="Experiment",
        feature_columns=features,
        drop_incomplete_rows=True,
    )

    if feature_set == "no_history":
        keep_cols = [
            c
            for c in X.columns
            if not (
                c.startswith(f"{TARGET}_lag_")
                or c.startswith(f"{TARGET}_rolling_")
            )
        ]
        X = X[keep_cols]

    return X, y, meta


def evaluate_chronological(
    X, y, meta, horizon, feature_set, random_state, include_optional=True, selected_models=None
):
    Xtr, Xte, ytr, yte = chronological_train_test_split(X, y, meta, 0.2, "Experiment")
    rows = []
    models = build_candidate_models(random_state=random_state, include_optional=include_optional)
    if selected_models:
        missing = [m for m in selected_models if m not in models]
        if missing:
            raise ValueError(f"Unknown model(s): {missing}. Available: {list(models)}")
        models = {m: models[m] for m in selected_models}

    for name, model in models.items():
        t0 = time.perf_counter()
        fitted = clone(model).fit(Xtr, ytr)
        pred = fitted.predict(Xte)
        elapsed = time.perf_counter() - t0
        metrics = calculate_forecast_metrics(yte, pred)
        print(f"  - {name}: MAE={metrics['MAE']:.4f} ({elapsed:.1f}s)", flush=True)
        rows.append({
            "Model": name,
            **metrics,
            **prediction_diagnostics(pred),
        })

    persistence_pred = Xte[f"{TARGET}_current"].to_numpy()
    rows.append({
        "Model": "Persistence",
        **calculate_forecast_metrics(yte, persistence_pred),
        **prediction_diagnostics(persistence_pred),
    })

    mean_pred = mean_baseline(ytr, len(yte))
    rows.append({
        "Model": "TrainingMean",
        **calculate_forecast_metrics(yte, mean_pred),
        **prediction_diagnostics(mean_pred),
    })

    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    best_name = table.iloc[0]["Model"]
    if best_name in models:
        best_model = clone(models[best_name]).fit(Xtr, ytr)
        ForecastEngine(best_model, list(X.columns), TARGET, horizon_steps=int(horizon)).save(
            Path("models/forecasting") / f"nlr_{horizon}_step_{feature_set}_{best_name.lower()}.joblib"
        )
    return table


def evaluate_loeo(X, y, meta, random_state, include_optional=True, selected_models=None):
    rows = []
    model_templates = build_candidate_models(
        random_state=random_state, include_optional=include_optional
    )
    if selected_models:
        missing = [m for m in selected_models if m not in model_templates]
        if missing:
            raise ValueError(f"Unknown model(s): {missing}. Available: {list(model_templates)}")
        model_templates = {m: model_templates[m] for m in selected_models}

    groups = meta["Experiment"]

    for fold, held_out in enumerate(sorted(groups.unique()), start=1):
        print(f"  [LOEO {fold}/12] Held out: {held_out}", flush=True)
        test_idx = meta.index[groups == held_out]
        train_idx = meta.index[groups != held_out]
        Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
        ytr, yte = y.loc[train_idx], y.loc[test_idx]

        fold_rows = []
        for name, template in model_templates.items():
            t0 = time.perf_counter()
            fitted = clone(template).fit(Xtr, ytr)
            pred = fitted.predict(Xte)
            elapsed = time.perf_counter() - t0
            metrics = calculate_forecast_metrics(yte, pred)
            print(f"    - {name}: MAE={metrics['MAE']:.4f} ({elapsed:.1f}s)", flush=True)
            fold_rows.append({
                "Experiment": held_out,
                "Model": name,
                **metrics,
                **prediction_diagnostics(pred),
            })

        persistence_pred = Xte[f"{TARGET}_current"].to_numpy()
        fold_rows.append({
            "Experiment": held_out,
            "Model": "Persistence",
            **calculate_forecast_metrics(yte, persistence_pred),
            **prediction_diagnostics(persistence_pred),
        })
        rows.extend(fold_rows)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["chronological", "loeo", "both"], default="loeo")
    parser.add_argument("--feature-set", choices=["no_history", "full"], default="no_history")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Use only sklearn models; omit XGBoost/LightGBM/CatBoost.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of model names (e.g., --models Ridge HistGradientBoosting).",
    )
    args = parser.parse_args()

    out = Path("reports/forecasting")
    out.mkdir(parents=True, exist_ok=True)
    Path("models/forecasting").mkdir(parents=True, exist_ok=True)

    for horizon in args.horizons:
        if horizon < 1:
            raise ValueError("Horizons must be positive integers.")

        print(
            f"\n=== Horizon: {horizon} row(s) | Feature set: {args.feature_set} ===",
            flush=True,
        )
        X, y, meta = build_dataset(horizon, feature_set=args.feature_set)
        print(
            f"Samples={len(X):,}; features={X.shape[1]}; experiments={meta['Experiment'].nunique()}"
        )

        if args.mode in ("chronological", "both"):
            result = evaluate_chronological(
                X, y, meta, horizon, args.feature_set, args.random_state,
                include_optional=not args.skip_optional,
                selected_models=args.models,
            )
            result.to_csv(out / f"chronological_{horizon}_step_{args.feature_set}.csv", index=False)
            print("\nChronological results:\n", result.to_string(index=False))

        if args.mode in ("loeo", "both"):
            loeo = evaluate_loeo(
                X, y, meta, args.random_state,
                include_optional=not args.skip_optional,
                selected_models=args.models,
            )
            loeo.to_csv(
                out / f"loeo_{horizon}_step_{args.feature_set}_by_experiment.csv", index=False
            )
            summary = (
                loeo.groupby("Model")
                .agg(
                    MAE=("MAE", "mean"),
                    RMSE=("RMSE", "mean"),
                    R2=("R2", "mean"),
                    MAPE_percent=("MAPE_percent", "mean"),
                    sMAPE_percent=("sMAPE_percent", "mean"),
                    Total_Neg_Preds=("Negative_Predictions", "sum"),
                    Min_Pred=("Prediction_Min", "min"),
                )
                .sort_values("MAE")
            )
            summary.to_csv(out / f"loeo_{horizon}_step_{args.feature_set}_summary.csv")
            print("\nLOEO mean metrics:\n", summary.to_string())

    print(f"\nReports saved to: {out.resolve()}")


if __name__ == "__main__":
    main()