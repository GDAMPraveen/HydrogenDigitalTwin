"""Benchmark Pure AI vs. Physics-Informed vs. Physics-Residual Hybrid on 12-fold LOEO.

Run from project root:
python -m scripts.run_hybrid_loeo --horizons 1 3 5 10
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.forecasting.data_preparation import prepare_forecasting_data
from src.forecasting.feature_engineering import create_supervised_dataset
from src.forecasting.forecast_evaluation import calculate_forecast_metrics
from src.forecasting.train_models import build_candidate_models
from src.integration.hybrid_digital_twin import (
    FaradayPhysicsCalibrator,
    HybridPEMForecaster,
)

DATA = "data/processed/nlr/nlr_observation_clean.csv"
TARGET = "IVAL_f_FM011_Flow"

EXCLUDED = {
    "Unnamed: 0",
    "H2E_f_Elec_CalcProdRate",
    "Efficiency",
    "Efficiency (kWh/kg)",
    "Extreme_Efficiency_Flag",
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


def build_no_history_dataset(horizon: int):
    df, features = prepare_forecasting_data(DATA, TARGET, experiment_column="Experiment")
    features = [f for f in features if f not in EXCLUDED]
    df = df[df["Experiment"] != "characterization_200.csv"].copy()

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

    keep_cols = [
        c for c in X.columns
        if not (c.startswith(f"{TARGET}_lag_") or c.startswith(f"{TARGET}_rolling_"))
    ]
    return X[keep_cols], y, meta


def evaluate_hybrid_loeo(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, random_state: int):
    base_models = build_candidate_models(random_state=random_state, include_optional=False)
    selected = {k: base_models[k] for k in ("Ridge", "HistGradientBoosting")}
    current_col = f"{TARGET}_current"
    groups = meta["Experiment"]
    rows = []

    for fold, held_out in enumerate(sorted(groups.unique()), start=1):
        test_idx = meta.index[groups == held_out]
        train_idx = meta.index[groups != held_out]
        Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
        ytr, yte = y.loc[train_idx], y.loc[test_idx]

        # 1. Persistence baseline
        pers_pred = Xte[current_col].to_numpy()
        rows.append({
            "Experiment": held_out,
            "Model": "Persistence",
            **calculate_forecast_metrics(yte, pers_pred),
            "Neg_Preds": int((pers_pred < 0).sum()),
            "Min_Pred": float(pers_pred.min()),
        })

        # 2. Pure Physics (Calibrated Faraday) baseline
        cal = FaradayPhysicsCalibrator().fit(Xtr, Xtr[current_col])
        phys_pred = cal.predict_faraday_flow(Xte)
        rows.append({
            "Experiment": held_out,
            "Model": "Physics_Faraday_Only",
            **calculate_forecast_metrics(yte, phys_pred),
            "Neg_Preds": int((phys_pred < 0).sum()),
            "Min_Pred": float(phys_pred.min()),
        })

        # 3. Pure AI vs Physics-Informed vs Physics-Residual
        for name, template in selected.items():
            for mode in ("pure_ai", "physics_informed", "physics_residual"):
                forecaster = HybridPEMForecaster(template, mode=mode, current_flow_col=current_col)
                forecaster.fit(Xtr, ytr)
                pred = forecaster.predict(Xte)
                rows.append({
                    "Experiment": held_out,
                    "Model": f"{name}_{mode}",
                    **calculate_forecast_metrics(yte, pred),
                    "Neg_Preds": int((pred < 0).sum()),
                    "Min_Pred": float(pred.min()),
                })

        print(f"  [LOEO {fold}/12] {held_out} (k_eff={cal.k_eff_:.2f})", flush=True)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out = Path("reports/forecasting")
    out.mkdir(parents=True, exist_ok=True)

    for horizon in args.horizons:
        print(f"\n=== Hybrid LOEO Benchmark | Horizon: {horizon} row(s) ===", flush=True)
        X, y, meta = build_no_history_dataset(horizon)
        print(f"Samples={len(X):,}; base features={X.shape[1]}; experiments={meta['Experiment'].nunique()}")

        results = evaluate_hybrid_loeo(X, y, meta, args.random_state)
        results.to_csv(out / f"hybrid_loeo_{horizon}_step_by_experiment.csv", index=False)

        summary = (
            results.groupby("Model")
            .agg(
                MAE=("MAE", "mean"),
                RMSE=("RMSE", "mean"),
                R2=("R2", "mean"),
                sMAPE_percent=("sMAPE_percent", "mean"),
                Total_Neg_Preds=("Neg_Preds", "sum"),
                Min_Pred=("Min_Pred", "min"),
            )
            .sort_values("MAE")
        )
        summary.to_csv(out / f"hybrid_loeo_{horizon}_step_summary.csv")
        print("\nSummary:\n", summary.to_string())


if __name__ == "__main__":
    main()