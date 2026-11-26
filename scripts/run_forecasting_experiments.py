"""Train/evaluate direct multi-horizon hydrogen-flow forecasting models.

Run from project root:
python scripts/run_forecasting_experiments.py --mode chronological
python scripts/run_forecasting_experiments.py --mode loeo
python scripts/run_forecasting_experiments.py --mode both

All horizons are row-based steps, not seconds/minutes.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from src.forecasting.data_preparation import prepare_forecasting_data
from src.forecasting.feature_engineering import create_supervised_dataset
from src.forecasting.train_models import build_candidate_models
from src.forecasting.forecast_evaluation import calculate_forecast_metrics, chronological_train_test_split
from src.forecasting.forecast_engine import ForecastEngine
from src.forecasting.baseline_models import mean_baseline

DATA = "data/processed/nlr/nlr_observation_clean.csv"
TARGET = "IVAL_f_FM011_Flow"
EXCLUDED = {
    "Original_Row_Index", "Flow_Was_Interpolated",
    "Calculated_PSU_Power_W", "Power_Difference_W",
    "Power_Relative_Error_Percent", "Calculated_PSU_Power_kW",
    "Power_Difference_kW", "Power_Relative_Error_kW_Percent",
    "H2E_f_TT641_Temp",
    "Unnamed: 31",
}

def build_dataset(horizon):
    df, features = prepare_forecasting_data(DATA, TARGET, experiment_column="Experiment")
    features = [f for f in features if f not in EXCLUDED]
    df = df[df["Experiment"] != "characterization_200.csv"].copy()
    X, y, meta = create_supervised_dataset(
        df, TARGET, horizon_steps=horizon, lags=(1,2,3,5,10),
        rolling_windows=(3,5,10), experiment_column="Experiment",
        feature_columns=features, drop_incomplete_rows=True
    )
    return X, y, meta

def evaluate_chronological(X,y,meta,random_state,include_optional=True):
    Xtr,Xte,ytr,yte = chronological_train_test_split(X,y,meta,.2,"Experiment")
    rows = []
    models = build_candidate_models(random_state=random_state, include_optional=include_optional)
    for name, model in models.items():
        fitted = clone(model).fit(Xtr,ytr)
        pred = fitted.predict(Xte)
        rows.append({"Model":name, **calculate_forecast_metrics(yte,pred)})
    persistence_col = f"{TARGET}_current"
    rows.append({"Model":"Persistence", **calculate_forecast_metrics(yte,Xte[persistence_col])})
    rows.append({"Model":"TrainingMean", **calculate_forecast_metrics(yte,mean_baseline(ytr,len(yte)))})
    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    best_name = table.iloc[0]["Model"]
    if best_name in models:
        best_model = clone(models[best_name]).fit(Xtr,ytr)
        ForecastEngine(best_model,list(X.columns),TARGET,horizon_steps=int(HORIZON)).save(
            Path("models/forecasting") / f"nlr_{HORIZON}_step_{best_name.lower()}.joblib"
        )
    return table

def evaluate_loeo(X,y,meta,random_state,include_optional=True):
    rows = []
    model_templates = build_candidate_models(random_state=random_state, include_optional=include_optional)
    groups = meta["Experiment"]
    for held_out in groups.unique():
        test_idx = meta.index[groups == held_out]
        train_idx = meta.index[groups != held_out]
        Xtr,Xte,ytr,yte = X.loc[train_idx],X.loc[test_idx],y.loc[train_idx],y.loc[test_idx]
        fold_rows = []
        for name, template in model_templates.items():
            fitted = clone(template).fit(Xtr,ytr)
            metrics = calculate_forecast_metrics(yte,fitted.predict(Xte))
            fold_rows.append({"Experiment":held_out,"Model":name,**metrics})
        fold_rows.append({"Experiment":held_out,"Model":"Persistence",
                          **calculate_forecast_metrics(yte,Xte[f"{TARGET}_current"])})
        rows.extend(fold_rows)
    return pd.DataFrame(rows)

def main():
    global HORIZON
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["chronological","loeo","both"], default="both")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1,3,5,10])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-optional", action="store_true",
                        help="Use only sklearn models; omit XGBoost/LightGBM/CatBoost.")
    args = parser.parse_args()
    out = Path("reports/forecasting")
    out.mkdir(parents=True,exist_ok=True)
    Path("models/forecasting").mkdir(parents=True,exist_ok=True)
    for HORIZON in args.horizons:
        if HORIZON < 1:
            raise ValueError("Horizons must be positive integers.")
        print(f"\n=== Direct forecast horizon: {HORIZON} row(s) ===", flush=True)
        X,y,meta = build_dataset(HORIZON)
        print(f"Samples={len(X):,}; features={X.shape[1]}; experiments={meta.Experiment.nunique()}")
        if args.mode in ("chronological","both"):
            result = evaluate_chronological(X,y,meta,args.random_state,include_optional=not args.skip_optional)
            result.to_csv(out/f"chronological_{HORIZON}_step.csv",index=False)
            print("\nChronological results:\n",result.to_string(index=False))
        if args.mode in ("loeo","both"):
            loeo = evaluate_loeo(X,y,meta,args.random_state,include_optional=not args.skip_optional)
            loeo.to_csv(out/f"loeo_{HORIZON}_step_by_experiment.csv",index=False)
            summary = loeo.groupby("Model")[["MAE","RMSE","R2","MAPE_percent","sMAPE_percent"]].mean().sort_values("MAE")
            summary.to_csv(out/f"loeo_{HORIZON}_step_summary.csv")
            print("\nLOEO mean metrics:\n",summary.to_string())
    print(f"\nReports: {out.resolve()}")

if __name__ == "__main__":
    main()
