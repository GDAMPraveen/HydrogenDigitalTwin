"""Evaluate OOD detection and sensor-fault resilience on 12-fold LOEO.

Run from project root:
python -m scripts.run_resilience_benchmark --horizon 1
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.forecasting.forecast_evaluation import calculate_forecast_metrics
from src.forecasting.train_models import build_candidate_models
from src.integration.hybrid_digital_twin import (
    CURRENT_COL,
    VOLTAGE_COL,
    HybridPEMForecaster,
)
from src.resilience.ood_and_faults import OODDetector, inject_sensor_fault
from scripts.run_hybrid_loeo import TARGET, build_no_history_dataset


FAULTS = ("clean", "missing_20pct", "gaussian_noise", "stuck_sensor", "linear_drift")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out = Path("reports/resilience")
    out.mkdir(parents=True, exist_ok=True)

    X, y, meta = build_no_history_dataset(args.horizon)
    groups = meta["Experiment"]
    current_col = f"{TARGET}_current"
    base_hgb = build_candidate_models(
        random_state=args.random_state, include_optional=False
    )["HistGradientBoosting"]

    ood_rows = []
    fault_rows = []
    rng = np.random.default_rng(args.random_state)

    print(f"\n=== OOD & Sensor-Fault Resilience Benchmark | Horizon: {args.horizon} ===", flush=True)

    for fold, held_out in enumerate(sorted(groups.unique()), start=1):
        test_idx = meta.index[groups == held_out]
        train_idx = meta.index[groups != held_out]
        Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
        ytr, yte = y.loc[train_idx], y.loc[test_idx]

        # 1. OOD Detection (fit strictly on training folds)
        detector = OODDetector(
            contamination=args.contamination, random_state=args.random_state
        ).fit(Xtr)
        is_ood, scores = detector.predict_ood(Xte)
        ood_rate = float(np.mean(is_ood) * 100.0)
        ood_rows.append({
            "Experiment": held_out,
            "OOD_Flagged_%": ood_rate,
            "Mean_Anomaly_Score": float(np.mean(scores)),
            "Max_Anomaly_Score": float(np.max(scores)),
        })

        # 2. Sensor Fault Resilience (fit Hybrid model once per fold)
        model = HybridPEMForecaster(
            base_hgb, mode="physics_residual", current_flow_col=current_col
        ).fit(Xtr, ytr)

        for sensor in (CURRENT_COL, VOLTAGE_COL):
            for fault in FAULTS:
                X_fault = inject_sensor_fault(Xte, sensor, fault, rng)
                pred = model.predict(X_fault)
                metrics = calculate_forecast_metrics(yte, pred)
                fault_rows.append({
                    "Experiment": held_out,
                    "Sensor": sensor,
                    "Fault": fault,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                })

        print(f"  [{fold:02d}/12] {held_out}: OOD_Flagged={ood_rate:.1f}%", flush=True)

    ood_df = pd.DataFrame(ood_rows).sort_values("OOD_Flagged_%", ascending=False)
    fault_df = pd.DataFrame(fault_rows)
    fault_summary = (
        fault_df.groupby(["Sensor", "Fault"])[["MAE", "RMSE"]]
        .mean()
        .reset_index()
        .sort_values(["Sensor", "MAE"])
    )

    ood_df.to_csv(out / f"ood_loeo_{args.horizon}_step.csv", index=False)
    fault_df.to_csv(out / f"sensor_faults_loeo_{args.horizon}_step_detailed.csv", index=False)
    fault_summary.to_csv(out / f"sensor_faults_loeo_{args.horizon}_step_summary.csv", index=False)

    print("\n========== OOD DETECTION BY EXPERIMENT ==========")
    print(ood_df.to_string(index=False))

    print("\n========== SENSOR FAULT RESILIENCE SUMMARY ==========")
    print(fault_summary.to_string(index=False))
    print(f"\nReports saved to: {out.resolve()}")


if __name__ == "__main__":
    main()