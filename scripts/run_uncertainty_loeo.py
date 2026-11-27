"""Evaluate 90% Split Conformal Prediction intervals on 12-fold LOEO.

Run from project root:
python -m scripts.run_uncertainty_loeo --horizons 1 3 5 10 --alpha 0.10
"""
import argparse
from pathlib import Path
import pandas as pd

from src.forecasting.forecast_evaluation import calculate_forecast_metrics
from src.forecasting.train_models import build_candidate_models
from src.integration.hybrid_digital_twin import HybridPEMForecaster
from src.uncertainty.conformal import SplitConformalForecaster, interval_metrics
from scripts.run_hybrid_loeo import TARGET, build_no_history_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--alpha", type=float, default=0.10, help="Miscoverage rate (0.10 = 90%% PI).")
    parser.add_argument("--cal-fraction", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out = Path("reports/uncertainty")
    out.mkdir(parents=True, exist_ok=True)

    base_hgb = build_candidate_models(
        random_state=args.random_state, include_optional=False
    )["HistGradientBoosting"]
    current_col = f"{TARGET}_current"
    target_cov = (1.0 - args.alpha) * 100.0

    horizon_summaries = []

    for horizon in args.horizons:
        print(f"\n=== Conformal Uncertainty LOEO | Horizon: {horizon} row(s) | Target: {target_cov:.0f}% ===", flush=True)
        X, y, meta = build_no_history_dataset(horizon)
        groups = meta["Experiment"]
        rows = []

        for fold, held_out in enumerate(sorted(groups.unique()), start=1):
            test_idx = meta.index[groups == held_out]
            train_idx = meta.index[groups != held_out]

            conformal = SplitConformalForecaster(
                forecaster_factory=lambda: HybridPEMForecaster(
                    base_hgb, mode="physics_residual", current_flow_col=current_col
                ),
                alpha=args.alpha,
                cal_fraction=args.cal_fraction,
            )
            conformal.fit(X.loc[train_idx], y.loc[train_idx], meta.loc[train_idx])
            pred, lower, upper = conformal.predict_interval(X.loc[test_idx])

            y_test = y.loc[test_idx].to_numpy(dtype=float)
            pt_metrics = calculate_forecast_metrics(y_test, pred)
            pi_metrics = interval_metrics(y_test, lower, upper, alpha=args.alpha)

            rows.append({
                "Horizon": horizon,
                "Experiment": held_out,
                "MAE": pt_metrics["MAE"],
                "RMSE": pt_metrics["RMSE"],
                "Quantile_q": conformal.quantile_,
                **pi_metrics,
                "Lower_Min": float(lower.min()),
            })
            print(
                f"  [{fold:02d}/12] {held_out}: Cov={pi_metrics['Coverage_Percent']:.1f}% | "
                f"Width={pi_metrics['Avg_Width_kg_hr']:.3f} | q={conformal.quantile_:.3f}",
                flush=True,
            )

        df_h = pd.DataFrame(rows)
        df_h.to_csv(out / f"conformal_loeo_{horizon}_step_by_experiment.csv", index=False)

        horizon_summaries.append({
            "Horizon": horizon,
            "Target_Coverage_%": target_cov,
            "Mean_Coverage_%": df_h["Coverage_Percent"].mean(),
            "Min_Exp_Coverage_%": df_h["Coverage_Percent"].min(),
            "Worst_Experiment": df_h.loc[df_h["Coverage_Percent"].idxmin(), "Experiment"],
            "Mean_Width_kg_hr": df_h["Avg_Width_kg_hr"].mean(),
            "Mean_Winkler": df_h["Winkler_Score"].mean(),
            "Mean_MAE": df_h["MAE"].mean(),
        })

    summary_df = pd.DataFrame(horizon_summaries)
    summary_df.to_csv(out / "conformal_loeo_summary.csv", index=False)
    print("\n========== CONFORMAL UNCERTAINTY SUMMARY ==========")
    print(summary_df.to_string(index=False))
    print(f"\nReports saved to: {out.resolve()}")


if __name__ == "__main__":
    main()