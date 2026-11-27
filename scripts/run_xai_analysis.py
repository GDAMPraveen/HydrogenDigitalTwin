"""Compute out-of-experiment Permutation Feature Importance for the Hybrid PEM Twin.

Run from project root:
python -m scripts.run_xai_analysis --horizon 1 --repeats 5
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from src.forecasting.train_models import build_candidate_models
from src.integration.hybrid_digital_twin import FaradayPhysicsCalibrator
from scripts.run_hybrid_loeo import TARGET, build_no_history_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out = Path("reports/xai")
    out.mkdir(parents=True, exist_ok=True)

    X, y, meta = build_no_history_dataset(args.horizon)
    groups = meta["Experiment"]
    current_col = f"{TARGET}_current"
    base_hgb = build_candidate_models(
        random_state=args.random_state, include_optional=False
    )["HistGradientBoosting"]

    fold_importances = []
    print(f"\n=== XAI Permutation Importance (LOEO) | Horizon: {args.horizon} ===", flush=True)

    for fold, held_out in enumerate(sorted(groups.unique()), start=1):
        test_idx = meta.index[groups == held_out]
        train_idx = meta.index[groups != held_out]
        Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
        ytr, yte = y.loc[train_idx], y.loc[test_idx]

        cal = FaradayPhysicsCalibrator().fit(Xtr, Xtr[current_col])
        Xtr_aug = cal.augment_features(Xtr, current_col)
        Xte_aug = cal.augment_features(Xte, current_col)

        ytr_res = ytr.to_numpy(dtype=float) - Xtr_aug["phys_faraday_flow_kg_hr"].to_numpy()
        yte_res = yte.to_numpy(dtype=float) - Xte_aug["phys_faraday_flow_kg_hr"].to_numpy()

        model = clone(base_hgb).fit(Xtr_aug, ytr_res)

        perm = permutation_importance(
            model,
            Xte_aug,
            yte_res,
            scoring="neg_mean_absolute_error",
            n_repeats=args.repeats,
            random_state=args.random_state + fold,
            n_jobs=-1,
        )

        for feat, mean_imp, std_imp in zip(
            Xte_aug.columns, perm.importances_mean, perm.importances_std
        ):
            fold_importances.append({
                "Experiment": held_out,
                "Feature": feat,
                "MAE_Increase_kg_hr": float(mean_imp),
                "Std": float(std_imp),
            })

        print(f"  [{fold:02d}/12] {held_out} done", flush=True)

    df_imp = pd.DataFrame(fold_importances)
    summary = (
        df_imp.groupby("Feature")
        .agg(
            Mean_MAE_Increase=("MAE_Increase_kg_hr", "mean"),
            Std_Across_Folds=("MAE_Increase_kg_hr", "std"),
            Max_MAE_Increase=("MAE_Increase_kg_hr", "max"),
        )
        .sort_values("Mean_MAE_Increase", ascending=False)
        .reset_index()
    )

    df_imp.to_csv(out / f"xai_permutation_loeo_{args.horizon}_step_by_experiment.csv", index=False)
    summary.to_csv(out / f"xai_permutation_loeo_{args.horizon}_step_summary.csv", index=False)

    print("\n========== TOP 15 MOST IMPORTANT FEATURES (LOEO) ==========")
    print(summary.head(15).to_string(index=False))
    print(f"\nReports saved to: {out.resolve()}")


if __name__ == "__main__":
    main()