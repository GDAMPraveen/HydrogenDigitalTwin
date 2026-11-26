import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/processed/nlr/nlr_minute_candidate.csv")

target = "IVAL_f_FM011_Flow_mean"

features = [
    "H2E_f_NLR_CurrentCmd (A)_mean",
    "H2E_f_NLR_CurrentCmd_mean",
    "Power Supply Average Voltage (Vdc)_mean",
    "H2E_n_PSU_A_Power_mean",
    "H2E_f_PSU_A_Current_mean",
    "H2E_f_CG220_LFLPer_mean",
    "H2E_f_DPS340_DPdegC_mean",
    "H2E_f_DPS341_DPdegC_mean",
    "H2E_f_DPS341_DPppmV_mean",
    "H2E_f_LS201_H2OLevel_mean",
    "H2E_f_LS301_H2OLevel_mean",
    "H2E_f_PT264_Pressure_mean",
    "H2E_f_PT307_Pressure_mean",
    "H2E_f_PT312_Pressure_mean",
    "H2E_f_PT604_Pressure_mean",
    "H2E_f_RS209_Resistivity_mean",
    "H2E_f_RS507_Resistivity_mean",
    "H2E_f_TE218_Temp_mean",
    "H2E_f_TE219_Temp_mean",
    "H2E_f_TE338_Temp_mean",
    "H2E_f_TE601_Temp_mean",
    "H2E_f_TT645_Temp_mean",
    "H2E_f_TT646_Temp_mean",
    "H2E_f_TT647_Temp_mean"
]

X = df[features]
y = df[target]
groups = df["Experiment"]

logo = LeaveOneGroupOut()
results = []

for train_idx, test_idx in logo.split(X, y, groups):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=0.01))
    ])

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = model.predict(X.iloc[test_idx])

    results.append({
        "Experiment": groups.iloc[test_idx].iloc[0],
        "Samples": len(test_idx),
        "MAE": mean_absolute_error(y.iloc[test_idx], predictions),
        "RMSE": np.sqrt(mean_squared_error(y.iloc[test_idx], predictions)),
        "R2": r2_score(y.iloc[test_idx], predictions)
    })

experiment_results = pd.DataFrame(results)

print("ALPHA 0.01 — LEAVE-ONE-EXPERIMENT-OUT RESULTS")
print(experiment_results.to_string(index=False))

print("\nSUMMARY")
print(experiment_results[["MAE", "RMSE", "R2"]].describe().round(5))