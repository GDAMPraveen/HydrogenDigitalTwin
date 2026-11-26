from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ZIP_PATH = ROOT / "data/nlr/raw/characterization/characterization_200.zip"
REPORT_DIR = ROOT / "reports/physics"

FLOW = "IVAL_f_FM011_Flow"
CURRENT = "H2E_f_PSU_A_Current"

SIGNALS = {
    "actual_current_A": CURRENT,
    "command_current_A": "H2E_f_NLR_CurrentCmd (A)",
    "measured_flow_kg_hr": FLOW,
    "calculated_rate_kg_hr": "H2E_f_Elec_CalcProdRate",
    "voltage_V": "Power Supply Average Voltage (Vdc)",
    "reported_power_kW": "H2E_n_PSU_A_Power",
    "H2_separator_pressure_barg": "H2E_f_PT307_Pressure",
    "H2_outlet_pressure_barg": "H2E_f_PT312_Pressure",
    "stack_inlet_pressure_barg": "H2E_f_PT264_Pressure",
    "water_inlet_temp_C": "H2E_f_TE218_Temp",
    "water_return_temp_C": "H2E_f_TE219_Temp",
    "H2_after_cooling_temp_C": "H2E_f_TE338_Temp",
    "coolant_inlet_temp_C": "H2E_f_TE601_Temp",
    "H2_crossover_LFL": "H2E_f_CG220_LFLPer",
}


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with ZipFile(ZIP_PATH) as archive:
        data = pd.read_csv(
            BytesIO(archive.read("characterization_200.csv"))
        )

    required = ["Unnamed: 0", *SIGNALS.values()]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    result = data[required].copy()
    result["timestamp"] = pd.to_datetime(
        result["Unnamed: 0"], errors="coerce"
    )

    result["low_flow"] = result[FLOW] < 0.1
    result["flow_group"] = np.where(
        result["low_flow"], "near_zero_flow", "other_flow"
    )

    # Compare only observations with similar measured current.
    result["current_band_A"] = (
        (result[CURRENT] // 100) * 100
    ).astype("Int64")

    signal_names = list(SIGNALS.keys())

    summary = (
        result.groupby("flow_group")[list(SIGNALS.values())]
        .agg(["count", "mean", "median", "std"])
    )

    matched = (
        result.groupby(["flow_group", "current_band_A"])[
            list(SIGNALS.values())
        ]
        .agg(["count", "mean", "median"])
    )

    episode_id = result["low_flow"].ne(
        result["low_flow"].shift()
    ).cumsum()

    episodes = (
        result[result["low_flow"]]
        .groupby(episode_id)
        .agg(
            start=("timestamp", "first"),
            end=("timestamp", "last"),
            observations=(FLOW, "size"),
            mean_current_A=(CURRENT, "mean"),
            mean_H2_pressure_barg=(
                "H2E_f_PT307_Pressure", "mean"
            ),
            mean_H2_outlet_pressure_barg=(
                "H2E_f_PT312_Pressure", "mean"
            ),
            mean_stack_temp_C=(
                "H2E_f_TE218_Temp", "mean"
            ),
            mean_calculated_rate_kg_hr=(
                "H2E_f_Elec_CalcProdRate", "mean"
            ),
        )
    )

    result.to_csv(
        REPORT_DIR / "nlr_low_flow_process_observations.csv",
        index=False,
    )
    summary.to_csv(
        REPORT_DIR / "nlr_low_flow_group_summary.csv"
    )
    matched.to_csv(
        REPORT_DIR / "nlr_low_flow_current_matched_summary.csv"
    )
    episodes.to_csv(
        REPORT_DIR / "nlr_low_flow_process_episodes.csv"
    )

    print("LOW-FLOW PROCESS DIAGNOSTIC")
    print("Total observations:", len(result))
    print("Near-zero flow observations:", int(result["low_flow"].sum()))
    print("Other flow observations:", int((~result["low_flow"]).sum()))
    print("Near-zero flow episodes:", len(episodes))
    print("\nGROUP SUMMARY")
    print(summary.to_string())
    print("\nCURRENT-MATCHED GROUP COUNTS")
    print(
        result.groupby(["flow_group", "current_band_A"])
        .size().to_string()
    )
    print("\nReports saved to:", REPORT_DIR)


if __name__ == "__main__":
    main()
