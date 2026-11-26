from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[3]
ZIP_PATH = ROOT / "data/nlr/raw/characterization/characterization_200.zip"
REPORT_DIR = ROOT / "reports/physics"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    with ZipFile(ZIP_PATH, "r") as archive:
        data = pd.read_csv(
            BytesIO(archive.read("characterization_200.csv"))
        )

    data["timestamp"] = pd.to_datetime(
        data["Unnamed: 0"], errors="coerce"
    )

    return data


def main():
    data = load_data()

    # 1. Timestamp validation
    intervals = (
        data["timestamp"]
        .sort_values()
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    timestamp_report = pd.DataFrame([{
        "total_rows": len(data),
        "duplicate_timestamps": int(data["timestamp"].duplicated().sum()),
        "median_interval_seconds": intervals.median(),
        "minimum_interval_seconds": intervals.min(),
        "maximum_interval_seconds": intervals.max(),
        "one_second_intervals": int((intervals == 1).sum()),
    }])

    # 2. Electrical consistency
    power = data[[
        "timestamp",
        "H2E_f_PSU_A_Current",
        "Power Supply Average Voltage (Vdc)",
        "H2E_n_PSU_A_Power",
    ]].copy()

    power["VI_power_kW"] = (
        power["H2E_f_PSU_A_Current"]
        * power["Power Supply Average Voltage (Vdc)"]
        / 1000
    )

    power["power_error_kW"] = (
        power["H2E_n_PSU_A_Power"] - power["VI_power_kW"]
    )

    # 3. Measured flow vs unverified calculated rate
    flow = data[[
        "timestamp",
        "IVAL_f_FM011_Flow",
        "H2E_f_Elec_CalcProdRate",
    ]].copy()

    flow["flow_rate_error"] = (
        flow["H2E_f_Elec_CalcProdRate"]
        - flow["IVAL_f_FM011_Flow"]
    )

    # 4. Summary
    summary = pd.DataFrame([{
        "rows": len(data),
        "mean_reported_power_kW": data["H2E_n_PSU_A_Power"].mean(),
        "mean_VI_power_kW": power["VI_power_kW"].mean(),
        "mean_absolute_power_error_kW": power["power_error_kW"].abs().mean(),
        "mean_measured_flow_kg_hr": data["IVAL_f_FM011_Flow"].mean(),
        "mean_calculated_rate": data["H2E_f_Elec_CalcProdRate"].mean(),
        "flow_rate_correlation": data[
            ["IVAL_f_FM011_Flow", "H2E_f_Elec_CalcProdRate"]
        ].corr().iloc[0, 1],
    }])

    # Save reports
    timestamp_report.to_csv(
        REPORT_DIR / "nlr_timestamp_validation.csv", index=False
    )

    power.to_csv(
        REPORT_DIR / "nlr_power_consistency.csv", index=False
    )

    flow.to_csv(
        REPORT_DIR / "nlr_production_rate_validation.csv", index=False
    )

    summary.to_csv(
        REPORT_DIR / "nlr_physics_validation_summary.csv", index=False
    )

    print("\nNLR PHYSICS VALIDATION COMPLETED")
    print(summary.to_string(index=False))
    print(f"\nReports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
