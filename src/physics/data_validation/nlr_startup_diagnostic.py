from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

ZIP_PATH = (
    ROOT
    / "data/nlr/raw/characterization/characterization_200.zip"
)

REPORT_DIR = ROOT / "reports/physics"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD CHARACTERIZATION DATA
# --------------------------------------------------

def load_data():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {ZIP_PATH}")

    with ZipFile(ZIP_PATH, "r") as archive:
        data = pd.read_csv(
            BytesIO(archive.read("characterization_200.csv"))
        )

    data["timestamp"] = pd.to_datetime(
        data["Unnamed: 0"],
        errors="coerce"
    )

    return data.dropna(subset=["timestamp"]).sort_values("timestamp")


# --------------------------------------------------
# STARTUP TRANSIENT INSPECTION
# --------------------------------------------------

def inspect_startup(data):
    start_time = data["timestamp"].iloc[0]
    end_time = start_time + pd.Timedelta(seconds=120)

    startup = data[
        (data["timestamp"] >= start_time)
        & (data["timestamp"] <= end_time)
    ].copy()

    startup["elapsed_seconds"] = (
        startup["timestamp"] - start_time
    ).dt.total_seconds()

    # Select observations at 5-second intervals.
    startup = startup[
        startup["elapsed_seconds"] % 5 == 0
    ].copy()

    # Electrical consistency calculation.
    startup["VI_power_kW"] = (
        startup["Power Supply Average Voltage (Vdc)"]
        * startup["H2E_f_PSU_A_Current"]
        / 1000
    )

    startup["power_error_kW"] = (
        startup["H2E_n_PSU_A_Power"]
        - startup["VI_power_kW"]
    )

    # Compare measured flow and unverified calculated rate.
    startup["flow_rate_error"] = (
        startup["H2E_f_Elec_CalcProdRate"]
        - startup["IVAL_f_FM011_Flow"]
    )

    columns = [
        "timestamp",
        "elapsed_seconds",
        "H2E_f_PSU_A_Current",
        "Power Supply Average Voltage (Vdc)",
        "H2E_n_PSU_A_Power",
        "VI_power_kW",
        "power_error_kW",
        "IVAL_f_FM011_Flow",
        "H2E_f_Elec_CalcProdRate",
        "flow_rate_error",
    ]

    return startup[columns]


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

def create_summary(startup):
    summary = {
        "startup_observations": len(startup),
        "first_timestamp": startup["timestamp"].min(),
        "last_timestamp": startup["timestamp"].max(),
        "mean_measured_flow_kg_hr":
            startup["IVAL_f_FM011_Flow"].mean(),
        "mean_calculated_rate":
            startup["H2E_f_Elec_CalcProdRate"].mean(),
        "mean_absolute_flow_rate_error":
            startup["flow_rate_error"].abs().mean(),
        "mean_absolute_power_error_kW":
            startup["power_error_kW"].abs().mean(),
    }

    return pd.DataFrame([summary])


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("\nLoading NLR characterization data...")

    data = load_data()

    startup = inspect_startup(data)
    summary = create_summary(startup)

    startup.to_csv(
        REPORT_DIR / "nlr_startup_120s_diagnostic.csv",
        index=False
    )

    summary.to_csv(
        REPORT_DIR / "nlr_startup_diagnostic_summary.csv",
        index=False
    )

    print("\nSTARTUP DIAGNOSTIC")
    print(startup.to_string(index=False))

    print("\nSTARTUP SUMMARY")
    print(summary.to_string(index=False))

    print(f"\nReports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
