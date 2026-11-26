
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

ZIP_PATH = (
    ROOT / "data/nlr/raw/characterization/characterization_200.zip"
)

REPORT_DIR = ROOT / "reports/physics"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with ZipFile(ZIP_PATH, "r") as archive:
        data = pd.read_csv(
            BytesIO(archive.read("characterization_200.csv"))
        )

    data["timestamp"] = pd.to_datetime(data["Unnamed: 0"])

    data["elapsed_seconds"] = (
        data["timestamp"] - data["timestamp"].iloc[0]
    ).dt.total_seconds()

    data["VI_power_kW"] = (
        data["Power Supply Average Voltage (Vdc)"]
        * data["H2E_f_PSU_A_Current"]
        / 1000
    )

    data["power_error_kW"] = (
        data["H2E_n_PSU_A_Power"] - data["VI_power_kW"]
    )

    data["absolute_power_error_kW"] = (
        data["power_error_kW"].abs()
    )

    data["relative_power_error_pct"] = (
        data["absolute_power_error_kW"]
        / data["H2E_n_PSU_A_Power"].abs().replace(0, pd.NA)
    ) * 100

    data["operating_period"] = "steady_period"

    data.loc[
        data["elapsed_seconds"] <= 120,
        "operating_period"
    ] = "startup_period"

    summary = (
        data.groupby("operating_period")
        .agg(
            observations=("power_error_kW", "count"),
            mean_reported_power_kW=("H2E_n_PSU_A_Power", "mean"),
            mean_VI_power_kW=("VI_power_kW", "mean"),
            mean_power_error_kW=("power_error_kW", "mean"),
            mean_absolute_error_kW=("absolute_power_error_kW", "mean"),
            median_absolute_error_kW=("absolute_power_error_kW", "median"),
            maximum_absolute_error_kW=("absolute_power_error_kW", "max"),
            mean_relative_error_pct=("relative_power_error_pct", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        REPORT_DIR / "nlr_power_error_period_analysis.csv",
        index=False
    )

    print("\nPOWER ERROR BY OPERATING PERIOD")
    print(summary.to_string(index=False))

    print(f"\nSaved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
