from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

ZIP_PATH = (
    ROOT / "data/nlr/raw/characterization/characterization_200.zip"
)

REPORT_DIR = ROOT / "reports/physics"

CURRENT_COL = "H2E_f_PSU_A_Current"
FLOW_COL = "IVAL_f_FM011_Flow"
TIME_COL = "Unnamed: 0"

FARADAY_CONSTANT = 96485.33212
HYDROGEN_MOLAR_MASS = 0.00201588


def calculate_one_cell_ideal_rate(current_a):
    return (
        current_a
        * HYDROGEN_MOLAR_MASS
        * 3600
        / (2 * FARADAY_CONSTANT)
    )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        raise FileNotFoundError(ZIP_PATH)

    with ZipFile(ZIP_PATH, "r") as archive:
        data = pd.read_csv(
            BytesIO(archive.read("characterization_200.csv"))
        )

    required = [TIME_COL, CURRENT_COL, FLOW_COL]

    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    result = data[required].copy()

    result["timestamp"] = pd.to_datetime(
        result[TIME_COL], errors="coerce"
    )

    result["elapsed_seconds"] = (
        result["timestamp"] - result["timestamp"].iloc[0]
    ).dt.total_seconds()

    result["one_cell_ideal_h2_kg_hr"] = (
        result[CURRENT_COL].apply(calculate_one_cell_ideal_rate)
    )

    result["effective_cell_count_times_efficiency"] = np.where(
        (result[CURRENT_COL] > 0)
        & (result[FLOW_COL] > 0)
        & (result["one_cell_ideal_h2_kg_hr"] > 0),
        result[FLOW_COL] / result["one_cell_ideal_h2_kg_hr"],
        np.nan,
    )

    result["operating_period"] = np.where(
        result["elapsed_seconds"] <= 120,
        "startup_0_120s",
        "after_startup",
    )

    valid = result[
        "effective_cell_count_times_efficiency"
    ].dropna()

    summary_rows = []

    for period, group in result.groupby("operating_period"):
        values = group[
            "effective_cell_count_times_efficiency"
        ].dropna()

        summary_rows.append({
            "period": period,
            "total_observations": len(group),
            "valid_observations": len(values),
            "mean_effective_factor": values.mean(),
            "median_effective_factor": values.median(),
            "std_effective_factor": values.std(),
            "q05_effective_factor": values.quantile(0.05),
            "q95_effective_factor": values.quantile(0.95),
            "minimum_effective_factor": values.min(),
            "maximum_effective_factor": values.max(),
        })

    summary = pd.DataFrame(summary_rows)

    result.to_csv(
        REPORT_DIR / "nlr_effective_faraday_observations.csv",
        index=False,
    )

    summary.to_csv(
        REPORT_DIR / "nlr_effective_faraday_summary.csv",
        index=False,
    )

    print("\nEFFECTIVE FARADAY PARAMETER ANALYSIS")
    print(f"Original observations: {len(data)}")
    print(f"Valid effective-factor observations: {len(valid)}")
    print("\nPeriod summary:")
    print(summary.to_string(index=False))

    print("\nReports saved to:", REPORT_DIR)


if __name__ == "__main__":
    main()
