
"""
NLR dataset audit for the Hydrogen Digital Twin project.

This script:
- Inspects dataset dimensions, columns, data types and missing values.
- Reports duplicate rows and experiment counts.
- Audits the unresolved Unnamed: 31 sensor column.
- Saves audit reports without modifying the source CSV.
"""

from pathlib import Path
import json

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "nlr" / "combined_wind_experiments.csv"
REPORT_DIR = PROJECT_ROOT / "data" / "nlr" / "audit"

UNRESOLVED_SENSOR = "Unnamed: 31"
TEMPERATURE_SENSOR = "H2E_f_TT641_Temp"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"NLR dataset not found: {DATA_PATH}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NLR dataset...")
    df = pd.read_csv(DATA_PATH)

    # Basic dataset information
    missing = df.isna().sum()
    missing = missing[missing > 0]

    experiment_counts = df["Experiment"].value_counts(dropna=False)
    duplicate_count = int(df.duplicated().sum())

    # Unresolved sensor statistics
    sensor_stats = {}

    if UNRESOLVED_SENSOR in df.columns:
        sensor = df[UNRESOLVED_SENSOR].dropna()

        if not sensor.empty:
            sensor_stats = {
                "count": int(sensor.count()),
                "minimum": float(sensor.min()),
                "median": float(sensor.median()),
                "maximum": float(sensor.max()),
                "above_40": int((sensor > 40).sum()),
                "above_80": int((sensor > 80).sum()),
                "above_100": int((sensor > 100).sum()),
            }

    # Check whether the two temperature columns are mutually exclusive
    temperature_relationship = {}

    if TEMPERATURE_SENSOR in df.columns and UNRESOLVED_SENSOR in df.columns:
        named = df[TEMPERATURE_SENSOR]
        unnamed = df[UNRESOLVED_SENSOR]

        temperature_relationship = {
            "both_present": int((named.notna() & unnamed.notna()).sum()),
            "named_only": int((named.notna() & unnamed.isna()).sum()),
            "unnamed_only": int((named.isna() & unnamed.notna()).sum()),
            "both_missing": int((named.isna() & unnamed.isna()).sum()),
        }

    # Generate summary
    summary = {
        "dataset": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": duplicate_count,
        "missing_values": {
            str(column): int(count)
            for column, count in missing.items()
        },
        "experiment_counts": {
            str(name): int(count)
            for name, count in experiment_counts.items()
        },
        "unresolved_sensor_statistics": sensor_stats,
        "temperature_column_relationship": temperature_relationship,
        "source_modified": False,
    }

    # Save reports
    with open(REPORT_DIR / "nlr_audit_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    missing.rename("missing_count").to_csv(
        REPORT_DIR / "nlr_missing_values.csv"
    )

    experiment_counts.rename("row_count").to_csv(
        REPORT_DIR / "nlr_experiment_counts.csv"
    )

    df.dtypes.astype(str).rename("data_type").to_csv(
        REPORT_DIR / "nlr_column_types.csv"
    )

    # Display results
    print("\n========== NLR DATASET AUDIT ==========")
    print(f"Dataset shape: {df.shape}")
    print(f"Duplicate rows: {duplicate_count}")

    print("\nMissing values:")
    print(missing.to_string())

    print("\nExperiment counts:")
    print(experiment_counts.to_string())

    print("\nUnresolved sensor statistics:")
    print(sensor_stats)

    print("\nTemperature column relationship:")
    print(temperature_relationship)

    print("\nAudit reports saved in:")
    print(REPORT_DIR)

    print("\nOriginal dataset was not modified.")


if __name__ == "__main__":
    main()
