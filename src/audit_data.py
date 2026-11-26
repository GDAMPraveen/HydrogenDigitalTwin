from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
FILES = [
    "test4_subset.csv",
    "test2_subset.csv",
    "test3_subset.csv",
]

FEATURES = ["PS-I-MON", "H-P1", "O-P1", "T-ELY-CH1"]
TARGET = "CV-mean"
EXPECTED_COLUMNS = FEATURES + [TARGET]


def audit_file(filename):
    path = DATA_DIR / filename
    df = pd.read_csv(path)

    print("\n" + "=" * 72)
    print(f"DATASET: {filename}")
    print("=" * 72)

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        print("ERROR: Missing expected columns:", sorted(missing_columns))
        print("Actual columns:", df.columns.tolist())
        return

    print(f"Rows: {len(df):,}")
    print("Missing values:")
    print(df[EXPECTED_COLUMNS].isna().sum().to_string())

    print("\nMissing target rows:", int(df[TARGET].isna().sum()))
    print("Rows with any missing value:",
          int(df[EXPECTED_COLUMNS].isna().any(axis=1).sum()))

    print("\nFeature and target ranges:")
    print(df[EXPECTED_COLUMNS].describe().round(4).to_string())

    print("\nFirst 5 rows:")
    print(df[EXPECTED_COLUMNS].head().round(4).to_string(index=False))

    print("\nLast 5 rows:")
    print(df[EXPECTED_COLUMNS].tail().round(4).to_string(index=False))

    print("\nLowest and highest voltage observations:")
    valid = df.dropna(subset=[TARGET])
    if not valid.empty:
        print("Lowest:")
        print(valid.nsmallest(3, TARGET)[EXPECTED_COLUMNS].to_string(index=False))
        print("Highest:")
        print(valid.nlargest(3, TARGET)[EXPECTED_COLUMNS].to_string(index=False))

    # Save an audit summary; never overwrite the source CSV.
    summary = {
        "dataset": filename,
        "rows": len(df),
        "missing_cells": int(df[EXPECTED_COLUMNS].isna().sum().sum()),
        "rows_with_missing": int(
            df[EXPECTED_COLUMNS].isna().any(axis=1).sum()
        ),
        "missing_voltage": int(df[TARGET].isna().sum()),
        "voltage_min": df[TARGET].min(),
        "voltage_median": df[TARGET].median(),
        "voltage_max": df[TARGET].max(),
    }
    return summary


def main():
    summaries = []

    for filename in FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"WARNING: File not found: {path}")
            continue

        summary = audit_file(filename)
        if summary:
            summaries.append(summary)

    if summaries:
        output = DATA_DIR / "audit_summary.csv"
        pd.DataFrame(summaries).to_csv(output, index=False)
        print(f"\nAudit summary saved to: {output}")


if __name__ == "__main__":
    main()
