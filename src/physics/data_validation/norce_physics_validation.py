
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATASETS = {
    "test2": ROOT / "data/norce/test2_subset.csv",
    "test3": ROOT / "data/norce/test3_subset.csv",
    "test4": ROOT / "data/norce/test4_subset.csv",
}

REQUIRED_COLUMNS = [
    "PS-I-MON",
    "H-P1",
    "O-P1",
    "T-ELY-CH1",
    "CV-mean",
]

REPORT_DIR = ROOT / "reports/physics/norce"


def inspect_dataset(name, path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name}: Missing columns: {missing_columns}"
        )

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    summary = {
        "dataset": name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df[REQUIRED_COLUMNS].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for column in REQUIRED_COLUMNS:
        values = df[column].dropna()

        summary[f"{column}_min"] = (
            values.min() if not values.empty else np.nan
        )
        summary[f"{column}_mean"] = (
            values.mean() if not values.empty else np.nan
        )
        summary[f"{column}_max"] = (
            values.max() if not values.empty else np.nan
        )

    # Derived pressure difference.
    # Sign is kept as H-P1 minus O-P1, without assuming
    # a preferred operating direction.
    df["pressure_difference_H_minus_O_bar"] = (
        df["H-P1"] - df["O-P1"]
    )

    summary["pressure_difference_mean_bar"] = (
        df["pressure_difference_H_minus_O_bar"].mean()
    )

    # The README documents CV-mean as mean cell voltage.
    # This is not total stack voltage.
    summary["mean_cell_voltage_V"] = df["CV-mean"].mean()

    # Current density uses the README's documented
    # total active area of 50 cm².
    # This remains conditional on that specification
    # applying to the supplied experiment.
    summary["current_density_mean_A_cm2"] = (
        df["PS-I-MON"].mean() / 50.0
    )

    return summary, df


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    for name, path in DATASETS.items():
        summary, df = inspect_dataset(name, path)
        summaries.append(summary)

        print(f"\n{name.upper()}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Missing values: {summary['missing_values']}")
        print(f"Duplicate rows: {summary['duplicate_rows']}")

    summary_df = pd.DataFrame(summaries)

    output_path = REPORT_DIR / "nor​​ce_dataset_physics_validation.csv"
    summary_df.to_csv(output_path, index=False)

    print("\nNORCE DATASET PHYSICS VALIDATION COMPLETED")
    print(summary_df.to_string(index=False))
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
