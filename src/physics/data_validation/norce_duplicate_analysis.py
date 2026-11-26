
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATASETS = {
    "test2": ROOT / "data/norce/test2_subset.csv",
    "test3": ROOT / "data/norce/test3_subset.csv",
    "test4": ROOT / "data/norce/test4_subset.csv",
}

COLUMNS = [
    "PS-I-MON",
    "H-P1",
    "O-P1",
    "T-ELY-CH1",
    "CV-mean",
]

REPORT_DIR = ROOT / "reports/physics/norce"


def analyze_duplicates(name, path):
    df = pd.read_csv(path)

    duplicate_mask = df.duplicated(subset=COLUMNS, keep=False)
    duplicate_rows = df.loc[duplicate_mask].copy()

    duplicate_rows.insert(
        0, "original_row_index", duplicate_rows.index
    )

    exact_duplicate_count = int(
        df.duplicated(subset=COLUMNS, keep="first").sum()
    )

    consecutive_duplicate_count = int(
        (
            df[COLUMNS].eq(df[COLUMNS].shift()).all(axis=1)
        ).sum()
    )

    summary = {
        "dataset": name,
        "total_rows": len(df),
        "duplicate_rows_after_first": exact_duplicate_count,
        "rows_in_duplicate_groups": len(duplicate_rows),
        "consecutive_repeated_rows": consecutive_duplicate_count,
        "unique_measurement_rows": len(
            df.drop_duplicates(subset=COLUMNS)
        ),
    }

    return summary, duplicate_rows


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    duplicate_details = []

    for name, path in DATASETS.items():
        summary, details = analyze_duplicates(name, path)
        summaries.append(summary)

        if not details.empty:
            details.insert(0, "dataset", name)
            duplicate_details.append(details)

    summary_df = pd.DataFrame(summaries)

    summary_path = REPORT_DIR / "norce_duplicate_summary.csv"
    details_path = REPORT_DIR / "norce_duplicate_records.csv"

    summary_df.to_csv(summary_path, index=False)

    if duplicate_details:
        pd.concat(
            duplicate_details, ignore_index=True
        ).to_csv(details_path, index=False)

    print("NORCE DUPLICATE ANALYSIS")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")
    print(f"Details saved: {details_path}")


if __name__ == "__main__":
    main()
