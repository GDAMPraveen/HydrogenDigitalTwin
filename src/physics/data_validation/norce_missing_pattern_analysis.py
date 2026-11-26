
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


def analyze_patterns(name, path):
    df = pd.read_csv(path)
    missing = df[COLUMNS].isna()

    # Count rows with different numbers of missing variables.
    missing_per_row = missing.sum(axis=1)

    summary = {
        "dataset": name,
        "total_rows": len(df),
        "rows_with_any_missing": int((missing_per_row > 0).sum()),
        "rows_with_all_five_missing": int((missing_per_row == 5).sum()),
        "rows_with_partial_missing": int(
            ((missing_per_row > 0) & (missing_per_row < 5)).sum()
        ),
        "rows_with_no_missing": int((missing_per_row == 0).sum()),
    }

    patterns = (
        missing.astype(int)
        .groupby(COLUMNS)
        .size()
        .reset_index(name="row_count")
    )

    patterns["missing_variables"] = patterns.apply(
        lambda row: ", ".join(
            column for column in COLUMNS if row[column] == 1
        ) or "None",
        axis=1,
    )

    patterns.insert(0, "dataset", name)

    return summary, patterns


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_patterns = []

    for name, path in DATASETS.items():
        summary, patterns = analyze_patterns(name, path)
        summaries.append(summary)
        all_patterns.append(patterns)

    summary_df = pd.DataFrame(summaries)
    pattern_df = pd.concat(all_patterns, ignore_index=True)

    summary_path = REPORT_DIR / "norce_missing_row_summary.csv"
    pattern_path = REPORT_DIR / "norce_missing_patterns.csv"

    summary_df.to_csv(summary_path, index=False)
    pattern_df.to_csv(pattern_path, index=False)

    print("NORCE MISSING-ROW PATTERN ANALYSIS")
    print("\nROW SUMMARY")
    print(summary_df.to_string(index=False))

    print("\nMISSING VARIABLE COMBINATIONS")
    print(pattern_df.to_string(index=False))

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {pattern_path}")


if __name__ == "__main__":
    main()
