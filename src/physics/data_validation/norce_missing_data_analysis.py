
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


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reports = []

    for dataset_name, path in DATASETS.items():
        df = pd.read_csv(path)

        for column in COLUMNS:
            reports.append({
                "dataset": dataset_name,
                "variable": column,
                "missing_count": int(df[column].isna().sum()),
                "missing_percentage": (
                    df[column].isna().mean() * 100
                ),
                "valid_count": int(df[column].notna().sum()),
            })

    result = pd.DataFrame(reports)

    output = REPORT_DIR / "norce_missing_data_by_variable.csv"
    result.to_csv(output, index=False)

    print("NORCE MISSING DATA ANALYSIS")
    print(result.to_string(index=False))
    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
