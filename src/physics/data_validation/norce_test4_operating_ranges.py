
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data/norce/test4_subset.csv"
REPORT_DIR = ROOT / "reports/physics/norce"

COLUMNS = [
    "PS-I-MON",
    "H-P1",
    "O-P1",
    "T-ELY-CH1",
    "CV-mean",
]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    summaries = []

    for column in COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")

        summaries.append({
            "variable": column,
            "minimum": values.min(),
            "maximum": values.max(),
            "negative_count": int((values < 0).sum()),
            "zero_count": int((values == 0).sum()),
            "positive_count": int((values > 0).sum()),
            "near_zero_abs_le_0_05": int(
                (values.abs() <= 0.05).sum()
            ),
        })

    summary = pd.DataFrame(summaries)

    output_path = REPORT_DIR / "norce_test4_operating_ranges.csv"
    summary.to_csv(output_path, index=False)

    print("NORCE TEST4 OPERATING RANGE ANALYSIS")
    print(summary.to_string(index=False))

    print("\nFirst 20 rows with negative mean cell voltage:")
    negative_voltage = df[df["CV-mean"] < 0]
    print(negative_voltage.head(20).to_string(index=True))

    print("\nFirst 20 rows with negative H2 pressure:")
    negative_h2 = df[df["H-P1"] < 0]
    print(negative_h2.head(20).to_string(index=True))

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
