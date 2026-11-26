
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

    duplicate_mask = df.duplicated(
        subset=COLUMNS,
        keep="first",
    )

    duplicate_indices = df.index[duplicate_mask]

    details = df.loc[duplicate_indices].copy()
    details.insert(0, "original_row_index", duplicate_indices)

    # Include the immediately preceding and following observations.
    context_indices = set()

    for idx in duplicate_indices:
        context_indices.update(
            i for i in range(max(0, idx - 1), min(len(df), idx + 2))
        )

    context = df.loc[sorted(context_indices)].copy()
    context.insert(0, "original_row_index", context.index)

    details_path = REPORT_DIR / "norce_test4_duplicate_occurrences.csv"
    context_path = REPORT_DIR / "norce_test4_duplicate_context.csv"

    details.to_csv(details_path, index=False)
    context.to_csv(context_path, index=False)

    print("TEST4 DUPLICATE INSPECTION")
    print(f"Duplicate occurrences: {len(duplicate_indices)}")
    print(f"Unique context rows: {len(context)}")

    print("\nFirst 15 duplicate occurrences:")
    print(details.head(15).to_string(index=False))

    print(f"\nSaved: {details_path}")
    print(f"Saved: {context_path}")


if __name__ == "__main__":
    main()
