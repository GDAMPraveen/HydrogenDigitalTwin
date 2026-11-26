from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

FILES = [
    "test4_subset.csv",
    "test2_subset.csv",
    "test3_subset.csv",
]


def analyze_file(filename):
    df = pd.read_csv(DATA_DIR / filename)

    print("\n" + "=" * 65)
    print(filename)
    print("=" * 65)

    # Report missingness before any filtering.
    print("Total rows:", len(df))
    print("Rows with missing values:",
          int(df.isna().any(axis=1).sum()))

    # Descriptive operating-state counts; these are exploratory,
    # not claims of verified startup/shutdown labels.
    current = df["PS-I-MON"]
    voltage = df["CV-mean"]
    temperature = df["T-ELY-CH1"]

    categories = {
        "Low current (< 0.5 A)": current < 0.5,
        "Intermediate current (0.5–3 A)": (
            (current >= 0.5) & (current < 3)
        ),
        "Higher current (>= 3 A)": current >= 3,
        "Low voltage (< 1.0 V)": voltage < 1.0,
        "Temperature below 60 C": temperature < 60,
    }

    for label, mask in categories.items():
        print(f"{label}: {int(mask.sum())} rows")

    print("\nNumeric correlations with voltage:")
    print(
        df.corr(numeric_only=True)["CV-mean"]
        .sort_values(ascending=False)
        .round(3)
        .to_string()
    )


def main():
    for filename in FILES:
        if (DATA_DIR / filename).exists():
            analyze_file(filename)
        else:
            print("Missing file:", filename)


if __name__ == "__main__":
    main()