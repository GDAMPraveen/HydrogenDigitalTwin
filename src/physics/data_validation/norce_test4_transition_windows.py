from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data/norce/test4_subset.csv"
OUTPUT_PATH = (
    ROOT / "reports/physics/norce/"
    "norce_test4_transition_windows.csv"
)

df = pd.read_csv(DATA_PATH)

columns = [
    "PS-I-MON",
    "H-P1",
    "O-P1",
    "T-ELY-CH1",
    "CV-mean"
]

df = df[columns].copy()
df.insert(0, "Source_Row_Index", range(len(df)))

windows = [
    ("Hydrogen pressure transition", 141, 181),
    ("Cell voltage and current transition", 4548, 4681)
]

results = []

for name, start, end in windows:
    section = df.iloc[start:end].copy()
    section.insert(1, "Transition", name)

    results.append(section)

    print(f"\n{'=' * 75}")
    print(name)
    print(f"Rows {start} to {end - 1}")
    print("=" * 75)

    print(section.to_string(index=False))

    print("\nNegative-value counts in this window:")

    for column in ["PS-I-MON", "H-P1", "O-P1", "CV-mean"]:
        print(
            f"{column}: "
            f"{(section[column] < 0).sum()} negative observations"
        )

output = pd.concat(results, ignore_index=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(OUTPUT_PATH, index=False)

print(f"\nReport saved to: {OUTPUT_PATH}")