from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data" / "norce"
REFERENCE_DIR = ROOT / "references"

DATA_FILE = DATA_DIR / "test4_subset.csv"
README_FILE = ROOT / "data" / "dataset_README.md"

print("\nNORCE TEST4 TIMEBASE AUDIT")

# 1. Inspect local NORCE files
print("\n1. NORCE DATA FILES")

for path in sorted(DATA_DIR.rglob("*")):
    if path.is_file():
        print(f"{path.relative_to(ROOT)} | {path.stat().st_size} bytes")

# 2. Inspect reference files
print("\n2. LOCAL REFERENCE FILES")

if REFERENCE_DIR.exists():
    for path in sorted(REFERENCE_DIR.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(ROOT)} | {path.stat().st_size} bytes")
else:
    print("References directory not found.")

# 3. Inspect test4 columns and index
print("\n3. TEST4 DATA STRUCTURE")

df = pd.read_csv(DATA_FILE)

print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Index type: {type(df.index).__name__}")
print(f"Index range: {df.index.min()} to {df.index.max()}")

time_candidates = [
    col for col in df.columns
    if any(term in col.lower() for term in
           ["time", "date", "stamp", "second", "elapsed"])
]

print(f"Potential time columns: {time_candidates}")

# 4. Inspect README time-resolution statement
print("\n4. README TIMEBASE REFERENCES")

if README_FILE.exists():
    readme = README_FILE.read_text(
        encoding="utf-8",
        errors="replace"
    )

    for line_number, line in enumerate(readme.splitlines(), start=1):
        if any(term in line.lower() for term in
               ["time", "timestamp", "sampling", "1hz", "resolution"]):
            print(f"Line {line_number}: {line.strip()}")
else:
    print("Dataset README not found.")

print("\nTIMEBASE AUDIT COMPLETE")