
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FEATURES = [
    "PS-I-MON",
    "H-P1",
    "O-P1",
    "T-ELY-CH1",
    "CV-mean",
]

def main():
    results = []

    for name in ["test4", "test2", "test3"]:
        path = DATA_DIR / f"{name}_subset.csv"
        df = pd.read_csv(path)
        df = df.dropna(subset=FEATURES)
        low = df[df["PS-I-MON"] < 0.5]

        for col in FEATURES:
            s = low[col]
            results.append({
                "experiment": name.upper(),
                "feature": col,
                "low_current_rows": len(low),
                "mean": s.mean(),
                "std": s.std(),
                "p10": s.quantile(0.10),
                "median": s.median(),
                "p90": s.quantile(0.90),
                "min": s.min(),
                "max": s.max(),
            })

    result = pd.DataFrame(results).round(5)

    out = (
        DATA_DIR / "benchmark" / "temporal_experiment"
        / "low_current_distribution_comparison.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print("\n=== Low-current feature distributions ===")
    print(result.to_string(index=False))
    print("\nSaved:", out)

if __name__ == "__main__":
    main()
