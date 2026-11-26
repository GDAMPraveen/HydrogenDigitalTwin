from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "test4_subset.csv",
    "test2_subset.csv",
    "test3_subset.csv",
]

FEATURES = [
    "PS-I-MON",
    "H-P1",
    "O-P1",
    "T-ELY-CH1",
    "CV-mean",
]

LABELS = {
    "PS-I-MON": "Stack current (A)",
    "H-P1": "Hydrogen-side pressure (bar)",
    "O-P1": "Oxygen-side pressure (bar)",
    "T-ELY-CH1": "Stack temperature (°C)",
    "CV-mean": "Mean cell voltage (V)",
}


def visualize_file(filename):
    path = DATA_DIR / filename
    df = pd.read_csv(path)

    print(f"\nPlotting {filename}: {len(df):,} rows")

    fig, axes = plt.subplots(
        nrows=len(FEATURES),
        ncols=1,
        figsize=(14, 12),
        sharex=True,
    )

    for ax, column in zip(axes, FEATURES):
        ax.plot(df.index, df[column], linewidth=0.7)
        ax.set_ylabel(LABELS[column])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Row index (not a verified timestamp)")
    fig.suptitle(f"PEM Electrolyzer Experiment: {filename}")
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{Path(filename).stem}_timeseries.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {output_path}")


def main():
    for filename in FILES:
        if (DATA_DIR / filename).exists():
            visualize_file(filename)
        else:
            print(f"File not found: {DATA_DIR / filename}")


if __name__ == "__main__":
    main()