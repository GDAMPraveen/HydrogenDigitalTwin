from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/test4_subset.csv")
OUT_DIR = Path("data/benchmark/temporal_experiment")
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH).reset_index(names="raw_row_index")

current = "PS-I-MON"
voltage = "CV-mean"
temperature = "T-ELY-CH1"

# Smooth short-term sensor variation, then calculate row-to-row changes.
df["current_smooth"] = df[current].rolling(11, center=True, min_periods=1).median()
df["voltage_smooth"] = df[voltage].rolling(11, center=True, min_periods=1).median()
df["current_change"] = df["current_smooth"].diff()
df["voltage_change"] = df["voltage_smooth"].diff()

# Candidate phases; thresholds are exploratory and should be reviewed.
df["phase"] = "normal_or_transition"
df.loc[df[voltage] < 0.2, "phase"] = "startup_or_non_operating"
df.loc[df[current] < 0.5, "phase"] = "low_current"

# A candidate shutdown window starts when current falls substantially
# from its stable ~5 A operating region near the end of Test 4.
candidate_start = 196408
df.loc[df["raw_row_index"] >= candidate_start, "phase"] = "candidate_shutdown"

summary = (
    df.groupby("phase", sort=False)
    .agg(
        rows=("raw_row_index", "size"),
        first_row=("raw_row_index", "min"),
        last_row=("raw_row_index", "max"),
        current_mean_A=(current, "mean"),
        voltage_mean_V=(voltage, "mean"),
        voltage_min_V=(voltage, "min"),
        voltage_max_V=(voltage, "max"),
        temperature_mean_C=(temperature, "mean"),
    )
    .reset_index()
)

print("Exploratory phase summary:")
print(summary.round(5).to_string(index=False))

summary.to_csv(OUT_DIR / "exploratory_phase_summary.csv", index=False)

# Plot a focused view around the current/voltage transition.
view = df.loc[196300:197000]

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

axes[0].plot(view["raw_row_index"], view[current])
axes[0].set_ylabel("Current (A)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(view["raw_row_index"], view[voltage], label="Measured voltage")
axes[1].plot(view["raw_row_index"], view["voltage_smooth"], label="11-row median")
axes[1].set_ylabel("Voltage (V)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(view["raw_row_index"], view[temperature])
axes[2].set_ylabel("Temperature (°C)")
axes[2].set_xlabel("Raw row index (not verified elapsed time)")
axes[2].grid(True, alpha=0.3)

fig.suptitle("Test 4: Candidate operating transition")
fig.tight_layout()

plot_path = FIG_DIR / "candidate_shutdown_transition.png"
fig.savefig(plot_path, dpi=160)
plt.close(fig)

print(f"\nSaved phase summary: {OUT_DIR / 'exploratory_phase_summary.csv'}")
print(f"Saved transition plot: {plot_path}")
