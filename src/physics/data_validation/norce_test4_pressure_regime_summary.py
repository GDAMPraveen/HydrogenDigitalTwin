from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data/norce/test4_subset.csv"
OUTPUT_PATH = (
    ROOT / "reports/physics/norce/"
    "norce_test4_pressure_regime_summary.csv"
)

df = pd.read_csv(DATA_PATH)

df.insert(0, "Source_Row_Index", range(len(df)))

# Define joint pressure-sign regimes.
df["Pressure_Regime"] = "Both non-negative"

df.loc[
    (df["H-P1"] < 0) & (df["O-P1"] >= 0),
    "Pressure_Regime"
] = "H2 negative only"

df.loc[
    (df["H-P1"] >= 0) & (df["O-P1"] < 0),
    "Pressure_Regime"
] = "O2 negative only"

df.loc[
    (df["H-P1"] < 0) & (df["O-P1"] < 0),
    "Pressure_Regime"
] = "Both negative"

summary = (
    df.groupby("Pressure_Regime", observed=False)
    .agg(
        observations=("Source_Row_Index", "size"),
        first_row=("Source_Row_Index", "min"),
        last_row=("Source_Row_Index", "max"),
        mean_current=("PS-I-MON", "mean"),
        min_current=("PS-I-MON", "min"),
        max_current=("PS-I-MON", "max"),
        mean_cell_voltage=("CV-mean", "mean"),
        mean_temperature=("T-ELY-CH1", "mean"),
        mean_H2_pressure=("H-P1", "mean"),
        mean_O2_pressure=("O-P1", "mean")
    )
    .reset_index()
)

# Also summarize each contiguous pressure regime separately.
change = df["Pressure_Regime"].ne(
    df["Pressure_Regime"].shift()
)
change.iloc[0] = True

df["Regime_ID"] = change.cumsum()

episodes = (
    df.groupby("Regime_ID")
    .agg(
        Pressure_Regime=("Pressure_Regime", "first"),
        start_row=("Source_Row_Index", "min"),
        end_row=("Source_Row_Index", "max"),
        observations=("Source_Row_Index", "size"),
        mean_current=("PS-I-MON", "mean"),
        mean_cell_voltage=("CV-mean", "mean"),
        mean_temperature=("T-ELY-CH1", "mean"),
        mean_H2_pressure=("H-P1", "mean"),
        mean_O2_pressure=("O-P1", "mean")
    )
    .reset_index(drop=True)
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    OUTPUT_PATH.with_name("norce_test4_pressure_regime_summary.csv"),
    index=False
)

episodes.to_csv(
    OUTPUT_PATH.with_name("norce_test4_pressure_regime_episodes.csv"),
    index=False
)

print("NORCE TEST4 PRESSURE REGIME SUMMARY")
print(summary.to_string(index=False))

print("\nCONTIGUOUS PRESSURE REGIME EPISODES")
print(episodes.to_string(index=False))

print("\nReports saved in:", OUTPUT_PATH.parent)