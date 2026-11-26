from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data/norce/test4_subset.csv"
OUTPUT_PATH = (
    ROOT / "reports/physics/norce/"
    "norce_test4_pressure_sign_transitions.csv"
)

df = pd.read_csv(DATA_PATH)

signals = ["H-P1", "O-P1"]
results = []

for signal in signals:
    negative = df[signal] < 0

    # A transition occurs whenever the sign category changes.
    transition = negative.ne(negative.shift())
    transition.iloc[0] = True

    group_id = transition.cumsum()

    runs = (
        pd.DataFrame({
            "Source_Row_Index": df.index,
            "Pressure_Negative": negative,
            "Run_ID": group_id
        })
        .groupby("Run_ID")
        .agg(
            start_row=("Source_Row_Index", "min"),
            end_row=("Source_Row_Index", "max"),
            observations=("Source_Row_Index", "size"),
            negative=("Pressure_Negative", "first")
        )
        .reset_index(drop=True)
    )

    runs.insert(0, "Signal", signal)
    runs["Sign_State"] = runs["negative"].map({
        True: "Negative",
        False: "Non-negative"
    })

    results.append(runs)

    print(f"\n{signal} sign-run summary")
    print("Total runs:", len(runs))
    print("Negative runs:", (runs["negative"]).sum())
    print("Non-negative runs:", (~runs["negative"]).sum())

    print("\nLongest negative runs:")
    print(
        runs[runs["negative"]]
        .nlargest(10, "observations")
        .to_string(index=False)
    )

    print("\nLongest non-negative runs:")
    print(
        runs[~runs["negative"]]
        .nlargest(10, "observations")
        .to_string(index=False)
    )

output = pd.concat(results, ignore_index=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(OUTPUT_PATH, index=False)

print(f"\nReport saved to: {OUTPUT_PATH}")