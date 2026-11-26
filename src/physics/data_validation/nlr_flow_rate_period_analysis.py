from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = ROOT / "reports/physics/nlr_production_rate_validation.csv"
OUTPUT_PATH = ROOT / "reports/physics/nlr_flow_rate_period_analysis.csv"


def main():
    data = pd.read_csv(INPUT_PATH)

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["elapsed_seconds"] = (
        data["timestamp"] - data["timestamp"].min()
    ).dt.total_seconds()

    data["flow_error_kg_hr"] = (
        data["H2E_f_Elec_CalcProdRate"]
        - data["IVAL_f_FM011_Flow"]
    )

    data["absolute_error_kg_hr"] = data["flow_error_kg_hr"].abs()

    data["operating_period"] = np.where(
        data["elapsed_seconds"] <= 120,
        "startup_period",
        "steady_period"
    )

    results = []

    for period, group in data.groupby("operating_period", sort=False):
        measured = group["IVAL_f_FM011_Flow"]
        calculated = group["H2E_f_Elec_CalcProdRate"]
        error = group["flow_error_kg_hr"]

        results.append({
            "operating_period": period,
            "observations": len(group),
            "mean_measured_flow_kg_hr": measured.mean(),
            "mean_calculated_rate_kg_hr": calculated.mean(),
            "mean_error_kg_hr": error.mean(),
            "mean_absolute_error_kg_hr": error.abs().mean(),
            "median_absolute_error_kg_hr": error.abs().median(),
            "rmse_kg_hr": np.sqrt((error ** 2).mean()),
            "maximum_absolute_error_kg_hr": error.abs().max(),
            "correlation": measured.corr(calculated),
        })

    summary = pd.DataFrame(results)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("\nFULL-RESOLUTION HYDROGEN FLOW VALIDATION")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
