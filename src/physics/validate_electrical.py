"""Validate NLR electrical measurements against V × I."""

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ARCHIVE = (
    PROJECT_ROOT
    / "data"
    / "nlr"
    / "raw"
    / "characterization"
    / "characterization_200.zip"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "physics"
    / "nlr_electrical_validation.csv"
)

REQUIRED_COLUMNS = [
    "Power Supply Average Voltage (Vdc)",
    "H2E_f_PSU_A_Current",
    "H2E_n_PSU_A_Power",
]


def load_characterization_data():
    """Read the original characterization CSV without modifying it."""
    with zipfile.ZipFile(DATA_ARCHIVE) as archive:
        with archive.open("characterization_200.csv") as file:
            return pd.read_csv(file)


def validate_electrical_data(df):
    """Calculate physical power and compare it with reported power."""
    voltage = pd.to_numeric(
        df["Power Supply Average Voltage (Vdc)"],
        errors="coerce",
    )
    current = pd.to_numeric(
        df["H2E_f_PSU_A_Current"],
        errors="coerce",
    )
    reported = pd.to_numeric(
        df["H2E_n_PSU_A_Power"],
        errors="coerce",
    )

    valid = (
        voltage.notna()
        & current.notna()
        & reported.notna()
        & np.isfinite(voltage)
        & np.isfinite(current)
        & np.isfinite(reported)
    )

    if not valid.all():
        raise ValueError(
            f"Invalid electrical observations found: {(~valid).sum()}"
        )

    if (voltage < 0).any() or (current < 0).any():
        raise ValueError("Negative voltage or current detected.")

    calculated = voltage * current / 1000
    discrepancy = reported - calculated

    result = pd.DataFrame({
        "timestamp": df.iloc[:, 0],
        "voltage_V": voltage,
        "current_A": current,
        "calculated_power_kW": calculated,
        "reported_power_kW": reported,
        "discrepancy_kW": discrepancy,
        "absolute_discrepancy_kW": discrepancy.abs(),
    })

    result["absolute_percentage_error"] = (
        discrepancy.abs()
        / reported.abs().replace(0, np.nan)
        * 100
    )

    model = LinearRegression()
    model.fit(
        calculated.to_numpy().reshape(-1, 1),
        reported.to_numpy(),
    )

    result["regression_predicted_power_kW"] = model.predict(
        calculated.to_numpy().reshape(-1, 1)
    )

    result["regression_residual_kW"] = (
        reported - result["regression_predicted_power_kW"]
    )

    metrics = {
        "observations": len(result),
        "mean_reported_power_kW": reported.mean(),
        "mean_calculated_power_kW": calculated.mean(),
        "mean_signed_discrepancy_kW": discrepancy.mean(),
        "mean_absolute_discrepancy_kW": discrepancy.abs().mean(),
        "mean_absolute_percentage_error": (
            result["absolute_percentage_error"].mean()
        ),
        "maximum_positive_discrepancy_kW": discrepancy.max(),
        "maximum_negative_discrepancy_kW": discrepancy.min(),
        "regression_intercept_kW": model.intercept_,
        "regression_slope": model.coef_[0],
        "regression_r2": r2_score(
            reported,
            result["regression_predicted_power_kW"],
        ),
        "regression_rmse_kW": np.sqrt(
            np.mean(result["regression_residual_kW"] ** 2)
        ),
    }

    return result, metrics


def main():
    df = load_characterization_data()

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result, metrics = validate_electrical_data(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("=== NLR ELECTRICAL VALIDATION ===")
    for name, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{name}: {value}")
        else:
            print(f"{name}: {value:.6f}")

    print(f"\nCSV report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
