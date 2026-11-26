
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# FARADAY CONSTANTS
# --------------------------------------------------

FARADAY_CONSTANT = 96485.33212  # C/mol
HYDROGEN_MOLAR_MASS = 0.00201588  # kg/mol
ELECTRONS_PER_H2_MOLECULE = 2


# --------------------------------------------------
# THEORETICAL HYDROGEN PRODUCTION
# --------------------------------------------------

def calculate_hydrogen_rate(
    current_a: float,
    number_of_cells: int,
    faradaic_efficiency: float,
) -> float:
    """
    Calculate theoretical stack hydrogen production in kg/hr.

    Requires an explicitly verified or experimentally justified
    number of cells and Faradaic efficiency.
    """

    if current_a < 0:
        raise ValueError("Current cannot be negative.")

    if not isinstance(number_of_cells, int) or number_of_cells <= 0:
        raise ValueError("Number of cells must be a positive integer.")

    if not 0 < faradaic_efficiency <= 1:
        raise ValueError("Faradaic efficiency must be in (0, 1].")

    molar_rate = (
        number_of_cells
        * current_a
        * faradaic_efficiency
        / (ELECTRONS_PER_H2_MOLECULE * FARADAY_CONSTANT)
    )

    return molar_rate * HYDROGEN_MOLAR_MASS * 3600


def calculate_per_cell_ideal_rate(current_a: float) -> float:
    """
    Calculate ideal hydrogen production for ONE cell at 100%
    Faradaic efficiency.

    This is a reference coefficient, NOT the NLR stack output.
    """

    return calculate_hydrogen_rate(
        current_a=current_a,
        number_of_cells=1,
        faradaic_efficiency=1.0,
    )


# --------------------------------------------------
# NLR DATA REFERENCE CALCULATION
# --------------------------------------------------

def main():
    root = Path(__file__).resolve().parents[2]

    data_path = (
        root
        / "data/nlr/raw/characterization/characterization_200.zip"
    )

    report_dir = root / "reports/physics"
    report_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    import zipfile
    from io import BytesIO

    with zipfile.ZipFile(data_path, "r") as archive:
        data = pd.read_csv(
            BytesIO(archive.read("characterization_200.csv"))
        )

    current_column = "H2E_f_PSU_A_Current"

    if current_column not in data.columns:
        raise KeyError(f"Missing column: {current_column}")

    result = pd.DataFrame({
        "actual_stack_current_A": data[current_column]
    })

    result["per_cell_ideal_h2_kg_hr"] = (
        result["actual_stack_current_A"]
        .apply(calculate_per_cell_ideal_rate)
    )

    # This report deliberately does not calculate a full-stack
    # production rate because NLR cell count and Faradaic
    # efficiency have not been verified.

    output_path = (
        report_dir / "nlr_faraday_per_cell_reference.csv"
    )

    result.to_csv(output_path, index=False)

    print("\nFARADAY LAW REFERENCE CALCULATION")
    print(f"Records processed: {len(result)}")
    print(
        "Mean actual current (A): "
        f"{result['actual_stack_current_A'].mean():.4f}"
    )
    print(
        "Mean ideal per-cell production (kg/hr): "
        f"{result['per_cell_ideal_h2_kg_hr'].mean():.6f}"
    )
    print("\nStack-level production was NOT calculated.")
    print("Cell count and Faradaic efficiency remain unverified.")
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
