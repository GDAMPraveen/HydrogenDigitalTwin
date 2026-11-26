"""Electrochemical hydrogen-production calculations for PEM systems."""

import math
from numbers import Real

from .faraday_law import (
    FARADAY_CONSTANT,
    HYDROGEN_MOLAR_MASS,
    ELECTRONS_PER_H2_MOLECULE,
    calculate_hydrogen_rate,
    calculate_per_cell_ideal_rate,
)


def _validate_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def ideal_hydrogen_rate_per_cell_kg_hr(current_a):
    """Ideal hydrogen production for one cell at 100% efficiency."""
    current = _validate_finite(current_a, "current_a")

    if current < 0:
        raise ValueError("Current cannot be negative.")

    return calculate_per_cell_ideal_rate(current)


def theoretical_stack_hydrogen_rate_kg_hr(
    current_a,
    number_of_cells,
    faradaic_efficiency,
):
    """Theoretical stack production using explicitly supplied parameters."""
    current = _validate_finite(current_a, "current_a")
    efficiency = _validate_finite(
        faradaic_efficiency, "faradaic_efficiency"
    )

    if isinstance(number_of_cells, bool) or not isinstance(number_of_cells, int):
        raise TypeError("number_of_cells must be an integer.")

    if current < 0:
        raise ValueError("Current cannot be negative.")

    if number_of_cells <= 0:
        raise ValueError("number_of_cells must be positive.")

    if not 0 < efficiency <= 1:
        raise ValueError("Faradaic efficiency must be in (0, 1].")

    return calculate_hydrogen_rate(
        current,
        number_of_cells,
        efficiency,
    )


def effective_cell_efficiency_product(
    current_a,
    measured_hydrogen_kg_hr,
):
    """Estimate N_cells * eta_F from measured flow and actual current.

    This identifies only the combined parameter, not cell count and
    Faradaic efficiency independently.
    """
    current = _validate_finite(current_a, "current_a")
    measured_flow = _validate_finite(
        measured_hydrogen_kg_hr, "measured_hydrogen_kg_hr"
    )

    if current <= 0:
        raise ValueError("Current must be positive.")

    if measured_flow < 0:
        raise ValueError("Measured hydrogen flow cannot be negative.")

    return (
        measured_flow
        * ELECTRONS_PER_H2_MOLECULE
        * FARADAY_CONSTANT
        / (current * HYDROGEN_MOLAR_MASS * 3600)
    )


def hydrogen_rate_residual_kg_hr(
    measured_hydrogen_kg_hr,
    theoretical_hydrogen_kg_hr,
):
    """Return measured minus theoretical hydrogen flow in kg/hr."""
    measured = _validate_finite(
        measured_hydrogen_kg_hr, "measured_hydrogen_kg_hr"
    )
    theoretical = _validate_finite(
        theoretical_hydrogen_kg_hr, "theoretical_hydrogen_kg_hr"
    )

    if measured < 0 or theoretical < 0:
        raise ValueError("Hydrogen rates cannot be negative.")

    return measured - theoretical
