"""Hydrogen mass-balance calculations for PEM electrolyzer systems."""

import math
from numbers import Real


def _validate_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def calculate_hydrogen_mass_balance_rate_kg_hr(
    generation_rate_kg_hr,
    outlet_rate_kg_hr,
    loss_rate_kg_hr=0.0,
):
    """Calculate dM/dt = generation - outlet - losses, in kg/hr."""
    generation = _validate_finite(
        generation_rate_kg_hr, "generation_rate_kg_hr"
    )
    outlet = _validate_finite(
        outlet_rate_kg_hr, "outlet_rate_kg_hr"
    )
    losses = _validate_finite(loss_rate_kg_hr, "loss_rate_kg_hr")

    if min(generation, outlet, losses) < 0:
        raise ValueError("Mass flow rates cannot be negative.")

    return generation - outlet - losses


def calculate_inventory_change_kg(
    generation_rate_kg_hr,
    outlet_rate_kg_hr,
    duration_hours,
    loss_rate_kg_hr=0.0,
):
    """Calculate inventory change for constant rates over a duration.

    Positive values indicate accumulation; negative values indicate
    depletion. This is a change, not absolute gas inventory.
    """
    duration = _validate_finite(duration_hours, "duration_hours")

    if duration < 0:
        raise ValueError("Duration cannot be negative.")

    net_rate = calculate_hydrogen_mass_balance_rate_kg_hr(
        generation_rate_kg_hr,
        outlet_rate_kg_hr,
        loss_rate_kg_hr,
    )

    return net_rate * duration


def calculate_hydrogen_mass_from_flow_kg(
    flow_rate_kg_hr,
    duration_hours,
):
    """Calculate hydrogen mass transported at constant flow."""
    flow = _validate_finite(flow_rate_kg_hr, "flow_rate_kg_hr")
    duration = _validate_finite(duration_hours, "duration_hours")

    if flow < 0 or duration < 0:
        raise ValueError("Flow and duration cannot be negative.")

    return flow * duration


def calculate_mass_balance_residual_kg(
    generation_mass_kg,
    outlet_mass_kg,
    loss_mass_kg=0.0,
    inventory_change_kg=0.0,
):
    """Calculate closure residual for an integrated control volume.

    Residual = generation - outlet - losses - inventory change.
    A zero residual indicates mathematical mass-balance closure.
    """
    generation = _validate_finite(generation_mass_kg, "generation_mass_kg")
    outlet = _validate_finite(outlet_mass_kg, "outlet_mass_kg")
    losses = _validate_finite(loss_mass_kg, "loss_mass_kg")
    inventory = _validate_finite(inventory_change_kg, "inventory_change_kg")

    if min(generation, outlet, losses) < 0:
        raise ValueError("Mass quantities cannot be negative.")

    return generation - outlet - losses - inventory
