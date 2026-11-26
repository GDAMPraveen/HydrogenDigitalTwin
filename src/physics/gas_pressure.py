
"""Hydrogen gas inventory and pressure dynamics.

SI units:
    Pressure: Pa (absolute)
    Temperature: K
    Volume: m3
    Mass: kg
    Mass flow: kg/s
"""

import numpy as np


HYDROGEN_SPECIFIC_GAS_CONSTANT = 4124.0  # J/(kg K)


def _check(value, name):
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def celsius_to_kelvin(temperature_c):
    temperature = _check(temperature_c, "temperature_c")

    result = temperature + 273.15

    if np.any(result <= 0):
        raise ValueError("Temperature must be above absolute zero.")

    return result


def hydrogen_mass_from_pressure_kg(
    pressure_pa_absolute,
    volume_m3,
    temperature_k,
    gas_constant=HYDROGEN_SPECIFIC_GAS_CONSTANT,
):
    """Calculate hydrogen mass using the ideal gas equation."""
    pressure = _check(pressure_pa_absolute, "pressure")
    temperature = _check(temperature_k, "temperature")

    if np.any(pressure <= 0):
        raise ValueError("Absolute pressure must be positive.")

    if not np.isfinite(volume_m3) or volume_m3 <= 0:
        raise ValueError("Gas volume must be positive.")

    if np.any(temperature <= 0):
        raise ValueError("Temperature must be above zero kelvin.")

    return (
        pressure * volume_m3
        / (gas_constant * temperature)
    )


def hydrogen_pressure_from_mass_pa(
    hydrogen_mass_kg,
    volume_m3,
    temperature_k,
    gas_constant=HYDROGEN_SPECIFIC_GAS_CONSTANT,
):
    """Calculate absolute hydrogen pressure in Pa."""
    mass = _check(hydrogen_mass_kg, "hydrogen_mass_kg")
    temperature = _check(temperature_k, "temperature_k")

    if np.any(mass < 0):
        raise ValueError("Hydrogen mass cannot be negative.")

    if np.any(temperature <= 0):
        raise ValueError("Temperature must be positive.")

    if not np.isfinite(volume_m3) or volume_m3 <= 0:
        raise ValueError("Volume must be positive.")

    return (
        mass * gas_constant * temperature / volume_m3
    )


def gas_mass_balance_kg_s(
    inlet_mass_flow_kg_s,
    outlet_mass_flow_kg_s,
    leakage_mass_flow_kg_s=0.0,
):
    """Hydrogen inventory rate."""
    inlet = _check(inlet_mass_flow_kg_s, "inlet_mass_flow")
    outlet = _check(outlet_mass_flow_kg_s, "outlet_mass_flow")
    leakage = _check(leakage_mass_flow_kg_s, "leakage_mass_flow")

    if any(
        (value < 0).any()
        for value in (inlet, outlet, leakage)
    ):
        raise ValueError("Gas mass flow rates cannot be negative.")

    return inlet - outlet - leakage


def pressure_rate_pa_s(
    hydrogen_mass_kg,
    net_mass_flow_kg_s,
    volume_m3,
    temperature_k,
    temperature_rate_k_s=0.0,
    gas_constant=HYDROGEN_SPECIFIC_GAS_CONSTANT,
):
    """Calculate pressure derivative for a rigid gas volume.

    Derived from:
        P = m * R * T / V

    Assumes constant volume and ideal-gas behaviour.
    """
    mass = _check(hydrogen_mass_kg, "hydrogen_mass")
    mass_rate = _check(net_mass_flow_kg_s, "net_mass_flow")
    temperature = _check(temperature_k, "temperature")
    temperature_rate = _check(
        temperature_rate_k_s,
        "temperature_rate",
    )

    if np.any(mass < 0):
        raise ValueError("Gas mass cannot be negative.")

    if np.any(temperature <= 0):
        raise ValueError("Temperature must be positive.")

    if not np.isfinite(volume_m3) or volume_m3 <= 0:
        raise ValueError("Volume must be positive.")

    return (
        gas_constant / volume_m3
        * (
            temperature * mass_rate
            + mass * temperature_rate
        )
    )


def predict_pressure_step_pa(
    pressure_pa_absolute,
    pressure_rate_pa_s_value,
    dt_s,
):
    """Forward-Euler pressure update."""
    pressure = _check(
        pressure_pa_absolute,
        "pressure_pa_absolute",
    )
    rate = _check(
        pressure_rate_pa_s_value,
        "pressure_rate",
    )

    if np.any(pressure <= 0):
        raise ValueError("Absolute pressure must be positive.")

    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive.")

    updated = pressure + rate * dt_s

    if np.any(updated <= 0):
        raise ValueError("Predicted absolute pressure is non-positive.")

    return updated
