"""Electrical and energy calculations for PEM electrolyzer systems."""

import math
from numbers import Real


def _validate_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def calculate_dc_power_kw(voltage_v, current_a):
    """Calculate DC electrical power in kW from voltage and current."""
    voltage = _validate_finite(voltage_v, "voltage_v")
    current = _validate_finite(current_a, "current_a")

    if voltage < 0 or current < 0:
        raise ValueError("Voltage and current cannot be negative.")

    return voltage * current / 1000.0


def calculate_power_discrepancy_kw(
    reported_power_kw,
    calculated_power_kw,
):
    """Return reported power minus voltage-current calculated power."""
    reported = _validate_finite(reported_power_kw, "reported_power_kw")
    calculated = _validate_finite(calculated_power_kw, "calculated_power_kw")

    if reported < 0 or calculated < 0:
        raise ValueError("Power values cannot be negative.")

    return reported - calculated


def calculate_relative_power_discrepancy_percent(
    reported_power_kw,
    calculated_power_kw,
):
    """Return discrepancy as a percentage of reported power.

    Returns zero when both powers are zero. Raises an error when reported
    power is zero but calculated power is non-zero.
    """
    reported = _validate_finite(reported_power_kw, "reported_power_kw")
    calculated = _validate_finite(calculated_power_kw, "calculated_power_kw")

    if reported < 0 or calculated < 0:
        raise ValueError("Power values cannot be negative.")

    if reported == 0:
        if calculated == 0:
            return 0.0
        raise ValueError(
            "Relative discrepancy is undefined when reported power is zero."
        )

    return (reported - calculated) / reported * 100.0


def calculate_energy_kwh(power_kw, duration_hours):
    """Calculate energy in kWh for constant power over a duration.

    For time-varying power measurements, use timestamp-aware numerical
    integration instead of this constant-power calculation.
    """
    power = _validate_finite(power_kw, "power_kw")
    duration = _validate_finite(duration_hours, "duration_hours")

    if power < 0:
        raise ValueError("Power cannot be negative.")

    if duration < 0:
        raise ValueError("Duration cannot be negative.")

    return power * duration
