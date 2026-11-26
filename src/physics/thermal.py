
"""Thermal physics models for the hydrogen electrolyzer digital twin.

SI conventions:
    Temperature: deg C for measurements and differences
    Time: seconds
    Mass flow: kg/s
    Heat capacity: J/(kg K)
    Heat rate: W
    Thermal capacitance: J/K

Unknown equipment parameters must be supplied by the user.
"""

from dataclasses import dataclass
import numpy as np


WATER_CP_J_KG_K = 4182.0


def _finite(value, name):
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values.")
    return arr


def water_temperature_difference(inlet_temp_c, outlet_temp_c):
    """Return water outlet minus inlet temperature in deg C."""
    inlet = _finite(inlet_temp_c, "inlet_temp_c")
    outlet = _finite(outlet_temp_c, "outlet_temp_c")
    return outlet - inlet


def temperature_rate(temp_now_c, temp_previous_c, dt_s):
    """Estimate dT/dt in K/s using consecutive observations."""
    now = _finite(temp_now_c, "temp_now_c")
    previous = _finite(temp_previous_c, "temp_previous_c")

    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive and finite.")

    return (now - previous) / dt_s


def water_side_heat_transfer_w(
    water_mass_flow_kg_s,
    inlet_temp_c,
    outlet_temp_c,
    cp_j_kg_k=WATER_CP_J_KG_K,
):
    """Calculate water-side sensible heat transfer in watts."""
    flow = _finite(water_mass_flow_kg_s, "water_mass_flow_kg_s")
    delta_t = water_temperature_difference(
        inlet_temp_c, outlet_temp_c
    )

    if np.any(flow < 0):
        raise ValueError("Water mass flow cannot be negative.")

    if not np.isfinite(cp_j_kg_k) or cp_j_kg_k <= 0:
        raise ValueError("Specific heat must be positive.")

    return flow * cp_j_kg_k * delta_t


@dataclass(frozen=True)
class ThermalParameters:
    thermal_capacitance_j_k: float
    ambient_temperature_c: float
    heat_loss_coefficient_w_k: float
    water_cp_j_kg_k: float = WATER_CP_J_KG_K

    def __post_init__(self):
        for name in (
            "thermal_capacitance_j_k",
            "heat_loss_coefficient_w_k",
            "water_cp_j_kg_k",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive.")

        if not np.isfinite(self.ambient_temperature_c):
            raise ValueError("Ambient temperature must be finite.")


def thermal_energy_balance_w(
    stack_temperature_c,
    ambient_temperature_c,
    heat_generated_w,
    heat_removed_w,
    heat_loss_coefficient_w_k,
):
    """Net thermal power: generation - removal - ambient losses."""
    stack_t = _finite(stack_temperature_c, "stack_temperature_c")
    ambient_t = _finite(ambient_temperature_c, "ambient_temperature_c")
    generated = _finite(heat_generated_w, "heat_generated_w")
    removed = _finite(heat_removed_w, "heat_removed_w")

    if not np.isfinite(heat_loss_coefficient_w_k):
        raise ValueError("Heat-loss coefficient must be finite.")

    if heat_loss_coefficient_w_k < 0:
        raise ValueError("Heat-loss coefficient cannot be negative.")

    heat_loss = heat_loss_coefficient_w_k * (stack_t - ambient_t)

    return generated - removed - heat_loss


def predict_temperature_step_c(
    temperature_c,
    net_heat_w,
    dt_s,
    thermal_capacitance_j_k,
):
    """Forward-Euler temperature update using a lumped thermal model."""
    temperature = _finite(temperature_c, "temperature_c")
    net_heat = _finite(net_heat_w, "net_heat_w")

    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive.")

    if (
        not np.isfinite(thermal_capacitance_j_k)
        or thermal_capacitance_j_k <= 0
    ):
        raise ValueError("Thermal capacitance must be positive.")

    return temperature + (
        net_heat * dt_s / thermal_capacitance_j_k
    )


def thermal_state_rate(
    stack_temperature_c,
    heat_generated_w,
    heat_removed_w,
    parameters: ThermalParameters,
):
    """Return lumped stack temperature rate in K/s."""
    net_heat = thermal_energy_balance_w(
        stack_temperature_c=stack_temperature_c,
        ambient_temperature_c=parameters.ambient_temperature_c,
        heat_generated_w=heat_generated_w,
        heat_removed_w=heat_removed_w,
        heat_loss_coefficient_w_k=(
            parameters.heat_loss_coefficient_w_k
        ),
    )

    return net_heat / parameters.thermal_capacitance_j_k
