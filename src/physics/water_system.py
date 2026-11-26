
"""Water-system physics for the electrolyzer digital twin.

SI units:
    Mass: kg
    Time: seconds
    Mass flow: kg/s
    Volume: m3
    Density: kg/m3
"""
import numpy as np

def _check(value, name):
    import numpy as np

    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def volumetric_to_mass_flow_m3_s(
    volumetric_flow_m3_s,
    density_kg_m3=997.0,
):
    """Convert volumetric flow to mass flow in kg/s."""
    flow = _check(volumetric_flow_m3_s, "volumetric_flow_m3_s")

    if np.any(flow < 0):
        raise ValueError("Volumetric flow cannot be negative.")

    if not np.isfinite(density_kg_m3) or density_kg_m3 <= 0:
        raise ValueError("Density must be positive.")

    return flow * density_kg_m3


def water_mass_balance_kg_s(
    inlet_mass_flow_kg_s,
    outlet_mass_flow_kg_s,
    consumption_mass_flow_kg_s=0.0,
):
    """Water inventory rate = inlet - outlet - consumption."""
    inlet = _check(inlet_mass_flow_kg_s, "inlet_mass_flow")
    outlet = _check(outlet_mass_flow_kg_s, "outlet_mass_flow")
    consumption = _check(
        consumption_mass_flow_kg_s,
        "consumption_mass_flow",
    )

    if any(
        (value < 0).any()
        for value in (inlet, outlet, consumption)
    ):
        raise ValueError("Mass flow rates cannot be negative.")

    return inlet - outlet - consumption


def update_water_inventory_kg(
    previous_inventory_kg,
    net_mass_flow_kg_s,
    dt_s,
):
    """Update water inventory using a discrete mass balance."""
    inventory = _check(previous_inventory_kg, "inventory")
    net_flow = _check(net_mass_flow_kg_s, "net_mass_flow")

    if (inventory < 0).any():
        raise ValueError("Inventory cannot be negative.")

    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive.")

    updated = inventory + net_flow * dt_s

    if (updated < -1e-9).any():
        raise ValueError("Mass balance predicts negative inventory.")

    return updated.clip(min=0)


def water_side_heat_duty_w(
    mass_flow_kg_s,
    inlet_temperature_c,
    return_temperature_c,
    specific_heat_j_kg_k=4182.0,
):
    """Sensible heat transported by water."""
    flow = _check(mass_flow_kg_s, "mass_flow")
    inlet = _check(inlet_temperature_c, "inlet_temperature")
    returned = _check(return_temperature_c, "return_temperature")

    if (flow < 0).any():
        raise ValueError("Mass flow cannot be negative.")

    if (
        not np.isfinite(specific_heat_j_kg_k)
        or specific_heat_j_kg_k <= 0
    ):
        raise ValueError("Specific heat must be positive.")

    return flow * specific_heat_j_kg_k * (returned - inlet)


def hydrogen_water_consumption_kg_s(
    hydrogen_production_kg_s,
):
    """Stoichiometric water consumption for electrolysis.

    2 H2O -> 2 H2 + O2

    This is theoretical consumption, not total plant water demand.
    """
    hydrogen = _check(
        hydrogen_production_kg_s,
        "hydrogen_production_kg_s",
    )

    if (hydrogen < 0).any():
        raise ValueError("Hydrogen production cannot be negative.")

    water_to_hydrogen_mass_ratio = 18.01528 / (2 * 1.00794)

    return hydrogen * water_to_hydrogen_mass_ratio
