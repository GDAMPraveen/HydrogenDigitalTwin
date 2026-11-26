
from dataclasses import dataclass
from typing import Optional


@dataclass
class PEMParameters:
    number_of_cells: Optional[int] = None
    faradaic_efficiency: Optional[float] = None

    thermal_capacitance_j_k: Optional[float] = None
    ambient_temperature_c: Optional[float] = None
    heat_loss_coefficient_w_k: Optional[float] = None

    hydrogen_gas_volume_m3: Optional[float] = None


@dataclass
class PEMInputs:
    current_a: Optional[float] = None
    voltage_v: Optional[float] = None

    measured_hydrogen_flow_kg_s: Optional[float] = None

    hydrogen_inlet_flow_kg_s: Optional[float] = None
    hydrogen_outlet_flow_kg_s: Optional[float] = None
    hydrogen_mass_kg: Optional[float] = None

    water_inventory_kg: Optional[float] = None
    water_inlet_flow_kg_s: Optional[float] = None
    water_outlet_flow_kg_s: Optional[float] = None
    water_mass_flow_kg_s: Optional[float] = None

    water_inlet_temperature_c: Optional[float] = None
    water_return_temperature_c: Optional[float] = None

    hydrogen_pressure_pa: Optional[float] = None
    hydrogen_temperature_c: Optional[float] = None

    heat_generated_w: Optional[float] = None

    dt_s: Optional[float] = None


@dataclass
class PEMState:
    electrical_power_kw: Optional[float] = None
    ideal_hydrogen_rate_kg_h: Optional[float] = None

    hydrogen_mass_balance_rate_kg_s: Optional[float] = None
    hydrogen_inventory_kg: Optional[float] = None
    hydrogen_pressure_pa: Optional[float] = None

    water_mass_balance_rate_kg_s: Optional[float] = None
    water_inventory_kg: Optional[float] = None
    water_heat_duty_w: Optional[float] = None

    thermal_energy_balance_w: Optional[float] = None
    temperature_rate_c_s: Optional[float] = None
