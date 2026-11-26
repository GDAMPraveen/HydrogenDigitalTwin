
from src.physics.electrical import calculate_dc_power_kw
from src.physics.faraday_law import calculate_hydrogen_rate
from src.physics.mass_balance import (
    calculate_hydrogen_mass_balance_rate_kg_hr,
)

from src.physics.water_system import (
    water_mass_balance_kg_s,
    update_water_inventory_kg,
    water_side_heat_duty_w,
)

from src.physics.thermal import (
    thermal_energy_balance_w,
    thermal_state_rate,
    ThermalParameters,
)

from src.physics.gas_pressure import (
    hydrogen_pressure_from_mass_pa,
    hydrogen_mass_from_pressure_kg,
)

from src.integration.pem_state import (
    PEMInputs,
    PEMParameters,
    PEMState,
)


class PEMPhysicsEngine:
    """Integrates available PEM electrolyzer physics calculations."""

    def __init__(self, parameters: PEMParameters):
        self.parameters = parameters

    def calculate(self, inputs: PEMInputs) -> PEMState:
        state = PEMState()

        # 1. Electrical power
        if inputs.current_a is not None and inputs.voltage_v is not None:
            state.electrical_power_kw = calculate_dc_power_kw(
                inputs.voltage_v,
                inputs.current_a,
            )

        # 2. Faraday hydrogen production
        if (
            inputs.current_a is not None
            and self.parameters.number_of_cells is not None
            and self.parameters.faradaic_efficiency is not None
        ):
            state.ideal_hydrogen_rate_kg_h = calculate_hydrogen_rate(
                inputs.current_a,
                self.parameters.number_of_cells,
                self.parameters.faradaic_efficiency,
            )

        # 3. Hydrogen mass balance
        # Convert kg/s inputs to kg/hr for the existing physics function.
        if (
            inputs.hydrogen_inlet_flow_kg_s is not None
            and inputs.hydrogen_outlet_flow_kg_s is not None
        ):
            generation_kg_hr = inputs.hydrogen_inlet_flow_kg_s * 3600.0
            outlet_kg_hr = inputs.hydrogen_outlet_flow_kg_s * 3600.0

            net_rate_kg_hr = calculate_hydrogen_mass_balance_rate_kg_hr(
                generation_rate_kg_hr=generation_kg_hr,
                outlet_rate_kg_hr=outlet_kg_hr,
            )

            # Return to the integration layer's kg/s convention.
            state.hydrogen_mass_balance_rate_kg_s = net_rate_kg_hr / 3600.0

        # 4. Hydrogen pressure and inventory
        gas_inputs_available = (
            inputs.hydrogen_temperature_c is not None
            and self.parameters.hydrogen_gas_volume_m3 is not None
        )

        if gas_inputs_available and inputs.hydrogen_mass_kg is not None:
            state.hydrogen_pressure_pa = hydrogen_pressure_from_mass_pa(
                inputs.hydrogen_mass_kg,
                self.parameters.hydrogen_gas_volume_m3,
                inputs.hydrogen_temperature_c + 273.15,
            )

        elif gas_inputs_available and inputs.hydrogen_pressure_pa is not None:
            state.hydrogen_inventory_kg = hydrogen_mass_from_pressure_kg(
                inputs.hydrogen_pressure_pa,
                self.parameters.hydrogen_gas_volume_m3,
                inputs.hydrogen_temperature_c + 273.15,
            )

        # 5. Water mass balance
        if (
            inputs.water_inlet_flow_kg_s is not None
            and inputs.water_outlet_flow_kg_s is not None
        ):
            state.water_mass_balance_rate_kg_s = water_mass_balance_kg_s(
                inputs.water_inlet_flow_kg_s,
                inputs.water_outlet_flow_kg_s,
            )

            if (
                inputs.water_inventory_kg is not None
                and inputs.dt_s is not None
            ):
                state.water_inventory_kg = update_water_inventory_kg(
                    inputs.water_inventory_kg,
                    state.water_mass_balance_rate_kg_s,
                    inputs.dt_s,
                )

        # 6. Water-side heat duty
        if (
            inputs.water_mass_flow_kg_s is not None
            and inputs.water_inlet_temperature_c is not None
            and inputs.water_return_temperature_c is not None
        ):
            state.water_heat_duty_w = water_side_heat_duty_w(
                inputs.water_mass_flow_kg_s,
                inputs.water_inlet_temperature_c,
                inputs.water_return_temperature_c,
            )

        # 7. Thermal energy balance
        thermal_parameters_available = all(
            value is not None
            for value in (
                self.parameters.thermal_capacitance_j_k,
                self.parameters.ambient_temperature_c,
                self.parameters.heat_loss_coefficient_w_k,
            )
        )

        if (
            thermal_parameters_available
            and inputs.heat_generated_w is not None
            and state.water_heat_duty_w is not None
            and inputs.water_return_temperature_c is not None
        ):
            params = ThermalParameters(
                thermal_capacitance_j_k=self.parameters.thermal_capacitance_j_k,
                ambient_temperature_c=self.parameters.ambient_temperature_c,
                heat_loss_coefficient_w_k=self.parameters.heat_loss_coefficient_w_k,
            )

            state.thermal_energy_balance_w = thermal_energy_balance_w(
                inputs.water_return_temperature_c,
                params.ambient_temperature_c,
                inputs.heat_generated_w,
                state.water_heat_duty_w,
                params.heat_loss_coefficient_w_k,
            )

            state.temperature_rate_c_s = thermal_state_rate(
                inputs.water_return_temperature_c,
                inputs.heat_generated_w,
                state.water_heat_duty_w,
                params,
            )

        return state
