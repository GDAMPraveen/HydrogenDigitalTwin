
import pytest

from src.integration.pem_state import PEMInputs, PEMParameters
from src.integration.pem_engine import PEMPhysicsEngine


def test_electrical_power_integration():
    engine = PEMPhysicsEngine(PEMParameters())

    inputs = PEMInputs(current_a=100, voltage_v=50)
    state = engine.calculate(inputs)

    assert state.electrical_power_kw == pytest.approx(5.0)


def test_faraday_production_integration():
    engine = PEMPhysicsEngine(
        PEMParameters(
            number_of_cells=10,
            faradaic_efficiency=0.95,
        )
    )

    inputs = PEMInputs(current_a=100)
    state = engine.calculate(inputs)

    assert state.ideal_hydrogen_rate_kg_h is not None
    assert state.ideal_hydrogen_rate_kg_h > 0


def test_faraday_not_calculated_without_cell_parameters():
    engine = PEMPhysicsEngine(PEMParameters())

    state = engine.calculate(PEMInputs(current_a=100))

    assert state.ideal_hydrogen_rate_kg_h is None


def test_hydrogen_mass_balance():
    engine = PEMPhysicsEngine(PEMParameters())

    inputs = PEMInputs(
        hydrogen_inlet_flow_kg_s=0.5,
        hydrogen_outlet_flow_kg_s=0.2,
    )

    state = engine.calculate(inputs)

    assert state.hydrogen_mass_balance_rate_kg_s == pytest.approx(0.3)


def test_hydrogen_pressure_estimation():
    engine = PEMPhysicsEngine(
        PEMParameters(hydrogen_gas_volume_m3=1.0)
    )

    inputs = PEMInputs(
        hydrogen_mass_kg=0.1,
        hydrogen_temperature_c=25,
    )

    state = engine.calculate(inputs)

    assert state.hydrogen_pressure_pa is not None
    assert state.hydrogen_pressure_pa > 0


def test_hydrogen_inventory_estimation():
    engine = PEMPhysicsEngine(
        PEMParameters(hydrogen_gas_volume_m3=1.0)
    )

    inputs = PEMInputs(
        hydrogen_pressure_pa=200000,
        hydrogen_temperature_c=25,
    )

    state = engine.calculate(inputs)

    assert state.hydrogen_inventory_kg is not None
    assert state.hydrogen_inventory_kg > 0


def test_water_mass_balance():
    engine = PEMPhysicsEngine(PEMParameters())

    inputs = PEMInputs(
        water_inlet_flow_kg_s=0.5,
        water_outlet_flow_kg_s=0.2,
    )

    state = engine.calculate(inputs)

    assert state.water_mass_balance_rate_kg_s == pytest.approx(0.3)


def test_water_inventory_update():
    engine = PEMPhysicsEngine(PEMParameters())

    inputs = PEMInputs(
        water_inventory_kg=100,
        water_inlet_flow_kg_s=0.5,
        water_outlet_flow_kg_s=0.2,
        dt_s=10,
    )

    state = engine.calculate(inputs)

    assert state.water_inventory_kg == pytest.approx(103)


def test_water_inventory_not_updated_without_outlet_flow():
    engine = PEMPhysicsEngine(PEMParameters())

    inputs = PEMInputs(
        water_inventory_kg=100,
        water_inlet_flow_kg_s=0.5,
        dt_s=10,
    )

    state = engine.calculate(inputs)

    assert state.water_inventory_kg is None


def test_water_heat_duty():
    engine = PEMPhysicsEngine(PEMParameters())

    inputs = PEMInputs(
        water_mass_flow_kg_s=0.1,
        water_inlet_temperature_c=50,
        water_return_temperature_c=55,
    )

    state = engine.calculate(inputs)

    assert state.water_heat_duty_w == pytest.approx(2091)


def test_thermal_calculation():
    engine = PEMPhysicsEngine(
        PEMParameters(
            thermal_capacitance_j_k=10000,
            ambient_temperature_c=25,
            heat_loss_coefficient_w_k=100,
        )
    )

    inputs = PEMInputs(
        water_mass_flow_kg_s=0.1,
        water_inlet_temperature_c=50,
        water_return_temperature_c=55,
        heat_generated_w=5000,
    )

    state = engine.calculate(inputs)

    assert state.thermal_energy_balance_w is not None
    assert state.temperature_rate_c_s is not None


def test_thermal_not_calculated_without_heat_generation():
    engine = PEMPhysicsEngine(
        PEMParameters(
            thermal_capacitance_j_k=10000,
            ambient_temperature_c=25,
            heat_loss_coefficient_w_k=100,
        )
    )

    inputs = PEMInputs(
        water_mass_flow_kg_s=0.1,
        water_inlet_temperature_c=50,
        water_return_temperature_c=55,
    )

    state = engine.calculate(inputs)

    assert state.thermal_energy_balance_w is None
    assert state.temperature_rate_c_s is None


def test_empty_inputs_return_empty_state():
    engine = PEMPhysicsEngine(PEMParameters())

    state = engine.calculate(PEMInputs())

    assert state.electrical_power_kw is None
    assert state.ideal_hydrogen_rate_kg_h is None
    assert state.hydrogen_pressure_pa is None
    assert state.water_heat_duty_w is None
