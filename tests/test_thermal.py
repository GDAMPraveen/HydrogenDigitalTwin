
import pytest

from src.physics.thermal import (
    ThermalParameters,
    water_temperature_difference,
    temperature_rate,
    water_side_heat_transfer_w,
    thermal_energy_balance_w,
    predict_temperature_step_c,
    thermal_state_rate,
)


def test_water_temperature_difference():
    assert water_temperature_difference(50, 55) == 5


def test_temperature_rate():
    assert temperature_rate(60, 55, 5) == 1


def test_water_heat_transfer():
    result = water_side_heat_transfer_w(0.1, 50, 55)
    assert result == pytest.approx(2091.0)


def test_thermal_energy_balance():
    result = thermal_energy_balance_w(
        60, 25, 10000, 2000, 100
    )
    assert result == pytest.approx(4500)


def test_temperature_prediction():
    result = predict_temperature_step_c(
        50, 1000, 10, 10000
    )
    assert result == pytest.approx(51)


def test_thermal_state_rate():
    params = ThermalParameters(10000, 25, 100)
    result = thermal_state_rate(60, 10000, 2000, params)
    assert result == pytest.approx(0.45)


def test_invalid_time_interval():
    with pytest.raises(ValueError):
        temperature_rate(50, 40, 0)


def test_invalid_heat_capacity():
    with pytest.raises(ValueError):
        ThermalParameters(0, 25, 100)


def test_negative_water_flow():
    with pytest.raises(ValueError):
        water_side_heat_transfer_w(-0.1, 50, 55)


def test_non_finite_temperature():
    with pytest.raises(ValueError):
        water_temperature_difference(float("nan"), 55)
