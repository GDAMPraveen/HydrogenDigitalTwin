import pytest

from src.physics.electrical import (
    calculate_dc_power_kw,
    calculate_power_discrepancy_kw,
    calculate_relative_power_discrepancy_percent,
    calculate_energy_kwh,
)


def test_dc_power_calculation():
    assert calculate_dc_power_kw(400, 1500) == pytest.approx(600)


def test_zero_power():
    assert calculate_dc_power_kw(0, 1500) == 0


def test_power_discrepancy():
    assert calculate_power_discrepancy_kw(650, 620) == pytest.approx(30)


def test_relative_power_discrepancy():
    result = calculate_relative_power_discrepancy_percent(650, 620)
    assert result == pytest.approx(30 / 650 * 100)


def test_energy_calculation():
    assert calculate_energy_kwh(600, 2) == pytest.approx(1200)


def test_zero_duration():
    assert calculate_energy_kwh(600, 0) == 0


def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        calculate_dc_power_kw(-400, 1500)

    with pytest.raises(ValueError):
        calculate_energy_kwh(600, -1)


def test_relative_discrepancy_zero_reference():
    assert calculate_relative_power_discrepancy_percent(0, 0) == 0

    with pytest.raises(ValueError):
        calculate_relative_power_discrepancy_percent(0, 10)


def test_negative_power_rejected():
    with pytest.raises(ValueError):
        calculate_power_discrepancy_kw(-10, 20)
