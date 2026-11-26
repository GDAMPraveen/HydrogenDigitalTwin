
import pytest

from src.physics.gas_pressure import (
    celsius_to_kelvin,
    hydrogen_mass_from_pressure_kg,
    hydrogen_pressure_from_mass_pa,
    gas_mass_balance_kg_s,
    pressure_rate_pa_s,
    predict_pressure_step_pa,
)


def test_celsius_to_kelvin():
    assert celsius_to_kelvin(25) == pytest.approx(298.15)


def test_hydrogen_mass_from_pressure():
    result = hydrogen_mass_from_pressure_kg(
        200000, 1.0, 300
    )
    assert result == pytest.approx(
        200000 / (4124.0 * 300)
    )


def test_pressure_from_hydrogen_mass():
    mass = hydrogen_mass_from_pressure_kg(
        200000, 1.0, 300
    )

    result = hydrogen_pressure_from_mass_pa(
        mass, 1.0, 300
    )

    assert result == pytest.approx(200000)


def test_gas_mass_balance():
    result = gas_mass_balance_kg_s(
        0.5, 0.2, 0.1
    )
    assert result == pytest.approx(0.2)


def test_pressure_rate():
    result = pressure_rate_pa_s(
        0.1, 0.001, 1.0, 300
    )

    expected = 4124.0 * 300 * 0.001
    assert result == pytest.approx(expected)


def test_pressure_prediction():
    result = predict_pressure_step_pa(
        200000, 1000, 5
    )
    assert result == pytest.approx(205000)


def test_negative_absolute_pressure():
    with pytest.raises(ValueError):
        hydrogen_mass_from_pressure_kg(
            -100000, 1.0, 300
        )


def test_zero_gas_volume():
    with pytest.raises(ValueError):
        hydrogen_mass_from_pressure_kg(
            200000, 0, 300
        )


def test_negative_hydrogen_mass():
    with pytest.raises(ValueError):
        hydrogen_pressure_from_mass_pa(
            -0.1, 1.0, 300
        )


def test_negative_gas_flow():
    with pytest.raises(ValueError):
        gas_mass_balance_kg_s(
            0.5, -0.2
        )


def test_non_positive_temperature():
    with pytest.raises(ValueError):
        celsius_to_kelvin(-300)
