
import pytest

from src.physics.water_system import (
    volumetric_to_mass_flow_m3_s,
    water_mass_balance_kg_s,
    update_water_inventory_kg,
    water_side_heat_duty_w,
    hydrogen_water_consumption_kg_s,
)


def test_volumetric_to_mass_flow():
    result = volumetric_to_mass_flow_m3_s(0.001, 997)
    assert result == pytest.approx(0.997)


def test_water_mass_balance():
    result = water_mass_balance_kg_s(
        0.5, 0.2, 0.1
    )
    assert result == pytest.approx(0.2)


def test_water_inventory_update():
    result = update_water_inventory_kg(100, 0.5, 10)
    assert result == pytest.approx(105)


def test_water_heat_duty():
    result = water_side_heat_duty_w(
        0.1, 50, 55
    )
    assert result == pytest.approx(2091)


def test_hydrogen_water_consumption():
    result = hydrogen_water_consumption_kg_s(0.001)
    assert result == pytest.approx(
        0.008936, rel=1e-3
    )


def test_negative_volumetric_flow():
    with pytest.raises(ValueError):
        volumetric_to_mass_flow_m3_s(-0.001)


def test_negative_water_inventory():
    with pytest.raises(ValueError):
        update_water_inventory_kg(-10, 0.5, 10)


def test_negative_water_mass_balance_flow():
    with pytest.raises(ValueError):
        water_mass_balance_kg_s(0.5, -0.2)


def test_zero_time_interval():
    with pytest.raises(ValueError):
        update_water_inventory_kg(100, 0.5, 0)


def test_negative_hydrogen_production():
    with pytest.raises(ValueError):
        hydrogen_water_consumption_kg_s(-0.001)
