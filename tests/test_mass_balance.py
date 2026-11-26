import pytest

from src.physics.mass_balance import (
    calculate_hydrogen_mass_balance_rate_kg_hr,
    calculate_inventory_change_kg,
    calculate_hydrogen_mass_from_flow_kg,
    calculate_mass_balance_residual_kg,
)


def test_generation_exceeds_outflow():
    result = calculate_hydrogen_mass_balance_rate_kg_hr(12, 10)
    assert result == pytest.approx(2)


def test_outflow_exceeds_generation():
    result = calculate_hydrogen_mass_balance_rate_kg_hr(8, 10)
    assert result == pytest.approx(-2)


def test_losses_are_accounted_for():
    result = calculate_hydrogen_mass_balance_rate_kg_hr(12, 9, 1)
    assert result == pytest.approx(2)


def test_inventory_accumulation():
    result = calculate_inventory_change_kg(12, 10, 0.5)
    assert result == pytest.approx(1)


def test_inventory_depletion():
    result = calculate_inventory_change_kg(8, 10, 0.5)
    assert result == pytest.approx(-1)


def test_hydrogen_mass_from_constant_flow():
    result = calculate_hydrogen_mass_from_flow_kg(12, 2)
    assert result == pytest.approx(24)


def test_mass_balance_closure():
    result = calculate_mass_balance_residual_kg(
        generation_mass_kg=12,
        outlet_mass_kg=9,
        loss_mass_kg=1,
        inventory_change_kg=2,
    )
    assert result == pytest.approx(0)


def test_mass_balance_nonzero_residual():
    result = calculate_mass_balance_residual_kg(
        generation_mass_kg=12,
        outlet_mass_kg=9,
        loss_mass_kg=1,
        inventory_change_kg=1,
    )
    assert result == pytest.approx(1)


def test_negative_flow_rejected():
    with pytest.raises(ValueError):
        calculate_hydrogen_mass_balance_rate_kg_hr(-1, 2)


def test_negative_duration_rejected():
    with pytest.raises(ValueError):
        calculate_inventory_change_kg(12, 10, -1)
