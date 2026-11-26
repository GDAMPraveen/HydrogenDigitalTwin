
import pytest

from src.physics.faraday_law import (
    FARADAY_CONSTANT,
    HYDROGEN_MOLAR_MASS,
    ELECTRONS_PER_H2_MOLECULE,
    calculate_hydrogen_rate,
    calculate_per_cell_ideal_rate,
)


def test_faraday_constants():
    assert FARADAY_CONSTANT == pytest.approx(96485.33212)
    assert HYDROGEN_MOLAR_MASS == pytest.approx(0.00201588)
    assert ELECTRONS_PER_H2_MOLECULE == 2


def test_zero_current_produces_zero_hydrogen():
    result = calculate_hydrogen_rate(0, 10, 1.0)
    assert result == 0


def test_single_cell_ideal_hydrogen_rate():
    current = 1000

    expected = (
        current
        * HYDROGEN_MOLAR_MASS
        * 3600
        / (2 * FARADAY_CONSTANT)
    )

    result = calculate_hydrogen_rate(current, 1, 1.0)

    assert result == pytest.approx(expected)


def test_multiple_cells_increase_hydrogen_rate():
    single_cell = calculate_hydrogen_rate(1000, 1, 1.0)
    ten_cells = calculate_hydrogen_rate(1000, 10, 1.0)

    assert ten_cells == pytest.approx(single_cell * 10)


def test_faradaic_efficiency_scales_production():
    ideal = calculate_hydrogen_rate(1000, 10, 1.0)
    reduced = calculate_hydrogen_rate(1000, 10, 0.8)

    assert reduced == pytest.approx(ideal * 0.8)


def test_per_cell_ideal_rate():
    result = calculate_per_cell_ideal_rate(1000)

    expected = calculate_hydrogen_rate(1000, 1, 1.0)

    assert result == pytest.approx(expected)


def test_negative_current_rejected():
    with pytest.raises(ValueError):
        calculate_hydrogen_rate(-100, 10, 1.0)


def test_zero_cell_count_rejected():
    with pytest.raises(ValueError):
        calculate_hydrogen_rate(1000, 0, 1.0)


def test_non_integer_cell_count_rejected():
    with pytest.raises(ValueError):
        calculate_hydrogen_rate(1000, 10.5, 1.0)


def test_invalid_faradaic_efficiency_rejected():
    with pytest.raises(ValueError):
        calculate_hydrogen_rate(1000, 10, 1.2)


def test_zero_faradaic_efficiency_rejected():
    with pytest.raises(ValueError):
        calculate_hydrogen_rate(1000, 10, 0)
