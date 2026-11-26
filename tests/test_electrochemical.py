import math
import pytest

from src.physics.electrochemical import (
    ideal_hydrogen_rate_per_cell_kg_hr,
    theoretical_stack_hydrogen_rate_kg_hr,
    effective_cell_efficiency_product,
    hydrogen_rate_residual_kg_hr,
)


def test_zero_current():
    assert ideal_hydrogen_rate_per_cell_kg_hr(0) == 0.0


def test_stack_rate_scales_correctly():
    current = 1000.0
    per_cell = ideal_hydrogen_rate_per_cell_kg_hr(current)

    stack = theoretical_stack_hydrogen_rate_kg_hr(
        current, 10, 0.95
    )

    assert stack == pytest.approx(per_cell * 10 * 0.95)


def test_effective_parameter_recovery():
    current = 1000.0
    expected = 10 * 0.95

    measured = (
        ideal_hydrogen_rate_per_cell_kg_hr(current) * expected
    )

    result = effective_cell_efficiency_product(
        current, measured
    )

    assert result == pytest.approx(expected)


def test_zero_measured_flow():
    assert effective_cell_efficiency_product(1000, 0) == 0.0


def test_effective_parameter_rejects_zero_current():
    with pytest.raises(ValueError):
        effective_cell_efficiency_product(0, 0)


def test_invalid_stack_parameters():
    with pytest.raises(ValueError):
        theoretical_stack_hydrogen_rate_kg_hr(1000, 0, 0.95)

    with pytest.raises(ValueError):
        theoretical_stack_hydrogen_rate_kg_hr(1000, 10, 1.01)


def test_non_finite_current():
    with pytest.raises(ValueError):
        ideal_hydrogen_rate_per_cell_kg_hr(math.nan)


def test_hydrogen_flow_residual():
    assert hydrogen_rate_residual_kg_hr(12.0, 11.5) == pytest.approx(0.5)
