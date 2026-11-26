import pytest
from src.forecasting.forecast_horizons import DEFAULT_HORIZONS, ForecastHorizon, create_horizons, get_horizon

def test_default_horizons():
    assert [DEFAULT_HORIZONS[k].steps for k in ("one_step","three_steps","five_steps","ten_steps")] == [1,3,5,10]

def test_create_and_get_horizon():
    h = create_horizons({"seven":7})
    assert get_horizon("seven",h).steps == 7

def test_invalid_horizon_rejected():
    with pytest.raises(ValueError):
        ForecastHorizon("bad",0)
