import numpy as np
import pytest
from src.forecasting.baseline_models import persistence_forecast, mean_baseline, seasonal_naive_forecast

def test_persistence():
    np.testing.assert_allclose(persistence_forecast([1,2]), [1,2])

def test_mean_baseline():
    np.testing.assert_allclose(mean_baseline([1,3],3), [2,2,2])

def test_seasonal_naive():
    np.testing.assert_allclose(seasonal_naive_forecast([1,2,3,4],2,5), [3,4,3,4,3])

def test_empty_persistence_rejected():
    with pytest.raises(ValueError):
        persistence_forecast([])
