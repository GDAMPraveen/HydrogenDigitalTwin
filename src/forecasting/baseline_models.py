"""Simple forecasting baselines."""
import numpy as np

def persistence_forecast(current_values):
    values = np.asarray(current_values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("current_values must be non-empty and finite.")
    return values.copy()

def mean_baseline(training_target, number_of_predictions):
    values = np.asarray(training_target, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("training_target must be non-empty and finite.")
    if not isinstance(number_of_predictions, int) or number_of_predictions < 1:
        raise ValueError("number_of_predictions must be a positive integer.")
    return np.full(number_of_predictions, values.mean(), dtype=float)

def seasonal_naive_forecast(history, seasonal_period, number_of_predictions):
    values = np.asarray(history, dtype=float)
    if not isinstance(seasonal_period, int) or seasonal_period < 1:
        raise ValueError("seasonal_period must be a positive integer.")
    if not isinstance(number_of_predictions, int) or number_of_predictions < 1:
        raise ValueError("number_of_predictions must be a positive integer.")
    if values.size < seasonal_period or not np.isfinite(values).all():
        raise ValueError("history must contain at least one finite season.")
    season = values[-seasonal_period:]
    return np.resize(season, number_of_predictions)
