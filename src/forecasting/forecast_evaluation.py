"""Forecast metrics and chronological/group-aware splitting."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_forecast_metrics(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("y_true and y_pred must be non-empty and have equal length.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric inputs must be finite.")
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted)) if actual.size >= 2 else float("nan")
    nonzero = np.abs(actual) > 1e-12
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    denom = np.abs(actual) + np.abs(predicted)
    valid = denom > 1e-12
    smape = float(np.mean(2 * np.abs(actual[valid] - predicted[valid]) / denom[valid]) * 100) if valid.any() else float("nan")
    return {"MAE": float(mae), "RMSE": rmse, "R2": r2, "MAPE_percent": mape, "sMAPE_percent": smape}

def evaluate_models(models, X_test, y_test, persistence_column=None):
    rows = []
    for name, model in models.items():
        pred = model.predict(X_test)
        rows.append({"Model": name, **calculate_forecast_metrics(y_test, pred)})
    if persistence_column and persistence_column in X_test.columns:
        pred = X_test[persistence_column].to_numpy()
        rows.append({"Model": "Persistence", **calculate_forecast_metrics(y_test, pred)})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

def chronological_train_test_split(X, y, metadata, test_fraction=0.2,
                                   experiment_column="Experiment"):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if len(X) != len(y) or len(X) != len(metadata):
        raise ValueError("X, y and metadata lengths must match.")
    if experiment_column and experiment_column in metadata.columns:
        train_idx, test_idx = [], []
        for _, group in metadata.groupby(experiment_column, sort=False):
            idx = group.index.to_list()
            cut = int(len(idx) * (1 - test_fraction))
            if cut < 1 or cut >= len(idx):
                raise ValueError("Each experiment needs enough rows for both splits.")
            train_idx.extend(idx[:cut])
            test_idx.extend(idx[cut:])
    else:
        idx = list(X.index)
        cut = int(len(idx) * (1 - test_fraction))
        train_idx, test_idx = idx[:cut], idx[cut:]
    return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]
