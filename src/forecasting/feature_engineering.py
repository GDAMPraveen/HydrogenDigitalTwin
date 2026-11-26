"""Leakage-aware supervised time-series feature engineering."""
import pandas as pd
from src.forecasting.data_preparation import select_feature_columns

def create_supervised_dataset(df, target_column, horizon_steps=1,
                              lags=(1, 2, 3, 5, 10),
                              rolling_windows=(3, 5, 10),
                              experiment_column="Experiment",
                              timestamp_column=None, feature_columns=None,
                              drop_incomplete_rows=True):
    if not isinstance(horizon_steps, int) or isinstance(horizon_steps, bool) or horizon_steps < 1:
        raise ValueError("horizon_steps must be a positive integer.")
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")
    if experiment_column and experiment_column not in df.columns:
        raise ValueError(f"Experiment column not found: {experiment_column}")
    if timestamp_column and timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column not found: {timestamp_column}")
    if any(not isinstance(x, int) or x < 1 for x in lags):
        raise ValueError("All lags must be positive integers.")
    if any(not isinstance(x, int) or x < 2 for x in rolling_windows):
        raise ValueError("Rolling windows must be integers >= 2.")
    data = df.copy()
    if timestamp_column:
        data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors="raise")
        sort_cols = ([experiment_column] if experiment_column else []) + [timestamp_column]
        data = data.sort_values(sort_cols, kind="stable")
    elif experiment_column:
        data = data.sort_values(experiment_column, kind="stable")
    data = data.reset_index(drop=True)
    features = feature_columns or select_feature_columns(
        data, target_column, experiment_column, timestamp_column
    )
    for col in features:
        if col not in data.columns:
            raise ValueError(f"Feature column not found: {col}")
    group = data.groupby(experiment_column, sort=False) if experiment_column else None
    X = pd.DataFrame(index=data.index)
    for col in features:
        X[col] = pd.to_numeric(data[col], errors="coerce")
    target = pd.to_numeric(data[target_column], errors="coerce")
    # The measured target at issue time t is a valid persistence reference,
    # provided it is actually available to the deployed forecasting system.
    X[f"{target_column}_current"] = target
    if experiment_column:
        for lag in lags:
            X[f"{target_column}_lag_{lag}"] = group[target_column].shift(lag)
        for window in rolling_windows:
            shifted = group[target_column].shift(1)
            # Grouped rolling is calculated independently per experiment.
            roll = shifted.groupby(data[experiment_column], sort=False).rolling(
                window=window, min_periods=window
            )
            mean = roll.mean().reset_index(level=0, drop=True).sort_index()
            std = roll.std().reset_index(level=0, drop=True).sort_index()
            X[f"{target_column}_rolling_mean_{window}"] = mean
            X[f"{target_column}_rolling_std_{window}"] = std
        y = group[target_column].shift(-horizon_steps)
    else:
        for lag in lags:
            X[f"{target_column}_lag_{lag}"] = target.shift(lag)
        for window in rolling_windows:
            prior = target.shift(1)
            X[f"{target_column}_rolling_mean_{window}"] = prior.rolling(window, min_periods=window).mean()
            X[f"{target_column}_rolling_std_{window}"] = prior.rolling(window, min_periods=window).std()
        y = target.shift(-horizon_steps)
    y = y.rename("future_target")
    # Retain rows with a valid future target. Missing sensor inputs are left for
    # training-only imputation inside model pipelines.
    valid = y.notna()
    if drop_incomplete_rows:
        required_engineered = [c for c in X.columns if c.startswith(f"{target_column}_lag_")
                               or c.startswith(f"{target_column}_rolling_")]
        valid &= X[required_engineered].notna().all(axis=1)
    X, y = X.loc[valid].copy(), y.loc[valid].copy()
    meta_cols = ([experiment_column] if experiment_column else [])
    if timestamp_column:
        meta_cols.append(timestamp_column)
    metadata = data.loc[valid, meta_cols].copy()
    return X, y, metadata
