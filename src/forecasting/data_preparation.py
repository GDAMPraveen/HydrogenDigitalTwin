"""Leakage-aware data loading and feature selection."""
from pathlib import Path
import pandas as pd

DEFAULT_EXCLUDED_FEATURES = {"h2e_f_elec_calcprodrate", "efficiency"}
LEAKAGE_TOKENS = ("calcprodrate", "efficiency")

def load_forecasting_data(file_path, target_column, experiment_column="Experiment",
                          timestamp_column=None):
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = [target_column]
    if experiment_column:
        required.append(experiment_column)
    if timestamp_column:
        required.append(timestamp_column)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column])
    if experiment_column:
        df = df.dropna(subset=[experiment_column])
    if timestamp_column:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="raise")
        df = df.sort_values([experiment_column, timestamp_column], kind="stable")
    elif experiment_column:
        # Preserve original row order within each experiment.
        df = df.sort_values(experiment_column, kind="stable")
    return df.reset_index(drop=True)

def select_feature_columns(df, target_column, experiment_column="Experiment",
                            timestamp_column=None, feature_columns=None,
                            excluded_features=None):
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")
    excluded = set(DEFAULT_EXCLUDED_FEATURES)
    if excluded_features:
        excluded.update(str(x).strip().lower() for x in excluded_features)
    excluded.update({target_column.strip().lower(), "unnamed: 31"})
    if experiment_column:
        excluded.add(experiment_column.strip().lower())
    if timestamp_column:
        excluded.add(timestamp_column.strip().lower())
    candidates = list(feature_columns) if feature_columns is not None else list(df.columns)
    selected = []
    for col in candidates:
        if col not in df.columns:
            raise ValueError(f"Requested feature not found: {col}")
        name = str(col).strip().lower()
        if name in excluded or any(token in name for token in LEAKAGE_TOKENS):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            selected.append(col)
    if not selected:
        raise ValueError("No usable numeric feature columns were selected.")
    return selected

def prepare_forecasting_data(file_path, target_column, experiment_column="Experiment",
                             timestamp_column=None, feature_columns=None,
                             excluded_features=None):
    df = load_forecasting_data(file_path, target_column, experiment_column, timestamp_column)
    features = select_feature_columns(
        df, target_column, experiment_column, timestamp_column,
        feature_columns, excluded_features
    )
    return df, features
