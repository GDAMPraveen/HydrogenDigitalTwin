"""Reusable forecast inference and trusted local model persistence."""
from pathlib import Path
import joblib
import pandas as pd

class ForecastEngine:
    def __init__(self, model, feature_columns, target_name=None, horizon_steps=1):
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        if not isinstance(horizon_steps, int) or isinstance(horizon_steps, bool) or horizon_steps < 1:
            raise ValueError("horizon_steps must be a positive integer.")
        self.model = model
        self.feature_columns = list(feature_columns)
        self.target_name = target_name
        self.horizon_steps = horizon_steps

    def predict(self, input_data):
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError("input_data must be a pandas DataFrame.")
        missing = [c for c in self.feature_columns if c not in input_data.columns]
        if missing:
            raise ValueError(f"Missing required forecast features: {missing}")
        return self.model.predict(input_data.loc[:, self.feature_columns])

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_columns": self.feature_columns,
                     "target_name": self.target_name, "horizon_steps": self.horizon_steps},
                    destination)
        return destination

    @classmethod
    def load(cls, path):
        # Only load joblib files created by a trusted local process.
        artifact = joblib.load(path)
        required = {"model", "feature_columns", "target_name", "horizon_steps"}
        if not isinstance(artifact, dict) or not required.issubset(artifact):
            raise ValueError("Invalid forecasting model artifact.")
        return cls(**artifact)
