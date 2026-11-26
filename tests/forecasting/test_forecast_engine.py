import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.forecasting.forecast_engine import ForecastEngine

def test_predict_and_save_load(tmp_path):
    X = pd.DataFrame({"x":[0,1,2,3]})
    y = np.array([0,2,4,6])
    model = LinearRegression().fit(X,y)
    engine = ForecastEngine(model,["x"],"flow",horizon_steps=3)
    expected = engine.predict(X)
    path = tmp_path/"model.joblib"
    engine.save(path)
    loaded = ForecastEngine.load(path)
    np.testing.assert_allclose(loaded.predict(X),expected)
    assert loaded.horizon_steps == 3

def test_missing_feature_rejected():
    engine = ForecastEngine(LinearRegression(),["required"])
    import pytest
    with pytest.raises(ValueError):
        engine.predict(pd.DataFrame({"other":[1]}))
