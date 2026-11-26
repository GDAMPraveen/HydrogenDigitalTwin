import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from src.forecasting.forecast_evaluation import calculate_forecast_metrics, chronological_train_test_split

def test_metrics_known_values():
    m = calculate_forecast_metrics([1,2,3],[1,2,4])
    assert np.isclose(m["MAE"], 1/3)
    assert m["RMSE"] > 0

def test_chronological_split_within_groups():
    X = pd.DataFrame({"x":range(10)})
    y = pd.Series(range(10))
    meta = pd.DataFrame({"Experiment":["a"]*5+["b"]*5})
    Xtr,Xte,ytr,yte = chronological_train_test_split(X,y,meta,.2)
    assert len(Xte)==2
    assert set(Xtr.index).isdisjoint(set(Xte.index))
