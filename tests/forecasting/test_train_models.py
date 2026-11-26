import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from src.forecasting.train_models import train_models, build_candidate_models

def test_candidate_models_exist():
    models = build_candidate_models(include_optional=False)
    assert {"Ridge","RandomForest","ExtraTrees","GradientBoosting","HistGradientBoosting"} <= set(models)

def test_training_fits_supplied_model():
    X = pd.DataFrame({"x":np.arange(20,dtype=float)})
    y = 2*X["x"] + 1
    trained = train_models(X,y,{"dummy":DummyRegressor(strategy="mean")})
    assert len(trained["dummy"].predict(X)) == 20
