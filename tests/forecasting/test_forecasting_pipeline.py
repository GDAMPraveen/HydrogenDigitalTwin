import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.forecasting.feature_engineering import create_supervised_dataset
from src.forecasting.forecast_evaluation import chronological_train_test_split, calculate_forecast_metrics
from src.forecasting.forecast_engine import ForecastEngine

def test_synthetic_end_to_end_pipeline(tmp_path):
    n = 100
    df = pd.DataFrame({"Experiment":["A"]*50+["B"]*50,
                       "flow":np.r_[np.linspace(1,5,50),np.linspace(2,6,50)],
                       "sensor":np.arange(n,dtype=float)})
    X,y,meta = create_supervised_dataset(df,"flow",horizon_steps=1,lags=(1,2),rolling_windows=(3,),feature_columns=["sensor"])
    Xtr,Xte,ytr,yte = chronological_train_test_split(X,y,meta,.2)
    model = Pipeline([("imputer",SimpleImputer(strategy="median")),("regressor",Ridge())]).fit(Xtr,ytr)
    engine = ForecastEngine(model,list(X.columns),"flow",1)
    pred = engine.predict(Xte)
    assert np.isfinite(pred).all()
    assert calculate_forecast_metrics(yte,pred)["MAE"] >= 0
    p = tmp_path/"pipeline.joblib"
    engine.save(p)
    assert len(ForecastEngine.load(p).predict(Xte)) == len(Xte)
