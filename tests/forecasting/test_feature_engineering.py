import pandas as pd
from src.forecasting.feature_engineering import create_supervised_dataset

def sample():
    return pd.DataFrame({"Experiment":["a"]*8+["b"]*8,
                         "flow":list(range(8))+list(range(100,108)),
                         "sensor":list(range(16))})

def test_target_is_future_and_lags_are_past():
    X,y,m = create_supervised_dataset(sample(),"flow",horizon_steps=1,lags=(1,),rolling_windows=(3,),feature_columns=["sensor"])
    row = X.index[0]
    assert X.loc[row,"flow_lag_1"] == 2
    assert y.loc[row] == 4

def test_experiments_do_not_share_lag_history():
    X,y,m = create_supervised_dataset(sample(),"flow",horizon_steps=1,lags=(1,),rolling_windows=(3,),feature_columns=["sensor"])
    b_first = m.index[m["Experiment"]=="b"][0]
    assert X.loc[b_first,"flow_lag_1"] == 102

def test_missing_sensor_values_are_preserved_for_imputation():
    df = sample()
    df.loc[4,"sensor"] = float("nan")
    X,y,m = create_supervised_dataset(df,"flow",lags=(1,),rolling_windows=(3,),feature_columns=["sensor"])
    assert X["sensor"].isna().any()
