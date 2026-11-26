import pandas as pd
import pytest
from src.forecasting.data_preparation import select_feature_columns, load_forecasting_data

def test_selects_numeric_and_excludes_target_leakage():
    df = pd.DataFrame({"Experiment":["a","a"], "flow":[1,2], "sensor":[3,4],
                       "Efficiency (kWh/kg)":[2,2], "H2E_f_Elec_CalcProdRate":[1,1]})
    assert select_feature_columns(df, "flow") == ["sensor"]

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_forecasting_data("missing_file.csv", "flow")

def test_load_drops_missing_target_and_keeps_group():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "d.csv"
        pd.DataFrame({"Experiment":["a","a"], "flow":[1,None], "sensor":[4,5]}).to_csv(p,index=False)
        df = load_forecasting_data(p, "flow")
        assert len(df) == 1
