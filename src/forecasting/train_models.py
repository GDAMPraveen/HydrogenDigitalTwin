"""Leakage-safe candidate model construction and training."""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

def build_candidate_models(random_state=42, include_optional=True):
    models = {
        "Ridge": Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler()),
                           ("model", Ridge(alpha=0.01))]),
        "RandomForest": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("model", RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                                                                  random_state=random_state, n_jobs=-1))]),
        "ExtraTrees": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                ("model", ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2,
                                                              random_state=random_state, n_jobs=-1))]),
        "GradientBoosting": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                      ("model", GradientBoostingRegressor(n_estimators=150,
                                                                          learning_rate=0.05, max_depth=3,
                                                                          random_state=random_state))]),
        "HistGradientBoosting": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                          ("model", HistGradientBoostingRegressor(max_iter=150,
                                                                                  learning_rate=0.05,
                                                                                  random_state=random_state))]),
    }
    if include_optional:
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = Pipeline([("imputer", SimpleImputer(strategy="median")),
                ("model", XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05,
                                       subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
                                       random_state=random_state, n_jobs=-1))])
        except ImportError:
            pass
        try:
            from lightgbm import LGBMRegressor
            models["LightGBM"] = Pipeline([("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMRegressor(n_estimators=150, learning_rate=0.05,
                                        random_state=random_state, n_jobs=-1, verbosity=-1))])
        except ImportError:
            pass
        try:
            from catboost import CatBoostRegressor
            models["CatBoost"] = Pipeline([("imputer", SimpleImputer(strategy="median")),
                ("model", CatBoostRegressor(iterations=150, depth=6, learning_rate=0.05,
                                            random_seed=random_state, verbose=False, allow_writing_files=False))])
        except ImportError:
            pass
    return models

def train_models(X_train, y_train, models=None, random_state=42, include_optional=True):
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data cannot be empty.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train lengths must match.")
    candidates = build_candidate_models(random_state, include_optional=include_optional) if models is None else models
    trained = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained
