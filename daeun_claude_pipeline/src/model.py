import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    # Letting LightGBM default to all logical cores causes severe OpenMP thread
    # oversubscription under WSL2 (measured: 252s vs 1.2s for the same 300-round fit
    # on ~26k rows) -> capped explicitly rather than left to auto-detect.
    "num_threads": 4,
}


def train_lgbm(X: pd.DataFrame, y: pd.Series, params: dict = None, num_boost_round: int = 200) -> lgb.Booster:
    """Train a LightGBM regressor on a capacity-fraction target (docs/PLAN.md §1.3:
    train on target/capacity, not raw kWh, so error contributions are comparable
    across groups of slightly different capacity)."""
    merged_params = {**DEFAULT_PARAMS, **(params or {})}
    dataset = lgb.Dataset(X, label=y)
    return lgb.train(merged_params, dataset, num_boost_round=num_boost_round)


def predict_lgbm(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)
