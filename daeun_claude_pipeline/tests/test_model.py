import numpy as np
import pandas as pd

from src.model import predict_lgbm, train_lgbm


def _synthetic_power_curve_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    wind_speed = rng.uniform(0, 20, size=n)
    temp = rng.uniform(-5, 25, size=n)
    noise = rng.normal(0, 0.01, size=n)
    # monotonic-ish capacity-fraction target driven mostly by wind speed, like a power curve.
    target_frac = np.clip(1 / (1 + np.exp(-(wind_speed - 9))) + noise, 0, 1)
    X = pd.DataFrame({"wind_speed": wind_speed, "temp": temp})
    y = pd.Series(target_frac)
    return X, y


def test_predict_output_has_one_prediction_per_row():
    X, y = _synthetic_power_curve_data()
    model = train_lgbm(X, y, num_boost_round=20)
    preds = predict_lgbm(model, X)
    assert len(preds) == len(X)


def test_trained_model_beats_predicting_the_mean():
    X, y = _synthetic_power_curve_data()
    split = 300
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    model = train_lgbm(X_train, y_train, num_boost_round=50)
    preds = predict_lgbm(model, X_test)

    model_mae = np.mean(np.abs(preds - y_test.values))
    baseline_mae = np.mean(np.abs(y_train.mean() - y_test.values))
    assert model_mae < baseline_mae
