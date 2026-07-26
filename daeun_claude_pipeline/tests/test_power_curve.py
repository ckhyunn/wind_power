import numpy as np
import pandas as pd

from src.power_curve import (
    apply_speed_bias_correction,
    fit_binned_power_curve,
    fit_speed_bias_correction,
    group_scada_hourly,
    power_curve_feature,
)


def test_group_scada_hourly_sums_10min_energy_to_hourly_kwh():
    # 2 turbines, 6 readings/hour, each reading = 10-min energy (per EDA §6: *_power_kw10m
    # is energy not instantaneous power) -> hourly kWh is the SUM of the 6, not the mean.
    idx = pd.date_range("2022-01-01 00:10", periods=12, freq="10min")
    df = pd.DataFrame(
        {
            "kst_dtm": idx,
            "wtg01_power": [100.0] * 12,
            "wtg02_power": [50.0] * 12,
            "wtg01_ws": [8.0] * 12,
            "wtg02_ws": [8.0] * 12,
        }
    )
    hourly = group_scada_hourly(
        df, power_cols=["wtg01_power", "wtg02_power"], ws_cols=["wtg01_ws", "wtg02_ws"],
        unit_capacity_kw=3600,
    )
    # each hour: 6 readings * (100+50) = 900 kWh
    assert list(hourly["power_kwh"]) == [900.0, 900.0]
    assert list(hourly["wind_speed"]) == [8.0, 8.0]


def test_group_scada_hourly_filters_glitch_readings():
    idx = pd.date_range("2022-01-01 00:10", periods=6, freq="10min")
    power = [100.0] * 5 + [5e7]  # last reading is a sensor glitch, way past rated
    df = pd.DataFrame({"kst_dtm": idx, "wtg01_power": power, "wtg01_ws": [8.0] * 6})
    hourly = group_scada_hourly(
        df, power_cols=["wtg01_power"], ws_cols=["wtg01_ws"], unit_capacity_kw=3600
    )
    # glitch reading dropped (treated as missing), so sum is only the 5 good readings
    assert hourly["power_kwh"].iloc[0] == 500.0


def test_speed_bias_correction_recovers_known_linear_relationship():
    rng = np.random.default_rng(0)
    forecast_ws = rng.uniform(0, 20, size=200)
    scada_ws = 0.9 * forecast_ws + 0.5  # known systematic relationship
    bias = fit_speed_bias_correction(forecast_ws, scada_ws)
    corrected = apply_speed_bias_correction(forecast_ws, bias)
    assert np.allclose(corrected, scada_ws, atol=1e-6)


def test_binned_power_curve_is_monotonic_and_matches_sigmoid_shape():
    rng = np.random.default_rng(1)
    wind_speed = rng.uniform(0, 20, size=2000)
    capacity_fraction = 1 / (1 + np.exp(-(wind_speed - 10))) + rng.normal(0, 0.02, size=2000)
    capacity_fraction = np.clip(capacity_fraction, 0, 1)

    curve = fit_binned_power_curve(wind_speed, capacity_fraction)
    grid = np.linspace(0, 20, 41)
    predicted = power_curve_feature(grid, curve)

    assert np.all(np.diff(predicted) >= -1e-9)  # monotonic non-decreasing
    assert predicted[0] < 0.3  # near cut-in, low output
    assert predicted[-1] > 0.7  # near rated, high output


def test_power_curve_feature_output_matches_input_length():
    rng = np.random.default_rng(2)
    wind_speed = rng.uniform(0, 20, size=500)
    capacity_fraction = np.clip(wind_speed / 20 + rng.normal(0, 0.05, size=500), 0, 1)
    curve = fit_binned_power_curve(wind_speed, capacity_fraction)

    forecast_ws = rng.uniform(-2, 25, size=10)  # includes out-of-range values
    out = power_curve_feature(forecast_ws, curve)
    assert len(out) == 10
    assert np.all((out >= 0) & (out <= 1.01))
