import numpy as np
import pandas as pd

from src.features import (
    build_design_matrix,
    build_weather_features,
    calendar_features,
    reduce_grid,
    scada_hourly_energy,
    uv_to_speed_dir,
)


def test_uv_to_speed_dir_matches_meteorological_from_convention():
    # wind blowing FROM the north (u=0, v=-5, i.e. moving southward) -> reported direction 0deg (N).
    u = pd.Series([0.0, -5.0, 5.0])
    v = pd.Series([-5.0, 0.0, 0.0])
    speed, direction = uv_to_speed_dir(u, v)
    assert np.allclose(speed, [5.0, 5.0, 5.0])
    assert np.isclose(direction.iloc[0], 0.0)  # from the north
    assert np.isclose(direction.iloc[1], 90.0)  # from the east
    assert np.isclose(direction.iloc[2], 270.0)  # from the west


def test_calendar_features_hour_and_weekend():
    dt = pd.Series(pd.to_datetime(["2024-01-06 13:00:00", "2024-06-12 00:00:00"]))
    cal = calendar_features(dt)
    assert cal["hour"].tolist() == [13, 0]
    assert cal["month"].tolist() == [1, 6]
    assert cal["is_weekend"].tolist() == [1, 0]  # 2024-01-06 Sat, 2024-06-12 Wed
    assert np.isclose(cal["hour_sin"].iloc[1], 0.0)


def test_reduce_grid_mean_nearest_idw_differ_when_grids_are_spatially_spread():
    df = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(["2024-01-01 00:00:00"] * 2),
            "grid_id": [1, 2],
            "latitude": [37.0, 37.5],
            "longitude": [129.0, 129.5],
            "speed": [2.0, 10.0],
        }
    )
    farm_lat, farm_lon = 37.01, 129.01  # very close to grid 1

    mean_result = reduce_grid(df, "speed", farm_lat, farm_lon, method="mean")
    nearest_result = reduce_grid(df, "speed", farm_lat, farm_lon, method="nearest")
    idw_result = reduce_grid(df, "speed", farm_lat, farm_lon, method="idw")

    assert np.isclose(mean_result.iloc[0], 6.0)
    assert np.isclose(nearest_result.iloc[0], 2.0)
    # IDW should sit between nearest and mean, pulled toward the closer grid.
    assert 2.0 < idw_result.iloc[0] < 6.0


def test_build_weather_features_merges_ldaps_gfs_and_adds_disagreement():
    common_dt = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"])
    ldaps = pd.DataFrame(
        {
            "forecast_kst_dtm": common_dt,
            "grid_id": [1, 1],
            "latitude": [37.0, 37.0],
            "longitude": [129.0, 129.0],
            "heightAboveGround_10_10u": [3.0, 4.0],
            "heightAboveGround_10_10v": [0.0, 0.0],
        }
    )
    gfs = pd.DataFrame(
        {
            "forecast_kst_dtm": common_dt,
            "grid_id": [1, 1],
            "latitude": [37.0, 37.0],
            "longitude": [129.0, 129.0],
            "heightAboveGround_10_10u": [1.0, 1.0],
            "heightAboveGround_10_10v": [0.0, 0.0],
        }
    )
    merged = build_weather_features(ldaps, gfs, farm_lat=37.0, farm_lon=129.0, method="mean")
    assert "ldaps_ws10" in merged.columns
    assert "gfs_ws10" in merged.columns
    assert "speed_disagree_10m" in merged.columns
    assert np.allclose(merged["speed_disagree_10m"], [2.0, 3.0])


def test_scada_hourly_energy_sums_not_averages_10min_readings():
    # per EDA finding: *_power_kw10m is 10-min ENERGY, hourly kWh = sum of the 6 readings, not mean.
    minutes = pd.date_range("2024-01-01 00:10:00", "2024-01-01 01:00:00", freq="10min")
    df = pd.DataFrame({"kst_dtm": minutes, "wtg01_power_kw10m": [10.0] * 6})
    hourly = scada_hourly_energy(df, power_cols=["wtg01_power_kw10m"])
    assert hourly["wtg01_power_kw10m"].iloc[0] == 60.0  # sum of 6 x 10.0, not mean (10.0)


def test_build_design_matrix_drops_and_adds_calendar():
    base = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
            "kpx_group_1": [100.0, 200.0],
        }
    )
    weather = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
            "ldaps_ws10": [1.0, 2.0],
        }
    )
    design = build_design_matrix(base, weather, "forecast_kst_dtm", drop_cols=["kpx_group_1"])
    assert "kpx_group_1" not in design.columns
    assert "forecast_kst_dtm" not in design.columns
    assert "ldaps_ws10" in design.columns
    assert "hour_sin" in design.columns
    assert len(design) == 2
