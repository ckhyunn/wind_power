import pandas as pd

from src.config import CAPACITY_KWH, TARGET_COLS
from src.data import (
    dms_to_decimal,
    load_sample_submission,
    load_scada,
    load_train_labels,
    load_turbine_info,
    load_weather,
)


def test_target_cols_and_capacity():
    assert TARGET_COLS == ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
    assert CAPACITY_KWH == {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}


def test_load_train_labels(tiny_data_dir):
    df = load_train_labels(tiny_data_dir)
    assert list(df.columns) == ["kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]
    assert pd.api.types.is_datetime64_any_dtype(df["kst_dtm"])
    assert len(df) == 40


def test_load_sample_submission(tiny_data_dir):
    df = load_sample_submission(tiny_data_dir)
    assert pd.api.types.is_datetime64_any_dtype(df["forecast_kst_dtm"])
    assert len(df) == 8


def test_load_weather_train_and_test(tiny_data_dir):
    ldaps_train = load_weather(tiny_data_dir, "ldaps", "train")
    gfs_test = load_weather(tiny_data_dir, "gfs", "test")
    assert pd.api.types.is_datetime64_any_dtype(ldaps_train["forecast_kst_dtm"])
    assert len(ldaps_train) == 40 * 4
    assert len(gfs_test) == 8 * 1


def test_load_scada(tiny_data_dir):
    scada_v = load_scada(tiny_data_dir, "vestas")
    scada_u = load_scada(tiny_data_dir, "unison")
    assert pd.api.types.is_datetime64_any_dtype(scada_v["kst_dtm"])
    assert "vestas_wtg01_power_kw10m" in scada_v.columns
    assert "unison_wtg01_power_kw10m" in scada_u.columns


def test_dms_to_decimal():
    lat, lon = dms_to_decimal("37°16'55.61\"N 128°57'02.10\"E")
    assert abs(lat - 37.28211) < 1e-3
    assert abs(lon - 128.95058) < 1e-3

    lat_s, lon_w = dms_to_decimal("1°00'00.00\"S 2°00'00.00\"W")
    assert lat_s < 0
    assert lon_w < 0


def test_load_turbine_info(tiny_info_xlsx):
    info = load_turbine_info(tiny_info_xlsx)
    assert list(info["kpx_group"]) == [1, 1, 3]
    # merged-cell group_capacity_mw must be forward/back-filled within each group.
    assert info.loc[info["kpx_group"] == 1, "group_capacity_mw"].tolist() == [7.2, 7.2]
    assert info.loc[info["kpx_group"] == 3, "group_capacity_mw"].tolist() == [4.2]
    assert "lat" in info.columns and "lon" in info.columns
    assert info["lat"].notna().all()
    assert info["lon"].notna().all()
