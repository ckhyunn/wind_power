TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}

GROUP_MAKER = {
    "kpx_group_1": "vestas",
    "kpx_group_2": "vestas",
    "kpx_group_3": "unison",
}

# Non-numeric / identifier columns present in the raw LDAPS/GFS long-format files.
WEATHER_META_COLS = {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude", "longitude"}
