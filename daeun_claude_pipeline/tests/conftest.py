import pandas as pd
import pytest


@pytest.fixture
def tiny_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    hours = pd.date_range("2024-12-01 01:00:00", periods=48, freq="h")
    train_hours = hours[:40]
    test_hours = hours[40:]

    # 4 LDAPS grids at two farm sites (2 grids near each site), so nearest/IDW reduction is testable.
    ldaps_grids = [
        {"grid_id": 1, "latitude": 37.30, "longitude": 128.95},
        {"grid_id": 2, "latitude": 37.31, "longitude": 128.96},
        {"grid_id": 3, "latitude": 37.20, "longitude": 129.05},
        {"grid_id": 4, "latitude": 37.21, "longitude": 129.06},
    ]

    def make_ldaps(hours):
        rows = []
        for i, dt in enumerate(hours):
            for g in ldaps_grids:
                rows.append(
                    {
                        "forecast_kst_dtm": dt,
                        "data_available_kst_dtm": dt - pd.Timedelta(hours=12),
                        "grid_id": g["grid_id"],
                        "latitude": g["latitude"],
                        "longitude": g["longitude"],
                        "heightAboveGround_10_10u": 3.0 + 0.1 * i + g["grid_id"],
                        "heightAboveGround_10_10v": 1.0 + 0.05 * i,
                        "heightAboveGround_50_50MUmax": 5.0 + g["grid_id"],
                        "heightAboveGround_50_50MUmin": 4.0 + g["grid_id"],
                        "heightAboveGround_50_50MVmax": 2.0,
                        "heightAboveGround_50_50MVmin": 1.0,
                        "heightAboveGround_2_t": 280.0 + i,
                    }
                )
        return pd.DataFrame(rows)

    gfs_grids = [
        {"grid_id": 1, "latitude": 37.25, "longitude": 129.00},
    ]

    def make_gfs(hours):
        rows = []
        for i, dt in enumerate(hours):
            for g in gfs_grids:
                rows.append(
                    {
                        "forecast_kst_dtm": dt,
                        "data_available_kst_dtm": dt - pd.Timedelta(hours=12),
                        "grid_id": g["grid_id"],
                        "latitude": g["latitude"],
                        "longitude": g["longitude"],
                        "heightAboveGround_10_10u": 2.0 + 0.1 * i,
                        "heightAboveGround_10_10v": 0.5,
                        "heightAboveGround_100_100u": 3.0 + 0.1 * i,
                        "heightAboveGround_100_100v": 0.8,
                        "surface_0_gust": 6.0 + 0.1 * i,
                    }
                )
        return pd.DataFrame(rows)

    make_ldaps(train_hours).to_csv(train_dir / "ldaps_train.csv", index=False, encoding="utf-8-sig")
    make_gfs(train_hours).to_csv(train_dir / "gfs_train.csv", index=False, encoding="utf-8-sig")
    make_ldaps(test_hours).to_csv(test_dir / "ldaps_test.csv", index=False, encoding="utf-8-sig")
    make_gfs(test_hours).to_csv(test_dir / "gfs_test.csv", index=False, encoding="utf-8-sig")

    hour_of_day = pd.Series(range(len(train_hours))) % 24
    labels = pd.DataFrame(
        {
            "kst_dtm": train_hours,
            "kpx_group_1": 5000 + 100 * hour_of_day.values,
            "kpx_group_2": 4000 + 80 * hour_of_day.values,
            "kpx_group_3": 3000 + 60 * hour_of_day.values,
        }
    )
    labels.to_csv(train_dir / "train_labels.csv", index=False, encoding="utf-8-sig")

    submission = pd.DataFrame(
        {
            "forecast_id": [f"forecast_{i:04d}" for i in range(len(test_hours))],
            "forecast_kst_dtm": test_hours,
            "kpx_group_1": 0,
            "kpx_group_2": 0,
            "kpx_group_3": 0,
        }
    )
    submission.to_csv(data_dir / "sample_submission.csv", index=False, encoding="utf-8-sig")

    # 10-minute SCADA: 6 vestas turbines (group_1), each reading = kWh-per-10min (per EDA §6 finding).
    scada_minutes = pd.date_range("2024-12-01 00:10:00", "2024-12-02 00:00:00", freq="10min")
    v_cols = {f"vestas_wtg{n:02d}_power_kw10m": 50.0 + n for n in range(1, 7)}
    v_cols.update({f"vestas_wtg{n:02d}_ws": 6.0 for n in range(1, 7)})
    v_cols.update({f"vestas_wtg{n:02d}_wd": 200.0 for n in range(1, 7)})
    scada_v = pd.DataFrame({"kst_dtm": scada_minutes})
    for col, val in v_cols.items():
        scada_v[col] = val
    scada_v.to_csv(train_dir / "scada_vestas_train.csv", index=False, encoding="utf-8-sig")

    u_cols = {f"unison_wtg{n:02d}_power_kw10m": 30.0 + n for n in range(1, 6)}
    u_cols.update({f"unison_wtg{n:02d}_ws": 5.0 for n in range(1, 6)})
    u_cols.update({f"unison_wtg{n:02d}_wd": -60.0 for n in range(1, 6)})
    scada_u = pd.DataFrame({"kst_dtm": scada_minutes})
    for col, val in u_cols.items():
        scada_u[col] = val
    scada_u.to_csv(train_dir / "scada_unison_train.csv", index=False, encoding="utf-8-sig")

    return data_dir


@pytest.fixture
def tiny_info_xlsx(tmp_path):
    import openpyxl

    path = tmp_path / "info.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "info"
    # Rows 1-3: title/blank rows (mirrors real file's offset before the header).
    ws.append([None])
    ws.append([None, "title"])
    ws.append([None])
    ws.append([None, "stage", "name", "maker", "model", "unit_no", "coord", "kpx_group",
               "hub_height", "rotor_d", "unit_capacity_mw", "group_capacity_mw"])
    rows = [
        [None, 1, "farm-a", "VESTAS", "V126", 1, "37°16'55.61\"N 128°57'02.10\"E", 1, 117, 126, 3.6, 7.2],
        [None, 1, "farm-a", "VESTAS", "V126", 2, "37°17'04.05\"N 128°56'58.35\"E", None, 117, 126, 3.6, None],
        [None, 2, "farm-b", "UNISON", "U136", 1, "37°16'59.73\"N 128°57'44.97\"E", 3, 117, 136, 4.2, 4.2],
    ]
    for row in rows:
        ws.append(row)
    wb.save(path)
    return path
