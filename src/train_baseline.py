"""
RandomForest 베이스라인 개선 버전.

공식 baseline(codeshare 14031) 대비 바뀐 점:
  1. LDAPS/GFS를 전국 평균 1개로 뭉치지 않고, 그룹별로 가장 가까운 격자를 골라
     그룹 전용 날씨 피처를 구성 (info.xlsx 기반 그룹 좌표 계산)
  2. U/V 바람 성분을 풍속 크기 + 풍속^3 + 풍향(sin/cos)으로 변환 (파워커브 근사)
  3. 제출 전에 시계열 holdout으로 로컬 1-NMAE를 계산해서 리더보드 제출 전 확인 가능

실행:
    python src/train_baseline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from features import (
    TARGET_COLS,
    CAPACITY_KWH,
    load_turbine_table,
    compute_group_coords,
    nearest_grid_ids,
    add_wind_features,
    aggregate_weather_for_group,
    calendar_features,
)
from evaluate import nmae_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
SUBMISSION_DIR = Path(__file__).resolve().parent.parent / "submissions"

N_NEAREST_GRIDS = 3       # 그룹 좌표에서 가장 가까운 격자 몇 개를 쓸지
HOLDOUT_DAYS = 90         # 로컬 검증용 최근 N일을 holdout으로 분리 (시계열이므로 랜덤분할 금지)


def build_group_weather(ldaps: pd.DataFrame, gfs: pd.DataFrame, group_coords: dict) -> dict:
    """그룹별로 (LDAPS+GFS 병합, 풍속 파생피처까지 포함된) 날씨 피처 테이블 생성"""
    group_weather = {}
    for group, coord in group_coords.items():
        lat, lon = coord["lat"], coord["lon"]

        ldaps_grids = nearest_grid_ids(ldaps, lat, lon, k=N_NEAREST_GRIDS)
        gfs_grids = nearest_grid_ids(gfs, lat, lon, k=N_NEAREST_GRIDS)

        ldaps_agg = aggregate_weather_for_group(ldaps, ldaps_grids, "ldaps")
        gfs_agg = aggregate_weather_for_group(gfs, gfs_grids, "gfs")

        weather = ldaps_agg.merge(gfs_agg, on="forecast_kst_dtm", how="inner")

        weather = add_wind_features(
            weather, "ldaps_heightAboveGround_10_10u_mean", "ldaps_heightAboveGround_10_10v_mean", "ldaps_ws10"
        )
        weather = add_wind_features(
            weather, "gfs_heightAboveGround_10_10u_mean", "gfs_heightAboveGround_10_10v_mean", "gfs_ws10"
        )
        weather = add_wind_features(
            weather, "gfs_heightAboveGround_100_100u_mean", "gfs_heightAboveGround_100_100v_mean", "gfs_ws100"
        )

        group_weather[group] = weather
    return group_weather


def build_features(labels_or_submission: pd.DataFrame, weather: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    df = labels_or_submission.merge(weather, left_on=dt_col, right_on="forecast_kst_dtm", how="left")
    cal = calendar_features(df[dt_col])
    feature_cols = [c for c in weather.columns if c != "forecast_kst_dtm"]
    X = pd.concat([cal, df[feature_cols]], axis=1)
    return X


def main():
    train_labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    ldaps_train = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv(TEST_DIR / "ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv(TEST_DIR / "gfs_test.csv", encoding="utf-8-sig")

    train_labels["kst_dtm"] = pd.to_datetime(train_labels["kst_dtm"])
    sample_submission["forecast_kst_dtm"] = pd.to_datetime(sample_submission["forecast_kst_dtm"])

    turbine_df = load_turbine_table(DATA_DIR / "info.xlsx")
    group_coords = compute_group_coords(turbine_df)
    print("그룹별 대표 좌표:", group_coords)

    train_weather = build_group_weather(ldaps_train, gfs_train, group_coords)
    test_weather = build_group_weather(ldaps_test, gfs_test, group_coords)

    cutoff = train_labels["kst_dtm"].max() - pd.Timedelta(days=HOLDOUT_DAYS)

    predictions_test = pd.DataFrame({"forecast_kst_dtm": sample_submission["forecast_kst_dtm"]})
    predictions_holdout = {}
    holdout_actual = {}

    for target in TARGET_COLS:
        weather = train_weather[target]
        X_all = build_features(train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"}), weather, "forecast_kst_dtm")
        y_all = train_labels[target]
        dt_all = train_labels["kst_dtm"]

        mask_label = y_all.notna()
        X_all, y_all, dt_all = X_all[mask_label], y_all[mask_label], dt_all[mask_label]

        is_holdout = dt_all > cutoff
        X_tr, y_tr = X_all[~is_holdout], y_all[~is_holdout]
        X_ho, y_ho = X_all[is_holdout], y_all[is_holdout]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns)
        X_ho_imp = pd.DataFrame(imputer.transform(X_ho), columns=X_ho.columns)

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=6,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr_imp, y_tr)

        pred_ho = np.clip(model.predict(X_ho_imp), 0, CAPACITY_KWH[target])
        predictions_holdout[target] = pred_ho
        holdout_actual[target] = y_ho.values

        X_all_imp = pd.DataFrame(imputer.fit_transform(X_all), columns=X_all.columns)
        model.fit(X_all_imp, y_all)

        X_test = build_features(sample_submission[["forecast_id", "forecast_kst_dtm"]], test_weather[target], "forecast_kst_dtm")
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        pred_test = np.clip(model.predict(X_test_imp), 0, CAPACITY_KWH[target])
        predictions_test[target] = pred_test

        print(f"[{target}] train={len(X_tr)}, holdout={len(X_ho)}")

    pred_df = pd.DataFrame(predictions_holdout)
    actual_df = pd.DataFrame(holdout_actual)
    print("\n[로컬 검증 - holdout 구간 1-NMAE]")
    print(nmae_score(pred_df, actual_df, CAPACITY_KWH))

    SUBMISSION_DIR.mkdir(exist_ok=True)
    submission = sample_submission[["forecast_id", "forecast_kst_dtm"]].copy()
    for target in TARGET_COLS:
        submission[target] = predictions_test[target]
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    out_path = SUBMISSION_DIR / "baseline_v2_submit.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
