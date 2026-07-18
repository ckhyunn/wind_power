"""
RandomForest 베이스라인 개선 버전.

공식 baseline(codeshare 14031) 대비 바뀐 점:
  1. LDAPS/GFS를 전국 평균 1개로 뭉치지 않고, 그룹별로 가장 가까운 격자를 골라
     그룹 전용 날씨 피처를 구성 (info.xlsx 기반 그룹 좌표 계산)
  2. U/V 바람 성분을 풍속 크기 + 풍속^3 + 풍향(sin/cos)으로 변환 (파워커브 근사)
  3. 공식 평가 산식(1-NMAE, FICR)으로 홀드아웃 검증 + 오차율/편향 진단 출력
  4. v3: 홀드아웃 검증 결과 뚜렷한 '과소예측' 편향이 확인되어, 학습/보정(calib)/최종검증(holdout)
     3구간으로 나누고 보정용 구간에서 선형 보정식(실제=a+b*예측)을 학습해 적용.
     보정 전/후 Score를 같이 출력해서 실제로 개선되는지 검증함.

실행:
    python src/train_baseline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

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
from evaluate import metric, error_rate_breakdown, bias_diagnosis

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
SUBMISSION_DIR = Path(__file__).resolve().parent.parent / "submissions"

N_NEAREST_GRIDS = 3       # 그룹 좌표에서 가장 가까운 격자 몇 개를 쓸지
HOLDOUT_DAYS = 90         # 로컬 검증용 최근 N일을 holdout으로 분리 (시계열이므로 랜덤분할 금지)
CALIB_DAYS = 60           # holdout 바로 이전 N일을 '보정식 학습용'으로 별도 분리 (holdout과 절대 겹치지 않음)


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
    calib_start = cutoff - pd.Timedelta(days=CALIB_DAYS)

    predictions_test = pd.DataFrame({"forecast_kst_dtm": sample_submission["forecast_kst_dtm"]})
    predictions_holdout_raw = {}
    predictions_holdout_calibrated = {}
    holdout_actual = {}
    calibrators = {}

    for target in TARGET_COLS:
        weather = train_weather[target]
        X_all = build_features(train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"}), weather, "forecast_kst_dtm")
        y_all = train_labels[target]
        dt_all = train_labels["kst_dtm"]

        mask_label = y_all.notna()
        X_all, y_all, dt_all = X_all[mask_label], y_all[mask_label], dt_all[mask_label]

        # 3구간 분리: 학습 / 보정식 학습(calib) / 최종 검증(holdout) - 서로 절대 겹치지 않음
        is_holdout = dt_all > cutoff
        is_calib = (dt_all > calib_start) & (dt_all <= cutoff)
        is_train = dt_all <= calib_start

        X_tr, y_tr = X_all[is_train], y_all[is_train]
        X_cal, y_cal = X_all[is_calib], y_all[is_calib]
        X_ho, y_ho = X_all[is_holdout], y_all[is_holdout]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns)
        X_cal_imp = pd.DataFrame(imputer.transform(X_cal), columns=X_cal.columns)
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

        # calib 구간에서 '보정식' 학습: 실제값 = a + b * 모델예측값
        pred_cal = model.predict(X_cal_imp)
        calibrator = LinearRegression()
        calibrator.fit(pred_cal.reshape(-1, 1), y_cal)
        calibrators[target] = calibrator
        print(f"[{target}] 보정식: 실제값 = {calibrator.intercept_:.1f} + {calibrator.coef_[0]:.4f} * 예측값")

        # holdout에서 raw 예측 vs 보정 후 예측 둘 다 계산해서 개선 여부 확인
        pred_ho_raw = np.clip(model.predict(X_ho_imp), 0, CAPACITY_KWH[target])
        pred_ho_calibrated = np.clip(calibrator.predict(pred_ho_raw.reshape(-1, 1)), 0, CAPACITY_KWH[target])

        predictions_holdout_raw[target] = pred_ho_raw
        predictions_holdout_calibrated[target] = pred_ho_calibrated
        holdout_actual[target] = y_ho.values

        # 최종 제출용 모델: holdout을 제외한 전체(train+calib)로 재학습 후, 같은 보정식 적용
        X_fit = pd.concat([X_tr, X_cal])
        y_fit = pd.concat([y_tr, y_cal])
        X_fit_imp = pd.DataFrame(imputer.fit_transform(X_fit), columns=X_fit.columns)
        model.fit(X_fit_imp, y_fit)

        X_test = build_features(sample_submission[["forecast_id", "forecast_kst_dtm"]], test_weather[target], "forecast_kst_dtm")
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        pred_test_raw = np.clip(model.predict(X_test_imp), 0, CAPACITY_KWH[target])
        pred_test_calibrated = np.clip(calibrator.predict(pred_test_raw.reshape(-1, 1)), 0, CAPACITY_KWH[target])
        predictions_test[target] = pred_test_calibrated

        print(f"[{target}] train={len(X_tr)}, calib={len(X_cal)}, holdout={len(X_ho)}")

    # 그룹별 holdout 길이가 다를 수 있음(그룹3은 라벨 시작이 늦음).
    # metric()은 컬럼(그룹)별로 독립 계산하므로, 길이가 짧은 그룹은 NaN으로 채워도
    # 10% 미만 마스크에서 자동 제외되어 안전함 (데이터 손실 없이 정렬).
    def to_df(d):
        return pd.concat([pd.Series(v, name=k) for k, v in d.items()], axis=1)

    actual_df = to_df(holdout_actual)
    pred_df_raw = to_df(predictions_holdout_raw)
    pred_df_calibrated = to_df(predictions_holdout_calibrated)

    score_raw, nmae_raw, ficr_raw = metric(actual_df, pred_df_raw)
    score_cal, nmae_cal, ficr_cal = metric(actual_df, pred_df_calibrated)

    print("\n[보정 전 vs 보정 후 - holdout 구간, 공식 산식 기준]")
    print(f"{'지표':<10}{'보정 전':>12}{'보정 후':>12}")
    print(f"{'Score':<10}{score_raw:>12.4f}{score_cal:>12.4f}")
    print(f"{'1-NMAE':<10}{nmae_raw:>12.4f}{nmae_cal:>12.4f}")
    print(f"{'FICR':<10}{ficr_raw:>12.4f}{ficr_cal:>12.4f}")

    print("\n[오차율 구간별 분포 - 보정 후]")
    print(error_rate_breakdown(actual_df, pred_df_calibrated).to_string(index=False))
    print("\n[편향 진단 - 보정 후]")
    print(bias_diagnosis(actual_df, pred_df_calibrated).to_string(index=False))

    SUBMISSION_DIR.mkdir(exist_ok=True)
    submission = sample_submission[["forecast_id", "forecast_kst_dtm"]].copy()
    for target in TARGET_COLS:
        submission[target] = predictions_test[target]
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    out_path = SUBMISSION_DIR / "baseline_v3_submit.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
