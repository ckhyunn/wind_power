"""
LightGBM 베이스라인 (v7).

공식 baseline(codeshare 14031) 대비 바뀐 점:
  1. LDAPS/GFS를 전국 평균 1개로 뭉치지 않고, 그룹별로 가장 가까운 격자를 골라
     그룹 전용 날씨 피처를 구성 (info.xlsx 기반 그룹 좌표 계산)
  2. U/V 바람 성분을 풍속 크기 + 풍속^3 + 풍향(sin/cos)으로 변환 (파워커브 근사)
  3. 공식 평가 산식(1-NMAE, FICR)으로 홀드아웃 검증 + 오차율/편향 진단 출력
  4. v3: 홀드아웃 검증 결과 뚜렷한 '과소예측' 편향이 확인되어, 학습/보정(calib)/최종검증(holdout)
     3구간으로 나누고 보정용 구간에서 선형 보정식(실제=a+b*예측)을 학습해 적용.
     보정 전/후 Score를 같이 출력해서 실제로 개선되는지 검증함.
  5. v4: 예측값 구간별 편향 진단 결과, 편향의 크기/부호가 구간마다 달라(중간 구간이 가장 심함,
     일부 그룹은 고발전 구간에서 부호까지 반전) 선형 보정의 한계가 확인됨.
     -> IsotonicRegression(단조 비선형 보정)으로 교체했으나, calib 데이터(약 1,440시간)가
        부족해 오히려 과적합되어 v3보다 성능이 떨어짐을 확인.
  6. v5: calib 구간을 다시 반으로 나눠(calib_fit/calib_val) 선형/isotonic을 둘 다 학습해보고
     calib_val(둘 다 안 본 데이터)에서 더 나은 쪽을 그룹별로 자동 선택. 데이터가 충분치 않을 때
     isotonic이 과적합하는 문제를 자동으로 회피.
  7. v6: 근소한 차이(노이즈)로 isotonic이 선택되는 것을 막기 위해 5% 마진 추가.
  8. v7: 보정(후처리)만으로는 Score가 0.58~0.62 사이에서 정체됨을 확인.
     -> RandomForest를 LightGBM으로 교체 (부스팅 계열은 극단값을 평균으로 당기는 경향이 적어
        RandomForest 특유의 과소예측 편향 자체를 줄이는 효과를 기대). calib을 조기종료
        검증셋으로 사용해 과적합 방지.

실행:
    python src/train_baseline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
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
from evaluate import metric, error_rate_breakdown, bias_diagnosis, bias_by_prediction_quantile

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
SUBMISSION_DIR = Path(__file__).resolve().parent.parent / "submissions"

N_NEAREST_GRIDS = 3       # 그룹 좌표에서 가장 가까운 격자 몇 개를 쓸지
HOLDOUT_DAYS = 90         # 로컬 검증용 최근 N일을 holdout으로 분리 (시계열이므로 랜덤분할 금지)
CALIB_DAYS = 60           # holdout 바로 이전 N일을 '보정식 학습용'으로 별도 분리 (holdout과 절대 겹치지 않음)


class _Reshape1DWrapper:
    """LinearRegression은 2D 입력을 요구하는데, IsotonicRegression은 1D를 요구함.
    호출부에서 매번 분기하지 않도록 1D predict 인터페이스로 감싸는 얇은 wrapper."""

    def __init__(self, fitted_linear_model):
        self._model = fitted_linear_model

    def predict(self, x_1d):
        return self._model.predict(np.asarray(x_1d).reshape(-1, 1))


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

        model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        # calib을 조기종료 검증셋으로 사용 (holdout은 절대 학습/조기종료에 쓰지 않음)
        model.fit(
            X_tr_imp, y_tr,
            eval_set=[(X_cal_imp, y_cal)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        best_iteration = model.best_iteration_
        print(f"[{target}] LightGBM 조기종료 - best_iteration={best_iteration}")

        # calib 구간에서 '보정식' 학습.
        # v4에서 isotonic이 calib 데이터 부족으로 과적합해 오히려 성능이 나빠진 것을 확인했음.
        # -> calib을 다시 앞/뒤로 나눠(calib_fit/calib_val), 선형과 isotonic 둘 다 calib_fit으로
        #    학습한 뒤 calib_val(둘 다 안 본 데이터)에서 어느 쪽이 더 나은지 비교해서 자동 선택.
        #    선택된 방식은 calib 전체(calib_fit+calib_val)로 다시 학습해서 최종 사용.
        pred_cal = model.predict(X_cal_imp)
        n_cal = len(pred_cal)
        split = n_cal // 2
        pred_cal_fit, pred_cal_val = pred_cal[:split], pred_cal[split:]
        y_cal_fit, y_cal_val = y_cal.values[:split], y_cal.values[split:]

        def calib_error(calib_model) -> float:
            pred_val = np.clip(calib_model.predict(pred_cal_val), 0, CAPACITY_KWH[target])
            return np.mean(np.abs(pred_val - y_cal_val)) / CAPACITY_KWH[target]

        linear_probe = LinearRegression()
        linear_probe.fit(pred_cal_fit.reshape(-1, 1), y_cal_fit)
        linear_probe_wrapped = _Reshape1DWrapper(linear_probe)
        linear_err = calib_error(linear_probe_wrapped)

        isotonic_probe = IsotonicRegression(out_of_bounds="clip", increasing=True)
        isotonic_probe.fit(pred_cal_fit, y_cal_fit)
        isotonic_err = calib_error(isotonic_probe)

        # v5: 근소한 차이(예: 0.0537 vs 0.0543)로 isotonic이 선택되면 노이즈에 취약함.
        # -> isotonic이 linear보다 최소 SWITCH_MARGIN(상대 개선폭) 이상 확실히 나을 때만 전환.
        #    애매하면 더 단순하고 안정적인 linear를 기본값으로 유지.
        SWITCH_MARGIN = 0.05  # isotonic이 linear보다 5% 이상 더 낮은 오차를 보여야 전환
        if isotonic_err < linear_err * (1 - SWITCH_MARGIN):
            chosen = "isotonic"
            calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(pred_cal, y_cal)
        else:
            chosen = "linear"
            calibrator = _Reshape1DWrapper(LinearRegression().fit(pred_cal.reshape(-1, 1), y_cal))

        calibrators[target] = calibrator
        print(f"[{target}] 보정 방식 선택: {chosen} (calib_val 오차율 - linear={linear_err:.4f}, isotonic={isotonic_err:.4f})")

        # holdout에서 raw 예측 vs 보정 후 예측 둘 다 계산해서 개선 여부 확인
        pred_ho_raw = np.clip(model.predict(X_ho_imp), 0, CAPACITY_KWH[target])
        pred_ho_calibrated = np.clip(calibrator.predict(pred_ho_raw), 0, CAPACITY_KWH[target])

        predictions_holdout_raw[target] = pred_ho_raw
        predictions_holdout_calibrated[target] = pred_ho_calibrated
        holdout_actual[target] = y_ho.values

        # 최종 제출용 모델: holdout을 제외한 전체(train+calib)로 재학습 후, 같은 보정식 적용
        # 이 단계는 별도 검증셋이 없으므로 조기종료 대신 앞서 찾은 best_iteration을 그대로 사용
        model.set_params(n_estimators=best_iteration)
        X_fit = pd.concat([X_tr, X_cal])
        y_fit = pd.concat([y_tr, y_cal])
        X_fit_imp = pd.DataFrame(imputer.fit_transform(X_fit), columns=X_fit.columns)
        model.fit(X_fit_imp, y_fit)

        X_test = build_features(sample_submission[["forecast_id", "forecast_kst_dtm"]], test_weather[target], "forecast_kst_dtm")
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        pred_test_raw = np.clip(model.predict(X_test_imp), 0, CAPACITY_KWH[target])
        pred_test_calibrated = np.clip(calibrator.predict(pred_test_raw), 0, CAPACITY_KWH[target])
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
    print("\n[예측값 구간별(quantile) 편향 - 보정 후, 선형 보정이 구간별로도 고르게 먹혔는지 확인]")
    print(bias_by_prediction_quantile(actual_df, pred_df_calibrated).to_string(index=False))

    SUBMISSION_DIR.mkdir(exist_ok=True)
    submission = sample_submission[["forecast_id", "forecast_kst_dtm"]].copy()
    for target in TARGET_COLS:
        submission[target] = predictions_test[target]
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    out_path = SUBMISSION_DIR / "baseline_v7_submit.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
