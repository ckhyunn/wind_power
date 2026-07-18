"""
LightGBM+XGBoost 블렌딩 베이스라인 (v18).

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
     (v8 IDW, v9 고풍속플래그, v10 하이퍼파라미터탐색 -> 셋 다 calib 기준 개선이 holdout엔
      반영 안 됨. calib 60일로 '여러 선택지 중 고르기'를 반복한 게 원인으로 진단됨)
  9. v11: 진단에 따라 방향 전환 - '선택/탐색' 대신 '원칙에 기반한 확정 변경 + 무조건 앙상블'로.
     - objective를 기본값(L2) 대신 'mae'(L1)로 고정: 평가지표(NMAE, FICR)가 절대오차
       기반이라 학습 목적함수를 일치시킴 (탐색 아님, 원리적 결정)
     - 시드 3개 앙상블 (평균): '고르기'가 아니라 '항상 평균'이라 과적합 위험 없이 분산 감소
     - 리드타임(lead_time_hours) 피처 추가
     - 최근접 격자 원본값(평균으로 뭉개지 않은 값)을 별도 피처로 추가
     (v12: holdout을 1년 전 구간으로 이동 + 시드 7개 확장 -> 실제 리더보드 하락, 미채택)
     (backtest.py: 5개 시기 평균 검증 인프라 구축. raw vs 보정 실제 비교 결과 차이가
      0.007 수준 노이즈로 확인되어 보정 있는 v11 방식 유지)
  10. v14: SCADA 파워커브(물리 기반) 피처 추가. SCADA 실측 풍속은 예보 풍속과 스케일이
      안 맞아 직접 못 쓰므로(v9에서 확인), 대신 '예보 풍속(gfs_ws100_speed) -> 실제
      발전량'의 경험적 관계를 학습구간에서 isotonic으로 직접 적합해 강력한 단일 피처로 추가.
      -> backtest.py로 검증(파워커브 유무 비교) 결과 평균 Score 사실상 동일, 성능 기여 없음
         (LightGBM이 원본 풍속 피처만으로 이미 학습하고 있던 것으로 추정). 해롭지 않아 유지.
  11. v15: 모델 학습 로직을 modeling.py로 공용화 (train_baseline.py/backtest.py가 반드시
      같은 함수를 쓰도록 강제 - 파워커브 때 두 스크립트가 어긋났던 문제 재발 방지).
      LightGBM 단독 앙상블에 XGBoost도 섞은 '모델 블렌딩'으로 확장. 서로 다른 알고리즘은
      오차 패턴도 달라 시드 앙상블(v11)보다 분산 감소 효과가 클 것으로 기대.
      (v16: SCADA 2단계 물리 피처(예보풍속->실측풍속 환산->진짜 파워커브) 추가.
       diagnose_vestas.py로 VESTAS 이상치를 걷어내고 만든 피처. holdout에서는 최고치였지만
       backtest 5윈도우 평균은 v15와 완전히 동일 -> 유효한 개선 아님으로 결론)
      v8~v16까지 9번의 서로 다른 시도(격자집계/컷아웃/하이퍼파라미터/파워커브/모델블렌딩/
      SCADA물리피처)가 전부 backtest 평균 0.586~0.592 근처에서 정체. '이미 있는 정보를
      모델이 더 잘 쓰게 돕는' 접근은 한계에 도달한 것으로 진단, 방향 전환.
  12. v17: 지금까지 안 써본 새로운 정보를 추가.
      - 공기밀도(온도+기압, 이상기체법칙) - 같은 풍속이라도 밀도 높은 겨울 공기가 더 많은
        운동에너지를 전달하는데 이 정보가 지금까지 전혀 없었음
      - 윈드시어(100m-10m 풍속차) - 대기 안정도/난류 강도의 대리지표
      - 인접 시간대(전/후 1시간) 풍속 - 지금까지 매 시간을 독립적으로 취급했는데, 처음으로
        시간적 흐름 정보를 제공 (같은 예보 배치 안이라 데이터 누수 아님)

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
    nearest_grids_with_distance,
    add_wind_features,
    aggregate_weather_for_group,
    aggregate_weather_dispersion,
    lead_time_feature,
    nearest_grid_raw_features,
    add_air_density_feature,
    add_wind_shear_feature,
    add_lag_lead_features,
    calendar_features,
    fit_power_curve,
    apply_power_curve,
)
from evaluate import metric, error_rate_breakdown, bias_diagnosis, bias_by_prediction_quantile
from modeling import train_blended_ensemble, ensemble_predict, refit_final_models, final_predict

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
    """그룹별로 (LDAPS+GFS 병합, 풍속 파생피처 + 리드타임 + 최근접 격자 원본값까지 포함된)
    날씨 피처 테이블 생성"""
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

        # 리드타임 피처 (LDAPS 기준 - LDAPS/GFS 모두 동일한 발표 스케줄을 쓰므로 대표로 사용)
        lead = lead_time_feature(ldaps)
        weather = weather.merge(lead, on="forecast_kst_dtm", how="left")

        # 최근접 격자 원본값 (평균으로 뭉개지 않은 값도 별도로 제공)
        ldaps_grid_dist = nearest_grids_with_distance(ldaps, lat, lon, k=N_NEAREST_GRIDS)
        gfs_grid_dist = nearest_grids_with_distance(gfs, lat, lon, k=N_NEAREST_GRIDS)
        ldaps_nearest1 = nearest_grid_raw_features(ldaps, ldaps_grid_dist, "ldaps", top_n=1)
        gfs_nearest1 = nearest_grid_raw_features(gfs, gfs_grid_dist, "gfs", top_n=1)
        weather = weather.merge(ldaps_nearest1, on="forecast_kst_dtm", how="left")
        weather = weather.merge(gfs_nearest1, on="forecast_kst_dtm", how="left")

        # 최근접 격자 원본 U/V에도 풍속/풍향 파생피처 적용
        weather = add_wind_features(
            weather, "ldaps_nearest1_heightAboveGround_10_10u", "ldaps_nearest1_heightAboveGround_10_10v", "ldaps_nearest1_ws10"
        )
        weather = add_wind_features(
            weather, "gfs_nearest1_heightAboveGround_100_100u", "gfs_nearest1_heightAboveGround_100_100v", "gfs_nearest1_ws100"
        )

        # v17: 공기밀도 (온도+기압, 이상기체 법칙) - 같은 풍속이라도 밀도 높은 겨울 공기가
        # 더 많은 운동에너지를 전달함. 지금까지 풍속 위주 피처에는 없던 정보.
        weather = add_air_density_feature(
            weather, "ldaps_heightAboveGround_2_t_mean", "ldaps_surface_0_sp_mean", "ldaps"
        )
        weather = add_air_density_feature(
            weather, "gfs_heightAboveGround_2_2t_mean", "gfs_surface_0_sp_mean", "gfs"
        )

        # v17: 윈드시어 (고도별 풍속 차이) - 대기 안정도/난류 강도의 대리지표
        weather = add_wind_shear_feature(weather, "gfs_ws10_speed", "gfs_ws100_speed", "gfs")

        # v17: 인접 시간대(전/후 1시간) 풍속 - 지금까지 매 시간을 독립적으로 취급했는데,
        # 처음으로 시간적 흐름(급변/돌풍 여부) 정보를 제공
        weather = add_lag_lead_features(weather, ["gfs_ws100_speed", "ldaps_ws10_speed"])

        # v18: 격자 간 분산(불확실성) 피처. 지금까지 최근접 격자들을 평균으로만 썼는데,
        # 격자 간 편차가 크다는 건 그 시간대 예보 자체가 불확실하거나 국지적으로
        # 변동성이 큰 상황이라는 신호. 완전히 새로운 종류의 정보.
        ldaps_disp = aggregate_weather_dispersion(
            ldaps, ldaps_grids, ["heightAboveGround_10_10u", "heightAboveGround_10_10v"], "ldaps"
        )
        gfs_disp = aggregate_weather_dispersion(
            gfs, gfs_grids, ["heightAboveGround_10_10u", "heightAboveGround_10_10v",
                              "heightAboveGround_100_100u", "heightAboveGround_100_100v"], "gfs"
        )
        weather = weather.merge(ldaps_disp, on="forecast_kst_dtm", how="left")
        weather = weather.merge(gfs_disp, on="forecast_kst_dtm", how="left")

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

        # v14: 파워커브(물리 기반) 피처.
        # SCADA 실측 풍속을 직접 쓰면 예보 풍속과 스케일이 안 맞는 문제가 있었음(v9에서 확인).
        # 대신 '예보 풍속(gfs_ws100_speed) -> 실제 그룹 발전량'의 경험적 관계를 학습구간에서
        # 직접 적합(isotonic, 단조증가)해서, "이 예보 풍속이면 과거엔 보통 이 정도 발전했다"는
        # 강력한 단일 피처를 추가. LightGBM이 이걸 기준점 삼아 나머지 오차만 보정하면 되므로
        # 학습이 쉬워질 것으로 기대. SCADA를 직접 쓰지 않아 스케일 불일치 문제를 우회함.
        # (fit_power_curve/apply_power_curve는 features.py 공용 함수 - backtest.py도 동일하게 사용)
        power_curve_col = "gfs_ws100_speed"
        power_curve, fallback_ws = fit_power_curve(X_all.loc[is_train, power_curve_col], y_all[is_train])
        X_all["power_curve_estimate"] = apply_power_curve(
            power_curve, fallback_ws, X_all[power_curve_col], CAPACITY_KWH[target]
        )

        X_tr, y_tr = X_all[is_train], y_all[is_train]
        X_cal, y_cal = X_all[is_calib], y_all[is_calib]
        X_ho, y_ho = X_all[is_holdout], y_all[is_holdout]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns)
        X_cal_imp = pd.DataFrame(imputer.transform(X_cal), columns=X_cal.columns)
        X_ho_imp = pd.DataFrame(imputer.transform(X_ho), columns=X_ho.columns)

        # v11: 두 가지 확정적 변경 (선택/탐색이 아니라 원칙에 기반한 결정이라 calib 과적합 위험 없음)
        #  1) objective를 기본값(L2, 제곱오차) 대신 'mae'(L1)로 고정.
        #     대회 평가지표(NMAE, FICR)가 전부 절대오차 기반이라 학습 목적함수를 일치시킴.
        #  2) 시드 여러 개 앙상블. '고르기'가 아니라 '항상 평균'이라 과적합 위험이 없고
        #     분산을 줄여줌 (v8~v10에서 반복된 calib 과적합 패턴과는 성격이 다름).
        # v15: LightGBM 단독 앙상블에 XGBoost도 섞은 '모델 블렌딩'으로 확장 (modeling.py 참고).
        #      train_baseline.py와 backtest.py가 반드시 같은 함수를 쓰도록 공용화함
        #      (파워커브 피처 때 두 스크립트 로직이 어긋났던 사고 재발 방지).
        trained = train_blended_ensemble(X_tr_imp, y_tr, X_cal_imp, y_cal)
        models = trained  # 아래 최종 재학습 단계에서 재사용
        print(f"[{target}] 앙상블 구성: " + ", ".join(f"{k}(seed={s},iter={bi})" for k, s, _, bi in trained))

        # calib 구간에서 '보정식' 학습.
        # v4에서 isotonic이 calib 데이터 부족으로 과적합해 오히려 성능이 나빠진 것을 확인했음.
        # -> calib을 다시 앞/뒤로 나눠(calib_fit/calib_val), 선형과 isotonic 둘 다 calib_fit으로
        #    학습한 뒤 calib_val(둘 다 안 본 데이터)에서 어느 쪽이 더 나은지 비교해서 자동 선택.
        #    선택된 방식은 calib 전체(calib_fit+calib_val)로 다시 학습해서 최종 사용.
        pred_cal = ensemble_predict(models, X_cal_imp)
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
        pred_ho_raw = np.clip(ensemble_predict(models, X_ho_imp), 0, CAPACITY_KWH[target])
        pred_ho_calibrated = np.clip(calibrator.predict(pred_ho_raw), 0, CAPACITY_KWH[target])

        predictions_holdout_raw[target] = pred_ho_raw
        predictions_holdout_calibrated[target] = pred_ho_calibrated
        holdout_actual[target] = y_ho.values

        # 최종 제출용 모델: holdout을 제외한 전체(train+calib)로 재학습 후 같은 보정식 적용
        # 이 단계는 별도 검증셋이 없으므로 조기종료 대신 앞서 찾은 best_iteration을 그대로 사용
        X_fit = pd.concat([X_tr, X_cal])
        y_fit = pd.concat([y_tr, y_cal])
        X_fit_imp = pd.DataFrame(imputer.fit_transform(X_fit), columns=X_fit.columns)

        final_models = refit_final_models(trained, X_fit_imp, y_fit)

        X_test = build_features(sample_submission[["forecast_id", "forecast_kst_dtm"]], test_weather[target], "forecast_kst_dtm")
        X_test["power_curve_estimate"] = apply_power_curve(
            power_curve, fallback_ws, X_test[power_curve_col], CAPACITY_KWH[target]
        )
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        pred_test_raw = np.clip(final_predict(final_models, X_test_imp), 0, CAPACITY_KWH[target])
        pred_test_calibrated = np.clip(calibrator.predict(pred_test_raw), 0, CAPACITY_KWH[target])
        # v13에서 raw로 바꿔 실제 리더보드 테스트한 결과, 보정 버전(v11, 0.6146)이
        # raw(v13, 0.6073)보다 근소하게 나음. 백테스트는 반대였으나 차이(0.007)가 노이즈
        # 수준이라 보정 여부 자체는 큰 지렛대가 아닌 것으로 결론. v11 방식(보정)으로 복귀.
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

    out_path = SUBMISSION_DIR / "baseline_v18_submit.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
