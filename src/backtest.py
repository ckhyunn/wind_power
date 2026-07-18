"""
여러 시기(윈도우)에 걸쳐 반복 검증하는 백테스트 스크립트.

배경: v3~v11 아홉 번의 개선 시도를 전부 '최근 90일' 단일 holdout으로 판단해왔는데,
holdout을 1년 전 다른 시기로 옮기자(v12) 로컬/실제 리더보드 점수가 크게 달라졌음.
즉 단일 holdout 하나만 보고 판단하는 방식 자체가 특정 시기에 우연히 맞는 '가짜 개선'을
걸러내지 못한다는 게 확인됨.

이 스크립트는 서로 다른 계절/연도의 여러 윈도우에서 각각 학습->평가를 반복하고,
평균(과 표준편차)을 내서 훨씬 신뢰할 만한 로컬 지표를 만든다.
각 윈도우는 "그 구간만 빼고 나머지 전체로 학습 -> 그 구간에서 평가" 방식
(blocked time-series cross-validation)이라, 서로 다른 개선 아이디어를 비교할 때
이 스크립트의 평균 Score를 기준으로 판단하면 v8~v12에서 반복된 실수를 줄일 수 있음.

실행:
    python src/backtest.py

참고: 속도를 위해 앙상블 시드 수를 LGBM_SEEDS_BACKTEST/XGB_SEEDS_BACKTEST로 줄여서 사용.
실제 제출용 학습(train_baseline.py)은 시드를 더 많이 써도 됨 - 이 스크립트는
'여러 아이디어를 빠르게 비교'하는 용도.

v15: LightGBM 단독 앙상블에 XGBoost도 섞은 modeling.py의 블렌딩 함수를 그대로 사용하도록
     리팩터링 (train_baseline.py와 반드시 같은 모델 학습 로직을 공유해야 검증이 의미 있음).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from features import TARGET_COLS, CAPACITY_KWH, load_turbine_table, compute_group_coords, fit_power_curve, apply_power_curve
from train_baseline import (
    build_group_weather,
    build_features,
    _Reshape1DWrapper,
    DATA_DIR,
    TRAIN_DIR,
    N_NEAREST_GRIDS,
)
from modeling import train_blended_ensemble, ensemble_predict
from evaluate import metric

CALIB_DAYS = 45
# 백테스트는 속도 위해 시드 수를 줄임 (본 제출용 train_baseline.py는 더 많이 사용).
# modeling.py의 기본 시드보다 적게 쓰고 싶으면 아래처럼 슬라이스해서 넘기면 됨.
LGBM_SEEDS_BACKTEST = [42, 123, 2024]
XGB_SEEDS_BACKTEST = [7, 77]

# 서로 다른 연도/계절에 걸친 검증 윈도우.
# 그룹3은 2023년부터 라벨이 있어 2022년 윈도우에서는 자동으로 제외됨(데이터 없음 -> 스킵).
BACKTEST_WINDOWS = [
    ("2022-04-01", "2022-06-30"),   # 봄 2022 (그룹1,2만)
    ("2022-10-01", "2022-12-30"),   # 가을~겨울 2022 (그룹1,2만)
    ("2023-04-01", "2023-06-30"),   # 봄 2023 (전체 그룹)
    ("2023-10-04", "2024-01-02"),   # 가을~겨울 2023 (v12에서 썼던 것과 동일 - 비교용으로 유지)
    ("2024-04-01", "2024-06-30"),   # 봄 2024 (전체 그룹)
]


def evaluate_window(holdout_start: pd.Timestamp, holdout_end: pd.Timestamp,
                     train_labels: pd.DataFrame, train_weather: dict) -> dict | None:
    calib_end = holdout_start - pd.Timedelta(days=1)
    calib_start = calib_end - pd.Timedelta(days=CALIB_DAYS)

    pred_raw, pred_cal, actual = {}, {}, {}

    for target in TARGET_COLS:
        weather = train_weather[target]
        X_all = build_features(train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"}), weather, "forecast_kst_dtm")
        y_all = train_labels[target]
        dt_all = train_labels["kst_dtm"]

        mask_label = y_all.notna()
        X_all, y_all, dt_all = X_all[mask_label], y_all[mask_label], dt_all[mask_label]

        is_holdout = (dt_all >= holdout_start) & (dt_all <= holdout_end)
        is_calib = (dt_all >= calib_start) & (dt_all <= calib_end)
        is_train = ~is_holdout & ~is_calib

        if is_train.sum() == 0:
            print(f"  [{target}] 학습 구간에 데이터가 없어 건너뜀")
            continue

        # v14와 동일한 파워커브 피처 (train_baseline.py와 반드시 같은 로직 - features.py 공용 함수 사용)
        power_curve_col = "gfs_ws100_speed"
        power_curve, fallback_ws = fit_power_curve(X_all.loc[is_train, power_curve_col], y_all[is_train])
        X_all["power_curve_estimate"] = apply_power_curve(
            power_curve, fallback_ws, X_all[power_curve_col], CAPACITY_KWH[target]
        )

        X_tr, y_tr = X_all[is_train], y_all[is_train]
        X_cal, y_cal = X_all[is_calib], y_all[is_calib]
        X_ho, y_ho = X_all[is_holdout], y_all[is_holdout]

        if len(X_tr) == 0 or len(X_cal) < 10 or len(X_ho) == 0:
            print(f"  [{target}] 데이터 부족으로 건너뜀 (train={len(X_tr)}, calib={len(X_cal)}, holdout={len(X_ho)})")
            continue

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns)
        X_cal_imp = pd.DataFrame(imputer.transform(X_cal), columns=X_cal.columns)
        X_ho_imp = pd.DataFrame(imputer.transform(X_ho), columns=X_ho.columns)

        models = train_blended_ensemble(
            X_tr_imp, y_tr, X_cal_imp, y_cal,
            lgbm_seeds=LGBM_SEEDS_BACKTEST, xgb_seeds=XGB_SEEDS_BACKTEST,
        )

        pc = ensemble_predict(models, X_cal_imp)
        n_cal = len(pc)
        split = n_cal // 2

        if split < 5:
            calibrator = _Reshape1DWrapper(LinearRegression().fit(pc.reshape(-1, 1), y_cal))
        else:
            pc_fit, pc_val = pc[:split], pc[split:]
            y_cal_fit, y_cal_val = y_cal.values[:split], y_cal.values[split:]

            def calib_error(cm):
                pv = np.clip(cm.predict(pc_val), 0, CAPACITY_KWH[target])
                return np.mean(np.abs(pv - y_cal_val)) / CAPACITY_KWH[target]

            lin = _Reshape1DWrapper(LinearRegression().fit(pc_fit.reshape(-1, 1), y_cal_fit))
            lin_err = calib_error(lin)
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(pc_fit, y_cal_fit)
            iso_err = calib_error(iso)

            if iso_err < lin_err * 0.95:
                calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(pc, y_cal)
            else:
                calibrator = _Reshape1DWrapper(LinearRegression().fit(pc.reshape(-1, 1), y_cal))

        pr_raw = np.clip(ensemble_predict(models, X_ho_imp), 0, CAPACITY_KWH[target])
        pr_cal = np.clip(calibrator.predict(pr_raw), 0, CAPACITY_KWH[target])

        pred_raw[target] = pr_raw
        pred_cal[target] = pr_cal
        actual[target] = y_ho.values

    if not actual:
        return None

    def to_df(d):
        return pd.concat([pd.Series(v, name=k) for k, v in d.items()], axis=1)

    actual_df = to_df(actual)
    raw_df = to_df(pred_raw)
    cal_df = to_df(pred_cal)
    included_groups = list(actual.keys())
    cap_subset = {k: CAPACITY_KWH[k] for k in included_groups}

    score_raw, nmae_raw, ficr_raw = metric(actual_df, raw_df, target_cols=included_groups, capacity=cap_subset)
    score_cal, nmae_cal, ficr_cal = metric(actual_df, cal_df, target_cols=included_groups, capacity=cap_subset)

    return dict(
        window=f"{holdout_start.date()}~{holdout_end.date()}",
        groups=",".join(g.replace("kpx_group_", "g") for g in included_groups),
        score_raw=score_raw, nmae_raw=nmae_raw, ficr_raw=ficr_raw,
        score_cal=score_cal, nmae_cal=nmae_cal, ficr_cal=ficr_cal,
    )


def run_backtest() -> pd.DataFrame:
    """다른 스크립트에서 import해서 재사용할 수 있도록 함수로 분리."""
    train_labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
    train_labels["kst_dtm"] = pd.to_datetime(train_labels["kst_dtm"])
    ldaps_train = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")

    turbine_df = load_turbine_table(DATA_DIR / "info.xlsx")
    group_coords = compute_group_coords(turbine_df)
    train_weather = build_group_weather(ldaps_train, gfs_train, group_coords)

    results = []
    for i, (start_str, end_str) in enumerate(BACKTEST_WINDOWS, 1):
        holdout_start = pd.Timestamp(start_str)
        holdout_end = pd.Timestamp(end_str)
        print(f"\n=== 윈도우 {i}/{len(BACKTEST_WINDOWS)}: {holdout_start.date()} ~ {holdout_end.date()} ===")
        r = evaluate_window(holdout_start, holdout_end, train_labels, train_weather)
        if r:
            results.append(r)
            print(f"  포함 그룹: {r['groups']}")
            print(f"  Score(raw)={r['score_raw']:.4f}  Score(calibrated)={r['score_cal']:.4f}")
        else:
            print("  이 윈도우는 평가 가능한 그룹이 없어 건너뜀")

    return pd.DataFrame(results)


def main():
    df = run_backtest()
    if df.empty:
        print("\n평가 가능한 윈도우가 없습니다.")
        return

    print("\n" + "=" * 70)
    print("전체 윈도우 결과")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("평균 (표준편차) - 이 숫자를 개선 여부 판단 기준으로 사용")
    print("=" * 70)
    for col in ["score_raw", "nmae_raw", "ficr_raw", "score_cal", "nmae_cal", "ficr_cal"]:
        print(f"{col:12s}: {df[col].mean():.4f}  (±{df[col].std():.4f})")


if __name__ == "__main__":
    main()
