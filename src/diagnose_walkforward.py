"""
'진짜 walk-forward' vs '지금 backtest.py 방식(blocked-CV)' 비교 스크립트.

배경: 풍력발전량 예측 플레이북(다른 대회 경험 정리 문서)에서 지적된 원칙 -
"시계열 검증은 항상 시간 순서를 지켜야 하며, 검증 구간보다 미래의 데이터를
학습에 쓰면 실제보다 성능이 좋게 보이는 착시가 생긴다."

이걸 우리 backtest.py에 대조해보니, 실제로 이 문제가 있었음:
  is_train = ~is_holdout & ~is_calib
이 마스크는 평가 윈도우와 calib 기간만 빼고 '나머지 전부'를 학습에 씀 - 즉 평가
윈도우보다 미래의 데이터(예: 2022년 4~6월을 평가하면서 2023~2024년 데이터로 학습)도
포함됨. 실제 대회에서는 있을 수 없는 상황(2025년을 예측하는데 미래 데이터가 있을 리 없음).

이 스크립트는 같은 윈도우에 대해:
  - 지금 방식(blocked-CV, 미래 데이터 포함)
  - 진짜 walk-forward(그 시점 이전 데이터만 사용)
두 가지로 각각 학습/평가해서 점수가 얼마나 차이 나는지 직접 비교한다.

주의: 초기 윈도우(2022년 4~6월)는 그 이전 데이터가 거의 없어(3개월 미만) 진짜
walk-forward로는 평가 자체가 불가능해 제외함. 최소 9개월 이상의 과거 데이터가
있는 윈도우만 비교 대상으로 삼음.

실행:
    python src/diagnose_walkforward.py
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from features import (
    TARGET_COLS, CAPACITY_KWH, load_turbine_table, compute_group_coords,
    fit_power_curve, apply_power_curve,
)
from train_baseline import (
    build_group_weather, build_features, _Reshape1DWrapper,
    DATA_DIR, TRAIN_DIR, N_NEAREST_GRIDS, USE_LOG_TARGET,
)
from modeling import train_blended_ensemble, ensemble_predict
from evaluate import metric, find_best_ficr_adjustment

CALIB_DAYS = 45
LGBM_SEEDS = [42, 123, 2024]
XGB_SEEDS = [7, 77]
FICR_SHRINKAGE = 0.5  # 지금 채택된 값(v25/26)과 동일하게 맞춰서 비교

# 비교 대상 윈도우 (2022년 4~6월은 이전 데이터가 너무 적어 walk-forward 불가능이라 제외)
COMPARISON_WINDOWS = [
    ("2022-10-01", "2022-12-30"),   # 이전 데이터 약 9개월
    ("2023-04-01", "2023-06-30"),   # 이전 데이터 약 15개월
    ("2023-10-04", "2024-01-02"),   # 이전 데이터 약 21개월 (지금까지 실험에서 문제됐던 윈도우)
    ("2024-04-01", "2024-06-30"),   # 이전 데이터 약 27개월
]


def evaluate_one(holdout_start, holdout_end, train_labels, train_weather, walk_forward: bool):
    """walk_forward=True면 그 시점 이전 데이터만 학습에 사용 (진짜 walk-forward).
    walk_forward=False면 지금 backtest.py와 동일한 방식(미래 데이터 포함, blocked-CV)."""
    calib_end = holdout_start - pd.Timedelta(days=1)
    calib_start = calib_end - pd.Timedelta(days=CALIB_DAYS)

    pred_cal_all, pred_fa_all, actual_all = {}, {}, {}

    for target in TARGET_COLS:
        weather = train_weather[target]
        X_all = build_features(train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"}), weather, "forecast_kst_dtm")
        y_all = train_labels[target]
        dt_all = train_labels["kst_dtm"]

        mask_label = y_all.notna()
        X_all, y_all, dt_all = X_all[mask_label], y_all[mask_label], dt_all[mask_label]

        is_holdout = (dt_all >= holdout_start) & (dt_all <= holdout_end)
        is_calib = (dt_all >= calib_start) & (dt_all <= calib_end)
        if walk_forward:
            is_train = dt_all < calib_start  # 그 시점 이전 데이터만 (미래 데이터 절대 사용 안 함)
        else:
            is_train = ~is_holdout & ~is_calib  # 지금 backtest.py 방식 (미래 데이터 포함)

        if is_train.sum() < 100 or is_calib.sum() < 10 or is_holdout.sum() == 0:
            continue

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

        models = train_blended_ensemble(
            X_tr_imp, y_tr, X_cal_imp, y_cal,
            lgbm_seeds=LGBM_SEEDS, xgb_seeds=XGB_SEEDS, log_target=USE_LOG_TARGET,
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

        pc_calibrated = np.clip(calibrator.predict(pc), 0, CAPACITY_KWH[target])
        raw_scale, raw_shift, _ = find_best_ficr_adjustment(y_cal.values, pc_calibrated, CAPACITY_KWH[target])
        ficr_scale = 1.0 + FICR_SHRINKAGE * (raw_scale - 1.0)
        ficr_shift = FICR_SHRINKAGE * raw_shift

        pr_raw = np.clip(ensemble_predict(models, X_ho_imp), 0, CAPACITY_KWH[target])
        pr_cal = np.clip(calibrator.predict(pr_raw), 0, CAPACITY_KWH[target])
        pr_fa = np.clip(pr_cal * ficr_scale + ficr_shift, 0, CAPACITY_KWH[target])

        pred_cal_all[target] = pr_cal
        pred_fa_all[target] = pr_fa
        actual_all[target] = y_ho.values

    if not actual_all:
        return None

    def to_df(d):
        return pd.concat([pd.Series(v, name=k) for k, v in d.items()], axis=1)

    actual_df = to_df(actual_all)
    cal_df = to_df(pred_cal_all)
    fa_df = to_df(pred_fa_all)
    groups = list(actual_all.keys())
    cap_subset = {k: CAPACITY_KWH[k] for k in groups}

    score_cal, _, _ = metric(actual_df, cal_df, target_cols=groups, capacity=cap_subset)
    score_fa, _, _ = metric(actual_df, fa_df, target_cols=groups, capacity=cap_subset)
    n_train_days = None  # 참고용, 아래서 별도 출력
    return dict(groups=groups, score_cal=score_cal, score_fa=score_fa, n_train=len(actual_df))


def main():
    train_labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
    train_labels["kst_dtm"] = pd.to_datetime(train_labels["kst_dtm"])
    ldaps_train = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")

    turbine_df = load_turbine_table(DATA_DIR / "info.xlsx")
    group_coords = compute_group_coords(turbine_df)
    train_weather = build_group_weather(ldaps_train, gfs_train, group_coords)

    rows = []
    for start_str, end_str in COMPARISON_WINDOWS:
        holdout_start = pd.Timestamp(start_str)
        holdout_end = pd.Timestamp(end_str)
        prior_days = (holdout_start - train_labels["kst_dtm"].min()).days
        print(f"\n=== {holdout_start.date()} ~ {holdout_end.date()} (이전 데이터 약 {prior_days}일) ===")

        r_blocked = evaluate_one(holdout_start, holdout_end, train_labels, train_weather, walk_forward=False)
        r_wf = evaluate_one(holdout_start, holdout_end, train_labels, train_weather, walk_forward=True)

        if r_blocked is None or r_wf is None:
            print("  데이터 부족으로 건너뜀")
            continue

        gap_cal = r_blocked["score_cal"] - r_wf["score_cal"]
        gap_fa = r_blocked["score_fa"] - r_wf["score_fa"]
        print(f"  [지금 방식(미래데이터 포함)]  보정후={r_blocked['score_cal']:.4f}  FICR조정={r_blocked['score_fa']:.4f}")
        print(f"  [진짜 walk-forward]         보정후={r_wf['score_cal']:.4f}  FICR조정={r_wf['score_fa']:.4f}")
        print(f"  차이(지금방식 - walk-forward): 보정후={gap_cal:+.4f}  FICR조정={gap_fa:+.4f}")

        rows.append(dict(
            window=f"{holdout_start.date()}~{holdout_end.date()}",
            score_cal_blocked=r_blocked["score_cal"], score_cal_wf=r_wf["score_cal"], gap_cal=gap_cal,
            score_fa_blocked=r_blocked["score_fa"], score_fa_wf=r_wf["score_fa"], gap_fa=gap_fa,
        ))

    if not rows:
        print("\n비교 가능한 윈도우가 없습니다.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("전체 비교 결과")
    print("=" * 90)
    print(df.to_string(index=False))

    print("\n" + "=" * 90)
    print(f"평균 격차 (지금 방식이 walk-forward보다 얼마나 낙관적이었는지)")
    print("=" * 90)
    print(f"보정후 Score 평균 격차:   {df['gap_cal'].mean():+.4f}")
    print(f"FICR조정 Score 평균 격차: {df['gap_fa'].mean():+.4f}")
    print("(양수면 지금까지의 backtest 점수가 실제보다 낙관적이었다는 뜻)")


if __name__ == "__main__":
    main()
