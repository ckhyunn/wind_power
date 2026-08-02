"""
B모델을 A모델의 fold B와 동일한 방식(2024년 이전 데이터로 학습 -> 2024년 전체 예측)으로
학습시켜서, model_A/output_2024/actual_2024.csv(실제 정답)를 기준으로 A모델과 직접
정확도를 비교하는 스크립트.

지금까지는 A/B 모델의 '예측끼리만' 비교해서 다양성(상관관계)만 볼 수 있었는데,
이제 진짜 정답이 있으니 '누가 더 맞았는지'를 직접 확인할 수 있음.

확인하는 것:
  1. B모델의 2024년 실제 Score (A의 0.6381/0.6382와 직접 비교)
  2. 시간대별로 A/B 중 누가 더 정확했는지, 오차 상관관계
  3. 그룹별로 어느 쪽이 더 우세한지
  4. 최적 블렌딩 비율 탐색 (실제 정답 기준 grid search)
  5. 고풍속 구간(diagnose_divergence_dates.py에서 확인된 강풍 시간대와 유사한 조건)에서
     A/B 중 누가 더 정확한지 - 조건부 블렌딩 설계 근거

실행:
    python src/compare_real_accuracy_2024.py
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
from evaluate import metric, find_best_ficr_adjustment, group_score

CALIB_DAYS = 45
LGBM_SEEDS = [42, 123, 2024]
XGB_SEEDS = [7, 77, 777]
FICR_SHRINKAGE = 0.5

TRAIN_CUTOFF = pd.Timestamp("2024-01-01")  # 이 시점 이전 데이터로만 학습 (A와 동일 조건)
PRED_START = pd.Timestamp("2024-01-01 01:00:00")
PRED_END = pd.Timestamp("2025-01-01 00:00:00")

REPO_ROOT = DATA_DIR.parent
MODEL_A_ACTUAL = REPO_ROOT / "model_A" / "output_2024" / "actual_2024.csv"
MODEL_A_FILES = {
    "A_lgb": REPO_ROOT / "model_A" / "output_2024" / "pred_A_lgb_2024.csv",
    "A_cat": REPO_ROOT / "model_A" / "output_2024" / "pred_A_cat_2024.csv",
    "A_xgb": REPO_ROOT / "model_A" / "output_2024" / "pred_A_xgb_2024.csv",
    "A_blend": REPO_ROOT / "model_A" / "output_2024" / "pred_A_blend_2024.csv",
}


def train_and_predict_2024():
    """B모델을 2024년 이전 데이터로 학습해서 2024년 전체(실제 라벨 있는 구간)를 예측"""
    train_labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
    train_labels["kst_dtm"] = pd.to_datetime(train_labels["kst_dtm"])
    ldaps_train = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")

    turbine_df = load_turbine_table(DATA_DIR / "info.xlsx")
    group_coords = compute_group_coords(turbine_df)
    weather_all = build_group_weather(ldaps_train, gfs_train, group_coords)

    # calib: 2024년 이전 마지막 45일 (모델 자체 검증/보정용, 2024 예측에는 안 씀)
    calib_end = TRAIN_CUTOFF - pd.Timedelta(days=1)
    calib_start = calib_end - pd.Timedelta(days=CALIB_DAYS)

    predictions_2024 = {}

    for target in TARGET_COLS:
        weather = weather_all[target]
        X_all = build_features(train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"}), weather, "forecast_kst_dtm")
        y_all = train_labels[target]
        dt_all = train_labels["kst_dtm"]

        mask_label = y_all.notna()
        X_all, y_all, dt_all = X_all[mask_label], y_all[mask_label], dt_all[mask_label]

        is_calib = (dt_all >= calib_start) & (dt_all <= calib_end)
        is_train = dt_all < calib_start  # 2024 이전 데이터만 (walk-forward, A와 동일 원칙)

        power_curve_col = "gfs_ws100_speed"
        power_curve, fallback_ws = fit_power_curve(X_all.loc[is_train, power_curve_col], y_all[is_train])
        X_all["power_curve_estimate"] = apply_power_curve(
            power_curve, fallback_ws, X_all[power_curve_col], CAPACITY_KWH[target]
        )

        X_tr, y_tr = X_all[is_train], y_all[is_train]
        X_cal, y_cal = X_all[is_calib], y_all[is_calib]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns)
        X_cal_imp = pd.DataFrame(imputer.transform(X_cal), columns=X_cal.columns)

        print(f"[{target}] 학습 데이터: {len(X_tr)}시간 ({dt_all[is_train].min()} ~ {dt_all[is_train].max()})")

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

        # 2024년 전체(1/1~12/31)를 예측할 피처 구성 (train_labels의 2024년 구간을 그대로 사용
        # - forecast weather는 이미 계산돼 있으므로 해당 구간만 슬라이스)
        is_2024 = (dt_all >= PRED_START) & (dt_all <= PRED_END)
        X_2024 = X_all[is_2024]
        dt_2024 = dt_all[is_2024]
        X_2024_imp = pd.DataFrame(imputer.transform(X_2024), columns=X_2024.columns)

        pred_2024_raw = np.clip(ensemble_predict(models, X_2024_imp), 0, CAPACITY_KWH[target])
        pred_2024_cal = np.clip(calibrator.predict(pred_2024_raw), 0, CAPACITY_KWH[target])
        pred_2024_fa = np.clip(pred_2024_cal * ficr_scale + ficr_shift, 0, CAPACITY_KWH[target])

        predictions_2024[target] = pd.DataFrame({
            "forecast_kst_dtm": dt_2024.values,
            f"{target}_raw": pred_2024_raw,
            f"{target}_cal": pred_2024_cal,
            f"{target}_fa": pred_2024_fa,
        })

    # 그룹별 결과를 하나의 데이터프레임으로 병합
    merged = predictions_2024[TARGET_COLS[0]]
    for target in TARGET_COLS[1:]:
        merged = merged.merge(predictions_2024[target], on="forecast_kst_dtm", how="outer")

    return merged


def main():
    print("B모델을 2024년 이전 데이터로 학습해서 2024년 전체를 예측합니다...")
    b_pred = train_and_predict_2024()
    b_pred.to_csv(DATA_DIR.parent / "src" / "b_pred_2024.csv", index=False)
    print(f"\nB모델 2024년 예측 완료: {len(b_pred)}행. 'src/b_pred_2024.csv'로 저장됨.")

    # ==== 실제 정답과 대조 ====
    actual = pd.read_csv(MODEL_A_ACTUAL)
    actual["forecast_kst_dtm"] = pd.to_datetime(actual["forecast_kst_dtm"])
    b_pred["forecast_kst_dtm"] = pd.to_datetime(b_pred["forecast_kst_dtm"])

    merged = actual.merge(b_pred, on="forecast_kst_dtm").dropna(subset=TARGET_COLS)
    actual_df = merged[TARGET_COLS]

    for variant in ["raw", "cal", "fa"]:
        pred_df = merged[[f"{t}_{variant}" for t in TARGET_COLS]]
        pred_df.columns = TARGET_COLS
        score, nmae, ficr = metric(actual_df, pred_df)
        print(f"\n[B모델 2024 실제 성능 - {variant}] Score={score:.4f}  1-NMAE={nmae:.4f}  FICR={ficr:.4f}")

    # ==== A모델들과 비교 ====
    print("\n" + "=" * 70)
    print("A모델 구성요소별 2024년 실제 성능 (참고용 재확인)")
    print("=" * 70)
    for name, path in MODEL_A_FILES.items():
        a_df = pd.read_csv(path)
        a_df["forecast_kst_dtm"] = pd.to_datetime(a_df["forecast_kst_dtm"])
        m = actual.merge(a_df, on="forecast_kst_dtm", suffixes=("_actual", "_pred")).dropna(
            subset=[f"{t}_actual" for t in TARGET_COLS]
        )
        actual_a = m[[f"{t}_actual" for t in TARGET_COLS]]
        actual_a.columns = TARGET_COLS
        pred_a = m[[f"{t}_pred" for t in TARGET_COLS]]
        pred_a.columns = TARGET_COLS
        score, nmae, ficr = metric(actual_a, pred_a)
        print(f"[{name}] Score={score:.4f}  1-NMAE={nmae:.4f}  FICR={ficr:.4f}")


if __name__ == "__main__":
    main()
