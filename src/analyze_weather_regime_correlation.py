"""
[담당: 기상 레짐 상관분석]
analyze_A_residual_forensics.py에서 확인된 단서를 겨냥해서, 실제 기상 조건과
A모델 오차의 관계를 확인한다.

확인 순서:
  1. 2024-04-15, 2024-03-29 (그룹1·3 공통 대형오차일)의 실제 기상 조건
  2. 7월 vs 9월 기상 레짐 차이 (7월은 오차 크고 9월은 작은 이유)
  3. 오차율을 풍속/기압/풍향 구간별로 나눠서 어디에 몰리는지 (다른 팀의
     저기압·폭풍방향 방식 재현)
  4. 그룹3 과대예측 편향과 특정 조건의 연관성

실행:
    python src/analyze_weather_regime_correlation.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

from features import TARGET_COLS, CAPACITY_KWH, load_turbine_table, compute_group_coords
from train_baseline import build_group_weather, DATA_DIR, TRAIN_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTUAL_PATH = REPO_ROOT / "model_A" / "output_2024" / "actual_2024.csv"
PRED_PATH = REPO_ROOT / "model_A" / "output_2024" / "pred_A_blend_2024.csv"

MIN_OUTPUT_RATIO = 0.10
FLAGGED_DATES = ["2024-04-15", "2024-03-29"]


def load_merged_with_weather():
    actual = pd.read_csv(ACTUAL_PATH)
    pred = pd.read_csv(PRED_PATH)
    actual["forecast_kst_dtm"] = pd.to_datetime(actual["forecast_kst_dtm"])
    pred["forecast_kst_dtm"] = pd.to_datetime(pred["forecast_kst_dtm"])
    labels_pred = actual.merge(pred, on="forecast_kst_dtm", suffixes=("_actual", "_pred"))

    ldaps = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")
    turbine_df = load_turbine_table(DATA_DIR / "info.xlsx")
    group_coords = compute_group_coords(turbine_df)
    weather = build_group_weather(ldaps, gfs, group_coords)

    out = {}
    for target in TARGET_COLS:
        w = weather[target].rename(columns={"forecast_kst_dtm": "dt"})
        sub = labels_pred[["forecast_kst_dtm", f"{target}_actual", f"{target}_pred"]].copy()
        sub.columns = ["dt", "actual", "pred"]
        merged = sub.merge(w, on="dt", how="inner").dropna(subset=["actual"])
        out[target] = merged
    return out


def main():
    data = load_merged_with_weather()

    # ==== 1) 플래그된 날짜의 실제 기상 조건 ====
    print("=" * 80)
    print("1) 공통 대형오차일(2024-04-15, 2024-03-29) 실제 기상 조건")
    print("=" * 80)
    for target in ["kpx_group_1", "kpx_group_3"]:
        m = data[target]
        cap = CAPACITY_KWH[target]
        for date_str in FLAGGED_DATES:
            day = m[m["dt"].dt.date == pd.Timestamp(date_str).date()]
            if len(day) == 0:
                continue
            err = ((day["pred"] - day["actual"]).abs() / cap).mean() * 100
            print(f"\n[{target}] {date_str} (n={len(day)})")
            print(f"  평균 오차율: {err:.1f}%")
            print(f"  100m풍속: 평균={day['gfs_ws100_speed'].mean():.2f} 최대={day['gfs_ws100_speed'].max():.2f} m/s")
            if "gfs_ws850_speed" in day.columns:
                print(f"  850hPa풍속: 평균={day['gfs_ws850_speed'].mean():.2f} 최대={day['gfs_ws850_speed'].max():.2f} m/s")
            if "gfs_surface_0_sp_mean" in day.columns:
                print(f"  지표기압: 평균={day['gfs_surface_0_sp_mean'].mean()/100:.1f} hPa")
            print(f"  실제발전: 평균={day['actual'].mean():.0f}  예측: 평균={day['pred'].mean():.0f} "
                  f"(부호: {'과대' if day['pred'].mean()>day['actual'].mean() else '과소'}예측)")

    # ==== 2) 7월 vs 9월 기상 레짐 차이 ====
    print("\n" + "=" * 80)
    print("2) 7월 vs 9월 기상 레짐 비교")
    print("=" * 80)
    for target in TARGET_COLS:
        m = data[target].copy()
        m["month"] = m["dt"].dt.month
        jul = m[m["month"] == 7]
        sep = m[m["month"] == 9]
        print(f"\n[{target}]")
        for name, sub in [("7월", jul), ("9월", sep)]:
            print(f"  {name}: 100m풍속 평균={sub['gfs_ws100_speed'].mean():.2f}  "
                  f"850hPa풍속 평균={sub.get('gfs_ws850_speed', pd.Series([np.nan])).mean():.2f}  "
                  f"공기밀도 평균={sub.get('gfs_air_density', pd.Series([np.nan])).mean():.4f}")

    # ==== 3) 풍속/기압 구간별 오차율 ====
    print("\n" + "=" * 80)
    print("3) 풍속 구간별 오차율 (다른 팀 방식 재현)")
    print("=" * 80)
    for target in TARGET_COLS:
        m = data[target].copy()
        cap = CAPACITY_KWH[target]
        valid = m[m["actual"] >= cap * MIN_OUTPUT_RATIO].copy()
        valid["error_rate"] = (valid["pred"] - valid["actual"]).abs() / cap

        bins = [0, 3, 5, 7, 9, 11, 13, 15, 20, 40]
        valid["ws_bin"] = pd.cut(valid["gfs_ws100_speed"], bins=bins)
        result = valid.groupby("ws_bin", observed=True)["error_rate"].agg(["mean", "count"])
        result["mean"] = (result["mean"] * 100).round(1)
        print(f"\n[{target}] 풍속구간별 평균오차율(%)")
        print(result.to_string())

    # ==== 4) 기압 구간별 오차율 ====
    print("\n" + "=" * 80)
    print("4) 기압 구간별 오차율")
    print("=" * 80)
    for target in TARGET_COLS:
        m = data[target].copy()
        cap = CAPACITY_KWH[target]
        valid = m[m["actual"] >= cap * MIN_OUTPUT_RATIO].copy()
        if "gfs_surface_0_sp_mean" not in valid.columns:
            continue
        valid["error_rate"] = (valid["pred"] - valid["actual"]).abs() / cap
        valid["signed_error_rate"] = (valid["pred"] - valid["actual"]) / cap
        valid["pressure_hpa"] = valid["gfs_surface_0_sp_mean"] / 100

        pbins = pd.qcut(valid["pressure_hpa"], q=6, duplicates="drop")
        result = valid.groupby(pbins, observed=True).agg(
            평균오차율=("error_rate", lambda s: s.mean() * 100),
            평균부호오차=("signed_error_rate", lambda s: s.mean() * 100),
            시간수=("error_rate", "size"),
        )
        print(f"\n[{target}] 기압 6분위별 오차율(%) - 부호오차 양수=과대예측")
        print(result.round(2).to_string())


if __name__ == "__main__":
    main()
