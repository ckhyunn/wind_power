"""
'바람은 정상인데 발전량이 0에 가까운' 착빙(icing) 의심 패턴이 학습 기간 전체에서
얼마나 자주 나타나는지, 그리고 기온과 실제로 연관되는지 확인하는 스크립트.

정의: 예보 풍속이 컷인(3m/s) 이상인데(=발전 가능한 바람 조건인데) 실제 발전량이
설비용량의 5% 미만인 시간대를 '착빙 의심 시간대'로 정의.
이런 시간대가 기온이 낮을 때(특히 영하) 집중되는지 확인 -> 확인되면 새 피처 후보.

실행:
    python src/diagnose_icing_pattern.py
"""

import pandas as pd
import numpy as np

from features import TARGET_COLS, CAPACITY_KWH, load_turbine_table, compute_group_coords
from train_baseline import build_group_weather, build_features, DATA_DIR, TRAIN_DIR


def main():
    labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    ldaps = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")
    turbine_df = load_turbine_table(DATA_DIR / "info.xlsx")
    group_coords = compute_group_coords(turbine_df)
    weather = build_group_weather(ldaps, gfs, group_coords)

    for target in TARGET_COLS:
        cap = CAPACITY_KWH[target]
        w = weather[target].rename(columns={"forecast_kst_dtm": "kst_dtm"})
        merged = labels[["kst_dtm", target]].merge(w, on="kst_dtm", how="inner").dropna(subset=[target])

        wind_col = "gfs_ws100_speed"
        temp_col = "gfs_heightAboveGround_2_2t_mean"  # 켈빈 온도

        merged["cf"] = merged[target] / cap
        merged["temp_c"] = merged[temp_col] - 273.15  # 섭씨로 변환

        # 착빙 의심: 발전 가능한 바람(3m/s 이상)인데 발전량은 거의 0
        suspect = merged[(merged[wind_col] >= 3.0) & (merged["cf"] < 0.05)]
        normal = merged[(merged[wind_col] >= 3.0) & (merged["cf"] >= 0.05)]

        print(f"\n{'='*70}\n[{target}]\n{'='*70}")
        print(f"전체 '발전가능 바람(>=3m/s)' 시간대: {len(merged[merged[wind_col]>=3.0])}시간")
        print(f"  -> 그중 착빙의심(발전량<5%): {len(suspect)}시간 ({len(suspect)/max(len(merged[merged[wind_col]>=3.0]),1)*100:.2f}%)")

        if len(suspect) > 0 and len(normal) > 0:
            print(f"\n착빙의심 시간대 기온: 평균={suspect['temp_c'].mean():.1f}°C  중앙값={suspect['temp_c'].median():.1f}°C  "
                  f"영하비율={ (suspect['temp_c']<0).mean()*100:.1f}%")
            print(f"정상 발전 시간대 기온: 평균={normal['temp_c'].mean():.1f}°C  중앙값={normal['temp_c'].median():.1f}°C  "
                  f"영하비율={ (normal['temp_c']<0).mean()*100:.1f}%")

            # 월별 착빙의심 비율
            merged["month"] = merged["kst_dtm"].dt.month
            suspect_by_month = merged[(merged[wind_col]>=3.0)].groupby("month").apply(
                lambda g: (g["cf"] < 0.05).mean() * 100
            )
            print(f"\n월별 착빙의심 비율(%):")
            print(suspect_by_month.round(1).to_string())


if __name__ == "__main__":
    main()
