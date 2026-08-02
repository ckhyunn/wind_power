"""
diagnose_icing_pattern.py의 수정판.

문제: 원래 스크립트는 '풍속 3m/s 이상'을 '발전 가능'으로 정의했는데, 파워커브상
3~4m/s 구간은 원래 평균 발전량이 6kW(거의 0)라 정상적인 저출력이 대량으로
'착빙의심'에 섞여 들어감. 이게 계절 무관 6~8% 기저율, 기온 무상관의 원인으로 추정.

수정: 문턱을 8m/s(파워커브상 정격의 ~9%인 구간)로 올려서, '정상이라면 확실히
발전하고 있어야 할 풍속대'에서의 진짜 이상 저출력만 걸러냄.

실행:
    python src/diagnose_icing_pattern_v2.py
"""

import pandas as pd
import numpy as np

from features import TARGET_COLS, CAPACITY_KWH, load_turbine_table, compute_group_coords
from train_baseline import build_group_weather, DATA_DIR, TRAIN_DIR

WIND_THRESHOLD = 8.0   # m/s - 이 이상이면 파워커브상 확실히 발전하고 있어야 함
OUTPUT_THRESHOLD = 0.05  # 설비용량의 5% 미만이면 '이상 저출력'


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
        merged["temp_c"] = merged["gfs_heightAboveGround_2_2t_mean"] - 273.15
        merged["cf"] = merged[target] / cap

        strong_wind = merged[merged["gfs_ws100_speed"] >= WIND_THRESHOLD].copy()
        suspect = strong_wind[strong_wind["cf"] < OUTPUT_THRESHOLD]
        normal = strong_wind[strong_wind["cf"] >= OUTPUT_THRESHOLD]

        print(f"\n{'='*70}\n[{target}] 풍속 {WIND_THRESHOLD}m/s 이상 시간대 전수 확인\n{'='*70}")
        print(f"해당 시간대 총 {len(strong_wind)}시간")
        print(f"  -> 이상저출력(발전량<{OUTPUT_THRESHOLD*100:.0f}%): {len(suspect)}시간 ({len(suspect)/max(len(strong_wind),1)*100:.2f}%)")

        if len(suspect) > 0 and len(normal) > 0:
            print(f"\n이상저출력 시간대 기온: 평균={suspect['temp_c'].mean():.1f}°C  영하비율={(suspect['temp_c']<0).mean()*100:.1f}%")
            print(f"정상 발전 시간대 기온: 평균={normal['temp_c'].mean():.1f}°C  영하비율={(normal['temp_c']<0).mean()*100:.1f}%")

            merged["month"] = merged["kst_dtm"].dt.month
            strong_wind["month"] = strong_wind["kst_dtm"].dt.month
            by_month = strong_wind.groupby("month").apply(lambda g: (g["cf"] < OUTPUT_THRESHOLD).mean() * 100)
            print(f"\n월별 이상저출력 비율(%):")
            print(by_month.round(1).to_string())

            print(f"\n이상저출력 사례 날짜 상위 20개 (기온 순 정렬 - 가장 추운 사례부터):")
            print(suspect.nsmallest(20, "temp_c")[["kst_dtm", "gfs_ws100_speed", "temp_c", "cf"]].to_string(index=False))
        else:
            print("이상저출력 사례가 없거나 정상 사례가 없어 비교 불가")


if __name__ == "__main__":
    main()
