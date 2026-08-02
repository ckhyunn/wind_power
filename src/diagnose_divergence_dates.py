"""
A모델-B모델 예측이 크게 갈리는 특정 날짜(2025-03-02~05, 2025-02-08 등)에
실제로 어떤 기상 조건이었는지 우리 GFS/LDAPS 예보 원본으로 직접 확인하는 스크립트.

뉴스 검색으로는 특정되지 않는 조용한(비headline) 강풍 이벤트도, 우리가 가진
실제 예보 데이터에는 그대로 찍혀 있으므로 이게 가장 확실한 확인 방법.

실행:
    python src/diagnose_divergence_dates.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

from features import load_turbine_table, compute_group_coords
from train_baseline import build_group_weather, DATA_DIR, TRAIN_DIR

TEST_DIR = DATA_DIR / "test"

# compare_with_model_A.py의 top30 결과에서 확인된 쏠림 구간
CHECK_DATES = [
    ("2025-03-01", "2025-03-06"),   # 그룹1,3 공통 최대 쏠림 구간
    ("2025-02-07", "2025-02-09"),   # 그룹1,3 2차 쏠림
    ("2025-03-14", "2025-03-19"),   # 그룹2 소규모 쏠림
]


def main():
    ldaps_test = pd.read_csv(TEST_DIR / "ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv(TEST_DIR / "gfs_test.csv", encoding="utf-8-sig")

    turbine_df = load_turbine_table(DATA_DIR / "info.xlsx")
    group_coords = compute_group_coords(turbine_df)
    test_weather = build_group_weather(ldaps_test, gfs_test, group_coords)

    for group, weather in test_weather.items():
        print(f"\n{'='*80}\n[{group}] 예보 풍속/윈드시어/850hPa 전체 기간 통계 (비교 기준)\n{'='*80}")
        cols = ["gfs_ws100_speed", "gfs_ws850_speed", "gfs_100_850_wind_shear", "gfs_wind_shear"]
        cols = [c for c in cols if c in weather.columns]
        print(weather[cols].describe().round(2).to_string())

        for start, end in CHECK_DATES:
            mask = (weather["forecast_kst_dtm"] >= start) & (weather["forecast_kst_dtm"] <= end)
            sub = weather.loc[mask, cols]
            if len(sub) == 0:
                continue
            print(f"\n  -- {start} ~ {end} 구간 ({len(sub)}시간) --")
            print(f"  100m 풍속: 평균={sub['gfs_ws100_speed'].mean():.2f}  최대={sub['gfs_ws100_speed'].max():.2f} m/s")
            if "gfs_ws850_speed" in sub:
                print(f"  850hPa 풍속: 평균={sub['gfs_ws850_speed'].mean():.2f}  최대={sub['gfs_ws850_speed'].max():.2f} m/s")
            if "gfs_100_850_wind_shear" in sub:
                print(f"  100m-850hPa 시어: 평균={sub['gfs_100_850_wind_shear'].mean():.2f}  최대={sub['gfs_100_850_wind_shear'].max():.2f} m/s")

            # 전체 기간 대비 상위 몇 %에 해당하는지
            pct = (weather["gfs_ws100_speed"] < sub["gfs_ws100_speed"].mean()).mean() * 100
            print(f"  -> 이 구간 평균 100m풍속은 전체 기간의 상위 {100-pct:.1f}% 수준")


if __name__ == "__main__":
    main()
