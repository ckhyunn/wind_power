"""
발견된 초장기 무발전 구간(2024-02-19~29, 2024-08-10~21, 2024-01-18~24)을 정밀 진단.

확인 순서:
  1. 이 구간 동안의 실제 라벨값 시계열을 그대로 나열 (정확히 0인지, 아주 미세하게라도
     변하는지 - 완전히 똑같은 값이 반복되면 센서/통신 결측을 0으로 채운 데이터 문제일
     가능성, 미세하게라도 변하면 진짜 발전량이 낮은 물리적 현상일 가능성)
  2. 같은 구간 예보 풍속 시계열 (풍속 자체가 낮았는지, 아니면 바람은 있는데 발전 안 됐는지)
  3. 같은 구간 기온 시계열
  4. 다른 그룹은 같은 시점에 정상 발전 중이었는지 (한 그룹만 문제면 그룹 특유 원인,
     전 그룹이면 광역 기상/계통 요인)

실행:
    python src/diagnose_long_downtime.py
"""

import pandas as pd
import numpy as np

from features import TARGET_COLS, CAPACITY_KWH, load_turbine_table, compute_group_coords
from train_baseline import build_group_weather, DATA_DIR, TRAIN_DIR

LONG_WINDOWS = [
    ("2024-01-18", "2024-01-24", "1월 혹한기 무발전"),
    ("2024-02-19", "2024-02-29", "2월 초장기 무발전 (9~10일)"),
    ("2024-08-10", "2024-08-21", "8월 그룹3 장기 무발전 (11일)"),
]


def main():
    labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    ldaps = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")
    turbine_df = load_turbine_table(DATA_DIR / "info.xlsx")
    group_coords = compute_group_coords(turbine_df)
    weather = build_group_weather(ldaps, gfs, group_coords)

    for start, end, name in LONG_WINDOWS:
        print(f"\n{'#'*80}\n[{name}] {start} ~ {end}\n{'#'*80}")

        mask = (labels["kst_dtm"] >= start) & (labels["kst_dtm"] <= end)
        sub = labels.loc[mask, ["kst_dtm"] + TARGET_COLS].copy()

        for target in TARGET_COLS:
            cap = CAPACITY_KWH[target]
            vals = sub[target].dropna()
            if len(vals) == 0:
                print(f"\n  [{target}] 라벨 없음")
                continue

            n_unique = vals.nunique()
            n_exact_zero = (vals == 0).sum()
            print(f"\n  [{target}] {len(vals)}시간 중 고유값 {n_unique}개, 정확히 0인 값 {n_exact_zero}개")
            print(f"    값 범위: {vals.min():.2f} ~ {vals.max():.2f} (설비용량 {cap})")
            if n_unique <= 3:
                print(f"    -> 고유값이 매우 적음(반복값 의심): {sorted(vals.unique())}")

            # 풍속 시계열 (앞부분만 샘플로)
            w = weather[target]
            wmask = (w["forecast_kst_dtm"] >= start) & (w["forecast_kst_dtm"] <= end)
            wsub = w.loc[wmask, ["forecast_kst_dtm", "gfs_ws100_speed"]]
            if len(wsub) > 0:
                print(f"    예보풍속: 평균={wsub['gfs_ws100_speed'].mean():.2f}  "
                      f"최소={wsub['gfs_ws100_speed'].min():.2f}  최대={wsub['gfs_ws100_speed'].max():.2f} m/s")

        # 처음 24시간 상세 (그룹별 나란히)
        print(f"\n  -- 구간 시작 24시간 상세 (그룹별 나란히) --")
        print(sub.head(24).to_string(index=False))


if __name__ == "__main__":
    main()
