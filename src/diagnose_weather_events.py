"""
검색으로 확인된 실제 기상/지형 이벤트 날짜가 우리 학습 라벨(train_labels.csv)과
예보 데이터에 어떻게 찍혀 있는지 직접 대조하는 스크립트.

확인된 이벤트:
  - 2022-09-04~06: 태풍 힌남노 (강원도 4대 국립공원 태백산 포함 탐방로 통제)
  - 2023-08-10: 태풍 카눈 (태백산 탐방로 통제, 태백선 철도 운행 중단)
  - 2024-01-18~19: 태백 매봉산 인근 혹한/결빙 (터빈 착빙 의심 정황)

각 이벤트 구간에서:
  1. 실제 발전량이 정상 범위인지, 아니면 급격히 0으로 떨어지거나(강제정지) 정격을
     초과하는 이상 패턴이 있는지
  2. 그 시점 예보 풍속이 실제로 극단적이었는지 (전체 기간 대비 상위 %)
  3. 편향 진단에서 이 구간이 유독 오차가 컸던 시기와 겹치는지

실행:
    python src/diagnose_weather_events.py
"""

import pandas as pd
import numpy as np

from features import TARGET_COLS, CAPACITY_KWH, load_turbine_table, compute_group_coords
from train_baseline import build_group_weather, DATA_DIR, TRAIN_DIR

EVENTS = [
    ("2022-09-03", "2022-09-07", "태풍 힌남노"),
    ("2023-08-09", "2023-08-11", "태풍 카눈"),
    ("2024-01-17", "2024-01-20", "태백 매봉산 혹한/결빙"),
]


def main():
    labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])

    ldaps = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")
    turbine_df = load_turbine_table(DATA_DIR / "info.xlsx")
    group_coords = compute_group_coords(turbine_df)
    weather = build_group_weather(ldaps, gfs, group_coords)

    for start, end, name in EVENTS:
        print(f"\n{'='*80}\n[{name}] {start} ~ {end}\n{'='*80}")

        for target in TARGET_COLS:
            cap = CAPACITY_KWH[target]
            mask = (labels["kst_dtm"] >= start) & (labels["kst_dtm"] <= end)
            sub = labels.loc[mask, target].dropna()
            if len(sub) == 0:
                print(f"  [{target}] 이 기간 라벨 없음 (그룹3은 2023년부터 시작이라 2022년엔 없을 수 있음)")
                continue

            cf = sub / cap  # 설비이용률
            all_cf = labels[target].dropna() / cap
            pct_rank = (all_cf < cf.mean()).mean() * 100

            print(f"\n  [{target}] 실제 발전량 (n={len(sub)}시간)")
            print(f"    이용률: 평균={cf.mean()*100:.1f}%  최소={cf.min()*100:.1f}%  최대={cf.max()*100:.1f}%")
            print(f"    0 근처(용량 5%미만) 시간대: {(cf < 0.05).sum()}시간 ({(cf<0.05).mean()*100:.1f}%)")
            print(f"    -> 이 구간 평균 이용률은 전체 기간 대비 상위 {100-pct_rank:.1f}% 수준")

            # 같은 기간 예보 풍속
            w = weather[target]
            wmask = (w["forecast_kst_dtm"] >= start) & (w["forecast_kst_dtm"] <= end)
            wsub = w.loc[wmask, "gfs_ws100_speed"].dropna()
            if len(wsub) > 0:
                all_ws = w["gfs_ws100_speed"].dropna()
                ws_pct = (all_ws < wsub.mean()).mean() * 100
                print(f"    예보 100m풍속: 평균={wsub.mean():.2f}  최대={wsub.max():.2f} m/s "
                      f"(전체 대비 상위 {100-ws_pct:.1f}%)")


if __name__ == "__main__":
    main()
