"""
다른 대회(Phase2 Task2) 방법론 참고 - 두 가지 진단을 결합:
  1. 3년 전체에서 '무발전에 가까운 구간'을 전수 조사해서 주기적 패턴(정비 등)이 있는지 확인
     (그 팀의 '70일 주기 정비' 발견 방법론을 그대로 적용)
  2. 잔차/발전 저하를 기온 구간별로 정밀하게 나눠서 착빙 문턱을 통계적으로 확정
     (그 팀의 '-10도 이하 성능저하' 발견 방법론을 그대로 적용)

실행:
    python src/diagnose_downtime_patterns.py
"""

import pandas as pd
import numpy as np

from features import TARGET_COLS, CAPACITY_KWH, load_turbine_table, compute_group_coords
from train_baseline import build_group_weather, DATA_DIR, TRAIN_DIR


def find_zero_output_runs(labels: pd.DataFrame, target: str, cap: float, threshold: float = 0.05):
    """무발전(설비용량 threshold 미만) 연속 구간을 찾아서 시작일/길이 리스트로 반환.
    예보 풍속과 무관하게 순수 발전량만 보고 찾음 (외부요인 정비 등도 잡아내기 위함)."""
    df = labels[["kst_dtm", target]].dropna().sort_values("kst_dtm").reset_index(drop=True)
    is_low = (df[target] / cap) < threshold

    runs = []
    start_idx = None
    for i, low in enumerate(is_low):
        if low and start_idx is None:
            start_idx = i
        elif not low and start_idx is not None:
            length = i - start_idx
            if length >= 6:  # 6시간 이상만 (짧은 무풍은 정상이므로 제외)
                runs.append((df["kst_dtm"].iloc[start_idx], df["kst_dtm"].iloc[i - 1], length))
            start_idx = None
    if start_idx is not None:
        length = len(is_low) - start_idx
        if length >= 6:
            runs.append((df["kst_dtm"].iloc[start_idx], df["kst_dtm"].iloc[-1], length))
    return runs


def main():
    labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    ldaps = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")
    turbine_df = load_turbine_table(DATA_DIR / "info.xlsx")
    group_coords = compute_group_coords(turbine_df)
    weather = build_group_weather(ldaps, gfs, group_coords)

    print("=" * 80)
    print("1) 3년 전체 무발전(6시간+ 연속) 구간 전수 조사 - 주기성 확인")
    print("=" * 80)
    for target in TARGET_COLS:
        cap = CAPACITY_KWH[target]
        runs = find_zero_output_runs(labels, target, cap)
        print(f"\n[{target}] 6시간+ 무발전 구간 {len(runs)}건")
        starts = []
        for s, e, length in runs:
            print(f"  {s.date()} ~ {e.date()}  ({length}시간)")
            starts.append(s)
        if len(starts) >= 3:
            gaps = [(starts[i+1] - starts[i]).days for i in range(len(starts)-1)]
            print(f"  -> 시작일 간격(일): {gaps}")

    print("\n" + "=" * 80)
    print("2) 기온 구간별 저출력 비율 - 착빙 문턱 정밀 확인")
    print("=" * 80)
    for target in TARGET_COLS:
        cap = CAPACITY_KWH[target]
        w = weather[target].rename(columns={"forecast_kst_dtm": "kst_dtm"})
        merged = labels[["kst_dtm", target]].merge(w, on="kst_dtm", how="inner").dropna(subset=[target])
        merged["temp_c"] = merged["gfs_heightAboveGround_2_2t_mean"] - 273.15
        merged["cf"] = merged[target] / cap

        # 발전 가능한 바람(3m/s 이상)에서만 비교 (바람 자체가 약해서 저출력인 경우 제외)
        windy = merged[merged["gfs_ws100_speed"] >= 3.0].copy()
        bins = [-30, -15, -10, -7, -5, -3, 0, 5, 10, 40]
        windy["temp_bin"] = pd.cut(windy["temp_c"], bins=bins)
        result = windy.groupby("temp_bin", observed=True).apply(
            lambda g: pd.Series({
                "시간수": len(g),
                "저출력비율(%)": (g["cf"] < 0.05).mean() * 100,
            })
        )
        print(f"\n[{target}]")
        print(result.to_string())


if __name__ == "__main__":
    main()
