"""
[재학습 없음] 그룹2(wtg07~12) 개별 터빈이 날씨와 무관하게 서로 다른 파워커브
(풍속-발전량 관계)를 갖는지 확인 - "날씨로는 절대 못 잡는" 이상치 후보를 찾음.

실행:
    python check_turbine_individual_curve.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\doon2\Desktop\wind_power\data")


def main():
    scada = pd.read_csv(DATA_DIR / "train" / "scada_vestas_train.csv", encoding="utf-8-sig")
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])

    turbines = [f"vestas_wtg{i:02d}" for i in range(7, 13)]
    ws_cols = [f"{t}_ws" for t in turbines]
    pw_cols = [f"{t}_power_kw10m" for t in turbines]

    scada["group_avg_ws"] = scada[ws_cols].mean(axis=1, skipna=True)
    ws_bins = [0, 3, 5, 7, 9, 11, 13, 16, 25]
    scada["ws_bin"] = pd.cut(scada["group_avg_ws"], bins=ws_bins)

    print("=" * 90)
    print("풍속 구간별 각 터빈의 상대적 발전량 비율 (그룹2 평균 대비 %)")
    print("=" * 90)

    group_mean_pw = scada[pw_cols].clip(lower=0).mean(axis=1, skipna=True)
    results = {}
    for t, pw_col in zip(turbines, pw_cols):
        pw = scada[pw_col].clip(lower=0)
        ratio = np.where(group_mean_pw > 10, pw / group_mean_pw, np.nan)
        scada[f"{t}_ratio"] = ratio

    summary = scada.groupby("ws_bin", observed=True)[[f"{t}_ratio" for t in turbines]].mean()
    summary.columns = turbines
    print((summary * 100).round(1).to_string())

    print("\n전체기간 평균 상대비율(%) - 100%보다 확실히 낮거나 높으면 그 터빈만 특이:")
    overall = scada[[f"{t}_ratio" for t in turbines]].mean() * 100
    overall.index = turbines
    print(overall.round(1).to_string())

    print("\n전체기간 발전량 절대 합계 비교 (터빈 노후/고장 등으로 총량 자체가 낮은지):")
    totals = scada[pw_cols].clip(lower=0).sum()
    totals.index = turbines
    print(totals.round(0).to_string())


if __name__ == "__main__":
    main()
