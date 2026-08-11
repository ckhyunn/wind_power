"""
[재학습 없음] 그룹2의 wtg07 발견과 동일한 방식으로, 그룹1(VESTAS wtg01~06)과
그룹3(UNISON)에도 문제 터빈이 있는지 확인.

주의: UNISON 컬럼명은 정확한 명명 규칙을 모르니, 실행 전 
scada_unison_train.csv의 실제 컬럼명을 먼저 확인해서 UNISON_TURBINES 리스트를
맞게 수정해야 할 수 있음.

실행:
    python check_all_turbines.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\doon2\Desktop\wind_power\data")


def analyze_group(scada_path, turbine_prefix, turbine_ids, ws_suffix, pw_suffix, label):
    scada = pd.read_csv(scada_path, encoding="utf-8-sig")
    turbines = [f"{turbine_prefix}{i:02d}" for i in turbine_ids]
    pw_cols = [f"{t}{pw_suffix}" for t in turbines]

    missing = [c for c in pw_cols if c not in scada.columns]
    if missing:
        print(f"[{label}] 컬럼을 못 찾음: {missing}")
        print(f"  실제 컬럼 목록(일부): {[c for c in scada.columns if 'power' in c.lower() or 'kw' in c.lower()][:15]}")
        return

    group_mean_pw = scada[pw_cols].clip(lower=0).mean(axis=1, skipna=True)
    ws_cols = [f"{t}{ws_suffix}" for t in turbines]
    scada["group_avg_ws"] = scada[ws_cols].mean(axis=1, skipna=True)
    ws_bins = [0, 3, 5, 7, 9, 11, 13, 16, 25]
    scada["ws_bin"] = pd.cut(scada["group_avg_ws"], bins=ws_bins)

    for t, pw_col in zip(turbines, pw_cols):
        pw = scada[pw_col].clip(lower=0)
        scada[f"{t}_ratio"] = np.where(group_mean_pw > 10, pw / group_mean_pw, np.nan)

    print(f"\n{'='*90}\n[{label}] 풍속구간별 상대발전비율(%, 그룹평균=100)\n{'='*90}")
    summary = scada.groupby("ws_bin", observed=True)[[f"{t}_ratio" for t in turbines]].mean()
    summary.columns = turbines
    print((summary * 100).round(1).to_string())

    overall = scada[[f"{t}_ratio" for t in turbines]].mean() * 100
    overall.index = turbines
    print(f"\n전체기간 평균 상대비율(%):")
    print(overall.round(1).to_string())

    totals = scada[pw_cols].clip(lower=0).sum()
    totals.index = turbines
    print(f"\n전체 발전량 합계:")
    print(totals.round(0).to_string())

    neg_ratio = (scada[pw_cols] < 0).mean() * 100
    neg_ratio.index = turbines
    print(f"\n음수값(고장추정) 비율(%):")
    print(neg_ratio.round(1).to_string())


def main():
    # 그룹1
    analyze_group(
        DATA_DIR / "train" / "scada_vestas_train.csv",
        "vestas_wtg", range(1, 7), "_ws", "_power_kw10m", "그룹1(VESTAS wtg01~06)"
    )
    # 그룹3 - UNISON 컬럼명 확인 필요, 우선 5기로 가정
    analyze_group(
        DATA_DIR / "train" / "scada_unison_train.csv",
        "unison_wtg", range(1, 6), "_ws", "_power_kw10m", "그룹3(UNISON wtg01~05)"
    )


if __name__ == "__main__":
    main()
