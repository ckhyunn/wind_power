"""
[재학습 없음] wtg05(그룹1)와 unison_wtg02(그룹3)의 저성능이 만성적인지,
특정 시점부터 시작된 것인지 월별로 확인 (wtg07 조사와 동일 방식).

실행:
    python check_low_performers_timeline.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\doon2\Desktop\wind_power\data")


def check_timeline(scada_path, group_prefix, group_ids, target_turbine, ws_suffix, pw_suffix, label):
    scada = pd.read_csv(scada_path, encoding="utf-8-sig")
    dt_col = "kst_dtm" if "kst_dtm" in scada.columns else [c for c in scada.columns if "dtm" in c.lower()][0]
    scada[dt_col] = pd.to_datetime(scada[dt_col])

    turbines = [f"{group_prefix}{i:02d}" for i in group_ids]
    pw_cols = [f"{t}{pw_suffix}" for t in turbines]
    missing = [c for c in pw_cols if c not in scada.columns]
    if missing:
        print(f"[{label}] 컬럼을 못 찾음: {missing}")
        return

    group_mean_pw = scada[pw_cols].clip(lower=0).mean(axis=1, skipna=True)
    target_pw = scada[f"{target_turbine}{pw_suffix}"].clip(lower=0)
    scada["ratio"] = np.where(group_mean_pw > 10, target_pw / group_mean_pw, np.nan)

    scada["year_month"] = scada[dt_col].dt.to_period("M")
    monthly = scada.groupby("year_month")["ratio"].agg(["mean", "count"])
    monthly["mean"] = (monthly["mean"] * 100).round(1)
    print(f"\n{'='*70}\n[{label}] {target_turbine} 월별 상대발전비율(%) - 그룹평균 대비\n{'='*70}")
    print(monthly.to_string())

    scada["negative"] = scada[f"{target_turbine}{pw_suffix}"] < 0
    neg_monthly = scada.groupby("year_month")["negative"].mean() * 100
    print(f"\n{target_turbine} 월별 음수값 비율(%):")
    print(neg_monthly.round(1).to_string())

    # 연도별 요약
    scada["year"] = scada[dt_col].dt.year
    yearly = scada.groupby("year")["ratio"].mean() * 100
    print(f"\n연도별 평균 요약: {yearly.round(1).to_dict()}")


def main():
    check_timeline(
        DATA_DIR / "train" / "scada_vestas_train.csv",
        "vestas_wtg", range(1, 7), "vestas_wtg05", "_ws", "_power_kw10m", "그룹1"
    )
    check_timeline(
        DATA_DIR / "train" / "scada_unison_train.csv",
        "unison_wtg", range(1, 6), "unison_wtg02", "_ws", "_power_kw10m", "그룹3"
    )


if __name__ == "__main__":
    main()
