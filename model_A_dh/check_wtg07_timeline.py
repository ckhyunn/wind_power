"""
[재학습 없음] wtg07의 저성능이 학습기간(2022~2024) 내내 만성적이었는지,
아니면 특정 시점부터 시작된 고장/열화(regime shift)인지 시기별로 확인.

실행:
    python check_wtg07_timeline.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\doon2\Desktop\wind_power\data")


def main():
    scada = pd.read_csv(DATA_DIR / "train" / "scada_vestas_train.csv", encoding="utf-8-sig")
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])

    turbines = [f"vestas_wtg{i:02d}" for i in range(7, 13)]
    pw_cols = [f"{t}_power_kw10m" for t in turbines]

    group_mean_pw = scada[pw_cols].clip(lower=0).mean(axis=1, skipna=True)
    wtg07_pw = scada["vestas_wtg07_power_kw10m"].clip(lower=0)
    scada["wtg07_ratio"] = np.where(group_mean_pw > 10, wtg07_pw / group_mean_pw, np.nan)

    scada["year_month"] = scada["kst_dtm"].dt.to_period("M")
    monthly = scada.groupby("year_month")["wtg07_ratio"].agg(["mean", "count"])
    monthly["mean"] = (monthly["mean"] * 100).round(1)
    print("=" * 60)
    print("wtg07 월별 상대발전비율(%) - 그룹평균 대비")
    print("=" * 60)
    print(monthly.to_string())

    # 음수(고장추정) 비율도 월별로
    scada["wtg07_negative"] = scada["vestas_wtg07_power_kw10m"] < 0
    neg_monthly = scada.groupby("year_month")["wtg07_negative"].mean() * 100
    print("\nwtg07 월별 음수값(고장추정) 비율(%):")
    print(neg_monthly.round(1).to_string())


if __name__ == "__main__":
    main()
