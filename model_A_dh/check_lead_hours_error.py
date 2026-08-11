"""
[재학습 없음, 새로운 EDA 각도] 예보 리드타임(lead_hours: 예보발표시각~대상시각 
차이)이 클수록 오차가 커지는지 확인. 세션 내 처음 확인하는 각도.

실행:
    python check_lead_hours_error.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\doon2\Desktop\wind_power\data")

CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}


def main():
    actual = pd.read_csv(DATA_DIR.parent / "model_A" / "output_2024" / "actual_2024.csv")
    pred = pd.read_csv(DATA_DIR.parent / "model_A" / "output_2024" / "pred_A_blend_2024.csv")
    actual["forecast_kst_dtm"] = pd.to_datetime(actual["forecast_kst_dtm"])
    pred["forecast_kst_dtm"] = pd.to_datetime(pred["forecast_kst_dtm"])
    m = actual.merge(pred, on="forecast_kst_dtm", suffixes=("_actual", "_pred"))

    gfs = pd.read_csv(DATA_DIR / "train" / "gfs_train.csv", encoding="utf-8-sig")
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])
    gfs["data_available_kst_dtm"] = pd.to_datetime(gfs["data_available_kst_dtm"])
    gfs["lead_hours"] = (gfs["forecast_kst_dtm"] - gfs["data_available_kst_dtm"]).dt.total_seconds() / 3600
    lead = gfs.groupby("forecast_kst_dtm")["lead_hours"].first().reset_index()

    m = m.merge(lead, on="forecast_kst_dtm", how="inner")

    for g, cap in CAP.items():
        valid = m[m[f"{g}_actual"] >= cap * 0.10].copy()
        valid["er"] = (valid[f"{g}_pred"] - valid[f"{g}_actual"]).abs() / cap

        corr = valid["lead_hours"].corr(valid["er"])
        print(f"\n{'='*70}\n[{g}] 리드타임 vs 오차율 상관계수: {corr:.4f}\n{'='*70}")

        bins = [0, 6, 12, 18, 24, 30, 36]
        valid["lead_bin"] = pd.cut(valid["lead_hours"], bins=bins, include_lowest=True)
        result = valid.groupby("lead_bin", observed=True).agg(
            평균오차율=("er", lambda s: s.mean() * 100),
            시간수=("er", "size"),
        )
        print(result.round(2).to_string())


if __name__ == "__main__":
    main()
