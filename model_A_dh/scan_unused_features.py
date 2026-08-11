"""
[전처리 재검토] 아직 피처로 안 쓴 원본 변수들(구름량, 강수, 경계층높이, 
바람방향변화(veer), 돌풍/평균풍속 비율 등)을 한 번에 훑어서 오차율과 
상관관계가 있는지 확인. 재학습 없음, 몇 초 내 완료.

실행:
    python scan_unused_features.py
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
    print("GFS 원본 컬럼 목록:")
    print(list(gfs.columns))

    # [동훈수정] 실제 GFS 컬럼명에 맞게 교정 (원래 후보명은 존재하지 않아 조용히 스킵되고 있었음)
    candidates = {
        "atmosphere_0_tcc": "전운량(구름)",
        "lowCloudLayer_0_lcc": "하층운량",
        "middleCloudLayer_0_mcc": "중층운량",
        "highCloudLayer_0_hcc": "상층운량",
        "surface_0_dswrf": "하향단파복사(일사량)",
        "surface_0_dlwrf": "하향장파복사",
        "surface_0_tp": "누적강수",
        "surface_0_prate": "강수강도",
        "planetaryBoundaryLayer_0_VRATE": "경계층환기율(높이 컬럼 없어 대체)",
        "surface_0_gust": "돌풍",
        "isobaricInhPa_850_r": "850hPa 상대습도",
        "isobaricInhPa_700_t": "700hPa 기온",
        "isobaricInhPa_500_gh": "500hPa 지위고도",
        "heightAboveGround_100_100u": "u100", "heightAboveGround_100_100v": "v100",
        "heightAboveGround_10_10u": "u10", "heightAboveGround_10_10v": "v10",
        "isobaricInhPa_850_u": "u850", "isobaricInhPa_850_v": "v850",
    }
    available = {k: v for k, v in candidates.items() if k in gfs.columns}
    agg = gfs.groupby("forecast_kst_dtm")[list(available.keys())].mean()

    # 파생: 풍향변화(veer, 10m vs 850hPa), 돌풍/평균풍속 비율(난류도)
    if "heightAboveGround_10_10u" in agg.columns and "isobaricInhPa_850_u" in agg.columns:
        dir10 = np.degrees(np.arctan2(agg["heightAboveGround_10_10u"], agg["heightAboveGround_10_10v"]))
        dir850 = np.degrees(np.arctan2(agg["isobaricInhPa_850_u"], agg["isobaricInhPa_850_v"]))
        veer = (dir850 - dir10 + 180) % 360 - 180  # -180~180로 정규화
        agg["wind_veer_10_850"] = veer.abs()

    if "heightAboveGround_100_100u" in agg.columns:
        ws100 = np.sqrt(agg["heightAboveGround_100_100u"]**2 + agg["heightAboveGround_100_100v"]**2)
        agg["ws100"] = ws100
        if "surface_0_gust" in agg.columns:
            agg["gust_ratio"] = agg["surface_0_gust"] / (ws100 + 0.1)

    m = m.merge(agg.reset_index(), on="forecast_kst_dtm", how="inner")

    feature_cols = [c for c in agg.columns if c not in ["heightAboveGround_100_100u","heightAboveGround_100_100v",
                    "heightAboveGround_10_10u","heightAboveGround_10_10v","isobaricInhPa_850_u","isobaricInhPa_850_v"]]

    print(f"\n{'='*80}\n확인할 변수: {feature_cols}\n{'='*80}")

    for t, cap in CAP.items():
        valid = m[m[f"{t}_actual"] >= cap * 0.10].copy()
        valid["er"] = (valid[f"{t}_pred"] - valid[f"{t}_actual"]).abs() / cap
        print(f"\n[{t}] 오차율과의 상관계수:")
        for col in feature_cols:
            if col not in valid.columns:
                continue
            corr = valid[col].corr(valid["er"])
            flag = " <<<" if abs(corr) > 0.10 else ""
            print(f"  {col:<25}: {corr:>+.4f}{flag}")


if __name__ == "__main__":
    main()
