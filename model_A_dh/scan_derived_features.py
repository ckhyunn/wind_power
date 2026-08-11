"""
[전처리 재검토 2단계] 팀원 제안 중 '새로운 종류'인 3개 파생변수를 먼저 
단독으로 빠르게 검증 (복합피처로 바로 안 가고): 
  1. 연직 풍속전단 지수(alpha, 로그비율)
  2. 연직 대기불안정도(T지표-T700)
  3. 500hPa 기압경향(3시간)
재학습 없음, 몇 초 내 완료.

실행:
    python scan_derived_features.py
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

    # 필요 컬럼 존재 확인 (환경마다 실제 이름 다를 수 있어 후보 나열)
    col_candidates = {
        "ws10": ["heightAboveGround_10_10u", "heightAboveGround_10_10v"],
        "ws100": ["heightAboveGround_100_100u", "heightAboveGround_100_100v"],
        "t_surface": ["heightAboveGround_2_2t"],
        "t_700": ["isobaricInhPa_700_t", "isobaricInhPa_700_700t"],
        "z_500": ["isobaricInhPa_500_gh", "isobaricInhPa_500_z"],
    }
    print("실제 GFS 컬럼 중 700/500hPa, 기온 관련:")
    print([c for c in gfs.columns if "700" in c or "500" in c or "_t" in c.lower()])

    agg_cols = []
    for key, cands in col_candidates.items():
        if key in ("ws10", "ws100"):
            # [동훈수정] u/v 둘 다 필요한데 첫 매치에서 break해서 v가 누락되던 버그 수정
            for c in cands:
                if c in gfs.columns:
                    agg_cols.append(c)
        else:
            for c in cands:
                if c in gfs.columns:
                    agg_cols.append(c)
                    break
    agg = gfs.groupby("forecast_kst_dtm")[agg_cols].mean().sort_index()

    # 1) 풍속전단 지수(alpha)
    if "heightAboveGround_10_10u" in agg.columns and "heightAboveGround_100_100u" in agg.columns:
        ws10 = np.sqrt(agg["heightAboveGround_10_10u"]**2 + agg["heightAboveGround_10_10v"]**2)
        ws100 = np.sqrt(agg["heightAboveGround_100_100u"]**2 + agg["heightAboveGround_100_100v"]**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            agg["shear_alpha"] = np.log(ws100 / ws10.replace(0, np.nan)) / np.log(100/10)

    # 2) 연직 불안정도(T지표 - T700)
    t_surf_col = "heightAboveGround_2_2t"
    t700_col = next((c for c in ["isobaricInhPa_700_t", "isobaricInhPa_700_700t"] if c in agg.columns), None)
    if t_surf_col in agg.columns and t700_col:
        agg["t_instability"] = agg[t_surf_col] - agg[t700_col]

    # 3) 500hPa 기압경향(3시간)
    z500_col = next((c for c in ["isobaricInhPa_500_gh", "isobaricInhPa_500_z"] if c in agg.columns), None)
    if z500_col:
        agg["z500_tendency_3h"] = agg[z500_col] - agg[z500_col].shift(3)

    new_features = [c for c in ["shear_alpha", "t_instability", "z500_tendency_3h"] if c in agg.columns]
    print(f"\n생성된 파생변수: {new_features}")
    if not new_features:
        print("필요 원본 컬럼을 못 찾음 - 위 컬럼목록 참고해서 코드 수정 필요")
        return

    m = m.merge(agg[new_features].reset_index(), on="forecast_kst_dtm", how="inner")

    for t, cap in CAP.items():
        valid = m[m[f"{t}_actual"] >= cap * 0.10].copy()
        valid["er"] = (valid[f"{t}_pred"] - valid[f"{t}_actual"]).abs() / cap
        print(f"\n[{t}] 오차율과의 상관계수:")
        for col in new_features:
            corr = valid[col].corr(valid["er"])
            flag = " <<<" if abs(corr) > 0.10 else ""
            print(f"  {col:<20}: {corr:>+.4f}{flag}")


if __name__ == "__main__":
    main()
