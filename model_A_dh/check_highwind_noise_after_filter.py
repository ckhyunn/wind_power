"""
[진단] 이상치 필터(착빙+태풍) 적용 후에도, 고풍속(18m/s+) 구간이 여전히
'두 갈래(정상발전 vs 저출력)'로 갈리는지 확인. 갈리면 필터를 더 세게 해야
하고, 안 갈리면(고르게 낮음) 필터 문제가 아니라 다른 원인.

실행:
    python check_highwind_noise_after_filter.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\doon2\Desktop\wind_power\data")
CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}
ICING_WS_THRESHOLD = 8.0
ICING_OUTPUT_THRESHOLD = 0.30
TYPHOON_SP_THRESHOLD = 94000
TYPHOON_RH_THRESHOLD = 90


def main():
    labels = pd.read_csv(DATA_DIR / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm")

    gfs = pd.read_csv(DATA_DIR / "train" / "gfs_train.csv", encoding="utf-8-sig")
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])
    agg = gfs.groupby("forecast_kst_dtm").agg(
        u=("heightAboveGround_100_100u", "mean"), v=("heightAboveGround_100_100v", "mean"),
        sp=("surface_0_sp", "mean"), rh=("heightAboveGround_2_2r", "mean"),
    )
    agg["ws"] = np.sqrt(agg["u"] ** 2 + agg["v"] ** 2)

    for g, cap in CAP.items():
        y = labels[g].dropna()
        df = agg.join(y.rename("y"), how="inner").dropna()
        df = df[df["y"] >= cap * 0.10]

        is_icing = (df["ws"] >= ICING_WS_THRESHOLD) & (df["y"] < cap * ICING_OUTPUT_THRESHOLD)
        is_typhoon = (df["sp"] < TYPHOON_SP_THRESHOLD) & (df["rh"] > TYPHOON_RH_THRESHOLD)
        df_clean = df[~is_icing & ~is_typhoon].copy()

        for lo, label in [(18, "18m/s+"), (20, "20m/s+")]:
            before = df[df["ws"] >= lo]["y"] / cap * 100
            after = df_clean[df_clean["ws"] >= lo]["y"] / cap * 100

            print(f"\n[{g}] {label} 구간")
            print(f"  필터 전: n={len(before)}  평균={before.mean():.1f}%  표준편차={before.std():.1f}  "
                  f"(최소{before.min():.0f}~최대{before.max():.0f})")
            print(f"  필터 후: n={len(after)}  평균={after.mean():.1f}%  표준편차={after.std():.1f}  "
                  f"(최소{after.min():.0f}~최대{after.max():.0f})")
            # 이봉분포(두 갈래) 여부 - 저출력(<40%)과 고출력(>60%) 비율
            if len(after) > 0:
                low = (after < 40).mean() * 100
                high = (after > 60).mean() * 100
                mid = 100 - low - high
                print(f"  필터후 분포: 저출력(<40%) {low:.0f}%  중간(40~60%) {mid:.0f}%  고출력(>60%) {high:.0f}%")


if __name__ == "__main__":
    main()
