"""
[2단계] 이상치(착빙/태풍) 제거 후 깨끗한 파워커브 곡선 재피팅.
어제 만든 이상치 규칙(model_A_dh.py의 outlier_mask/typhoon_weight_factor와 동일 기준)
그대로 적용. 재학습 없음, 몇 초 내 완료.

실행:
    python fit_power_curve_clean.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

DATA_DIR = Path(r"C:\Users\doon2\Desktop\wind_power\data")
CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}
RHO_STD = 1.225

# 어제 확정한 이상치 기준
ICING_WS_THRESHOLD = 8.0
ICING_OUTPUT_THRESHOLD = 0.30
TYPHOON_SP_THRESHOLD = 94000
TYPHOON_RH_THRESHOLD = 90


def power_curve(v_adj, v0, k, v_cut_start, v_cut_width):
    rise = 1 / (1 + np.exp(-(v_adj - v0) / k))
    gate = 1 / (1 + np.exp((v_adj - v_cut_start) / max(v_cut_width, 0.1)))
    return rise * gate


def main():
    labels = pd.read_csv(DATA_DIR / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("kst_dtm")

    gfs = pd.read_csv(DATA_DIR / "train" / "gfs_train.csv", encoding="utf-8-sig")
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])
    agg = gfs.groupby("forecast_kst_dtm").agg(
        u=("heightAboveGround_100_100u", "mean"), v=("heightAboveGround_100_100v", "mean"),
        temp_k=("heightAboveGround_2_2t", "mean"), sp=("surface_0_sp", "mean"),
        rh=("heightAboveGround_2_2r", "mean"),
    )
    agg["ws"] = np.sqrt(agg["u"] ** 2 + agg["v"] ** 2)
    R = 287.05
    agg["rho"] = agg["sp"] / (R * agg["temp_k"])
    agg["ws_adj"] = agg["ws"] * (agg["rho"] / RHO_STD) ** (1 / 3)

    params_all = {}
    viz_data = {"ws_bins": list(range(0, 31, 2))}

    for g, cap in CAP.items():
        y = labels[g].dropna()
        df = agg.join(y.rename("y"), how="inner").dropna()
        df = df[df["y"] >= cap * 0.10]

        # 이상치 제거
        is_icing = (df["ws"] >= ICING_WS_THRESHOLD) & (df["y"] < cap * ICING_OUTPUT_THRESHOLD)
        is_typhoon = (df["sp"] < TYPHOON_SP_THRESHOLD) & (df["rh"] > TYPHOON_RH_THRESHOLD)
        df_raw = df.copy()
        df_clean = df[~is_icing & ~is_typhoon].copy()

        print(f"\n{'='*70}\n[{g}] 이상치 제거: 원본 {len(df_raw)}건 -> 정제후 {len(df_clean)}건 "
              f"(착빙 {is_icing.sum()}건, 태풍 {is_typhoon.sum()}건 제외)\n{'='*70}")

        x = df_clean["ws_adj"].to_numpy()
        yy = (df_clean["y"] / cap).to_numpy()

        p0 = [7.0, 2.0, 22.0, 2.0]
        try:
            popt, _ = curve_fit(power_curve, x, yy, p0=p0, maxfev=5000,
                                 bounds=([3, 0.5, 15, 0.5], [12, 6, 30, 8]))
        except Exception as e:
            print(f"피팅 실패: {e}")
            continue

        params_all[g] = popt.tolist()
        pred = power_curve(x, *popt) * cap
        actual = yy * cap
        mae = np.abs(pred - actual).mean()
        r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)
        print(f"  v0={popt[0]:.2f}  k={popt[1]:.2f}  컷아웃시작={popt[2]:.2f}  컷아웃폭={popt[3]:.2f}")
        print(f"  MAE={mae:.0f}kWh ({mae/cap*100:.1f}%)  R²={r2:.3f}")

        # 시각화용: 정제전/후 실측 평균 + 곡선값을 풍속 구간별로
        bins = list(range(0, 31, 2))
        df_raw["ws_bin"] = pd.cut(df_raw["ws"], bins=bins)
        df_clean["ws_bin"] = pd.cut(df_clean["ws"], bins=bins)
        raw_avg = (df_raw.groupby("ws_bin", observed=True)["y"].mean() / cap * 100).reindex(
            pd.IntervalIndex.from_breaks(bins)).tolist()
        clean_avg = (df_clean.groupby("ws_bin", observed=True)["y"].mean() / cap * 100).reindex(
            pd.IntervalIndex.from_breaks(bins)).tolist()
        curve_vals = [power_curve(np.array([b + 1]), *popt)[0] * 100 for b in bins[:-1]]

        viz_data[g] = {"raw_avg": raw_avg, "clean_avg": clean_avg, "curve": curve_vals}

        print(f"\n  풍속구간별 비교(정제전실측 / 정제후실측 / 피팅곡선) %:")
        for i, b in enumerate(bins[:-1]):
            r = raw_avg[i] if raw_avg[i] == raw_avg[i] else None
            c = clean_avg[i] if clean_avg[i] == clean_avg[i] else None
            print(f"    {b}~{b+2}m/s: 정제전={r}  정제후={c}  곡선={curve_vals[i]:.1f}")

    print("\n\n=== 시각화용 데이터(JSON 형태로 복사해서 알려주세요) ===")
    import json
    print(json.dumps(viz_data, default=lambda x: None if pd.isna(x) else x))
    print("\n최종 파라미터:", params_all)


if __name__ == "__main__":
    main()
