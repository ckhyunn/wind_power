"""
[3단계] 파워커브 수식을 '산모양'에서 '사다리꼴'로 교체해 재피팅.
이유: 로지스틱상승x로지스틱하강은 구조적으로 '넓은 평평구간'을 표현 못함.
실제 터빈은 정격 도달 후 컷아웃 직전까지 쭉 평평하게 유지되다 급락하는 사다리꼴.
이상치(착빙/태풍) 제거는 이전 단계와 동일하게 적용.

실행:
    python fit_power_curve_trapezoid.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

DATA_DIR = Path(r"C:\Users\doon2\Desktop\wind_power\data")
CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}
RHO_STD = 1.225
ICING_WS_THRESHOLD = 8.0
ICING_OUTPUT_THRESHOLD = 0.30
TYPHOON_SP_THRESHOLD = 94000
TYPHOON_RH_THRESHOLD = 90


def power_curve_trapezoid(v_adj, v0, k, v_cut, cut_width):
    """상승 후 1에서 포화(정격유지) + 컷아웃 폭도 데이터가 직접 찾게 자유변수로 둠
    (임의 고정값 대신 - 이전 시도에서 폭을 0.75로 고정했다가 R2가 오히려 나빠졌음)."""
    rise = np.clip(1 / (1 + np.exp(-(v_adj - v0) / k)), 0, 1)
    gate = 1 / (1 + np.exp((v_adj - v_cut) / np.clip(cut_width, 0.3, 6)))
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
    for g, cap in CAP.items():
        y = labels[g].dropna()
        df = agg.join(y.rename("y"), how="inner").dropna()
        df = df[df["y"] >= cap * 0.10]

        is_icing = (df["ws"] >= ICING_WS_THRESHOLD) & (df["y"] < cap * ICING_OUTPUT_THRESHOLD)
        is_typhoon = (df["sp"] < TYPHOON_SP_THRESHOLD) & (df["rh"] > TYPHOON_RH_THRESHOLD)
        df_clean = df[~is_icing & ~is_typhoon].copy()

        x = df_clean["ws_adj"].to_numpy()
        yy = (df_clean["y"] / cap).to_numpy()

        p0 = [7.0, 2.0, 25.0, 1.5]
        popt, _ = curve_fit(power_curve_trapezoid, x, yy, p0=p0, maxfev=5000,
                             bounds=([3, 0.5, 18, 0.3], [12, 6, 28, 6]))
        params_all[g] = popt.tolist()

        pred = power_curve_trapezoid(x, *popt) * cap
        actual = yy * cap
        mae = np.abs(pred - actual).mean()
        r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)

        print(f"\n[{g}] v0={popt[0]:.2f}  k={popt[1]:.2f}  컷아웃={popt[2]:.2f}  컷아웃폭={popt[3]:.2f}")
        print(f"  MAE={mae:.0f}kWh ({mae/cap*100:.1f}%)  R²={r2:.3f}")

        print(f"  풍속별 이용률(%):")
        for v_check in [3, 5, 8, 12, 16, 18, 20, 23, 26]:
            frac = power_curve_trapezoid(np.array([v_check]), *popt)[0]
            print(f"    {v_check}m/s -> {frac*100:.1f}%")

    print("\n최종 파라미터:", params_all)


if __name__ == "__main__":
    main()
