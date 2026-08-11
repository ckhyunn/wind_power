"""
[1단계] 그룹별 물리기반 파워커브(풍속->발전량) 곡선 피팅.
재학습 없음, 2022~2024 실제 라벨+GFS 풍속으로 몇 초 내 피팅.

곡선 형태: 정격용량 x 로지스틱상승(풍속) x 컷아웃게이트(풍속)
공기밀도 보정: 보정풍속 = 풍속 x (밀도/표준밀도)^(1/3)

실행:
    python fit_power_curve.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

DATA_DIR = Path(r"C:\Users\doon2\Desktop\wind_power\data")
CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}
RHO_STD = 1.225  # 표준 공기밀도(kg/m^3)


def power_curve(v_adj, v0, k, v_cut_start, v_cut_width):
    """로지스틱 상승 x 컷아웃 게이트 (0~1 사이 비율로 반환, 정격용량은 나중에 곱함)."""
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
    )
    agg["ws"] = np.sqrt(agg["u"] ** 2 + agg["v"] ** 2)
    R = 287.05
    agg["rho"] = agg["sp"] / (R * agg["temp_k"])
    agg["ws_adj"] = agg["ws"] * (agg["rho"] / RHO_STD) ** (1 / 3)

    params_all = {}
    for g, cap in CAP.items():
        y = labels[g].dropna()
        df = agg.join(y.rename("y"), how="inner").dropna()
        df = df[df["y"] >= cap * 0.10]  # 평가기준과 동일하게

        x = df["ws_adj"].to_numpy()
        yy = (df["y"] / cap).to_numpy()  # 0~1 비율로 정규화

        p0 = [7.0, 2.0, 22.0, 2.0]  # v0, k, cutout시작, cutout폭 (초기값)
        try:
            popt, _ = curve_fit(power_curve, x, yy, p0=p0, maxfev=5000,
                                 bounds=([3, 0.5, 15, 0.5], [12, 6, 30, 8]))
        except Exception as e:
            print(f"[{g}] 피팅 실패: {e}")
            continue

        params_all[g] = popt
        pred = power_curve(x, *popt) * cap
        actual = yy * cap
        mae = np.abs(pred - actual).mean()
        r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)

        print(f"\n[{g}] 피팅 결과")
        print(f"  v0(중간점)={popt[0]:.2f}  k(기울기)={popt[1]:.2f}  "
              f"컷아웃시작={popt[2]:.2f}  컷아웃폭={popt[3]:.2f}")
        print(f"  MAE={mae:.0f}kWh ({mae/cap*100:.1f}%)  R²={r2:.3f}")

        # 참고: 몇 개 풍속 지점에서 곡선값 확인
        for v_check in [3, 5, 8, 12, 16, 20, 23, 26]:
            frac = power_curve(np.array([v_check]), *popt)[0]
            print(f"    풍속{v_check}m/s -> 이용률 {frac*100:.1f}%")

    print("\n\n최종 파라미터 (model_A_dh.py에 넣을 값):")
    print(params_all)


if __name__ == "__main__":
    main()
