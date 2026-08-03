"""
[후류 가설 실측 검증] 이론(위경도 방위각 115도)이 아니라, 실제 SCADA 풍속으로
그룹1이 그룹2보다 상류에 있을 때(풍향 NE-E) 그룹2가 진짜로 바람을 덜 받는지 확인.

방법: VESTAS SCADA(그룹1=wtg01~06, 그룹2=wtg07~12)의 같은 시각 실측 풍속을
그룹별로 평균 낸 뒤, GFS 예보 풍향이 NE-E(45~135도, wake_align_g1_g2 높은 구간)일
때와 아닐 때로 나눠서 '그룹2 실측풍속 / 그룹1 실측풍속' 비율을 비교.

만약 NE-E일 때 이 비율이 확실히 1보다 작으면(그룹2가 더 약함) -> 후류효과 실증.
그 크기(몇 %나 약한지)까지 나오면, 지금의 대칭적 cos() 공식보다 훨씬 정교한
피처(실측 기반 보정계수)를 만들 근거가 됨.

실행:
    python src/verify_wake_with_scada.py
"""

import pandas as pd
import numpy as np

from features import load_turbine_table, compute_group_coords
from train_baseline import build_group_weather, DATA_DIR, TRAIN_DIR

WAKE_BEARING = 115.0  # 그룹1->그룹2 방위각 (위경도로 계산)


def main():
    scada = pd.read_csv(TRAIN_DIR / "scada_vestas_train.csv", encoding="utf-8-sig")
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])

    g1_ws_cols = [f"vestas_wtg{i:02d}_ws" for i in range(1, 7)]
    g2_ws_cols = [f"vestas_wtg{i:02d}_ws" for i in range(7, 13)]

    scada["g1_ws_actual"] = scada[g1_ws_cols].mean(axis=1, skipna=True)
    scada["g2_ws_actual"] = scada[g2_ws_cols].mean(axis=1, skipna=True)

    # 10분 단위 -> 시간 단위로 평균 집계 (예보 풍향과 매칭하기 위해)
    hourly = scada.set_index("kst_dtm")[["g1_ws_actual", "g2_ws_actual"]].resample("h").mean().reset_index()
    hourly = hourly.dropna()

    # GFS 예보 풍향(그룹1 기준) 가져오기
    ldaps = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")
    turbine_df = load_turbine_table(DATA_DIR / "info.xlsx")
    group_coords = compute_group_coords(turbine_df)
    weather = build_group_weather(ldaps, gfs, group_coords)
    w = weather["kpx_group_1"][["forecast_kst_dtm", "gfs_ws100_speed"]].rename(
        columns={"forecast_kst_dtm": "kst_dtm", "gfs_ws100_speed": "forecast_ws"}
    )
    # 풍향 재계산 (원본 u/v 필요 - build_group_weather 결과에서 재구성)
    w_full = weather["kpx_group_1"]
    if "gfs_ws100_dir_sin" in w_full.columns:
        w_full = w_full[["forecast_kst_dtm", "gfs_ws100_dir_sin", "gfs_ws100_dir_cos", "gfs_ws100_speed"]].copy()
        w_full["wind_dir_deg"] = (np.degrees(np.arctan2(w_full["gfs_ws100_dir_sin"], w_full["gfs_ws100_dir_cos"])) + 360) % 360
        w_full = w_full.rename(columns={"forecast_kst_dtm": "kst_dtm"})
    else:
        print("풍향 컬럼 없음 - 종료")
        return

    merged = hourly.merge(w_full, on="kst_dtm", how="inner")
    merged = merged[(merged["g1_ws_actual"] > 0.5)]  # 완전 정지 시간대 제외 (0으로 나누기 방지)
    merged["g2_g1_ratio"] = merged["g2_ws_actual"] / merged["g1_ws_actual"]

    # wake 정렬 지수 (연속값)
    merged["wake_align"] = np.cos(np.radians(merged["wind_dir_deg"] - WAKE_BEARING))

    print("=" * 80)
    print("전체 기간 - 풍향 구간별 실측 풍속 비율(그룹2/그룹1)")
    print("=" * 80)
    dir_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    dir_labels = ["N-NE", "NE-E", "E-SE", "SE-S", "S-SW", "SW-W", "W-NW", "NW-N"]
    merged["dir_bin"] = pd.cut(merged["wind_dir_deg"], bins=dir_bins, labels=dir_labels, include_lowest=True)
    result = merged.groupby("dir_bin", observed=True).agg(
        평균비율_g2div1=("g2_g1_ratio", "mean"),
        중앙값비율=("g2_g1_ratio", "median"),
        시간수=("g2_g1_ratio", "size"),
    )
    print(result.round(3).to_string())

    print("\n" + "=" * 80)
    print("wake_align 연속값과 g2/g1 실측풍속비율의 상관관계")
    print("=" * 80)
    corr = merged["wake_align"].corr(merged["g2_g1_ratio"])
    print(f"상관계수: {corr:.4f} (음수면 wake_align이 높을수록(=풍향이 그룹1->그룹2 방향) "
          f"그룹2 풍속이 상대적으로 약해진다는 뜻 - 후류가설 지지)")

    # wake_align 상위 25% vs 하위 25% 비교
    q75 = merged["wake_align"].quantile(0.75)
    q25 = merged["wake_align"].quantile(0.25)
    high_wake = merged[merged["wake_align"] >= q75]
    low_wake = merged[merged["wake_align"] <= q25]
    print(f"\nwake_align 상위25%(풍향이 그룹1->2 방향에 가까움): 평균비율={high_wake['g2_g1_ratio'].mean():.3f} (n={len(high_wake)})")
    print(f"wake_align 하위25%(반대방향): 평균비율={low_wake['g2_g1_ratio'].mean():.3f} (n={len(low_wake)})")
    print(f"차이: {(high_wake['g2_g1_ratio'].mean() - low_wake['g2_g1_ratio'].mean())*100:.1f}%p")


if __name__ == "__main__":
    main()
