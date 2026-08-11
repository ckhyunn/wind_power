"""
[재학습 없음] 풍속 하나가 아니라, 풍속/기온/기압/습도/풍향을 전부 종합해서
'이 시간대가 전체적으로 이례적인가'를 통계적으로 판단(Isolation Forest).

방식: 순환논리 없음 - 모델 예측이 아니라, 각 시간대의 (기상조건 + 실제발전량)
벡터가 전체 분포에서 얼마나 동떨어져 있는지를 비지도학습으로 계산.
여러 변수의 '조합'이 이상한 경우(예: 풍속은 정상인데 기압이 극단적으로 낮으면서
발전량도 이상한 경우)까지 잡아낼 수 있음 - 단일 조건 규칙보다 포괄적.

검증: 걸러진 이상치가 실제로 알려진 사건(2024-01-20~25 혹한, 2023-11-06~07 등)과
겹치는지 확인해서, 이 방법이 신뢰할 만한지부터 체크.

실행 (몇 초~수십 초):
    python detect_outliers_multivariate.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

DATA_DIR = Path(r"C:\Users\doon2\Desktop\wind_power\data")
CLEAN_DIR = Path(__file__).resolve().parent / "cleaned"

CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}
CONTAMINATION = 0.03  # 전체 중 상위 3%를 이상치 후보로 (튜닝 가능)

KNOWN_EVENTS = [
    ("2024-01-18", "2024-01-25", "1월 혹한기 착빙"),
    ("2023-11-06", "2023-11-07", "11월 착빙"),
    ("2024-02-19", "2024-02-29", "2월 초장기 무발전"),
]


def main():
    labels = pd.read_csv(CLEAN_DIR / "train_labels_clean.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")

    gfs = pd.read_csv(DATA_DIR / "train/gfs_train.csv", encoding="utf-8-sig")
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])
    agg = gfs.groupby("forecast_kst_dtm").agg(
        u=("heightAboveGround_100_100u", "mean"), v=("heightAboveGround_100_100v", "mean"),
        temp_k=("heightAboveGround_2_2t", "mean"), rh=("heightAboveGround_2_2r", "mean"),
        sp=("surface_0_sp", "mean"),
    )
    agg["ws"] = np.sqrt(agg["u"] ** 2 + agg["v"] ** 2)
    agg["wind_dir_sin"] = agg["u"] / agg["ws"].replace(0, np.nan)
    agg["wind_dir_cos"] = agg["v"] / agg["ws"].replace(0, np.nan)
    agg["temp_c"] = agg["temp_k"] - 273.15

    for g, cap in CAP.items():
        y = labels[g].dropna()
        df = agg.join(y.rename("y"), how="inner").dropna()
        # 발전 가능한 바람(컷인 이상)에서만 판단 - 무풍 정상정지는 애초에 이상치 대상 아님
        df = df[df["ws"] >= 3.0].copy()
        df["output_ratio"] = df["y"] / cap

        features = ["ws", "temp_c", "sp", "rh", "wind_dir_sin", "wind_dir_cos", "output_ratio"]
        Xf = df[features].dropna()

        iso = IsolationForest(contamination=CONTAMINATION, random_state=42, n_estimators=200)
        pred = iso.fit_predict(Xf)  # -1 = 이상치, 1 = 정상
        df.loc[Xf.index, "is_outlier"] = (pred == -1)

        n_outlier = df["is_outlier"].sum()
        print(f"\n{'='*70}\n[{g}] 전체 {len(df)}건 중 이상치 {n_outlier}건 ({n_outlier/len(df)*100:.2f}%)\n{'='*70}")

        outliers = df[df["is_outlier"]]
        print(f"이상치 시간대 프로파일: 평균풍속={outliers['ws'].mean():.1f}m/s  "
              f"평균기온={outliers['temp_c'].mean():.1f}°C  평균습도={outliers['rh'].mean():.1f}%  "
              f"평균이용률={outliers['output_ratio'].mean()*100:.1f}%")

        print("\n알려진 사건과의 일치 여부:")
        for start, end, name in KNOWN_EVENTS:
            in_event = outliers[(outliers.index >= start) & (outliers.index <= end)]
            total_in_event = df[(df.index >= start) & (df.index <= end)]
            if len(total_in_event) > 0:
                print(f"  [{name}] 이 기간 {len(total_in_event)}시간 중 {len(in_event)}건이 이상치로 잡힘 "
                      f"({len(in_event)/len(total_in_event)*100:.1f}%)")

        print("\n이상치 상위 10건 (가장 이례적인 순):")
        top = df[df["is_outlier"]].copy()
        top["anomaly_score"] = iso.decision_function(Xf.loc[top.index])
        top = top.sort_values("anomaly_score").head(10)
        print(top[["ws", "temp_c", "rh", "sp", "output_ratio"]].round(2).to_string())


if __name__ == "__main__":
    main()
