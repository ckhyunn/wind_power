"""
[재학습 없음] 이상치 필터 문턱값(풍속>=8m/s AND 발전량<X%)이 실제로 몇 시간이나
걸러내는지, 그리고 그게 진짜 착빙/이상 사례처럼 보이는지 즉시 확인.
model_A_dh.py의 CLEAN_DIR/DATA_DIR 그대로 재사용, 학습 없이 원본 데이터만 읽음.

실행:
    python quick_check_outlier_threshold.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\doon2\Desktop\wind_power\data")  # model_A_dh.py와 동일하게 맞춰서 실행 전 확인 필요
CLEAN_DIR = Path(__file__).resolve().parent / "cleaned"

CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}
WS_THRESHOLD = 8.0
OUTPUT_THRESHOLDS = [0.10, 0.20, 0.30, 0.40]


def main():
    labels = pd.read_csv(CLEAN_DIR / "train_labels_clean.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")

    gfs = pd.read_csv(DATA_DIR / "train/gfs_train.csv", encoding="utf-8-sig")
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])
    agg = gfs.groupby("forecast_kst_dtm").agg(
        u=("heightAboveGround_100_100u", "mean"), v=("heightAboveGround_100_100v", "mean"),
        temp_k=("heightAboveGround_2_2t", "mean"), rh=("heightAboveGround_2_2r", "mean"),
    )
    agg["ws"] = np.sqrt(agg["u"] ** 2 + agg["v"] ** 2)
    agg["temp_c"] = agg["temp_k"] - 273.15

    for g, cap in CAP.items():
        y = labels[g].dropna()
        df = agg.join(y.rename("y"), how="inner")
        windy = df[df["ws"] >= WS_THRESHOLD]

        print(f"\n{'='*70}\n[{g}] 풍속>={WS_THRESHOLD}m/s 시간대 총 {len(windy)}건\n{'='*70}")
        for th in OUTPUT_THRESHOLDS:
            caught = windy[windy["y"] < cap * th]
            below10 = windy[windy["y"] < cap * 0.10]
            new_catch = len(caught) - len(below10)  # 10%컷과 겹치지 않는 순수 추가분
            print(f"  발전량<{th*100:.0f}%: 총 {len(caught)}건 (10%컷 이미제외분 {len(below10)}건, 순수추가분 {new_catch}건)")

        # 10~30% 구간(새로 걸러질 후보)의 실제 기온/습도 프로파일
        mid = windy[(windy["y"] >= cap*0.10) & (windy["y"] < cap*0.30)]
        if len(mid) > 0:
            print(f"\n  10~30%구간(순수추가분) 프로파일: n={len(mid)}")
            print(f"    평균기온={mid['temp_c'].mean():.1f}°C  평균습도={mid['rh'].mean():.1f}%  평균풍속={mid['ws'].mean():.1f}m/s")
            print(f"    기온<=5도 비율: {(mid['temp_c']<=5).mean()*100:.1f}%  (착빙 의심 신호)")


if __name__ == "__main__":
    main()
