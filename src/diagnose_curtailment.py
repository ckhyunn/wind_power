"""
'출력제어(curtailment)' 흔적 진단 스크립트.

가설: 계통 안정성 등의 이유로 바람이 충분한데도 발전량이 인위적으로 제한되는
시간대가 있다면, 날씨 기반 예측으로는 원천적으로 못 맞히는 '설명 불가능한 오차'가
섞여있는 것이고 이게 FICR 정체의 숨은 원인일 수 있음.

확인 방법: 예보 풍속이 충분히 높아서(파워커브 상 발전량이 높아야 정상인) 구간인데도
실제 발전량이 뚜렷하게 낮게 '눌려있는' 시간대가 있는지 찾는다.
(컷아웃과 구분: 컷아웃은 극단적 고풍속에서만 발생. 이 진단은 중간 풍속대에서도
출력이 눌려있는 패턴을 찾음 - 컷아웃으로는 설명 안 되는 억제)

실행:
    python src/diagnose_curtailment.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train"

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}


def main():
    labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])

    for target in TARGET_COLS:
        cap = CAPACITY_KWH[target]
        y = labels[target].dropna()
        cf = y / cap  # capacity factor (이용률)

        print(f"\n{'='*70}")
        print(f"[{target}] 설비이용률(CF) 분포")
        print(f"{'='*70}")
        print(cf.describe().round(3).to_string())

        # 의심 패턴 1: 정확히 같은 값이 3시간 이상 연속되는 '평평한 구간'
        # (정상적인 바람 변동이라면 이렇게 오래 완전히 똑같기 어려움 - 인위적 캡 의심)
        vals = y.values
        same_run = 1
        flat_runs = []
        for i in range(1, len(vals)):
            if vals[i] == vals[i-1] and vals[i] > cap * 0.05:  # 너무 낮은 값(정지)은 제외
                same_run += 1
            else:
                if same_run >= 3:
                    flat_runs.append(same_run)
                same_run = 1
        print(f"\n동일값 3시간+ 연속(평평한 구간) 발생 횟수: {len(flat_runs)}")
        if flat_runs:
            print(f"  최장 연속: {max(flat_runs)}시간, 평균: {np.mean(flat_runs):.1f}시간")

        # 의심 패턴 2: 설비이용률이 '정확히' 특정 라운드넘버(예: 정격의 정확히 50%, 80%)에서
        # 비정상적으로 많이 몰려있는지 (인위적 출력제한 지시값의 흔적)
        cf_rounded = (cf * 100).round(0)
        common = cf_rounded.value_counts().head(5)
        print(f"\n가장 빈번한 CF(%) 값 top5 (특정 값에 비정상적으로 몰려있으면 의심):")
        print(common.to_string())


if __name__ == "__main__":
    main()
