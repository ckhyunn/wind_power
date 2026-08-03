"""
[담당: 잔차 시계열 forensics]
A모델(현윤) 실제 2024년 예측 vs 실제 라벨로 잔차(오차)를 계산해서,
큰 오차가 특정 날짜에 몰리는지, 주기성(다른 대회 팀의 70일 정비 사례처럼)이
있는지 전수조사한다.

특히 group3(kpx_group_3)에 집중 - 담당 지시사항 반영.

확인 순서:
  1. 그룹별 오차율(|예측-실제|/설비용량) 전체 분포
  2. 오차율 상위 N% 시간대를 날짜별로 집계 - 특정 날짜에 몰리는지
  3. 상위 오차 날짜들의 시작일 간격을 계산해서 주기성(예: 70일) 있는지 확인
  4. 월별 오차율 분포 - 계절 패턴 확인
  5. group3 vs group1/2 비교 - group3만의 특이 패턴이 있는지

실행:
    python src/analyze_A_residual_forensics.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTUAL_PATH = REPO_ROOT / "model_A" / "output_2024" / "actual_2024.csv"
PRED_PATH = REPO_ROOT / "model_A" / "output_2024" / "pred_A_blend_2024.csv"

# 평가 기준(10% 미만 시간대 제외)과 동일한 원칙 적용
MIN_OUTPUT_RATIO = 0.10
TOP_N_ERROR_PCT = 0.10  # 오차율 상위 10%를 '큰 오차'로 정의


def load_merged():
    actual = pd.read_csv(ACTUAL_PATH)
    pred = pd.read_csv(PRED_PATH)
    actual["forecast_kst_dtm"] = pd.to_datetime(actual["forecast_kst_dtm"])
    pred["forecast_kst_dtm"] = pd.to_datetime(pred["forecast_kst_dtm"])
    merged = actual.merge(pred, on="forecast_kst_dtm", suffixes=("_actual", "_pred"))
    return merged.dropna(subset=[f"{t}_actual" for t in TARGET_COLS])


def main():
    m = load_merged()
    print(f"병합된 행수: {len(m)} ({m['forecast_kst_dtm'].min()} ~ {m['forecast_kst_dtm'].max()})\n")

    for target in TARGET_COLS:
        cap = CAPACITY_KWH[target]
        sub = m[[f"forecast_kst_dtm", f"{target}_actual", f"{target}_pred"]].copy()
        sub.columns = ["dt", "actual", "pred"]

        # 평가지표와 동일하게 10% 미만 시간대 제외
        valid = sub[sub["actual"] >= cap * MIN_OUTPUT_RATIO].copy()
        valid["error_rate"] = (valid["pred"] - valid["actual"]).abs() / cap
        valid["signed_error_rate"] = (valid["pred"] - valid["actual"]) / cap

        print(f"\n{'='*80}\n[{target}] 평가대상 {len(valid)}시간, 평균오차율={valid['error_rate'].mean()*100:.2f}%\n{'='*80}")

        # === 1) 오차율 상위 10% 시간대 ===
        n_top = max(int(len(valid) * TOP_N_ERROR_PCT), 1)
        top_error = valid.nlargest(n_top, "error_rate").copy()
        print(f"\n오차율 상위 {TOP_N_ERROR_PCT*100:.0f}%({n_top}시간) 기준:")
        print(f"  오차율 범위: {top_error['error_rate'].min()*100:.1f}% ~ {top_error['error_rate'].max()*100:.1f}%")
        print(f"  과대예측(양수) 비율: {(top_error['signed_error_rate']>0).mean()*100:.1f}%")

        # === 2) 날짜별 집계 - 몰림 확인 ===
        top_error["date"] = top_error["dt"].dt.date
        by_date = top_error.groupby("date").size().sort_values(ascending=False)
        print(f"\n  큰오차 시간대가 가장 많이 몰린 날짜 top 15:")
        print(f"  {by_date.head(15).to_string()}")

        # 하루 20시간 이상 큰오차인 '이상일' 식별 (거의 하루 종일 크게 틀린 날)
        bad_days = by_date[by_date >= 15].index.tolist()
        print(f"\n  하루 15시간+ 큰오차인 '이상일' 개수: {len(bad_days)}")
        if bad_days:
            bad_days_sorted = sorted(bad_days)
            print(f"  이상일 목록: {bad_days_sorted}")

            # === 3) 주기성 확인 ===
            if len(bad_days_sorted) >= 3:
                gaps = [(bad_days_sorted[i+1] - bad_days_sorted[i]).days for i in range(len(bad_days_sorted)-1)]
                print(f"  이상일 간 간격(일): {gaps}")

        # === 4) 월별 분포 ===
        valid["month"] = valid["dt"].dt.month
        monthly_error = valid.groupby("month")["error_rate"].mean() * 100
        monthly_bigerror_rate = valid.groupby("month").apply(
            lambda g: (g["error_rate"] >= valid["error_rate"].quantile(1 - TOP_N_ERROR_PCT)).mean() * 100
        )
        print(f"\n  월별 평균오차율(%):")
        print(f"  {monthly_error.round(2).to_string()}")
        print(f"\n  월별 '상위10% 큰오차' 시간대 비율(%) - 월별로 고르면 10% 근처, 몰리면 특정월 급증:")
        print(f"  {monthly_bigerror_rate.round(1).to_string()}")


if __name__ == "__main__":
    main()
