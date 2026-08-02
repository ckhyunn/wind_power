"""
B모델(동훈)과 A모델(현윤, submit_v32.csv)의 예측을 비교하는 스크립트.

주의: 둘 다 2025년 test 기간의 '예측값'이라 실제 정답을 모름.
따라서 여기서 확인 가능한 건 "누가 더 맞았나"가 아니라:
  1. 두 모델 예측의 상관관계 (다양성이 있는지)
  2. 그룹별로 얼마나 다르게 예측하는지
  3. 시간대/계절별로 차이가 큰 구간이 있는지
  4. 간단한 블렌딩 미리보기
"누가 더 맞았나"는 실제 라벨이 있는 backtest 구간에서 A모델도 예측을 내야 확인 가능함.

실행:
    python src/compare_with_model_A.py <B모델_제출파일_경로>

예:
    python src/compare_with_model_A.py submissions/baseline_v25_submit.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_A_PATH = REPO_ROOT / "model_A" / "submit_v32.csv"


def main():
    if len(sys.argv) < 2:
        print("사용법: python src/compare_with_model_A.py <B모델_제출파일_경로>")
        sys.exit(1)

    b_path = Path(sys.argv[1])
    a_df = pd.read_csv(MODEL_A_PATH)
    b_df = pd.read_csv(b_path)

    a_df["forecast_kst_dtm"] = pd.to_datetime(a_df["forecast_kst_dtm"])
    b_df["forecast_kst_dtm"] = pd.to_datetime(b_df["forecast_kst_dtm"])

    merged = a_df.merge(b_df, on="forecast_kst_dtm", suffixes=("_A", "_B"))
    print(f"병합된 행 수: {len(merged)} (A={len(a_df)}, B={len(b_df)})")
    if len(merged) != len(a_df) or len(merged) != len(b_df):
        print("⚠️ 경고: 행 수가 안 맞습니다 - 시간 범위나 형식을 확인하세요.")

    print("\n" + "=" * 70)
    print("1) 그룹별 상관관계 (A예측 vs B예측)")
    print("=" * 70)
    for target in TARGET_COLS:
        a_col, b_col = f"{target}_A", f"{target}_B"
        corr = merged[a_col].corr(merged[b_col])
        print(f"[{target}] 상관계수 = {corr:.4f}")

    print("\n" + "=" * 70)
    print("2) 그룹별 예측값 차이 (|A-B|, 설비용량 대비 %)")
    print("=" * 70)
    for target in TARGET_COLS:
        a_col, b_col = f"{target}_A", f"{target}_B"
        cap = CAPACITY_KWH[target]
        diff_pct = (merged[a_col] - merged[b_col]).abs() / cap * 100
        print(f"[{target}] 평균 차이={diff_pct.mean():.2f}%  "
              f"중앙값={diff_pct.median():.2f}%  최대={diff_pct.max():.2f}%")
        print(f"          차이가 10%p 넘는 시간대: {(diff_pct > 10).sum()}건 "
              f"({(diff_pct > 10).mean()*100:.1f}%)")

    # diff_pct 컬럼을 먼저 전부 계산 (3번, 4번에서 공통으로 사용)
    for target in TARGET_COLS:
        a_col, b_col = f"{target}_A", f"{target}_B"
        cap = CAPACITY_KWH[target]
        merged[f"{target}_diff_pct"] = (merged[a_col] - merged[b_col]).abs() / cap * 100

    print("\n" + "=" * 70)
    print("3) 차이가 가장 큰 시간대 top 30 (그룹별, 날짜순 정렬 - 특정 기간 쏠림 확인용)")
    print("=" * 70)
    for target in TARGET_COLS:
        a_col, b_col = f"{target}_A", f"{target}_B"
        diff_col = f"{target}_diff_pct"
        print(f"\n--- [{target}] ---")
        top30 = merged.nlargest(30, diff_col)[["forecast_kst_dtm", a_col, b_col, diff_col]]
        top30 = top30.sort_values("forecast_kst_dtm")
        print(top30.to_string(index=False))

    print("\n" + "=" * 70)
    print("4) 월별 평균 차이 (계절에 따라 다양성이 달라지는지)")
    print("=" * 70)
    merged["month"] = merged["forecast_kst_dtm"].dt.month
    monthly = merged.groupby("month")[[f"{t}_diff_pct" for t in TARGET_COLS]].mean()
    print(monthly.round(2).to_string())

    print("\n" + "=" * 70)
    print("5) 간단 블렌딩 미리보기 (50:50, 두 예측의 평균)")
    print("=" * 70)
    for target in TARGET_COLS:
        a_col, b_col = f"{target}_A", f"{target}_B"
        cap = CAPACITY_KWH[target]
        blended = np.clip((merged[a_col] + merged[b_col]) / 2, 0, cap)
        print(f"[{target}] 블렌딩 평균값={blended.mean():.1f} (A평균={merged[a_col].mean():.1f}, B평균={merged[b_col].mean():.1f})")

    print("\n" + "=" * 70)
    print("해석 가이드")
    print("=" * 70)
    print("- 상관계수가 0.9 이상이면: 두 모델이 매우 비슷하게 예측 -> 블렌딩 효과 제한적일 가능성")
    print("- 상관계수가 0.7~0.9면: 어느 정도 다양성 있음 -> 블렌딩 시도해볼 가치 있음")
    print("- 차이 큰 시간대가 특정 월/계절에 몰려있다면: 그 구간에서 다양성이 크다는 신호")
    print("- '누가 더 맞았는지'는 이 스크립트로 알 수 없음 (실제 정답 모름) - 리더보드로 최종 확인 필요")


if __name__ == "__main__":
    main()
