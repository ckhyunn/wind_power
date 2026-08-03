"""
'오차(NMAE)를 줄이면 FICR도 같이 오르는가'를 실제 A모델 2024 데이터로 검증.

방법: 모든 시간대의 오차를 균일하게 X%씩 줄였다고 가정(시뮬레이션)했을 때,
1-NMAE와 FICR이 각각 얼마나 개선되는지 비교. 균일하게 비례해서 오르면
"오차 줄이기 = FICR 올리기"가 성립하지만, 그렇지 않다면(계단효과) 정확히
어느 오차율 구간을 줄여야 FICR에 leverage가 있는지 확인.

실행:
    python src/analyze_nmae_vs_ficr.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

from features import TARGET_COLS, CAPACITY_KWH
from evaluate import metric

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTUAL_PATH = REPO_ROOT / "model_A" / "output_2024" / "actual_2024.csv"
PRED_PATH = REPO_ROOT / "model_A" / "output_2024" / "pred_A_blend_2024.csv"


def load_merged():
    actual = pd.read_csv(ACTUAL_PATH)
    pred = pd.read_csv(PRED_PATH)
    actual["forecast_kst_dtm"] = pd.to_datetime(actual["forecast_kst_dtm"])
    pred["forecast_kst_dtm"] = pd.to_datetime(pred["forecast_kst_dtm"])
    m = actual.merge(pred, on="forecast_kst_dtm", suffixes=("_actual", "_pred"))
    return m.dropna(subset=[f"{t}_actual" for t in TARGET_COLS])


def main():
    m = load_merged()
    actual_df = m[[f"{t}_actual" for t in TARGET_COLS]].rename(columns=lambda c: c.replace("_actual", ""))
    pred_df = m[[f"{t}_pred" for t in TARGET_COLS]].rename(columns=lambda c: c.replace("_pred", ""))

    base_score, base_nmae, base_ficr = metric(actual_df, pred_df)
    print(f"기준(A모델 실제 2024): Score={base_score:.4f}  1-NMAE={base_nmae:.4f}  FICR={base_ficr:.4f}\n")

    print("=" * 80)
    print("실험 1: 모든 오차를 균일하게 X%씩 줄이면 (예측을 실제값 쪽으로 X%만큼 당김)")
    print("=" * 80)
    print(f"{'오차감소율':>10}{'1-NMAE':>12}{'1-NMAE변화':>14}{'FICR':>10}{'FICR변화':>12}{'변화비율(FICR/NMAE)':>20}")
    for shrink in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]:
        adjusted = {}
        for t in TARGET_COLS:
            adjusted[t] = np.clip(
                actual_df[t] + (pred_df[t] - actual_df[t]) * (1 - shrink), 0, CAPACITY_KWH[t]
            )
        adj_df = pd.DataFrame(adjusted)
        score, nmae, ficr = metric(actual_df, adj_df)
        d_nmae = nmae - base_nmae
        d_ficr = ficr - base_ficr
        ratio = d_ficr / d_nmae if d_nmae != 0 else float("nan")
        print(f"{shrink*100:>9.0f}%{nmae:>12.4f}{d_nmae:>+14.4f}{ficr:>10.4f}{d_ficr:>+12.4f}{ratio:>20.2f}")

    print("\n(비율이 1.0보다 훨씬 크면 FICR이 오차감소에 더 민감하게 반응한다는 뜻,")
    print(" 1.0보다 작으면 오차를 줄여도 FICR은 상대적으로 둔감하다는 뜻)")

    print("\n" + "=" * 80)
    print("실험 2: 오차율 분포 - '문턱 근처(6~10%)'에 얼마나 많은 시간대가 몰려있는가")
    print("=" * 80)
    print("(이 구간에 있는 시간대를 줄이면 FICR에 즉각 leverage가 있음)")
    for target in TARGET_COLS:
        cap = CAPACITY_KWH[target]
        actual = m[f"{target}_actual"]
        pred = m[f"{target}_pred"]
        valid = actual >= cap * 0.10
        error_rate = (pred[valid] - actual[valid]).abs() / cap

        print(f"\n[{target}] (평가대상 {valid.sum()}시간)")
        bins = [0, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 1.0]
        labels = ["<=6%(만점)", "6~8%(3점)", "8~10%(0점,문턱바로위)", "10~15%(0점)", "15~20%(0점)", "20~30%(0점)", "30%+(0점)"]
        counts = pd.cut(error_rate, bins=bins, labels=labels).value_counts().reindex(labels)
        pct = (counts / len(error_rate) * 100).round(1)
        for label, c, p in zip(labels, counts, pct):
            print(f"  {label:<22}: {c:>5}시간 ({p:>5.1f}%)")
        # 8~10% 구간(문턱 바로 위, 조금만 줄이면 3점권 진입 가능)의 비중을 강조
        near_threshold = ((error_rate > 0.08) & (error_rate <= 0.10)).sum()
        print(f"  -> '8~10%(문턱 바로 위)' 시간대가 {near_threshold}건 - 이게 가장 손쉬운 FICR 개선 타깃")


if __name__ == "__main__":
    main()
