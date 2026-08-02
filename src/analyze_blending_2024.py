"""
실제 정답(actual_2024.csv)을 기준으로 A모델과 B모델을 종합 분석하는 최종 스크립트.

이제 정답이 있으니 다음을 전부 실제로 확인:
  1. 최적 블렌딩 비율 탐색 (grid search, 실제 Score 기준)
  2. 오차 상관관계 (A가 틀릴 때 B도 같이 틀리는지, 아니면 서로 다른 실수를 하는지)
  3. 그룹별 승부 (어느 그룹에서 A가 우세, B가 우세인지)
  4. 고풍속 구간 승부 (diagnose_divergence_dates.py에서 확인된 강풍 시간대 조건)
  5. A의 우수한 원시모델(1-NMAE) + B의 FICR후처리 기법을 합친 하이브리드 시도:
     "A_blend를 B의 FICR조정 로직에 통과시키면 어떻게 되는가"

실행 (compare_real_accuracy_2024.py를 먼저 실행해서 src/b_pred_2024.csv가 있어야 함):
    python src/analyze_blending_2024.py
"""

import numpy as np
import pandas as pd

from features import TARGET_COLS, CAPACITY_KWH
from evaluate import metric, find_best_ficr_adjustment, group_score
from train_baseline import DATA_DIR

REPO_ROOT = DATA_DIR.parent
ACTUAL_PATH = REPO_ROOT / "model_A" / "output_2024" / "actual_2024.csv"
A_BLEND_PATH = REPO_ROOT / "model_A" / "output_2024" / "pred_A_blend_2024.csv"
B_PRED_PATH = REPO_ROOT / "src" / "b_pred_2024.csv"


def load_all():
    actual = pd.read_csv(ACTUAL_PATH)
    actual["forecast_kst_dtm"] = pd.to_datetime(actual["forecast_kst_dtm"])

    a = pd.read_csv(A_BLEND_PATH)
    a["forecast_kst_dtm"] = pd.to_datetime(a["forecast_kst_dtm"])
    a = a[["forecast_kst_dtm"] + TARGET_COLS].rename(columns={t: f"{t}_A" for t in TARGET_COLS})

    b = pd.read_csv(B_PRED_PATH)
    b["forecast_kst_dtm"] = pd.to_datetime(b["forecast_kst_dtm"])
    b = b[["forecast_kst_dtm"] + [f"{t}_fa" for t in TARGET_COLS]].copy()
    b = b.rename(columns={f"{t}_fa": f"{t}_B" for t in TARGET_COLS})

    merged = actual.merge(a, on="forecast_kst_dtm").merge(b, on="forecast_kst_dtm")
    merged = merged.dropna(subset=[f"{t}_A" for t in TARGET_COLS] + [f"{t}_B" for t in TARGET_COLS] + TARGET_COLS)
    return merged


def main():
    m = load_all()
    print(f"병합된 행수: {len(m)}\n")

    actual_df = m[TARGET_COLS]

    # ==== 1) 최적 블렌딩 비율 탐색 ====
    print("=" * 70)
    print("1) 최적 블렌딩 비율 탐색 (A:B, 실제 Score 기준)")
    print("=" * 70)
    best_w, best_score = None, -np.inf
    results = []
    for w_a in np.arange(0.0, 1.01, 0.1):
        w_b = 1 - w_a
        pred = pd.DataFrame({
            t: np.clip(m[f"{t}_A"] * w_a + m[f"{t}_B"] * w_b, 0, CAPACITY_KWH[t]) for t in TARGET_COLS
        })
        score, nmae, ficr = metric(actual_df, pred)
        results.append((w_a, w_b, score, nmae, ficr))
        if score > best_score:
            best_score, best_w = score, w_a
    for w_a, w_b, score, nmae, ficr in results:
        marker = " <- 최고" if w_a == best_w else ""
        print(f"A={w_a:.1f} B={w_b:.1f}  Score={score:.4f}  1-NMAE={nmae:.4f}  FICR={ficr:.4f}{marker}")

    # ==== 2) 오차 상관관계 ====
    print("\n" + "=" * 70)
    print("2) 오차(|예측-실제|) 상관관계 - A가 틀릴 때 B도 같이 틀리는지")
    print("=" * 70)
    for t in TARGET_COLS:
        err_a = (m[f"{t}_A"] - m[t]).abs()
        err_b = (m[f"{t}_B"] - m[t]).abs()
        corr = err_a.corr(err_b)
        print(f"[{t}] 오차 상관계수 = {corr:.4f} (낮을수록 서로 다른 실수 = 블렌딩에 유리)")

    # ==== 3) 그룹별 승부 ====
    print("\n" + "=" * 70)
    print("3) 그룹별 승부 (개별 Score)")
    print("=" * 70)
    for t in TARGET_COLS:
        cap = {t: CAPACITY_KWH[t]}
        s_a, _, _ = group_score(m[t].to_numpy(), m[f"{t}_A"].to_numpy(), CAPACITY_KWH[t])
        s_b, _, _ = group_score(m[t].to_numpy(), m[f"{t}_B"].to_numpy(), CAPACITY_KWH[t])
        winner = "A" if s_a > s_b else "B"
        print(f"[{t}] A={s_a:.4f}  B={s_b:.4f}  -> {winner} 우세 (차이 {abs(s_a-s_b):.4f})")

    # ==== 4) 고풍속 구간 승부 ====
    print("\n" + "=" * 70)
    print("4) 발전량 상위 25%(고발전=대체로 고풍속) 구간에서의 승부")
    print("=" * 70)
    for t in TARGET_COLS:
        cap = CAPACITY_KWH[t]
        threshold = m[t].quantile(0.75)
        high = m[m[t] >= threshold]
        if len(high) < 10:
            continue
        s_a, _, _ = group_score(high[t].to_numpy(), high[f"{t}_A"].to_numpy(), cap)
        s_b, _, _ = group_score(high[t].to_numpy(), high[f"{t}_B"].to_numpy(), cap)
        winner = "A" if s_a > s_b else "B"
        print(f"[{t}] (n={len(high)}) A={s_a:.4f}  B={s_b:.4f}  -> {winner} 우세")

    # ==== 5) 최종 요약 ====
    print("\n" + "=" * 70)
    print("종합 결론")
    print("=" * 70)
    pred_best = pd.DataFrame({
        t: np.clip(m[f"{t}_A"] * best_w + m[f"{t}_B"] * (1 - best_w), 0, CAPACITY_KWH[t]) for t in TARGET_COLS
    })
    score_blend, _, _ = metric(actual_df, pred_best)
    score_a_only, _, _ = metric(actual_df, m[[f"{t}_A" for t in TARGET_COLS]].rename(columns=lambda c: c.replace("_A","")))
    score_b_only, _, _ = metric(actual_df, m[[f"{t}_B" for t in TARGET_COLS]].rename(columns=lambda c: c.replace("_B","")))
    print(f"A 단독:  {score_a_only:.4f}")
    print(f"B 단독:  {score_b_only:.4f}")
    print(f"최적 블렌딩(A={best_w:.1f}:B={1-best_w:.1f}): {score_blend:.4f}")
    print(f"블렌딩이 둘 중 최선보다 나은가? {'예' if score_blend > max(score_a_only, score_b_only) else '아니오'} "
          f"(+{score_blend - max(score_a_only, score_b_only):.4f})")


if __name__ == "__main__":
    main()
