"""
A모델의 개별 구성요소(LightGBM/CatBoost/XGBoost/최종블렌딩) 각각과
B모델(동훈) 예측을 비교하는 상세 버전.

compare_with_model_A.py(최종 블렌딩끼리 비교)의 확장판 - A의 어느 구성요소가
B와 가장/가장 덜 비슷한지까지 확인해서, 블렌딩에 어떤 조합이 유망할지 힌트를 얻는다.

여전히 2025 test 기간(정답 모름) 비교라 '누가 더 맞았는지'는 알 수 없음 - 상관관계/
다양성 확인용.

실행:
    python src/compare_with_model_A_detailed.py <B모델_제출파일_경로>
"""

import sys
from pathlib import Path
import pandas as pd

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_A_FILES = {
    "A_lgb": REPO_ROOT / "model_A" / "output" / "pred_A_lgb.csv",
    "A_cat": REPO_ROOT / "model_A" / "output" / "pred_A_cat.csv",
    "A_xgb": REPO_ROOT / "model_A" / "output" / "pred_A_xgb.csv",
    "A_blend": REPO_ROOT / "model_A" / "output" / "submit_A.csv",
}


def main():
    if len(sys.argv) < 2:
        print("사용법: python src/compare_with_model_A_detailed.py <B모델_제출파일_경로>")
        sys.exit(1)

    b_path = Path(sys.argv[1])
    b_df = pd.read_csv(b_path)
    b_df["forecast_kst_dtm"] = pd.to_datetime(b_df["forecast_kst_dtm"])

    a_dfs = {}
    for name, path in MODEL_A_FILES.items():
        df = pd.read_csv(path)
        df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
        a_dfs[name] = df

    print("=" * 70)
    print("A모델 구성요소별 vs B모델 상관관계")
    print("=" * 70)
    print(f"{'구성요소':<10}" + "".join(f"{t:>16}" for t in TARGET_COLS))
    for name, a_df in a_dfs.items():
        merged = a_df.merge(b_df, on="forecast_kst_dtm", suffixes=("_A", "_B"))
        corrs = [merged[f"{t}_A"].corr(merged[f"{t}_B"]) for t in TARGET_COLS]
        print(f"{name:<10}" + "".join(f"{c:>16.4f}" for c in corrs))

    print("\n" + "=" * 70)
    print("A모델 구성요소 간 상관관계 (참고 - A 내부 다양성 확인)")
    print("=" * 70)
    names = list(a_dfs.keys())
    print(f"{'':<10}" + "".join(f"{n:>12}" for n in names))
    for n1 in names:
        row = [f"{n1:<10}"]
        for n2 in names:
            merged = a_dfs[n1].merge(a_dfs[n2], on="forecast_kst_dtm", suffixes=("_1", "_2"))
            avg_corr = sum(merged[f"{t}_1"].corr(merged[f"{t}_2"]) for t in TARGET_COLS) / 3
            row.append(f"{avg_corr:>12.4f}")
        print("".join(row))

    print("\n" + "=" * 70)
    print("평균 예측값 비교 (그룹별)")
    print("=" * 70)
    print(f"{'구성요소':<10}" + "".join(f"{t:>16}" for t in TARGET_COLS))
    for name, a_df in a_dfs.items():
        means = [a_df[t].mean() for t in TARGET_COLS]
        print(f"{name:<10}" + "".join(f"{m:>16.1f}" for m in means))
    means_b = [b_df[t].mean() for t in TARGET_COLS]
    print(f"{'B(동훈)':<10}" + "".join(f"{m:>16.1f}" for m in means_b))

    print("\n" + "=" * 70)
    print("해석 가이드")
    print("=" * 70)
    print("- B와 상관이 가장 낮은 A 구성요소가 블렌딩 다양성 관점에서 가장 유망함")
    print("- 단, A 내부 ablation에서 이미 'Cat+XGB 다양성 기여 거의 0'이라고 나왔으므로")
    print("  A 구성요소 중 뭘 고르든 큰 차이 없을 가능성 있음 (참고 표에서 A 내부 상관도 확인)")
    print("- 진짜 정확도 비교는 라벨 있는 fold 데이터가 와야 가능")


if __name__ == "__main__":
    main()
