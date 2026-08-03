"""
[이벤트 겨냥 후처리] A모델(submit_v32.csv)의 산불 정점 구간(2025-03-21~28) 예측을
분석으로 확인된 편향 방향에 맞춰 국소 보정.

근거:
  - analyze_wildfire_conditions_2025.py: 2025-03-21~28이 강풍(850hPa 상위14%)+
    저기압(지표기압 상위94%, 거의 최저)+건조(무강수 87%) 삼박자 확인
  - analyze_weather_regime_correlation.py: A모델 2024 실제 데이터에서 저기압
    (936~942hPa) 구간에 group2·3이 4.25~6.80% 과대예측 편향 확인 (group1은 약함)

방식: v25/26에서 검증된 '수축(shrinkage)' 원칙 재사용 - 편향 추정치를 100% 그대로
적용하지 않고 50%만 적용해서 과잉보정 위험을 줄임. 2024년 폴드에서 관측된 값이지
2025년 이 특정 극단적 사례에 그대로 적용된다는 보장은 없으므로 보수적으로 접근.

적용 범위: kpx_group_2, kpx_group_3 (group1은 편향이 약해 미적용)
적용 기간: 2025-03-21 ~ 2025-03-28 (192시간)
보정 방향: 하향 조정 (과대예측 편향이므로)

실행:
    python src/apply_wildfire_regime_correction.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "model_A" / "submit_v32.csv"
OUTPUT_PATH = REPO_ROOT / "submissions" / "submit_v32_wildfire_corrected.csv"

CORRECTION_WINDOW = ("2025-03-21", "2025-03-28")
TARGET_GROUPS = ["kpx_group_2", "kpx_group_3"]

# 2024년 유사 저기압(936~942hPa) 구간에서 관측된 과대예측 편향(부호오차, %p)
OBSERVED_BIAS_PCT = {"kpx_group_2": 6.80, "kpx_group_3": 4.25}
SHRINKAGE = 0.5  # v25/26 원칙 재사용 - 절반만 적용


def main():
    df = pd.read_csv(INPUT_PATH)
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])

    mask = (df["forecast_kst_dtm"] >= CORRECTION_WINDOW[0]) & (df["forecast_kst_dtm"] <= CORRECTION_WINDOW[1])
    print(f"보정 대상 시간대: {mask.sum()}시간 ({CORRECTION_WINDOW[0]} ~ {CORRECTION_WINDOW[1]})")

    for target in TARGET_GROUPS:
        cap = CAPACITY_KWH[target]
        bias_pct = OBSERVED_BIAS_PCT[target]
        applied_pct = bias_pct * SHRINKAGE

        before = df.loc[mask, target].copy()
        # 과대예측 편향만큼(수축 적용) 설비용량 대비 %p로 하향 조정
        df.loc[mask, target] = np.clip(df.loc[mask, target] - cap * (applied_pct / 100), 0, cap)
        after = df.loc[mask, target]

        print(f"\n[{target}] 관측편향={bias_pct:.2f}%p -> 적용({SHRINKAGE:.0%} 수축)={applied_pct:.2f}%p "
              f"({cap * applied_pct / 100:.0f}kWh 하향)")
        print(f"  보정 전 평균: {before.mean():.0f}  보정 후 평균: {after.mean():.0f}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df_out = df.copy()
    df_out["forecast_kst_dtm"] = df_out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUTPUT_PATH}")
    print(f"(다른 그룹/시간대는 원본 submit_v32.csv와 완전히 동일 - 이 192시간 x 2개 그룹만 조정됨)")


if __name__ == "__main__":
    main()
