"""
LDAPS/GFS 예보 원본 데이터 이상치 진단 스크립트.

SCADA는 이상치를 찾아서 걷어내고 큰 개선 근거(diagnose_vestas.py)를 얻었는데,
정작 모델의 핵심 입력인 LDAPS/GFS 예보 원본 데이터는 한 번도 이상치 점검을 안 함.
풍속 컬럼 위주로 물리적으로 말이 안 되는 값(음수 풍속 크기, 비현실적으로 큰 값),
결측치 비율, 격자별/시간대별 이상치 분포를 확인한다.

실행:
    python src/diagnose_forecast_data.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# 물리적으로 타당한 범위 (참고치 - 지상 10m/100m 풍속 기준)
# 태풍급 순간풍속도 보통 60m/s를 넘지 않음. 이보다 크면 데이터 오류로 간주.
WIND_COMPONENT_MAX = 60.0


def check_file(path: Path, wind_pairs: list):
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"\n{'='*80}")
    print(f"[{path.name}] 행수={len(df)}, 기간={df['forecast_kst_dtm'].min()} ~ {df['forecast_kst_dtm'].max()}")
    print(f"{'='*80}")

    print("\n-- 결측치 비율 (0% 초과만 표시) --")
    na_ratio = (df.isna().sum() / len(df) * 100).round(3)
    na_ratio = na_ratio[na_ratio > 0]
    if len(na_ratio) == 0:
        print("결측치 없음")
    else:
        print(na_ratio.to_string())

    print("\n-- 풍속 성분(U/V) 이상치 점검 --")
    for u_col, v_col in wind_pairs:
        if u_col not in df.columns or v_col not in df.columns:
            continue
        speed = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)
        n_extreme = (speed > WIND_COMPONENT_MAX).sum()
        n_zero_exact = ((df[u_col] == 0) & (df[v_col] == 0)).sum()
        print(f"  [{u_col} / {v_col}]")
        print(f"    풍속 범위: {speed.min():.2f} ~ {speed.max():.2f} m/s")
        print(f"    {WIND_COMPONENT_MAX}m/s 초과(물리적 이상치 의심): {n_extreme}개 ({n_extreme/len(df)*100:.4f}%)")
        print(f"    U=V=0 정확히 0인 경우(센서 결측을 0으로 채웠을 가능성): {n_zero_exact}개 ({n_zero_exact/len(df)*100:.4f}%)")

    print("\n-- 격자(grid_id)별 결측/이상 비율 편차 확인 --")
    if "grid_id" in df.columns:
        u_col, v_col = wind_pairs[0]
        if u_col in df.columns:
            per_grid_na = df.groupby("grid_id")[u_col].apply(lambda s: s.isna().mean() * 100)
            print(per_grid_na.round(3).to_string())


def main():
    ldaps_wind_pairs = [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    ]
    gfs_wind_pairs = [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
        ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
    ]

    check_file(TRAIN_DIR / "ldaps_train.csv", ldaps_wind_pairs)
    check_file(TRAIN_DIR / "gfs_train.csv", gfs_wind_pairs)
    check_file(TEST_DIR / "ldaps_test.csv", ldaps_wind_pairs)
    check_file(TEST_DIR / "gfs_test.csv", gfs_wind_pairs)


if __name__ == "__main__":
    main()
