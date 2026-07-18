"""
대회 공식 평가 산식 구현 (codeshare 14035 "평가 산식 코드" 그대로 이식).

Score = 0.5 * (1-NMAE) + 0.5 * FICR

- 평가는 실제발전량이 '그룹 설비용량의 10% 이상'인 시간대만 대상
- FICR은 시간대별 오차율(error_rate = |pred-actual|/capacity)에 따라
  계단식 단가(4.0 / 3.0 / 0.0)가 적용되는 정산금 구조
    error_rate <= 0.06 -> 단가 4.0
    error_rate <= 0.08 -> 단가 3.0
    error_rate >  0.08 -> 단가 0.0  (정산금 0원)
"""

import numpy as np
import pandas as pd

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}


def metric(answer_df: pd.DataFrame, pred_df: pd.DataFrame, target_cols=TARGET_COLS, capacity=CAPACITY_KWH):
    """
    공식 평가 코드(codeshare 14035)를 그대로 이식한 함수.
    answer_df, pred_df: TARGET_COLS 컬럼을 가진 DataFrame (같은 순서/길이여야 함)

    반환: (total_score, one_minus_nmae, ficr)
    """
    group_nmae = []
    group_ficr = []

    for col in target_cols:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        cap = capacity[col]

        valid = actual >= cap * 0.10
        actual = actual[valid]
        forecast = forecast[valid]

        if len(actual) == 0:
            continue

        error_rate = np.abs(forecast - actual) / cap
        group_nmae.append(np.mean(error_rate))

        unit_price = np.select(
            [error_rate <= 0.06, error_rate <= 0.08],
            [4.0, 3.0],
            default=0.0,
        )

        earned_settlement = np.sum(actual * unit_price)
        max_settlement = np.sum(actual * 4.0)
        group_ficr.append(earned_settlement / max_settlement)

    one_minus_nmae = 1 - np.mean(group_nmae)
    ficr = np.mean(group_ficr)
    total_score = 0.5 * one_minus_nmae + 0.5 * ficr

    return total_score, one_minus_nmae, ficr


def error_rate_breakdown(answer_df: pd.DataFrame, pred_df: pd.DataFrame, target_cols=TARGET_COLS, capacity=CAPACITY_KWH) -> pd.DataFrame:
    """
    그룹별로 오차율 구간(<=6%, <=8%, >8%) 시간대 비율을 보여주는 진단용 함수.
    FICR을 깎아먹는 '8% 초과' 구간이 얼마나 되는지 확인할 때 사용.
    """
    rows = []
    for col in target_cols:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        cap = capacity[col]

        valid = actual >= cap * 0.10
        actual, forecast = actual[valid], forecast[valid]
        if len(actual) == 0:
            continue

        error_rate = np.abs(forecast - actual) / cap
        n = len(error_rate)
        rows.append({
            "group": col,
            "평가대상_시간수": n,
            "6%이하_비율": np.mean(error_rate <= 0.06),
            "6~8%_비율": np.mean((error_rate > 0.06) & (error_rate <= 0.08)),
            "8%초과_비율(정산금0)": np.mean(error_rate > 0.08),
        })
    return pd.DataFrame(rows)


def bias_diagnosis(answer_df: pd.DataFrame, pred_df: pd.DataFrame, target_cols=TARGET_COLS, capacity=CAPACITY_KWH) -> pd.DataFrame:
    """
    편향(bias) 진단용 함수.
    - 평균_편향률: (예측-실제)/설비용량의 평균. 0에 가까우면 편향 없음(무편향).
      양수면 과대예측(발전량을 실제보다 높게 부르는 경향), 음수면 과소예측.
    - 8%초과_시간_평균오차율: 8%를 넘는 시간대들만 모아서 평균 오차율이 얼마나 되는지.
      이게 8~10% 정도로 문턱에 살짝 걸쳐있으면 '조금만 개선해도 문턱을 넘길 여지'가 크다는 뜻이고,
      15% 이상으로 크면 단순 보정이 아니라 모델 자체를 개선해야 한다는 뜻.
    """
    rows = []
    for col in target_cols:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        cap = capacity[col]

        valid = actual >= cap * 0.10
        actual, forecast = actual[valid], forecast[valid]
        if len(actual) == 0:
            continue

        signed_error_rate = (forecast - actual) / cap
        error_rate = np.abs(signed_error_rate)
        over_mask = error_rate > 0.08

        rows.append({
            "group": col,
            "평균_편향률(부호있음)": np.mean(signed_error_rate),
            "편향_방향": "과대예측" if np.mean(signed_error_rate) > 0 else "과소예측",
            "8%초과_시간_평균오차율": np.mean(error_rate[over_mask]) if over_mask.any() else np.nan,
            "8%초과_시간_중앙값오차율": np.median(error_rate[over_mask]) if over_mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


# 하위호환: 기존 train_baseline.py가 import하던 이름 유지
def bias_by_prediction_quantile(answer_df: pd.DataFrame, pred_df: pd.DataFrame, target_cols=TARGET_COLS, capacity=CAPACITY_KWH, n_bins: int = 5) -> pd.DataFrame:
    """
    예측값 크기 구간(quantile)별로 편향을 나눠서 보여주는 진단 함수.

    목적: '일괄 선형 보정'이 맞는 방식인지, 아니면 '예측값 크기에 따라 다른 보정'이
    필요한지 판단하기 위함. RandomForest 계열은 통상 예측값이 클수록(고발전 구간)
    평균으로 수렴하려는 경향 때문에 과소예측이 더 커지는 경우가 많음.

    각 구간(bin)마다:
      - 편향률: (예측-실제)/설비용량의 평균
      - 8%초과_비율: 그 구간에서 오차율 8% 초과(정산금 0) 비율
    """
    rows = []
    for col in target_cols:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        cap = capacity[col]

        valid = actual >= cap * 0.10
        actual, forecast = actual[valid], forecast[valid]
        if len(actual) == 0:
            continue

        # 예측값 크기 기준으로 n_bins개 구간으로 분할 (같은 개수씩)
        try:
            bins = pd.qcut(forecast, q=n_bins, duplicates="drop")
        except ValueError:
            continue

        signed_error_rate = (forecast - actual) / cap
        error_rate = np.abs(signed_error_rate)

        df = pd.DataFrame({"bin": bins, "signed_error_rate": signed_error_rate, "error_rate": error_rate, "forecast": forecast})
        grouped = df.groupby("bin", observed=True).agg(
            예측값_범위_kWh=("forecast", lambda s: f"{s.min():.0f}~{s.max():.0f}"),
            시간수=("forecast", "size"),
            편향률=("signed_error_rate", "mean"),
            평균오차율=("error_rate", "mean"),
            **{"8%초과_비율": ("error_rate", lambda s: (s > 0.08).mean())},
        )
        grouped.insert(0, "group", col)
        rows.append(grouped.reset_index(drop=True))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def nmae_score(pred: pd.DataFrame, actual: pd.DataFrame, capacity: dict = CAPACITY_KWH) -> float:
    _, one_minus_nmae, _ = metric(actual, pred, target_cols=list(capacity.keys()), capacity=capacity)
    return one_minus_nmae


def total_score(pred: pd.DataFrame, actual: pd.DataFrame, capacity: dict = CAPACITY_KWH):
    return metric(actual, pred, target_cols=list(capacity.keys()), capacity=capacity)
