"""대회 평가 산식 구현.

Score = 0.5 x (1 - NMAE) + 0.5 x FICR

- 그룹별 NMAE = mean(|예측 - 실제| / 그룹 설비용량),
  단 실제발전량이 설비용량의 10% 이상인 시간대만 대상으로 계산한다.
- 1 - NMAE = 1 - (3개 그룹 NMAE 평균)
- FICR은 시간대별 정산금 계단 구간표를 기반으로 계산하나,
  아직 구간값이 공개되지 않아 미구현 상태로 남겨둔다.
"""

from collections.abc import Sequence

import numpy as np

CAPACITY_THRESHOLD_RATIO = 0.1


def nmae_score(pred: np.ndarray, actual: np.ndarray, capacity: float) -> float:
    """단일 그룹의 NMAE를 계산한다.

    실제발전량이 설비용량(capacity)의 10% 이상인 시간대만 대상으로 한다.
    """
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)

    mask = actual >= CAPACITY_THRESHOLD_RATIO * capacity
    if not mask.any():
        return 0.0

    return float(np.mean(np.abs(pred[mask] - actual[mask]) / capacity))


def group_nmae_score(group_nmae_scores: Sequence[float]) -> float:
    """KPX 3개 그룹의 NMAE 평균을 계산한다."""
    return float(np.mean(group_nmae_scores))


def ficr_score(*args, **kwargs) -> float:
    # TODO: 시간대별 정산금 계단 구간표가 공개되면 해당 구간값을 기준으로 구현한다.
    raise NotImplementedError("FICR 계단 구간표가 아직 공개되지 않아 미구현 상태입니다.")


def total_score(group_nmae_scores: Sequence[float], ficr: float) -> float:
    """0.5 x (1 - NMAE) + 0.5 x FICR."""
    one_minus_nmae = 1 - group_nmae_score(group_nmae_scores)
    return 0.5 * one_minus_nmae + 0.5 * ficr
