import numpy as np


def nmae(actual, pred, capacity: float, mask=None) -> float:
    """Normalized MAE: mean(|pred-actual|/capacity) over included hours.

    Matches the competition's NMAE definition (docs/PLAN.md §1.2): error is
    normalized by fixed group capacity, not by actual output.
    """
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        actual = actual[mask]
        pred = pred[mask]
    return float(np.mean(np.abs(pred - actual) / capacity))


def eligibility_mask(actual, capacity: float, threshold: float = 0.10):
    """Hours where actual >= threshold*capacity — the only hours the competition scores
    (docs/PLAN.md §1.2 "Scoring scope")."""
    actual = np.asarray(actual, dtype=float)
    return actual >= threshold * capacity


# FICR settlement bands confirmed from the evaluation page image (2026-07-26, see
# docs/PLAN.md §1.2 — the page renders this as a chart, so it's transcribed here):
# hourly NMAE <= 6% -> 4 won/kWh, 6-8% -> 3 won/kWh, > 8% -> 0 (no settlement).
_FICR_BANDS = [(0.06, 4.0), (0.08, 3.0)]
_FICR_BEST_RATE = _FICR_BANDS[0][1]


def _settlement_rate(hourly_nmae: np.ndarray) -> np.ndarray:
    rate = np.zeros_like(hourly_nmae)
    for upper, won_per_kwh in reversed(_FICR_BANDS):
        rate = np.where(hourly_nmae <= upper, won_per_kwh, rate)
    return rate


def ficr(actual, pred, capacity: float, mask=None) -> float:
    """Financial Cost Recovery Rate: earned settlement / theoretical-max settlement.

    Each eligible hour's per-hour NMAE (|pred-actual|/capacity) selects a settlement
    rate from `_FICR_BANDS`; theoretical max assumes every hour hits the best rate.
    """
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        actual = actual[mask]
        pred = pred[mask]
    hourly_nmae = np.abs(pred - actual) / capacity
    earned = np.sum(_settlement_rate(hourly_nmae) * actual)
    theoretical_max = np.sum(_FICR_BEST_RATE * actual)
    return float(earned / theoretical_max)


def group_scores(actual: dict, pred: dict, capacity: dict, threshold: float = 0.10) -> dict:
    """Per-group NMAE/FICR (masked to eligible hours), macro-averages, and total score."""
    out = {}
    for group, cap in capacity.items():
        mask = eligibility_mask(actual[group], cap, threshold)
        out[group] = {
            "nmae": nmae(actual[group], pred[group], cap, mask=mask),
            "ficr": ficr(actual[group], pred[group], cap, mask=mask),
        }
    out["macro_nmae"] = float(np.mean([out[g]["nmae"] for g in capacity]))
    out["macro_ficr"] = float(np.mean([out[g]["ficr"] for g in capacity]))
    out["total_score"] = 0.5 * (1 - out["macro_nmae"]) + 0.5 * out["macro_ficr"]
    return out
