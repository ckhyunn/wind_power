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


def group_scores(actual: dict, pred: dict, capacity: dict, threshold: float = 0.10) -> dict:
    """Per-group NMAE (masked to eligible hours) plus the macro-average across groups."""
    out = {}
    for group, cap in capacity.items():
        mask = eligibility_mask(actual[group], cap, threshold)
        out[group] = {"nmae": nmae(actual[group], pred[group], cap, mask=mask)}
    out["macro_nmae"] = float(np.mean([out[g]["nmae"] for g in capacity]))
    return out
