import pandas as pd

from src.config import TARGET_COLS


def build_submission(sample_submission: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """Build the final submission CSV from per-group predictions.

    `predictions` maps each column in TARGET_COLS to a pd.Series indexed by
    forecast_kst_dtm. forecast_id/forecast_kst_dtm and column order are taken verbatim
    from sample_submission (docs/PLAN.md §1.4: never hand-edit these). Negative
    predictions are clipped to 0 kWh (kWh cannot be negative); no upper clip is applied
    since EDA (docs/eda_report.md §3) found real settlement values occasionally exceed
    nameplate capacity by a small margin with no hard ceiling.
    """
    missing_groups = [g for g in TARGET_COLS if g not in predictions]
    if missing_groups:
        raise ValueError(f"predictions missing target group(s): {missing_groups}")

    out = sample_submission[["forecast_id", "forecast_kst_dtm"]].copy()
    for group in TARGET_COLS:
        series = predictions[group]
        aligned = series.reindex(sample_submission["forecast_kst_dtm"])
        if aligned.isna().any():
            missing_ts = sample_submission["forecast_kst_dtm"][aligned.isna().values].tolist()
            raise ValueError(f"predictions for {group!r} missing timestamps: {missing_ts}")
        out[group] = aligned.clip(lower=0).values

    return out
