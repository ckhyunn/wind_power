import numpy as np
import pandas as pd
import pytest

from src.submission import build_submission


def _sample_submission():
    return pd.DataFrame(
        {
            "forecast_id": ["forecast_0001", "forecast_0002", "forecast_0003"],
            "forecast_kst_dtm": pd.to_datetime(
                ["2025-01-01 01:00:00", "2025-01-01 02:00:00", "2025-01-01 03:00:00"]
            ),
            "kpx_group_1": [0, 0, 0],
            "kpx_group_2": [0, 0, 0],
            "kpx_group_3": [0, 0, 0],
        }
    )


def _predictions():
    dtm = pd.to_datetime(["2025-01-01 01:00:00", "2025-01-01 02:00:00", "2025-01-01 03:00:00"])
    return {
        "kpx_group_1": pd.Series([100.0, -50.0, 300.0], index=dtm),
        "kpx_group_2": pd.Series([200.0, 400.0, 600.0], index=dtm),
        "kpx_group_3": pd.Series([50.0, 60.0, 70.0], index=dtm),
    }


def test_preserves_forecast_id_and_dtm_exactly():
    sample = _sample_submission()
    out = build_submission(sample, _predictions())
    pd.testing.assert_series_equal(out["forecast_id"], sample["forecast_id"])
    pd.testing.assert_series_equal(out["forecast_kst_dtm"], sample["forecast_kst_dtm"])


def test_column_order_matches_sample_submission():
    sample = _sample_submission()
    out = build_submission(sample, _predictions())
    assert list(out.columns) == list(sample.columns)


def test_clips_negative_predictions_to_zero():
    sample = _sample_submission()
    out = build_submission(sample, _predictions())
    assert out["kpx_group_1"].min() >= 0
    assert out.loc[1, "kpx_group_1"] == 0.0


def test_raises_when_predictions_missing_a_timestamp():
    sample = _sample_submission()
    preds = _predictions()
    preds["kpx_group_1"] = preds["kpx_group_1"].iloc[:2]  # drop the last hour
    with pytest.raises(ValueError, match="missing"):
        build_submission(sample, preds)


def test_raises_when_a_target_group_is_absent_from_predictions():
    sample = _sample_submission()
    preds = _predictions()
    del preds["kpx_group_3"]
    with pytest.raises(ValueError, match="kpx_group_3"):
        build_submission(sample, preds)
