import numpy as np
import pandas as pd

from src.metrics import eligibility_mask, ficr, group_scores, nmae


def test_nmae_perfect_prediction_is_zero():
    actual = np.array([1000.0, 2000.0, 3000.0])
    pred = actual.copy()
    assert nmae(actual, pred, capacity=21600) == 0.0


def test_nmae_matches_hand_computed_value():
    # |pred-actual| = [500, 1000], capacity=1000 -> mean(0.5, 1.0) = 0.75
    actual = np.array([1000.0, 2000.0])
    pred = np.array([1500.0, 1000.0])
    assert nmae(actual, pred, capacity=1000) == 0.75


def test_nmae_applies_mask_before_averaging():
    actual = np.array([100.0, 5000.0, 9000.0])
    pred = np.array([9999.0, 5000.0, 9000.0])  # first prediction is wildly wrong
    mask = np.array([False, True, True])
    assert nmae(actual, pred, capacity=21600, mask=mask) == 0.0


def test_eligibility_mask_flags_hours_at_or_above_threshold():
    actual = pd.Series([0.0, 2159.9, 2160.0, 21600.0])
    mask = eligibility_mask(actual, capacity=21600, threshold=0.10)
    assert list(mask) == [False, False, True, True]


def test_ficr_perfect_prediction_gets_best_rate_everywhere():
    # every hour's per-hour NMAE is 0 -> best band (<=6%) -> rate=4 everywhere
    # -> earned == theoretical max -> FICR = 1.0
    actual = np.array([1000.0, 2000.0, 3000.0])
    pred = actual.copy()
    assert ficr(actual, pred, capacity=21600) == 1.0


def test_ficr_uses_evaluation_page_bands_confirmed_2026_07_26():
    # docs/PLAN.md §1.2 band table: NMAE<=6% -> 4won/kWh, 6-8% -> 3won/kWh, >8% -> 0.
    # capacity=1000 so hourly NMAE = |pred-actual|/1000 lands exactly on band edges.
    actual = np.array([1000.0, 1000.0, 1000.0])
    pred = np.array(
        [
            1060.0,  # NMAE=0.06 -> best band, rate=4
            1070.0,  # NMAE=0.07 -> mid band, rate=3
            1090.0,  # NMAE=0.09 -> no settlement, rate=0
        ]
    )
    # earned = 1000*4 + 1000*3 + 1000*0 = 7000; max = 1000*4*3 = 12000
    assert ficr(actual, pred, capacity=1000) == 7000.0 / 12000.0


def test_ficr_applies_mask_before_averaging():
    actual = np.array([100.0, 1000.0, 1000.0])
    pred = np.array([9999.0, 1000.0, 1090.0])  # first hour wildly wrong but masked out
    mask = np.array([False, True, True])
    # earned = 1000*4 + 1000*0 = 4000; max = 1000*4 + 1000*4 = 8000
    assert ficr(actual, pred, capacity=1000, mask=mask) == 4000.0 / 8000.0


def test_group_scores_masks_ineligible_hours_and_averages_across_groups():
    # capacity=21600 -> 10% threshold = 2160. Hour 0 (actual=2000) is below threshold
    # for both groups, so it must be excluded from both groups' NMAE (docs/PLAN.md §1.2).
    actual = {
        "kpx_group_1": np.array([2000.0, 20000.0]),
        "kpx_group_2": np.array([2000.0, 20000.0]),
    }
    pred = {
        "kpx_group_1": np.array([9999.0, 20000.0]),  # wrong on the excluded hour only
        "kpx_group_2": np.array([2000.0, 16000.0]),  # wrong on the included hour only
    }
    capacity = {"kpx_group_1": 21600, "kpx_group_2": 21600}

    scores = group_scores(actual, pred, capacity)

    assert scores["kpx_group_1"]["nmae"] == 0.0
    assert scores["kpx_group_2"]["nmae"] == 4000.0 / 21600
    assert scores["macro_nmae"] == (scores["kpx_group_1"]["nmae"] + scores["kpx_group_2"]["nmae"]) / 2

    # group_1 hour 1 is a perfect prediction -> FICR=1.0 on its only eligible hour.
    # group_2 hour 1's error (4000/21600 = 18.5%) is past the 8% band -> FICR=0.0.
    assert scores["kpx_group_1"]["ficr"] == 1.0
    assert scores["kpx_group_2"]["ficr"] == 0.0
    assert scores["macro_ficr"] == 0.5
    assert scores["total_score"] == 0.5 * (1 - scores["macro_nmae"]) + 0.5 * scores["macro_ficr"]
