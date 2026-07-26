import numpy as np
import pandas as pd

from src.metrics import eligibility_mask, group_scores, nmae


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
