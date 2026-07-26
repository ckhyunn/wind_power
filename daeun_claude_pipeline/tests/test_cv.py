import numpy as np
import pandas as pd

from src.cv import time_series_splits


def _dates(n):
    return pd.Series(pd.date_range("2024-01-01", periods=n, freq="h"))


def test_yields_requested_number_of_splits():
    dates = _dates(100)
    splits = list(time_series_splits(dates, n_splits=4))
    assert len(splits) == 4


def test_every_split_keeps_train_strictly_before_test():
    dates = _dates(100)
    for train_idx, test_idx in time_series_splits(dates, n_splits=4):
        assert dates[train_idx].max() < dates[test_idx].min()


def test_test_folds_are_disjoint_and_move_forward_in_time():
    dates = _dates(100)
    splits = list(time_series_splits(dates, n_splits=4))
    test_starts = [dates[test_idx].min() for _, test_idx in splits]
    assert test_starts == sorted(test_starts)
    all_test_idx = np.concatenate([test_idx for _, test_idx in splits])
    assert len(all_test_idx) == len(set(all_test_idx))


def test_works_when_input_is_not_sorted():
    dates = _dates(20)
    shuffled = dates.sample(frac=1, random_state=0).reset_index(drop=True)
    for train_idx, test_idx in time_series_splits(shuffled, n_splits=2):
        assert shuffled[train_idx].max() < shuffled[test_idx].min()
