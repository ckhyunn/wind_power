import numpy as np
import pandas as pd


def time_series_splits(dates, n_splits: int):
    """Expanding-window, time-respecting CV splits.

    Sorts rows by `dates`, cuts the sorted timeline into n_splits+1 equal-sized
    contiguous blocks, and for fold i (0-indexed) yields:
      train = every row in blocks [0..i], test = block i+1.
    So each fold's train set only ever contains hours strictly earlier than its
    test set (docs/PLAN.md §1.3 "trustworthy offline CV" / never random-shuffle).
    Returns (train_idx, test_idx) as positional integer arrays into the original
    (possibly unsorted) `dates` input.
    """
    dates = pd.Series(dates).reset_index(drop=True)
    order = np.argsort(dates.values, kind="mergesort")
    n = len(order)
    block_size = n // (n_splits + 1)

    for i in range(n_splits):
        train_end = block_size * (i + 1)
        test_end = block_size * (i + 2) if i < n_splits - 1 else n
        train_idx = order[:train_end]
        test_idx = order[train_end:test_end]
        yield train_idx, test_idx
