"""End-to-end pipeline: raw CSVs -> per-group LightGBM models -> submissions/*.csv.

Evidence for each design choice is in docs/eda_report.md (cited inline) and
docs/PLAN.md §1.2-1.3 (competition metric). Known simplifications for this
first cut are documented in docs/submission_log.md, not silently baked in here.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.config import CAPACITY_KWH, TARGET_COLS
from src.cv import time_series_splits
from src.data import (
    farm_centroids,
    load_sample_submission,
    load_train_labels,
    load_turbine_info,
    load_weather,
)
from src.features import build_design_matrix, build_weather_features
from src.metrics import group_scores
from src.model import predict_lgbm, train_lgbm
from src.submission import build_submission

# EDA Open Question #7 (docs/eda_report.md §Open Questions/7) found no single grid
# reduction method dominates for all 3 groups (mean best for group_1, nearest best for
# groups 2/3, gap ~0.02-0.15 correlation) and explicitly recommends letting per-group CV
# decide rather than hardcoding. This first cut hardcodes "mean" uniformly for simplicity
# and revisits per-group choice in a follow-up iteration (see docs/submission_log.md).
REDUCTION_METHOD = "mean"
N_CV_SPLITS = 4
NUM_BOOST_ROUND = 300


def _group_id(group_col: str) -> float:
    return float(group_col.rsplit("_", 1)[1])


def main():
    info = load_turbine_info(REPO_ROOT / "info.xlsx")
    centroids = farm_centroids(info)

    labels = load_train_labels(REPO_ROOT)
    ldaps_train = load_weather(REPO_ROOT, "ldaps", "train")
    gfs_train = load_weather(REPO_ROOT, "gfs", "train")
    ldaps_test = load_weather(REPO_ROOT, "ldaps", "test")
    gfs_test = load_weather(REPO_ROOT, "gfs", "test")
    sample_sub = load_sample_submission(REPO_ROOT)

    cv_actual, cv_pred, test_predictions = {}, {}, {}

    for group in TARGET_COLS:
        capacity = CAPACITY_KWH[group]
        lat, lon = centroids.loc[_group_id(group), ["lat", "lon"]]

        weather_train = build_weather_features(ldaps_train, gfs_train, lat, lon, REDUCTION_METHOD)
        weather_test = build_weather_features(ldaps_test, gfs_test, lat, lon, REDUCTION_METHOD)

        # Per EDA §5: label gaps are large contiguous blocks (telemetry outage / group_3's
        # pre-2023 non-reporting period), not scattered noise -> safe to drop, not interpolate.
        base = labels[["kst_dtm", group]].rename(columns={"kst_dtm": "forecast_kst_dtm"})
        base = base.dropna(subset=[group]).reset_index(drop=True)

        design = build_design_matrix(base, weather_train, "forecast_kst_dtm", drop_cols=[group])
        # Per PLAN.md §1.3: train on capacity fraction, not raw kWh, so NMAE contributions
        # are comparable across groups of slightly different capacity.
        y = base[group] / capacity
        dates = base["forecast_kst_dtm"]

        fold_actual, fold_pred = [], []
        for fold_i, (train_idx, test_idx) in enumerate(time_series_splits(dates, N_CV_SPLITS)):
            model = train_lgbm(design.iloc[train_idx], y.iloc[train_idx], num_boost_round=NUM_BOOST_ROUND)
            pred_frac = predict_lgbm(model, design.iloc[test_idx])
            fold_actual.append(base[group].iloc[test_idx].values)
            fold_pred.append(pred_frac * capacity)
            print(f"[{group}] fold {fold_i}: train={len(train_idx)} test={len(test_idx)}")

        cv_actual[group] = np.concatenate(fold_actual)
        cv_pred[group] = np.concatenate(fold_pred)

        final_model = train_lgbm(design, y, num_boost_round=NUM_BOOST_ROUND)

        test_design = build_design_matrix(
            sample_sub[["forecast_kst_dtm"]], weather_test, "forecast_kst_dtm", drop_cols=[]
        )
        test_design = test_design[design.columns]  # enforce identical column order
        test_pred_frac = predict_lgbm(final_model, test_design)
        test_predictions[group] = pd.Series(
            test_pred_frac * capacity, index=sample_sub["forecast_kst_dtm"].values
        )

    scores = group_scores(cv_actual, cv_pred, CAPACITY_KWH)
    print("\nCV NMAE (masked to eligible hours, docs/PLAN.md §1.2):")
    for group in TARGET_COLS:
        print(f"  {group}: {scores[group]['nmae']:.4f}  (1-NMAE = {1 - scores[group]['nmae']:.4f})")
    print(f"  macro_nmae: {scores['macro_nmae']:.4f}  (1-NMAE = {1 - scores['macro_nmae']:.4f})")

    submission = build_submission(sample_sub, test_predictions)
    out_dir = REPO_ROOT / "submissions"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"submission_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nwrote {out_path}  ({len(submission)} rows)")

    return scores, out_path


if __name__ == "__main__":
    main()
