import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def _filter_glitches(power_10m: pd.Series, unit_capacity_kw: float) -> pd.Series:
    """NaN out sensor-glitch power readings (docs/eda_report.md §7: Vestas SCADA has
    ~0.05% of readings at 1e7-5e7 scale, integer-overflow-like artifacts, always far past
    plausible rated output). Bounded generously above rated capacity so real short-term
    overspeed readings survive; NaN'd readings are excluded (not zeroed) from the hourly
    sum via skipna, matching the EDA's recommended filter."""
    threshold = unit_capacity_kw * 1.5
    return power_10m.where(power_10m.abs() <= threshold)


def group_scada_hourly(
    scada_df: pd.DataFrame, power_cols: list, ws_cols: list, unit_capacity_kw: float
) -> pd.DataFrame:
    """Hourly group-total power (kWh) and mean wind speed (m/s) from 10-min SCADA.

    Power: per docs/eda_report.md §6, `*_power_kw10m` columns are 10-min ENERGY, not
    instantaneous power, so hourly kWh is the SUM of the 6 readings/hour (not the mean) —
    sum each turbine's hourly energy, then across turbines for the group total.
    Wind speed: plain mean (not additive), across turbines and across the hour.
    Uses label="right"/closed="right" to match train_labels' end-of-hour convention
    (verified at r=0.9998 in the EDA).
    """
    filtered_power = scada_df[power_cols].apply(_filter_glitches, unit_capacity_kw=unit_capacity_kw)
    working = pd.DataFrame(
        {
            "kst_dtm": scada_df["kst_dtm"],
            "group_power_10m": filtered_power.sum(axis=1, skipna=True),
            "group_ws_10m": scada_df[ws_cols].mean(axis=1),
        }
    )
    working = working.set_index(pd.to_datetime(working["kst_dtm"]))
    hourly = working.resample("1h", label="right", closed="right").agg(
        power_kwh=("group_power_10m", "sum"), wind_speed=("group_ws_10m", "mean")
    )
    return hourly.reset_index()


def fit_speed_bias_correction(forecast_ws, scada_ws) -> tuple:
    """Linear forecast-wind-speed -> SCADA-measured-wind-speed correction.

    The two differ systematically: forecast is a grid-cell average at a different
    height/location than the turbine nacelle anemometer. A 2-parameter linear fit is
    deliberately low-capacity — docs/submission_log.md's isotonic-calibration rollback
    showed a flexible non-parametric fit here would risk overfitting the historical
    relationship instead of capturing a stable bias.
    """
    forecast_ws = np.asarray(forecast_ws, dtype=float)
    scada_ws = np.asarray(scada_ws, dtype=float)
    mask = ~(np.isnan(forecast_ws) | np.isnan(scada_ws))
    slope, intercept = np.polyfit(forecast_ws[mask], scada_ws[mask], 1)
    return slope, intercept


def apply_speed_bias_correction(forecast_ws, bias: tuple) -> np.ndarray:
    slope, intercept = bias
    return slope * np.asarray(forecast_ws, dtype=float) + intercept


def fit_binned_power_curve(
    wind_speed, capacity_fraction, bin_width: float = 0.5, max_speed: float = 30.0
) -> IsotonicRegression:
    """Empirical power curve: bin wind speed, take the MEDIAN capacity fraction per bin
    (robust to per-reading noise), then fit isotonic regression on the bin medians (one
    degree of freedom per *bin*, not per raw point) to enforce the physically expected
    monotonic shape. Deliberately not isotonic on raw points — docs/submission_log.md
    documents that overfitting isotonic regression on a real-valued residual failed to
    generalize to the true test period; binning first bounds the model's flexibility.
    """
    wind_speed = np.asarray(wind_speed, dtype=float)
    capacity_fraction = np.asarray(capacity_fraction, dtype=float)
    bins = np.arange(0, max_speed + bin_width, bin_width)
    bin_idx = np.digitize(wind_speed, bins)

    bin_centers, bin_medians = [], []
    for i in range(1, len(bins)):
        in_bin = bin_idx == i
        if not np.any(in_bin):
            continue
        bin_centers.append((bins[i - 1] + bins[i]) / 2)
        bin_medians.append(np.median(capacity_fraction[in_bin]))

    curve = IsotonicRegression(out_of_bounds="clip")
    curve.fit(np.array(bin_centers), np.array(bin_medians))
    return curve


def power_curve_feature(forecast_ws, curve: IsotonicRegression) -> np.ndarray:
    return curve.predict(np.asarray(forecast_ws, dtype=float))
