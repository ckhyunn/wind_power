import numpy as np
import pandas as pd

from src.config import WEATHER_META_COLS

# Columns that form a coherent (u, v) vector pair -> derive speed/direction from them.
# heightAboveGround_50_50M{U,V}{max,min} are deliberately excluded: EDA (docs/eda_report.md
# §7 / Open Question #2) found max-U and max-V are independent per-hour extremes, not a
# simultaneous vector, so combining them into a single "speed" overstates gusts. They are
# kept as four independent scalar features instead (see build_weather_features).
_UV_PAIRS = [
    ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "10"),
    ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "100"),
]

_INDEPENDENT_SCALAR_COLS = [
    "heightAboveGround_50_50MUmax",
    "heightAboveGround_50_50MUmin",
    "heightAboveGround_50_50MVmax",
    "heightAboveGround_50_50MVmin",
]


def uv_to_speed_dir(u, v):
    """(u, v) wind components -> (speed, meteorological "from" direction in compass degrees)."""
    u = pd.Series(u).astype(float)
    v = pd.Series(v).astype(float)
    speed = np.sqrt(u**2 + v**2)
    direction = np.degrees(np.arctan2(-u, -v)) % 360
    return speed, direction


def calendar_features(dt_series: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(dt_series)
    out = pd.DataFrame(index=dt.index)
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out.reset_index(drop=True)


def reduce_grid(df: pd.DataFrame, value_col: str, farm_lat: float, farm_lon: float, method: str = "mean") -> pd.Series:
    """Reduce a long-format (forecast_kst_dtm, grid_id) weather column to one value per
    forecast_kst_dtm, relative to a farm's (lat, lon) centroid. Per EDA Open Question #7,
    no single method dominates across all 3 groups (correlation proxy: mean 0.73-0.74,
    nearest 0.61-0.78, IDW 0.71-0.75, group-dependent) — all three are provided so CV
    picks per group rather than hard-coding one.
    """
    grids = df[["grid_id", "latitude", "longitude"]].drop_duplicates().set_index("grid_id")
    dist = np.sqrt((grids["latitude"] - farm_lat) ** 2 + (grids["longitude"] - farm_lon) ** 2)

    if method == "mean":
        return df.groupby("forecast_kst_dtm")[value_col].mean()

    if method == "nearest":
        nearest_grid_id = dist.idxmin()
        return df.loc[df["grid_id"] == nearest_grid_id].set_index("forecast_kst_dtm")[value_col]

    if method == "idw":
        weight = (1.0 / (dist + 1e-6)).rename("weight")
        tmp = df.merge(weight, left_on="grid_id", right_index=True)
        return tmp.groupby("forecast_kst_dtm")[[value_col, "weight"]].apply(
            lambda g: np.average(g[value_col], weights=g["weight"])
        )

    raise ValueError(f"Unknown reduction method: {method!r}. Expected 'mean', 'nearest', or 'idw'.")


def aggregate_weather(df: pd.DataFrame, prefix: str, farm_lat: float, farm_lon: float, method: str = "mean") -> pd.DataFrame:
    """Reduce every value column in a long-format weather file to one row per
    forecast_kst_dtm, deriving speed/direction for known (u, v) pairs and keeping the
    50 m max/min fields as independent scalars (see module docstring / Open Question #2).
    """
    out = pd.DataFrame({"forecast_kst_dtm": sorted(df["forecast_kst_dtm"].unique())}).set_index(
        "forecast_kst_dtm"
    )

    uv_cols_used = set()
    for u_col, v_col, height in _UV_PAIRS:
        if u_col not in df.columns or v_col not in df.columns:
            continue
        u_reduced = reduce_grid(df, u_col, farm_lat, farm_lon, method)
        v_reduced = reduce_grid(df, v_col, farm_lat, farm_lon, method)
        speed, direction = uv_to_speed_dir(u_reduced, v_reduced)
        out[f"{prefix}_ws{height}"] = speed
        out[f"{prefix}_wd{height}"] = direction
        uv_cols_used.update([u_col, v_col])

    other_cols = [
        c for c in df.columns
        if c not in WEATHER_META_COLS and c not in uv_cols_used and c not in _INDEPENDENT_SCALAR_COLS
    ]
    for col in other_cols:
        out[f"{prefix}_{col}"] = reduce_grid(df, col, farm_lat, farm_lon, method)

    for col in _INDEPENDENT_SCALAR_COLS:
        if col in df.columns:
            out[f"{prefix}_{col}"] = reduce_grid(df, col, farm_lat, farm_lon, method)

    return out.reset_index()


def build_weather_features(
    ldaps_df: pd.DataFrame, gfs_df: pd.DataFrame, farm_lat: float, farm_lon: float, method: str = "mean"
) -> pd.DataFrame:
    ldaps_agg = aggregate_weather(ldaps_df, "ldaps", farm_lat, farm_lon, method)
    gfs_agg = aggregate_weather(gfs_df, "gfs", farm_lat, farm_lon, method)
    merged = ldaps_agg.merge(gfs_agg, on="forecast_kst_dtm", how="inner")
    merged["speed_disagree_10m"] = merged["ldaps_ws10"] - merged["gfs_ws10"]
    return merged


def scada_hourly_energy(df: pd.DataFrame, power_cols: list) -> pd.DataFrame:
    """Aggregate 10-minute SCADA `*_power_kw10m` readings to hourly kWh.

    Per EDA §6: despite the "kw10m" name, each reading is 10-minute ENERGY, not
    instantaneous power — hourly kWh is the SUM of the 6 readings in the hour, not
    their mean (using mean silently understates output by ~6x). Uses
    label='right'/closed='right' to match train_labels' end-of-hour timestamp convention
    (verified against train_labels at r=0.9998 in the EDA).
    """
    hourly = (
        df.set_index("kst_dtm")[power_cols]
        .resample("1h", label="right", closed="right")
        .sum(min_count=1)
    )
    return hourly.reset_index()


def build_design_matrix(base_df: pd.DataFrame, weather_df: pd.DataFrame, dt_col: str, drop_cols: list) -> pd.DataFrame:
    merged = base_df.merge(weather_df, on=dt_col, how="left")
    calendar = calendar_features(merged[dt_col])
    exclude = {dt_col, *drop_cols}
    feature_cols = [c for c in merged.columns if c not in exclude]
    return pd.concat([calendar, merged[feature_cols].reset_index(drop=True)], axis=1)
