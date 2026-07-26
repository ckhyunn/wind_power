import re
from pathlib import Path

import openpyxl
import pandas as pd

from src.config import CAPACITY_KWH, TARGET_COLS  # noqa: F401  (re-exported for convenience)

_DMS_PATTERN = re.compile(r"(\d+)\D+(\d+)\D+([\d.]+)\D*([NSEW])")

_INFO_COLUMNS = [
    "_blank",
    "stage",
    "name",
    "maker",
    "model",
    "unit_no",
    "coord",
    "kpx_group",
    "hub_height",
    "rotor_d",
    "unit_capacity_mw",
    "group_capacity_mw",
]


def load_train_labels(data_dir) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "train" / "train_labels.csv", encoding="utf-8-sig")
    df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
    return df


def load_sample_submission(data_dir) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "sample_submission.csv", encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    return df


def load_weather(data_dir, source: str, split: str) -> pd.DataFrame:
    subdir = "train" if split == "train" else "test"
    path = Path(data_dir) / subdir / f"{source}_{split}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df["data_available_kst_dtm"] = pd.to_datetime(df["data_available_kst_dtm"])
    return df


def load_scada(data_dir, maker: str) -> pd.DataFrame:
    path = Path(data_dir) / "train" / f"scada_{maker}_train.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
    return df


def dms_to_decimal(coord: str):
    """Parse a "37°16'55.61\"N 128°57'02.10\"E" style string to (lat, lon) decimal degrees."""
    out = {}
    for deg, minutes, seconds, hemi in _DMS_PATTERN.findall(coord):
        value = float(deg) + float(minutes) / 60 + float(seconds) / 3600
        if hemi in ("S", "W"):
            value = -value
        out["lat" if hemi in ("N", "S") else "lon"] = value
    return out.get("lat"), out.get("lon")


def load_turbine_info(xlsx_path, sheet_name: str = "info") -> pd.DataFrame:
    """Parse info.xlsx's `info` sheet: 2 header rows + merged cells (group fields only
    populated on each group's first turbine row), so a flat pandas.read_excel is wrong —
    forward-fill `kpx_group`, then fill `group_capacity_mw` within each group.
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))

    header_idx = next(i for i, row in enumerate(rows) if row[1] == "stage" or row[1] == "단계")
    data_rows = rows[header_idx + 1 :]

    info = pd.DataFrame(data_rows, columns=_INFO_COLUMNS)
    info = info.dropna(how="all").reset_index(drop=True)
    info["kpx_group"] = info["kpx_group"].ffill()
    info["group_capacity_mw"] = info.groupby("kpx_group")["group_capacity_mw"].transform(
        lambda s: s.ffill().bfill()
    )
    info[["lat", "lon"]] = info["coord"].apply(lambda s: pd.Series(dms_to_decimal(s)))
    return info


def farm_centroids(info: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) centroid per kpx_group, averaged over that group's turbines."""
    return info.groupby("kpx_group")[["lat", "lon"]].mean()
