"""
그룹별(위치 기반) 피처 엔지니어링 공용 함수.

핵심 개선점:
- 공식 baseline은 LDAPS 16격자 / GFS 9격자를 전국 평균 하나로 뭉쳐서
  3개 그룹 모델이 전부 동일한 날씨 피처를 사용했음.
- info.xlsx 분석 결과, 3개 KPX 그룹은 서로 다른 위치의 터빈 묶음이므로
  그룹별로 가장 가까운 격자를 골라 날씨 피처를 따로 구성해야 함.

info.xlsx 구조 (직접 확인한 결과):
- VESTAS 1~6호기  -> KPX 그룹 1 (21.6MW)
- VESTAS 7~12호기 -> KPX 그룹 2 (21.6MW)
- UNISON 1~5호기  -> KPX 그룹 3 (21.0MW)
- '좌표(Google)' 컬럼은 "37°16'55.61"N 128°57'02.10"E" 형태의 DMS 문자열
"""

import re
import numpy as np
import pandas as pd

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}

# info.xlsx를 직접 열어 확인한 터빈-그룹 매핑 (그룹 라벨이 첫 호기에만 찍혀있어 수동 확정)
GROUP_TURBINE_PREFIX = {
    "kpx_group_1": [f"vestas_wtg{i:02d}" for i in range(1, 7)],
    "kpx_group_2": [f"vestas_wtg{i:02d}" for i in range(7, 13)],
    "kpx_group_3": [f"unison_wtg{i:02d}" for i in range(1, 6)],
}


def parse_dms(coord_str: str) -> tuple[float, float]:
    """'37°16'55.61"N 128°57'02.10"E' -> (37.28211..., 128.95058...) 형태로 변환"""
    pattern = r"(\d+)°(\d+)'([\d.]+)\"([NSEW])"
    matches = re.findall(pattern, coord_str)
    if len(matches) != 2:
        raise ValueError(f"좌표 파싱 실패: {coord_str}")

    def to_decimal(deg, minute, sec, direction):
        val = float(deg) + float(minute) / 60 + float(sec) / 3600
        if direction in ("S", "W"):
            val = -val
        return val

    lat_d, lat_m, lat_s, lat_dir = matches[0]
    lon_d, lon_m, lon_s, lon_dir = matches[1]
    lat = to_decimal(lat_d, lat_m, lat_s, lat_dir)
    lon = to_decimal(lon_d, lon_m, lon_s, lon_dir)
    return lat, lon


def load_turbine_table(info_path) -> pd.DataFrame:
    """
    info.xlsx를 읽어서 (그룹, 위경도) 테이블로 정리.
    엑셀이 병합 헤더 형태라 앞의 빈 행/열을 스킵하고 헤더를 다시 잡는다.
    """
    raw = pd.read_excel(info_path, header=None)
    header_row = raw[raw.iloc[:, 1] == "단계"].index[0]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = raw.iloc[header_row]
    df = df.reset_index(drop=True)
    df = df.rename(columns={"좌표(Google)": "좌표"})
    df[["lat", "lon"]] = df["좌표"].apply(lambda s: pd.Series(parse_dms(s)))
    return df


def compute_group_coords(turbine_df: pd.DataFrame) -> dict:
    """
    GROUP_TURBINE_PREFIX 매핑에 따라 그룹별 대표 좌표(터빈 좌표 평균)를 계산.
    turbine_df는 제작사+호기 순서가 info.xlsx 원본 순서와 같다고 가정.
    """
    vestas = turbine_df[turbine_df["제작사"] == "VESTAS"].reset_index(drop=True)
    unison = turbine_df[turbine_df["제작사"] == "UNISON"].reset_index(drop=True)

    group_coords = {
        "kpx_group_1": vestas.iloc[0:6][["lat", "lon"]].mean().to_dict(),
        "kpx_group_2": vestas.iloc[6:12][["lat", "lon"]].mean().to_dict(),
        "kpx_group_3": unison.iloc[0:5][["lat", "lon"]].mean().to_dict(),
    }
    return group_coords


def nearest_grid_ids(weather_df: pd.DataFrame, target_lat: float, target_lon: float, k: int = 3) -> list:
    """weather_df(ldaps 또는 gfs)에서 (target_lat, target_lon)에 가장 가까운 grid_id k개 반환"""
    grids = weather_df[["grid_id", "latitude", "longitude"]].drop_duplicates()
    grids["dist"] = np.sqrt((grids["latitude"] - target_lat) ** 2 + (grids["longitude"] - target_lon) ** 2)
    return grids.nsmallest(k, "dist")["grid_id"].tolist()


def add_wind_features(df: pd.DataFrame, u_col: str, v_col: str, prefix: str) -> pd.DataFrame:
    """U/V 성분 바람을 풍속 크기 + 풍향(sin/cos)으로 변환 (발전량은 물리적으로 풍속에 크게 의존)"""
    df = df.copy()
    u, v = df[u_col], df[v_col]
    df[f"{prefix}_speed"] = np.sqrt(u ** 2 + v ** 2)
    df[f"{prefix}_speed_cubed"] = df[f"{prefix}_speed"] ** 3  # 발전량 ~ 풍속^3 (파워커브 근사)
    wind_dir_rad = np.arctan2(u, v)
    df[f"{prefix}_dir_sin"] = np.sin(wind_dir_rad)
    df[f"{prefix}_dir_cos"] = np.cos(wind_dir_rad)
    return df


def aggregate_weather_for_group(df: pd.DataFrame, grid_ids: list, prefix: str) -> pd.DataFrame:
    """지정된 grid_ids만 필터링해서 forecast_kst_dtm별 평균을 낸 그룹 전용 날씨 피처"""
    df = df[df["grid_id"].isin(grid_ids)].copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    drop_cols = {"data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in {"forecast_kst_dtm", *drop_cols}]
    agg = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg.columns = [f"{prefix}_{c}_mean" for c in agg.columns]
    return agg.reset_index()


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
    return out
