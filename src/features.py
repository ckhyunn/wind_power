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
from sklearn.isotonic import IsotonicRegression

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


def nearest_grids_with_distance(weather_df: pd.DataFrame, target_lat: float, target_lon: float, k: int = 3) -> pd.DataFrame:
    """weather_df에서 (target_lat, target_lon)에 가장 가까운 grid_id k개를, 거리(度 단위)와 함께 반환.
    IDW(역거리가중) 집계에 사용."""
    grids = weather_df[["grid_id", "latitude", "longitude"]].drop_duplicates()
    grids["dist"] = np.sqrt((grids["latitude"] - target_lat) ** 2 + (grids["longitude"] - target_lon) ** 2)
    return grids.nsmallest(k, "dist")[["grid_id", "dist"]].reset_index(drop=True)


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


AIR_GAS_CONSTANT = 287.05  # J/(kg*K), 건조공기 기체상수


def add_air_density_feature(df: pd.DataFrame, temp_col: str, pressure_col: str, prefix: str) -> pd.DataFrame:
    """
    이상기체 법칙으로 공기밀도 계산: rho = P / (R * T)
    온도는 Kelvin, 기압은 Pa 단위여야 함 (GFS/LDAPS 원본 데이터 확인 결과 두 단위 모두 해당).

    발전량은 물리적으로 공기밀도에도 비례함 (같은 풍속이라도 겨울철 밀도 높은 공기가
    더 많은 운동에너지를 전달) - 지금까지 풍속 위주 피처에는 없던 정보.
    """
    df = df.copy()
    df[f"{prefix}_air_density"] = df[pressure_col] / (AIR_GAS_CONSTANT * df[temp_col])
    return df


def add_wind_shear_feature(df: pd.DataFrame, ws_low_col: str, ws_high_col: str, prefix: str) -> pd.DataFrame:
    """
    고도별 풍속 차이(윈드시어) = 고고도 풍속 - 저고도 풍속.
    대기 안정도/난류 강도의 대리지표. 시어가 크면 대기가 불안정(난류 심함)해서
    터빈이 겪는 실제 바람이 예보 평균값과 더 크게 어긋날 가능성이 있음.
    """
    df = df.copy()
    df[f"{prefix}_wind_shear"] = df[ws_high_col] - df[ws_low_col]
    return df


def add_lag_lead_features(weather: pd.DataFrame, cols: list, dt_col: str = "forecast_kst_dtm") -> pd.DataFrame:
    """
    인접 시간대(전/후 1시간) 값을 피처로 추가.
    기상예보는 하루치(24시간)가 한 번에 발표되므로, 특정 시각을 예측할 때 그 앞뒤 시간대
    예보값을 아는 것은 데이터 누수가 아님 (전부 같은 시점에 이미 확보된 예보 정보).
    지금까지 모델은 매 시간을 독립적으로 취급했는데, 이 피처로 시간적 흐름(급변 여부 등)
    정보를 처음으로 제공함.
    """
    weather = weather.sort_values(dt_col).reset_index(drop=True)
    for col in cols:
        weather[f"{col}_lag1"] = weather[col].shift(1)
        weather[f"{col}_lead1"] = weather[col].shift(-1)
        weather[f"{col}_diff1"] = weather[col] - weather[f"{col}_lag1"]  # 시간당 변화율 (급변/돌풍 대리지표)
    return weather


def aggregate_weather_for_group(df: pd.DataFrame, grid_ids: list, prefix: str) -> pd.DataFrame:
    """지정된 grid_ids만 필터링해서 forecast_kst_dtm별 단순평균을 낸 그룹 전용 날씨 피처.
    (참고용으로 남겨둠 - IDW 도입 전 버전과 비교하고 싶을 때 사용)"""
    df = df[df["grid_id"].isin(grid_ids)].copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    drop_cols = {"data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    value_cols = [c for c in df.columns if c not in {"forecast_kst_dtm", *drop_cols}]
    agg = df.groupby("forecast_kst_dtm")[value_cols].mean()
    agg.columns = [f"{prefix}_{c}_mean" for c in agg.columns]
    return agg.reset_index()


def aggregate_weather_dispersion(df: pd.DataFrame, grid_ids: list, cols: list, prefix: str) -> pd.DataFrame:
    """
    지정된 grid_ids 간의 값 차이(표준편차)를 피처로 만듦.
    지금까지는 격자들을 평균으로 뭉개기만 했는데, 격자 간 편차가 크다는 건
    '이 시간대는 지역별로 바람이 크게 다르다 = 예보 자체가 불확실하거나 국지적으로
    변동성이 큰 상황'이라는 신호일 수 있음. 지금까지 한 번도 사용하지 않은 정보.

    cols: 표준편차를 계산할 컬럼 목록 (예: ['heightAboveGround_10_10u', ...])
    """
    df = df[df["grid_id"].isin(grid_ids)].copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    disp = df.groupby("forecast_kst_dtm")[cols].std()
    disp.columns = [f"{prefix}_{c}_std" for c in disp.columns]
    return disp.reset_index()


def aggregate_weather_idw(df: pd.DataFrame, grid_dist: pd.DataFrame, prefix: str, power: float = 2.0) -> pd.DataFrame:
    """
    거리 가중 평균(IDW, Inverse Distance Weighting)으로 그룹 전용 날씨 피처를 만듦.
    가까운 격자일수록 가중치가 커짐: weight = 1 / distance^power

    grid_dist: nearest_grids_with_distance()의 결과 (grid_id, dist 컬럼)
    power: 거리 감쇠 강도. 2.0이 일반적인 기본값 (거리 2배면 가중치 1/4)
    """
    grid_ids = grid_dist["grid_id"].tolist()
    dist_map = dict(zip(grid_dist["grid_id"], grid_dist["dist"]))

    df = df[df["grid_id"].isin(grid_ids)].copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])

    # 거리 0(격자가 정확히 좌표와 일치)인 경우를 대비해 아주 작은 값으로 하한 처리
    eps = 1e-6
    df["_weight"] = df["grid_id"].map(dist_map).clip(lower=eps).pow(-power)

    drop_cols = {"data_available_kst_dtm", "grid_id", "latitude", "longitude", "_weight"}
    value_cols = [c for c in df.columns if c not in {"forecast_kst_dtm", *drop_cols}]

    # 가중평균 = sum(value * weight) / sum(weight), forecast_kst_dtm별로 계산
    weighted = df[value_cols].multiply(df["_weight"], axis=0)
    weighted["forecast_kst_dtm"] = df["forecast_kst_dtm"]
    weighted["_weight"] = df["_weight"]

    grouped = weighted.groupby("forecast_kst_dtm")
    weight_sum = grouped["_weight"].sum()
    agg = grouped[value_cols].sum().div(weight_sum, axis=0)
    agg.columns = [f"{prefix}_{c}_idw" for c in agg.columns]
    return agg.reset_index()


def lead_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    예보 리드타임(발표 후 몇 시간짜리 예측인지) 피처.
    forecast_kst_dtm 하나당 grid_id가 여러 개 있지만 data_available_kst_dtm은
    같은 시각 내에서 전부 동일하므로, 대표로 1개 grid만 남겨서 계산.

    일반적으로 예보는 리드타임이 길수록(먼 미래를 예측할수록) 부정확해지므로
    모델이 이 정보를 참고할 수 있게 함.
    """
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df["data_available_kst_dtm"] = pd.to_datetime(df["data_available_kst_dtm"])
    rep = df.drop_duplicates(subset="forecast_kst_dtm")[["forecast_kst_dtm", "data_available_kst_dtm"]].copy()
    rep["lead_time_hours"] = (rep["forecast_kst_dtm"] - rep["data_available_kst_dtm"]).dt.total_seconds() / 3600
    return rep[["forecast_kst_dtm", "lead_time_hours"]]


def nearest_grid_raw_features(df: pd.DataFrame, grid_dist: pd.DataFrame, prefix: str, top_n: int = 1) -> pd.DataFrame:
    """
    최근접 격자 top_n개의 '원본' 값을 각각 별도 컬럼으로 제공 (평균/가중평균으로 뭉개지 않음).
    평균은 정보 손실이 있을 수 있어, 가장 가까운 격자 원본값을 모델이 직접 보게 해서
    LightGBM이 스스로 유용한 조합을 찾을 여지를 줌.

    grid_dist: nearest_grids_with_distance()의 결과, 거리순 정렬되어 있다고 가정
    """
    grid_dist_sorted = grid_dist.sort_values("dist").reset_index(drop=True)
    top_grid_ids = grid_dist_sorted["grid_id"].tolist()[:top_n]

    drop_cols = {"data_available_kst_dtm", "grid_id", "latitude", "longitude"}
    result = None
    for rank, grid_id in enumerate(top_grid_ids, start=1):
        sub = df[df["grid_id"] == grid_id].copy()
        sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"])
        value_cols = [c for c in sub.columns if c not in {"forecast_kst_dtm", *drop_cols}]
        sub = sub[["forecast_kst_dtm", *value_cols]].copy()
        sub.columns = ["forecast_kst_dtm"] + [f"{prefix}_nearest{rank}_{c}" for c in value_cols]
        result = sub if result is None else result.merge(sub, on="forecast_kst_dtm", how="outer")

    return result


def fit_power_curve(wind_speed: pd.Series, generation: pd.Series):
    """
    '예보 풍속 -> 실제 발전량'의 경험적 관계를 isotonic(단조증가)으로 적합.
    SCADA 실측 풍속은 예보 풍속과 스케일이 안 맞아 직접 못 쓰므로(diagnose_cutout.py에서 확인),
    예보 풍속 자체를 입력으로 써서 스케일 불일치 문제를 우회함.

    반드시 학습 구간(train)의 wind_speed/generation만 넣어야 함 (calib/holdout/test에 새어나가지 않게).

    반환: (적합된 IsotonicRegression, 결측치 대체용 중앙값)
    """
    valid = wind_speed.notna()
    curve = IsotonicRegression(out_of_bounds="clip", increasing=True)
    curve.fit(wind_speed[valid], generation[valid])
    fallback_ws = wind_speed[valid].median()
    return curve, fallback_ws


def apply_power_curve(curve, fallback_ws: float, wind_speed: pd.Series, capacity: float) -> np.ndarray:
    """fit_power_curve()로 적합한 커브를 새 데이터(calib/holdout/test)에 적용"""
    return np.clip(curve.predict(wind_speed.fillna(fallback_ws)), 0, capacity)


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
