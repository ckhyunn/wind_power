# -*- coding: utf-8 -*-
"""
=========================================================================
 BARAM 2026 풍력발전량 예측 — A모델 기반 작업 사본 (동훈, model_A_dh)
=========================================================================
 원본: model_A/model_A.py (현윤, 리더보드 0.63625) - 원본은 절대 수정하지 않음
 이 사본에서는 '모델(LGB/CAT/XGB 파라미터, 앙상블 가중치, 학습로직)은 그대로 두고
 피처 엔지니어링/EDA만' 추가로 진행함.

 [동훈 추가분 - CFG 하단에 토글 추가됨]
   USE_850_DIR       : GFS 850hPa 격자별 풍향(sin/cos) 추가
                        (원본은 100m까지만 격자별 풍향이 있고 850hPa은 풍속만 있었음 -
                         analyze_A_residual_forensics.py 등에서 확인된 저기압/강풍 이벤트
                         분석과 B모델 v29 성공 경험에 근거)
   USE_PRESSURE_TEND : 기압 3시간/6시간 경향(tendency) 추가
                        (analyze_weather_regime_correlation.py에서 확인된 저기압 구간
                         group2·3 과대예측 편향 근거. 기존 add_lagleads()와 동일한
                         '같은 예보블록 내' 방식으로 계산 - 데이터 누수 아님)

 구성: LDAPS/GFS 예보 → 피처(격자별 포함) → LGBM+CatBoost+XGBoost 앙상블
       → group3 라벨 정제(pooled) + g1 편향 보정 → 제출
 사용법:
   1) 아래 CONFIG의 DATA_DIR 를 대회 원본 데이터(train/, test/) 위치로 설정
   2) python model_A_dh.py
 ablation(제거/변경 실험)은 CONFIG 토글만 바꾸면 됨 (README 참고)
 필요 패키지: requirements.txt
=========================================================================
"""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from pathlib import Path
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# ========================== CONFIG ==========================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "data"        # ← 대회 원본 데이터(train/, test/, sample_submission.csv) 위치 (B모델과 동일한 wind_power/data)
FEAT_DIR   = SCRIPT_DIR / "features"           # train_features.csv, test_features.csv (레포 포함)
CLEAN_DIR  = SCRIPT_DIR / "cleaned"            # train_labels_clean.csv, scada_unison_clean.csv (레포 포함)
OUT_DIR    = SCRIPT_DIR / "output"             # 결과 저장 위치
OUT_DIR.mkdir(exist_ok=True)

CFG = {
    # --- 앙상블 ---
    "MODELS":        ["lgb", "cat", "xgb"],          # 넣을 모델. 예: ["lgb"] / ["lgb","xgb"]
    "WEIGHTS":       {"lgb": 0.50, "cat": 0.10, "xgb": 0.40},  # 앙상블 비율(선택 모델만 자동 재정규화, 폴백/기본값)
    "GROUP_WEIGHTS": {  # [동훈추가] 그룹별 최적 가중치 - 실제 2024 fold 개별모델 예측으로
        # 그리드서치해서 확인 (재학습 없이 pred_A_lgb/cat/xgb_2024.csv + actual_2024.csv만
        # 사용). 그룹마다 최적 조합이 확연히 다름 - 전그룹 공통 고정값(0.5/0.1/0.4)이 최적이
        # 아니었음. group_score 기준 확인된 개선폭: g1 +0.0032, g2 +0.0019, g3 +0.0013.
        "kpx_group_1": {"lgb": 0.9, "cat": 0.0, "xgb": 0.1},
        "kpx_group_2": {"lgb": 0.5, "cat": 0.5, "xgb": 0.0},
        "kpx_group_3": {"lgb": 0.3, "cat": 0.6, "xgb": 0.1},
    },
    "N_SEEDS":       5,                               # LGBM 시드 평균 개수 (빠른 스크리닝용, 확정시 5로)
    # --- 피처 ---
    "USE_PERGRID":     True,   # 격자별 풍속 피처(ws10/ws50/ws50³, gfs ws100/80/850/ws100³)
    "USE_PERGRID_DIR": True,   # 격자별 풍향(sin/cos)
    "USE_850_DIR":       True,   # [동훈추가] GFS 850hPa 격자별 풍향(sin/cos) - 원본엔 없던 것
    "USE_T_INSTABILITY": False,  # 연직 대기불안정도(지표기온-700hPa기온) - fold개선(+0.001대)했으나
    # 실제 리더보드 하락(-0.00069)으로 기각
    "USE_PRESSURE_TEND": False,  # [동훈추가] 기압 3h/6h 경향 - 실험 결과 효과 없어 기각, 꺼둠
    "USE_OUTLIER_FILTER": True,  # [동훈추가] 학습단계 이상치 제거 (풍속충분+거의무발전=
    # 착빙/강제정지 등 물리적 이상 -> 평시 풍속-발전량 관계 오염 방지 위해 학습에서 제외)
    "OUTLIER_WS_THRESHOLD": 8.0,       # 이 풍속(m/s) 이상이면 '발전 가능한 바람'으로 간주
    "OUTLIER_OUTPUT_THRESHOLD": 0.30,  # [동훈수정] 0.10은 USE_10PCT_CUT과 완전히 겹쳐서
    # 무효(순수추가분 0건)였음 - quick_check_outlier_threshold.py로 확인 후 0.30으로 조정.
    "USE_TYPHOON_DOWNWEIGHT": True,   # [동훈추가] 태풍형(저기압+고습) 이상치는 완전제외가
    # 아니라 가중치만 낮춤 - detect_outliers_multivariate.py에서 확인: 착빙형(풍속대비
    # 발전량 이상)과 달리 태풍형은 개별변수가 실제로 극단적인 '진짜' 상황이라, 완전히
    # 배제하면 2025년 유사 강풍(산불 사건 등) 학습기회를 잃을 위험이 있음.
    "TYPHOON_SP_THRESHOLD": 94000,     # Pa - 이보다 낮으면 저기압
    "TYPHOON_RH_THRESHOLD": 90,        # % - 이보다 높으면 고습
    "TYPHOON_WEIGHT_FACTOR": 0.3,      # 조건 해당 시 가중치를 이 비율로 낮춤(완전제외 아님)
    "USE_WAKE_ALIGN":    False,  # [동훈추가] 그룹1->그룹2 후류 정렬 지수 - SCADA 실측 풍속으로 검증시 반대방향(기각)
    "PRESSURE_TEND_EXCLUDE_GROUPS": ["kpx_group_2"],  # (USE_PRESSURE_TEND=False라 현재 미사용)
    # --- 학습/후처리 기법 ---
    "USE_GEN_WEIGHT":  True,   # 발전량 가중치(0.5 + y/용량)
    "USE_10PCT_CUT":   True,   # 발전량<용량10% 시간 학습 제외
    "USE_G3_DENOISE":  True,   # group3 라벨 정제(unison SCADA와 1000kWh 초과 어긋난 시간 제외)
    "USE_G1_CALIB":    True,   # (레거시, 아래 CALIB_GROUPS로 대체됨 - 호환용으로만 유지)
    "CALIB_GROUPS": ["kpx_group_1", "kpx_group_2", "kpx_group_3"],  # [동훈재수정] group1 재포함
    # - 이전 실패는 shrinkage=1.0(전량적용) 기본값 때문이었던 것으로 판명. 3개 분할점
    # (1~3/6/9월 calib)으로 검증한 결과 shrinkage=0.5에서 전부 일관되게 Score 개선
    # (+0.009~0.014) 확인됨 - 우연이 아님. shrinkage=1.0(과거 실패했던 그대로)만 위험.
    "CALIB_SHRINKAGE": {"kpx_group_1": 0.5, "kpx_group_2": 1.0, "kpx_group_3": 1.0},
    "CALIB_DAYS_PER_GROUP": {"kpx_group_1": 30, "kpx_group_2": 45, "kpx_group_3": 45},  # [동훈추가]
    # 그룹1은 30일이 최적으로 확인됨(11/16 분할점에서 +0.0220, 45일보다 좋음). 전체 calib
    # 구간(45일)은 공통으로 확보하되, 그룹1만 그 중 최근 30일만 잘라서 fit_debias에 씀.
    "RECENCY_WEIGHT_GROUPS": {},  # [동훈수정] group1(wtg05)도 fold선 미세개선이었으나 실제
    # 리더보드는 하락(0.64129->0.64069) - 완전 기각. group2(wtg07)도 이미 기각됨.
    "CALIB_NBIN": {"kpx_group_2": 6, "kpx_group_3": 20},  # [동훈추가] fit_debias 구간 수.
    # 그룹3만 6->20으로 늘림 - gap감소율(5.8%->8.8%)과 Score(+0.013) 둘 다 뚜렷이 개선.
    # 그룹2는 구간을 늘려도 차이 없어 그대로(6) 유지.
    # (fold A/B 양쪽에서 계속 그룹1만 보정 후 하락하는 패턴이 반복됨 - 그룹1은 원래
    # 850hPa 단독 확정판 상태(보정 없음)가 최선이었던 것으로 판단, group2·3만 보정 적용)
    "OUTLIER_FILTER_GROUPS": ["kpx_group_2", "kpx_group_3"],  # [동훈추가] 이상치필터/태풍
    # 다운웨이트도 그룹1은 제외 - CALIB_GROUPS만 뺐더니 그룹1 학습데이터 자체가 여전히
    # 이상치필터의 영향을 받아 원래 850hPa 단독 확정판과 달라져 있었음(fold B g1이
    # 원기준선 0.6410보다도 낮은 0.6292로 나온 원인). 그룹1은 전부 원래 상태로 고정.
    # 그룹1에만 적용하던 것을 전 그룹으로 확장 (경계선 오차 시간대를 문턱 밑으로 밀어
    # FICR 개선을 겨냥 - check_temp_threshold_upperbound.py 등에서 확인된 대로
    # '경계선(8~12%) 시간대'가 가장 효율 높은 개선 대상이라는 분석 근거)
    "USE_SMOOTH":      True,   # 예보블록 내 3시간 이동중앙값 스무딩
    # --- 실행 옵션 ---
    "RUN_2FOLD":       True,   # 2-fold 검증(2023/2024) 실행
    "MAKE_SUBMISSION": True,   # 최종 제출 파일 생성 (빠른 스크리닝용, 확정시 True로)
    "SAVE_INDIVIDUAL": True,   # 각 모델 단독 예측(test) 저장 (동훈 분석/블렌딩용)
}

# --- 튜닝된 하이퍼파라미터 (Optuna로 탐색) ---
LGB_PARAMS = dict(objective="l1", subsample_freq=1, n_jobs=-1, verbosity=-1,
    n_estimators=500, learning_rate=0.02776406349122535, num_leaves=32,
    min_child_samples=37, subsample=0.7820145696884759,
    colsample_bytree=0.8538137732715506, reg_alpha=4.702394069679149, reg_lambda=1.101208820080058)
CAT_PARAMS = dict(loss_function="MAE", verbose=0, allow_writing_files=False,
    iterations=1100, learning_rate=0.029595812891825217, depth=6,
    l2_leaf_reg=1.556845906566105, random_strength=0.961019205773751)
XGB_PARAMS = dict(objective="reg:absoluteerror", tree_method="hist", n_jobs=-1, verbosity=0,
    n_estimators=700, learning_rate=0.023860339335477935, max_depth=5,
    subsample=0.6927606106505114, colsample_bytree=0.6597857945015654,
    min_child_weight=7, reg_lambda=3.4831272676354486)

TARGETS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}
FOLD_A = (pd.Timestamp("2023-01-01 01:00:00"), pd.Timestamp("2024-01-01 00:00:00"))
FOLD_B = (pd.Timestamp("2024-01-01 01:00:00"), pd.Timestamp("2025-01-01 00:00:00"))
# gain 중요도 하위 30% 가지치기에서 미리 제외했던 피처
REMOVED = ["ldaps_l50_ws3_max", "ldaps_l50_ws3_mean", "ldaps_l50_ws3_min", "ldaps_l50_ws_mean",
           "ldaps_l_power_proxy_min", "ldaps_surface_0_lssrate_mean",
           "_ws100_outlier_check", "_sp_outlier_check", "_rh_outlier_check"]  # [동훈추가] 필터링 전용

# ========================== 데이터 로드 ==========================
labels = pd.read_csv(CLEAN_DIR / "train_labels_clean.csv", encoding="utf-8-sig",
                     parse_dates=["kst_dtm"]).set_index("kst_dtm")
su = pd.read_csv(CLEAN_DIR / "scada_unison_clean.csv", encoding="utf-8-sig",
                 parse_dates=["kst_dtm"]).set_index("kst_dtm")
hu = su[[c for c in su.columns if "power" in c]].sum(axis=1).resample("1h", label="right", closed="right").sum()

# ========================== 피처 빌드 ==========================
def add_lagleads(X):
    """같은 예보블록 안 1~3시간 앞뒤 풍속(ramp)."""
    X = X.copy(); blk = (X.index - pd.to_timedelta(X["lead_hours"], unit="h"))
    for c in ["ldaps_l50_ws_gustmax_max", "gfs_g5_g850_ws"]:
        g = X[c].groupby(blk.values)
        for k in [1, 2, 3]:
            X[f"{c}_lag{k}"] = g.shift(k); X[f"{c}_lead{k}"] = g.shift(-k)
    return X

def add_pressure_tendency(X, hours=(3, 6)):
    """[동훈추가] 기압 경향(pressure tendency) - 저기압/전선 접근 감지용 표준 기상지표.
    add_lagleads()와 동일 원칙(같은 예보블록 안에서만 계산 - 데이터 누수 아님).
    base_means()로 만들어진 'base_gf_surface_0_sp' 컬럼을 대상으로 계산하므로
    반드시 base_means join 이후에 호출해야 함."""
    X = X.copy(); blk = (X.index - pd.to_timedelta(X["lead_hours"], unit="h"))
    col = "base_gf_surface_0_sp"
    if col in X.columns:
        g = X[col].groupby(blk.values)
        for h in hours:
            X[f"{col}_tendency{h}h"] = X[col] - g.shift(h)
    return X

WAKE_BEARING_G1_TO_G2 = 115.0  # 도(0=북,90=동,180=남,270=서) - 위경도로 계산된 실제 방위각

def add_wake_alignment_feature(X):
    """[동훈추가] group2의 후류(wake) 노출 지수.
    analyze_group2_forensics.py에서 발견: group2는 북동~동 계열 풍향(45~135도)에서
    압도적 표본(3,386시간)으로 오차가 크고 과대예측 편향이 뚜렷함. 그룹1->그룹2
    방위각을 위경도로 직접 계산하니 115도로, 정확히 그 구간 한가운데.
    -> group1이 상류(바람이 불어오는 쪽)에 있어 group2가 후류(바람그늘)에 들어가는
    지형 배치로 추정. cos(풍향-115도)를 연속값으로 제공 - 트리가 원본 u/v 분할만
    으로는 근사하기 어려운 대각선(oblique) 경계라 유효할 가능성 (v9/v30 같은 이산
    플래그와 달리, 다른 팀 문서의 '요정렬오차' 공식과 동일 원리의 연속 피처).
    +1에 가까우면 풍향이 정확히 그룹1->그룹2 방향(후류 노출 최대), -1이면 반대.
    """
    X = X.copy()
    u, v = X["base_gf_heightAboveGround_100_100u"], X["base_gf_heightAboveGround_100_100v"]
    wind_dir_rad = np.arctan2(u, v)
    bearing_rad = np.radians(WAKE_BEARING_G1_TO_G2)
    X["wake_align_g1_g2"] = np.cos(wind_dir_rad - bearing_rad)
    return X

def base_means(path, prefix):
    """베이스라인식 격자 전체 평균."""
    df = pd.read_csv(path, encoding="utf-8-sig"); df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    drop_c = {"data_available_kst_dtm", "grid_id", "latitude", "longitude", "surface_0_lsm"}
    cols = [c for c in df.columns if c not in {"forecast_kst_dtm", *drop_c}]
    g = df.groupby("forecast_kst_dtm")[cols].mean(); g.columns = [f"base_{prefix}_{c}" for c in g.columns]; return g

def pergrid(path, pre, pairs, cube_of=()):
    """격자별 풍속(+선택 ws³). 격자를 평균 안 내고 각각 컬럼으로."""
    usec = ["forecast_kst_dtm", "grid_id"] + [c for pr in pairs for c in pr[:2]]
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=usec); df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    for u, v, nm in pairs: df[nm] = np.sqrt(df[u] ** 2 + df[v] ** 2)
    for nm in cube_of: df[nm + "3"] = (df[nm] ** 3) / 1000.0
    valcols = [p[2] for p in pairs] + [nm + "3" for nm in cube_of]
    df = df.groupby(["forecast_kst_dtm", "grid_id"])[valcols].mean().reset_index(); outs = []
    for nm in valcols:
        w = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=nm)
        w.columns = [f"{pre}_g{c}_{nm}" for c in w.columns]; outs.append(w)
    return pd.concat(outs, axis=1)

def pergrid_dir(path, pre, u, v):
    """격자별 풍향 sin/cos."""
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["forecast_kst_dtm", "grid_id", u, v])
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    ws = np.sqrt(df[u] ** 2 + df[v] ** 2) + 1e-6; df["dsin"] = df[v] / ws; df["dcos"] = df[u] / ws
    df = df.groupby(["forecast_kst_dtm", "grid_id"])[["dsin", "dcos"]].mean().reset_index(); outs = []
    for nm in ["dsin", "dcos"]:
        w = df.pivot(index="forecast_kst_dtm", columns="grid_id", values=nm)
        w.columns = [f"{pre}_g{c}_{nm}" for c in w.columns]; outs.append(w)
    return pd.concat(outs, axis=1)

LP = [("heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
      ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "ws50")]
GP = [("heightAboveGround_100_100u", "heightAboveGround_100_100v", "ws100"),
      ("heightAboveGround_80_u", "heightAboveGround_80_v", "ws80"),
      ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "ws850")]

def build_features():
    print("피처 빌드...", flush=True)
    X = pd.read_csv(FEAT_DIR / "train_features.csv", encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"]).set_index("forecast_kst_dtm")
    Xte = pd.read_csv(FEAT_DIR / "test_features.csv", encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"]).set_index("forecast_kst_dtm")
    X = add_lagleads(X); Xte = add_lagleads(Xte)
    X = X.join(base_means(DATA_DIR / "train/ldaps_train.csv", "ld")).join(base_means(DATA_DIR / "train/gfs_train.csv", "gf"))
    Xte = Xte.join(base_means(DATA_DIR / "test/ldaps_test.csv", "ld")).join(base_means(DATA_DIR / "test/gfs_test.csv", "gf"))
    if CFG["USE_PRESSURE_TEND"]:
        # [동훈추가] base_means join 직후에 호출해야 base_gf_surface_0_sp 컬럼을 쓸 수 있음
        X = add_pressure_tendency(X); Xte = add_pressure_tendency(Xte)
    if CFG["USE_WAKE_ALIGN"]:
        # [동훈추가] 마찬가지로 base_gf_heightAboveGround_100_100u/v 컬럼이 필요해 이 위치
        X = add_wake_alignment_feature(X); Xte = add_wake_alignment_feature(Xte)
    if CFG["USE_T_INSTABILITY"]:
        X["t_instability"] = X["base_gf_heightAboveGround_2_2t"] - X["base_gf_isobaricInhPa_700_t"]
        Xte["t_instability"] = Xte["base_gf_heightAboveGround_2_2t"] - Xte["base_gf_isobaricInhPa_700_t"]
    if CFG["USE_PERGRID"]:
        X = X.join(pergrid(DATA_DIR / "train/ldaps_train.csv", "pgl", LP, cube_of=("ws50",))).join(pergrid(DATA_DIR / "train/gfs_train.csv", "pgg", GP, cube_of=("ws100",)))
        Xte = Xte.join(pergrid(DATA_DIR / "test/ldaps_test.csv", "pgl", LP, cube_of=("ws50",))).join(pergrid(DATA_DIR / "test/gfs_test.csv", "pgg", GP, cube_of=("ws100",)))
    if CFG["USE_PERGRID_DIR"]:
        X = X.join(pergrid_dir(DATA_DIR / "train/ldaps_train.csv", "pgld", "heightAboveGround_10_10u", "heightAboveGround_10_10v")).join(pergrid_dir(DATA_DIR / "train/gfs_train.csv", "pggd", "heightAboveGround_100_100u", "heightAboveGround_100_100v"))
        Xte = Xte.join(pergrid_dir(DATA_DIR / "test/ldaps_test.csv", "pgld", "heightAboveGround_10_10u", "heightAboveGround_10_10v")).join(pergrid_dir(DATA_DIR / "test/gfs_test.csv", "pggd", "heightAboveGround_100_100u", "heightAboveGround_100_100v"))
    if CFG["USE_850_DIR"]:
        # [동훈추가] GFS 850hPa 격자별 풍향 - 원본엔 100m까지만 있던 것을 확장
        X = X.join(pergrid_dir(DATA_DIR / "train/gfs_train.csv", "pgg850d", "isobaricInhPa_850_u", "isobaricInhPa_850_v"))
        Xte = Xte.join(pergrid_dir(DATA_DIR / "test/gfs_test.csv", "pgg850d", "isobaricInhPa_850_u", "isobaricInhPa_850_v"))

    # [동훈추가] 이상치(착빙/강제정지 등) 필터용 내부 풍속 컬럼. FEATS에는 안 넣음(피처가
    # 아니라 학습데이터 필터링에만 씀 - REMOVED에 추가해서 prune_features 이후에도 제외됨).
    X["_ws100_outlier_check"] = np.sqrt(X["base_gf_heightAboveGround_100_100u"]**2 + X["base_gf_heightAboveGround_100_100v"]**2)
    Xte["_ws100_outlier_check"] = np.sqrt(Xte["base_gf_heightAboveGround_100_100u"]**2 + Xte["base_gf_heightAboveGround_100_100v"]**2)
    # [동훈추가] 태풍형 다운웨이트(완전제외 아님)용 - detect_outliers_multivariate.py에서
    # 확인된 태풍 특징(저기압+고습도) 판정용 내부 컬럼
    X["_sp_outlier_check"] = X["base_gf_surface_0_sp"]
    Xte["_sp_outlier_check"] = Xte["base_gf_surface_0_sp"]
    X["_rh_outlier_check"] = X["base_gf_heightAboveGround_2_2r"]
    Xte["_rh_outlier_check"] = Xte["base_gf_heightAboveGround_2_2r"]
    return X, Xte

def prune_features(X):
    """gain 중요도 하위 30% 가지치기(피처 선택)."""
    FV = [c for c in X.columns if c not in REMOVED]
    Xtb = X[X.index < FOLD_B[0]]; gain = pd.Series(0.0, index=FV)
    for g in TARGETS[:2]:
        y = labels[g].reindex(Xtb.index); m = y.notna() & (y >= CAP[g] * 0.10)
        mo = lgb.LGBMRegressor(objective="l1", n_estimators=900, learning_rate=0.04, num_leaves=63,
                               min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               n_jobs=-1, verbosity=-1, random_state=42)
        mo.fit(Xtb.loc[m, FV], y[m])
        gs = pd.Series(mo.booster_.feature_importance("gain"), index=FV); gain += gs / gs.sum()
    gain /= 2
    return [f for f in FV if f not in gain.sort_values().head(int(len(FV) * 0.3)).index]

# ========================== 학습/예측 ==========================
def g3mask(idx):
    if not CFG["USE_G3_DENOISE"]: return pd.Series(True, index=idx)
    l3 = labels["kpx_group_3"].reindex(idx); s3 = hu.reindex(idx)
    return ~(((l3 - s3).abs() > 1000) & l3.notna() & s3.notna())

def outlier_mask(Xdf, y, cap, group=None):
    """[동훈추가] 풍속충분(>=OUTLIER_WS_THRESHOLD)한데 거의무발전(<OUTLIER_OUTPUT_THRESHOLD)
    = 착빙/강제정지 등 물리적 이상으로 간주해 학습에서 제외할 마스크(True=학습에 포함).
    오차율 기준이 아니라 물리적 개연성 기준이라 순환논리 없음 - 모델 예측 불필요.
    group이 OUTLIER_FILTER_GROUPS에 없으면 그 그룹은 원본 그대로(전부 True)."""
    if not CFG["USE_OUTLIER_FILTER"] or (group is not None and group not in CFG["OUTLIER_FILTER_GROUPS"]):
        return pd.Series(True, index=y.index)
    ws = Xdf["_ws100_outlier_check"].reindex(y.index)
    is_outlier = (ws >= CFG["OUTLIER_WS_THRESHOLD"]) & (y < cap * CFG["OUTLIER_OUTPUT_THRESHOLD"])
    return ~is_outlier.fillna(False)

def typhoon_weight_factor(Xdf, idx, group=None):
    """[동훈추가] 태풍형(저기압+고습) 다운웨이트 계수. 완전제외(outlier_mask)와 달리
    가중치만 낮춤 - 진짜 극단상황이라 완전배제 시 유사 미래사례 학습기회 손실 위험.
    group이 OUTLIER_FILTER_GROUPS에 없으면 그 그룹은 가중치 그대로(전부 1.0)."""
    if not CFG["USE_TYPHOON_DOWNWEIGHT"] or (group is not None and group not in CFG["OUTLIER_FILTER_GROUPS"]):
        return pd.Series(1.0, index=idx)
    sp = Xdf["_sp_outlier_check"].reindex(idx)
    rh = Xdf["_rh_outlier_check"].reindex(idx)
    is_typhoon = (sp < CFG["TYPHOON_SP_THRESHOLD"]) & (rh > CFG["TYPHOON_RH_THRESHOLD"])
    factor = pd.Series(1.0, index=idx)
    factor[is_typhoon.fillna(False)] = CFG["TYPHOON_WEIGHT_FACTOR"]
    return factor

def gscore(a, f, cap):
    """대회 지표(그룹별): 0.5*(1-평균오차) + 0.5*발전량가중 밴드점수(≤6%4,≤8%3,>8%0)."""
    v = (a >= cap * 0.10) & ~np.isnan(a); er = np.abs(f[v] - a[v]) / cap
    up = np.select([er <= 0.06, er <= 0.08], [4.0, 3.0], default=0.0)
    return 0.5 * (1 - er.mean()) + 0.5 * (np.sum(a[v] * up) / np.sum(a[v] * 4.0))

def _sw(y, cap):
    return (0.5 + y / cap).to_numpy() if CFG["USE_GEN_WEIGHT"] else np.ones(len(y))

def _recency_weight(idx, group):
    """[동훈추가] wtg07 만성저성능의 완만한 개선추세(2022 ~60%대 -> 2024 ~68~70%대)를
    반영해 group2 학습 시 최근(2024) 데이터에 더 높은 가중치를 줌. 다른 그룹은 영향없음."""
    factors = CFG.get("RECENCY_WEIGHT_GROUPS", {})
    if group not in factors:
        return np.ones(len(idx))
    year = idx.year
    w = np.ones(len(idx), dtype=float)
    w[year == 2022] = factors[group].get(2022, 1.0)
    w[year == 2023] = factors[group].get(2023, 1.0)
    w[year == 2024] = factors[group].get(2024, 1.0)
    return w

def predict_components(Xtr, Xout, FEATS):
    """그룹별·모델별 예측 {g:{'lgb','cat','xgb'}} (스무딩 전 raw)."""
    out = {g: {} for g in TARGETS}; dm = g3mask(Xtr.index)
    # pooled 스택(모든 그룹 합쳐 학습 — group3용)
    xs, ys, ws = [], [], []
    for gi, gg in enumerate(TARGETS):
        y = labels[gg].reindex(Xtr.index); m = y.notna()
        if CFG["USE_10PCT_CUT"]: m &= (y >= CAP[gg] * 0.10)
        if gg == "kpx_group_3": m &= dm
        m &= outlier_mask(Xtr, y, CAP[gg], group=gg)  # [동훈추가]
        xg = Xtr.loc[m, FEATS].copy(); xg["group_id"] = gi; xs.append(xg)
        yy = y[m] / CAP[gg]; ys.append(yy)
        w_typhoon = typhoon_weight_factor(Xtr, y[m].index, group=gg)  # [동훈추가]
        ws.append(_sw(y[m], CAP[gg]) * w_typhoon.to_numpy())
    SX, SY, SW = pd.concat(xs), pd.concat(ys), np.concatenate(ws)
    for g in TARGETS:
        pool = (g == "kpx_group_3"); y = labels[g].reindex(Xtr.index); m = y.notna()
        if CFG["USE_10PCT_CUT"]: m &= (y >= CAP[g] * 0.10)
        if pool: m &= dm
        m &= outlier_mask(Xtr, y, CAP[g], group=g)  # [동훈추가]
        # [동훈추가] group별 피처 제외 - group3(pool)은 구조상 배제 불가하므로 개별학습(g1/g2)만 적용
        group_feats = FEATS
        if (not pool) and g in CFG.get("PRESSURE_TEND_EXCLUDE_GROUPS", []):
            group_feats = [f for f in FEATS if "tendency" not in f]
        w = _sw(y[m], CAP[g]) * typhoon_weight_factor(Xtr, y[m].index, group=g).to_numpy() * _recency_weight(y[m].index, g)  # [동훈추가]
        Xtrm = Xtr.loc[m, group_feats]; Xof = Xout[group_feats]; cap = CAP[g]
        # LightGBM (시드 평균)
        if "lgb" in CFG["MODELS"]:
            ps = []
            for sd in range(CFG["N_SEEDS"]):
                seed = [42, 7, 2026, 777, 1234, 2024, 99, 5][sd % 8]
                mo = lgb.LGBMRegressor(**dict(LGB_PARAMS, random_state=seed))
                if pool: mo.fit(SX, SY, sample_weight=SW); xo = Xof.copy(); xo["group_id"] = 2; ps.append(mo.predict(xo) * cap)
                else: mo.fit(Xtrm, y[m], sample_weight=w); ps.append(mo.predict(Xof))
            out[g]["lgb"] = np.mean(ps, axis=0)
        # CatBoost
        if "cat" in CFG["MODELS"]:
            mc = CatBoostRegressor(**dict(CAT_PARAMS, random_seed=42))
            if pool: mc.fit(SX.fillna(-999), SY, sample_weight=SW); xo = Xof.fillna(-999).copy(); xo["group_id"] = 2; out[g]["cat"] = mc.predict(xo) * cap
            else: mc.fit(Xtrm.fillna(-999), y[m], sample_weight=w); out[g]["cat"] = mc.predict(Xof.fillna(-999))
        # XGBoost
        if "xgb" in CFG["MODELS"]:
            mx = XGBRegressor(**dict(XGB_PARAMS, random_state=123))
            if pool: mx.fit(SX, SY, sample_weight=SW); xo = Xof.copy(); xo["group_id"] = 2; out[g]["xgb"] = mx.predict(xo) * cap
            else: mx.fit(Xtrm, y[m], sample_weight=w); out[g]["xgb"] = mx.predict(Xof)
    return out

def _smooth(raw, Xout, cap):
    if CFG["USE_SMOOTH"]:
        blk = (Xout.index - pd.to_timedelta(Xout["lead_hours"], unit="h"))
        raw = pd.Series(raw, index=Xout.index).groupby(blk.values).transform(lambda z: z.rolling(3, center=True, min_periods=1).median()).to_numpy()
    return np.clip(raw, 0, cap)

def blend(comp, g, Xout):
    """선택 모델을 정규화된 가중치로 섞고 스무딩+클리핑.
    [동훈수정] 그룹별 최적 가중치(GROUP_WEIGHTS)를 우선 사용, 없으면 공통 WEIGHTS로 폴백."""
    cap = CAP[g]
    gw = CFG["GROUP_WEIGHTS"].get(g, CFG["WEIGHTS"])
    ws = {k: gw[k] for k in CFG["MODELS"]}; s = sum(ws.values())
    raw = sum(ws[k] / s * comp[g][k] for k in CFG["MODELS"])
    return _smooth(raw, Xout, cap)

def fit_debias(f, a, cap, nbin=6):
    d = pd.DataFrame({"f": f, "a": a}); d = d[(a >= cap * 0.10) & ~np.isnan(a)]
    d["b"] = pd.qcut(d.f, nbin, labels=False, duplicates="drop")
    gb = d.groupby("b").agg(fc=("f", "median"), dl=("a", lambda s: np.median(s.values - d.loc[s.index, "f"].values)))
    return gb["fc"].to_numpy(), gb["dl"].to_numpy()

def apply_cal(f, fc, dl, cap, shrinkage=1.0):
    # [동훈추가] shrinkage - v25/26(FICR조정)에서 검증된 원칙 재사용. Fold A에서
    # 편향보정이 과적합 위험을 보여(calib 구간이 2022년 11~12월, 데이터 초창기라
    # 대표성이 낮을 수 있음), 보정값을 100% 그대로 적용하지 않고 일부만 적용해서
    # 위험을 줄임.
    offset = np.interp(f, fc, dl, left=dl[0], right=dl[-1])
    return np.clip(f + offset * shrinkage, 0, cap)

# ========================== 실행 ==========================
def main():
    X, Xte = build_features()
    FEATS = prune_features(X)
    print(f"피처 {len(FEATS)}개 | 모델 {CFG['MODELS']} | 가중치 {CFG['WEIGHTS']}", flush=True)

    # --- 2-fold 검증 ---
    if CFG["RUN_2FOLD"]:
        print("[2-fold 검증]", flush=True)
        CALIB_DAYS = 45  # [동훈추가] fold와 최종제출이 반드시 같은 calib 방식을 쓰도록 통일
        for fold, (vs, ve), grp in [("A", FOLD_A, TARGETS[:2]), ("B", FOLD_B, TARGETS)]:
            calib_end = vs - pd.Timedelta(days=1)
            calib_start = calib_end - pd.Timedelta(days=CALIB_DAYS)
            Xtr = X[X.index < calib_start]
            Xcalib = X[(X.index >= calib_start) & (X.index <= calib_end)]
            va = (X.index >= vs) & (X.index <= ve); Xva = X[va]
            Xout_combined = pd.concat([Xcalib, Xva])
            comp = predict_components(Xtr, Xout_combined, FEATS)

            group_scores = {}
            fold_b_dump = {}
            for g in grp:
                p_all = blend(comp, g, Xout_combined)
                p_calib, p_va = p_all[:len(Xcalib)], p_all[len(Xcalib):]
                a_calib = labels[g].reindex(Xcalib.index).to_numpy(float)
                a_va = labels[g].reindex(Xva.index).to_numpy(float)

                p_va_raw = p_va.copy()  # [동훈추가] 보정 전 값 따로 보관
                if g in CFG["CALIB_GROUPS"]:
                    # [동훈추가] 그룹별 calib일수만큼만 최근 구간 사용
                    g_days = CFG["CALIB_DAYS_PER_GROUP"].get(g, CALIB_DAYS)
                    g_calib_start = calib_end - pd.Timedelta(days=g_days)
                    g_mask = Xcalib.index >= g_calib_start
                    fc, dl = fit_debias(p_calib[g_mask], a_calib[g_mask], CAP[g], nbin=CFG["CALIB_NBIN"].get(g, 6))
                    p_va = apply_cal(p_va, fc, dl, CAP[g], shrinkage=CFG["CALIB_SHRINKAGE"].get(g, 1.0))

                group_scores[g] = gscore(a_va, p_va, CAP[g])
                if fold == "B":
                    fold_b_dump[g] = {"actual": a_va, "pred_before": p_va_raw, "pred_after": p_va}

            if fold == "B":
                # [동훈추가] Phase1/Phase3 진단 도구(diagnose_error_matrix.py, slice_eval)에서
                # 쓸 수 있도록 보정 전/후 예측값을 CSV로 저장
                out_rows = []
                for g in grp:
                    d = fold_b_dump[g]
                    for i, dt in enumerate(Xva.index):
                        out_rows.append({"dt": dt, "group": g, "actual": d["actual"][i],
                                          "pred_before": d["pred_before"][i], "pred_after": d["pred_after"][i]})
                pd.DataFrame(out_rows).to_csv(OUT_DIR / "fold_b_calib_before_after.csv", index=False, encoding="utf-8-sig")
                print(f"  Fold B 보정전/후 예측값 저장: fold_b_calib_before_after.csv", flush=True)

            sc = np.nanmean(list(group_scores.values()))
            detail = ", ".join(f"{g}={s:.4f}" for g, s in group_scores.items())
            print(f"  fold {fold}: {sc:.4f}  ({detail})", flush=True)

    if not CFG["MAKE_SUBMISSION"]:
        return

    # --- 전체 학습 → 예측 ---
    print("[전체 학습 → 예측]", flush=True)
    compTe = predict_components(X, Xte, FEATS)

    # [동훈추가] fold 검증과 정확히 동일한 45일 calib 방식 (이전 FICR조정 실패의 원인이었던
    # 'fold와 제출 로직 불일치'를 반복하지 않기 위해 반드시 동일 로직 사용)
    calib_end_final = X.index.max() - pd.Timedelta(days=0)  # 학습데이터 마지막 시점
    calib_start_final = calib_end_final - pd.Timedelta(days=45)
    XtrFinal = X[X.index < calib_start_final]
    XcalibFinal = X[X.index >= calib_start_final]
    compFinalCalib = predict_components(XtrFinal, XcalibFinal, FEATS)

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    sdt = pd.to_datetime(sample["forecast_kst_dtm"])

    # 개별 모델 예측 저장 (동훈 분석/블렌딩용)
    if CFG["SAVE_INDIVIDUAL"]:
        for mdl in CFG["MODELS"]:
            sub = sample[["forecast_id", "forecast_kst_dtm"]].copy()
            for g in TARGETS:
                raw = _smooth(compTe[g][mdl], Xte, CAP[g])
                sub[g] = pd.Series(raw, index=Xte.index).reindex(sdt.values).to_numpy()
            sub.to_csv(OUT_DIR / f"pred_A_{mdl}.csv", index=False, encoding="utf-8-sig")
        print("  개별 예측 저장: pred_A_*.csv", flush=True)

    # 앙상블 + 그룹별 편향보정 (fold와 동일 45일 calib 로직)
    sub = sample[["forecast_id", "forecast_kst_dtm"]].copy()
    for g in TARGETS:
        sub[g] = pd.Series(blend(compTe, g, Xte), index=Xte.index).reindex(sdt.values).to_numpy()
    for g in CFG["CALIB_GROUPS"]:
        pC_full = blend(compFinalCalib, g, XcalibFinal)
        aC_full = labels[g].reindex(XcalibFinal.index).to_numpy(float)
        # [동훈추가] 그룹별 calib일수만큼만 최근 구간 사용 (fold와 동일 로직)
        g_days = CFG["CALIB_DAYS_PER_GROUP"].get(g, 45)
        g_calib_start = calib_end_final - pd.Timedelta(days=g_days)
        g_mask = XcalibFinal.index >= g_calib_start
        fc, dl = fit_debias(pC_full[g_mask], aC_full[g_mask], CAP[g], nbin=CFG["CALIB_NBIN"].get(g, 6))
        sub[g] = apply_cal(sub[g].to_numpy(float), fc, dl, CAP[g], shrinkage=CFG["CALIB_SHRINKAGE"].get(g, 1.0))
    for g in TARGETS: sub[g] = np.clip(sub[g], 0, CAP[g])

    # 검증 후 저장
    assert len(sub) == 8760 and not sub[TARGETS].isna().any().any()
    for g in TARGETS: assert (sub[g] >= 0).all() and (sub[g] <= CAP[g]).all()
    sub["forecast_kst_dtm"] = sdt.dt.strftime("%Y-%m-%d %H:%M:%S")
    sub.to_csv(OUT_DIR / "submit_A.csv", index=False, encoding="utf-8-sig")
    print("저장: submit_A.csv | 평균 " + ", ".join(f"{g[-1]}:{sub[g].mean():.0f}" for g in TARGETS), flush=True)

if __name__ == "__main__":
    main()
