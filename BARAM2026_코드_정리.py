# -*- coding: utf-8 -*-
"""
=========================================================================
 BARAM 2026 풍력발전량 예측 — 전체 코드 정리 (현윤)
 최종 리더보드: 0.64404 (v41)
=========================================================================
 이 파일은 두 부분으로 구성됨:
   [PART 1] 베이스 모델  — model_A.py 원본 그대로 (LGBM+CatBoost+XGBoost 앙상블)
   [PART 2] 후처리 캘리브레이션 — 베이스 모델 출력을 보정해서 0.63625 → 0.64404로 개선한 부분
            (실제 팀 최종 제출인 v41을 만든 단계별 코드)

 실행 순서: PART 1 실행 → submit_A.csv(베이스 예측) 생성 → PART 2가 그 위에 보정 적용
=========================================================================
"""

# #########################################################################
# PART 1. 베이스 모델 (model_A.py)
# #########################################################################
r"""
import numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# ========================== CONFIG ==========================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = Path(r"C:\Users\ckhyu")           # ← 대회 원본 데이터(train/, test/, sample_submission.csv) 위치
FEAT_DIR   = SCRIPT_DIR / "features"           # train_features.csv, test_features.csv
CLEAN_DIR  = SCRIPT_DIR / "cleaned"            # train_labels_clean.csv, scada_unison_clean.csv
OUT_DIR    = SCRIPT_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

CFG = {
    # --- 앙상블 ---
    "MODELS":        ["lgb", "cat", "xgb"],
    "WEIGHTS":       {"lgb": 0.50, "cat": 0.10, "xgb": 0.40},
    "N_SEEDS":       5,
    # --- 피처 ---
    "USE_PERGRID":     True,   # 격자별 풍속 피처(ws10/ws50/ws50³, gfs ws100/80/850/ws100³)
    "USE_PERGRID_DIR": True,   # 격자별 풍향(sin/cos)
    # --- 학습/후처리 기법 ---
    "USE_GEN_WEIGHT":  True,   # 발전량 가중치(0.5 + y/용량)
    "USE_10PCT_CUT":   True,   # 발전량<용량10% 시간 학습 제외
    "USE_G3_DENOISE":  True,   # group3 라벨 정제(unison SCADA와 1000kWh 초과 어긋난 시간 제외)
    "USE_G1_CALIB":    True,   # group1 편향 보정(전체 2024 기준, 기본형)
    "USE_SMOOTH":      True,   # 예보블록 내 3시간 이동중앙값 스무딩
    "RUN_2FOLD":       True,
    "MAKE_SUBMISSION": True,
    "SAVE_INDIVIDUAL": True,
}

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
REMOVED = ["ldaps_l50_ws3_max", "ldaps_l50_ws3_mean", "ldaps_l50_ws3_min", "ldaps_l50_ws_mean",
           "ldaps_l_power_proxy_min", "ldaps_surface_0_lssrate_mean"]

# ========================== 데이터 로드 ==========================
labels = pd.read_csv(CLEAN_DIR / "train_labels_clean.csv", encoding="utf-8-sig",
                     parse_dates=["kst_dtm"]).set_index("kst_dtm")
su = pd.read_csv(CLEAN_DIR / "scada_unison_clean.csv", encoding="utf-8-sig",
                 parse_dates=["kst_dtm"]).set_index("kst_dtm")
hu = su[[c for c in su.columns if "power" in c]].sum(axis=1).resample("1h", label="right", closed="right").sum()

# ========================== 피처 빌드 ==========================
def add_lagleads(X):
    \"\"\"같은 예보블록 안 1~3시간 앞뒤 풍속(ramp).\"\"\"
    X = X.copy(); blk = (X.index - pd.to_timedelta(X["lead_hours"], unit="h"))
    for c in ["ldaps_l50_ws_gustmax_max", "gfs_g5_g850_ws"]:
        g = X[c].groupby(blk.values)
        for k in [1, 2, 3]:
            X[f"{c}_lag{k}"] = g.shift(k); X[f"{c}_lead{k}"] = g.shift(-k)
    return X

def base_means(path, prefix):
    \"\"\"베이스라인식 격자 전체 평균.\"\"\"
    df = pd.read_csv(path, encoding="utf-8-sig"); df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    drop_c = {"data_available_kst_dtm", "grid_id", "latitude", "longitude", "surface_0_lsm"}
    cols = [c for c in df.columns if c not in {"forecast_kst_dtm", *drop_c}]
    g = df.groupby("forecast_kst_dtm")[cols].mean(); g.columns = [f"base_{prefix}_{c}" for c in g.columns]; return g

def pergrid(path, pre, pairs, cube_of=()):
    \"\"\"격자별 풍속(+선택 ws³). 격자를 평균 안 내고 각각 컬럼으로.\"\"\"
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
    \"\"\"격자별 풍향 sin/cos.\"\"\"
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
    X = pd.read_csv(FEAT_DIR / "train_features.csv", encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"]).set_index("forecast_kst_dtm")
    Xte = pd.read_csv(FEAT_DIR / "test_features.csv", encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"]).set_index("forecast_kst_dtm")
    X = add_lagleads(X); Xte = add_lagleads(Xte)
    X = X.join(base_means(DATA_DIR / "train/ldaps_train.csv", "ld")).join(base_means(DATA_DIR / "train/gfs_train.csv", "gf"))
    Xte = Xte.join(base_means(DATA_DIR / "test/ldaps_test.csv", "ld")).join(base_means(DATA_DIR / "test/gfs_test.csv", "gf"))
    if CFG["USE_PERGRID"]:
        X = X.join(pergrid(DATA_DIR / "train/ldaps_train.csv", "pgl", LP, cube_of=("ws50",))).join(pergrid(DATA_DIR / "train/gfs_train.csv", "pgg", GP, cube_of=("ws100",)))
        Xte = Xte.join(pergrid(DATA_DIR / "test/ldaps_test.csv", "pgl", LP, cube_of=("ws50",))).join(pergrid(DATA_DIR / "test/gfs_test.csv", "pgg", GP, cube_of=("ws100",)))
    if CFG["USE_PERGRID_DIR"]:
        X = X.join(pergrid_dir(DATA_DIR / "train/ldaps_train.csv", "pgld", "heightAboveGround_10_10u", "heightAboveGround_10_10v")).join(pergrid_dir(DATA_DIR / "train/gfs_train.csv", "pggd", "heightAboveGround_100_100u", "heightAboveGround_100_100v"))
        Xte = Xte.join(pergrid_dir(DATA_DIR / "test/ldaps_test.csv", "pgld", "heightAboveGround_10_10u", "heightAboveGround_10_10v")).join(pergrid_dir(DATA_DIR / "test/gfs_test.csv", "pggd", "heightAboveGround_100_100u", "heightAboveGround_100_100v"))
    return X, Xte

def prune_features(X):
    \"\"\"gain 중요도 하위 30% 가지치기(피처 선택).\"\"\"
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

def gscore(a, f, cap):
    \"\"\"대회 지표(그룹별): 0.5*(1-평균오차) + 0.5*발전량가중 밴드점수(≤6%4,≤8%3,>8%0).\"\"\"
    v = (a >= cap * 0.10) & ~np.isnan(a); er = np.abs(f[v] - a[v]) / cap
    up = np.select([er <= 0.06, er <= 0.08], [4.0, 3.0], default=0.0)
    return 0.5 * (1 - er.mean()) + 0.5 * (np.sum(a[v] * up) / np.sum(a[v] * 4.0))

def _sw(y, cap):
    return (0.5 + y / cap).to_numpy() if CFG["USE_GEN_WEIGHT"] else np.ones(len(y))

def predict_components(Xtr, Xout, FEATS):
    \"\"\"그룹별·모델별 예측 {g:{'lgb','cat','xgb'}} (스무딩 전 raw).\"\"\"
    out = {g: {} for g in TARGETS}; dm = g3mask(Xtr.index)
    xs, ys, ws = [], [], []
    for gi, gg in enumerate(TARGETS):
        y = labels[gg].reindex(Xtr.index); m = y.notna()
        if CFG["USE_10PCT_CUT"]: m &= (y >= CAP[gg] * 0.10)
        if gg == "kpx_group_3": m &= dm
        xg = Xtr.loc[m, FEATS].copy(); xg["group_id"] = gi; xs.append(xg)
        yy = y[m] / CAP[gg]; ys.append(yy); ws.append(_sw(y[m], CAP[gg]))
    SX, SY, SW = pd.concat(xs), pd.concat(ys), np.concatenate(ws)
    for g in TARGETS:
        pool = (g == "kpx_group_3"); y = labels[g].reindex(Xtr.index); m = y.notna()
        if CFG["USE_10PCT_CUT"]: m &= (y >= CAP[g] * 0.10)
        if pool: m &= dm
        w = _sw(y[m], CAP[g]); Xtrm = Xtr.loc[m, FEATS]; Xof = Xout[FEATS]; cap = CAP[g]
        if "lgb" in CFG["MODELS"]:
            ps = []
            for sd in range(CFG["N_SEEDS"]):
                seed = [42, 7, 2026, 777, 1234, 2024, 99, 5][sd % 8]
                mo = lgb.LGBMRegressor(**dict(LGB_PARAMS, random_state=seed))
                if pool: mo.fit(SX, SY, sample_weight=SW); xo = Xof.copy(); xo["group_id"] = 2; ps.append(mo.predict(xo) * cap)
                else: mo.fit(Xtrm, y[m], sample_weight=w); ps.append(mo.predict(Xof))
            out[g]["lgb"] = np.mean(ps, axis=0)
        if "cat" in CFG["MODELS"]:
            mc = CatBoostRegressor(**dict(CAT_PARAMS, random_seed=42))
            if pool: mc.fit(SX.fillna(-999), SY, sample_weight=SW); xo = Xof.fillna(-999).copy(); xo["group_id"] = 2; out[g]["cat"] = mc.predict(xo) * cap
            else: mc.fit(Xtrm.fillna(-999), y[m], sample_weight=w); out[g]["cat"] = mc.predict(Xof.fillna(-999))
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
    \"\"\"선택 모델을 정규화된 가중치로 섞고 스무딩+클리핑.\"\"\"
    cap = CAP[g]; ws = {k: CFG["WEIGHTS"][k] for k in CFG["MODELS"]}; s = sum(ws.values())
    raw = sum(ws[k] / s * comp[g][k] for k in CFG["MODELS"])
    return _smooth(raw, Xout, cap)

def fit_debias(f, a, cap, nbin=6):
    \"\"\"구간별 편향 보정 규칙 계산 (무가중 median). PART 2에서 이 구조를 가중버전으로 확장함.\"\"\"
    d = pd.DataFrame({"f": f, "a": a}); d = d[(a >= cap * 0.10) & ~np.isnan(a)]
    d["b"] = pd.qcut(d.f, nbin, labels=False, duplicates="drop")
    gb = d.groupby("b").agg(fc=("f", "median"), dl=("a", lambda s: np.median(s.values - d.loc[s.index, "f"].values)))
    return gb["fc"].to_numpy(), gb["dl"].to_numpy()

def apply_cal(f, fc, dl, cap):
    return np.clip(f + np.interp(f, fc, dl, left=dl[0], right=dl[-1]), 0, cap)

# ========================== 실행 ==========================
def main():
    X, Xte = build_features()
    FEATS = prune_features(X)

    if CFG["RUN_2FOLD"]:
        for fold, (vs, ve), grp in [("A", FOLD_A, TARGETS[:2]), ("B", FOLD_B, TARGETS)]:
            Xtr = X[X.index < vs]; va = (X.index >= vs) & (X.index <= ve); Xva = X[va]
            comp = predict_components(Xtr, Xva, FEATS)
            sc = np.nanmean([gscore(labels[g].reindex(Xva.index).to_numpy(float), blend(comp, g, Xva), CAP[g]) for g in grp])
            print(f"  fold {fold}: {sc:.4f}")

    compTe = predict_components(X, Xte, FEATS)
    XtrB = X[X.index < FOLD_B[0]]; vb = (X.index >= FOLD_B[0]) & (X.index <= FOLD_B[1]); XvaB = X[vb]
    if CFG["USE_G1_CALIB"]:
        compB = predict_components(XtrB, XvaB, FEATS)

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    sdt = pd.to_datetime(sample["forecast_kst_dtm"])

    sub = sample[["forecast_id", "forecast_kst_dtm"]].copy()
    for g in TARGETS:
        sub[g] = pd.Series(blend(compTe, g, Xte), index=Xte.index).reindex(sdt.values).to_numpy()
    if CFG["USE_G1_CALIB"]:
        pB1 = blend(compB, "kpx_group_1", XvaB); aB1 = labels["kpx_group_1"].reindex(XvaB.index).to_numpy(float)
        fc, dl = fit_debias(pB1, aB1, CAP["kpx_group_1"])
        sub["kpx_group_1"] = apply_cal(sub["kpx_group_1"].to_numpy(float), fc, dl, CAP["kpx_group_1"])
    for g in TARGETS: sub[g] = np.clip(sub[g], 0, CAP[g])

    sub["forecast_kst_dtm"] = sdt.dt.strftime("%Y-%m-%d %H:%M:%S")
    sub.to_csv(OUT_DIR / "submit_A.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
"""

# #########################################################################
# PART 2. 후처리 캘리브레이션 (베이스 모델 출력을 0.63625 -> 0.64404 로 개선)
# #########################################################################
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# 2-1. 공통 유틸: 구간별 편향 보정 (Quantile Bias Correction)
# -------------------------------------------------------------------------
def fit_debias_window(pred, actual, cap, nbin=6, weighted=False):
    """
    예측값을 크기순 nbin개 구간으로 나눠, 각 구간의 편향(실제-예측)을 계산.
    weighted=True 면 편향을 '발전량 가중 median'으로 계산 (FICR이 발전량가중이라 더 정확).

    pred, actual : 같은 길이의 numpy array (보정 규칙을 만들 기간의 예측/실제)
    cap          : 그룹 용량 (kWh)
    반환값       : (구간 대표 예측값 배열, 구간별 편향 배열) - interp에 사용
    """
    d = pd.DataFrame({"f": pred, "a": actual})
    d = d[(d.a >= cap * 0.10) & ~d.f.isna()]          # 채점 대상(용량 10% 이상)만 사용
    d["b"] = pd.qcut(d.f, nbin, labels=False, duplicates="drop")

    fc, dl = [], []
    for b, grp in d.groupby("b"):
        diff = (grp.a - grp.f).to_numpy()
        fc.append(grp.f.median())
        if weighted:
            dl.append(_weighted_median(diff, grp.a.to_numpy()))   # 발전량 가중
        else:
            dl.append(np.median(diff))                             # 무가중(팀원 기본 방식)
    fc, dl = np.array(fc), np.array(dl)
    order = np.argsort(fc)
    return fc[order], dl[order]


def _weighted_median(values, weights):
    """가중 중앙값: 누적가중치가 절반을 넘는 지점의 값을 반환."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return v[np.searchsorted(cw, cw[-1] / 2.0)]


def apply_bias(pred, fc, dl, cap, strength=1.0):
    """
    구간별 편향을 예측값에 적용. strength로 적용 강도 조절 가능
    (예: 팀원이 group1에 0.3~0.5만 적용했던 것 → 여기서 1.0 이상으로 강화).
    """
    adjustment = strength * np.interp(pred, fc, dl, left=dl[0], right=dl[-1])
    return np.clip(pred + adjustment, 0, cap)


# -------------------------------------------------------------------------
# 2-2. 실제 적용 — v41을 만든 단계 (베이스 예측 submit_A.csv 위에 순서대로 적용)
# -------------------------------------------------------------------------
"""
아래는 submit_A.csv(베이스 모델 앙상블+기본 g1보정 결과)를 입력으로 받아
최종 v41(0.64404)을 만드는 실제 순서. 2024년 OOF(pred_A_blend_2024.csv, actual_2024.csv)가
편향 계산에 필요함 (model_A.py의 predict_components로 2023 학습 -> 2024 예측해서 만든 것).

CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}

# --- STEP A. group1 최근창(30일) 구간별 편향으로 강화 (v35 -> v39) ---
# 2024년 실측 마지막 30일만 사용해 편향 계산 (전체기간이 아니라 '최근'이 핵심)
recent_g1 = actual_2024.index > (actual_2024.index.max() - pd.Timedelta(days=30))
fc1, dl1 = fit_debias_window(
    pred_A_blend_2024["kpx_group_1"][recent_g1],
    actual_2024["kpx_group_1"][recent_g1],
    cap=CAP["kpx_group_1"], nbin=6, weighted=False,
)
# 팀원 베이스(submit_A) g1에 편향을 추가로 더 반영 (강도를 단계적으로 올려가며 리더보드로 정점 탐색)
g1_v39 = apply_bias(submit_A["kpx_group_1"], fc1, dl1, CAP["kpx_group_1"], strength=0.3)
#  (팀원 원본이 이미 편향의 일부를 반영하고 있으므로, 여기서 strength는 '추가로 더 얹는 양')

# --- STEP B. group3 발전량 가중 편향 (v39 -> v41, 가장 큰 개선) ---
# 2024년 실측 마지막 45일 사용, 이번엔 weighted=True로 발전량 가중 median 사용
recent_g3 = actual_2024.index > (actual_2024.index.max() - pd.Timedelta(days=45))
fc3, dl3 = fit_debias_window(
    pred_A_blend_2024["kpx_group_3"][recent_g3],
    actual_2024["kpx_group_3"][recent_g3],
    cap=CAP["kpx_group_3"], nbin=20, weighted=True,     # ← 핵심: weighted=True
)
# raw(캘리브레이션 전 베이스) 예측에 발전량가중 보정을 적용
g3_v41 = apply_bias(raw_v32["kpx_group_3"], fc3, dl3, CAP["kpx_group_3"], strength=1.0)

# --- 최종 조합 ---
submit_v41 = submit_A.copy()
submit_v41["kpx_group_1"] = g1_v39      # STEP A 결과
submit_v41["kpx_group_2"] = submit_A["kpx_group_2"]   # 그대로 유지 (이미 최적)
submit_v41["kpx_group_3"] = g3_v41      # STEP B 결과
for g in CAP:
    submit_v41[g] = np.clip(submit_v41[g], 0, CAP[g])
"""


# -------------------------------------------------------------------------
# 2-3. 오프라인 검증 함수 (2023 -> 2024 시간분리 검증에 사용)
# -------------------------------------------------------------------------
def gscore(actual, forecast, cap):
    """대회 지표(그룹별 단일점수): 0.5*(1-평균오차율) + 0.5*발전량가중 밴드점수."""
    valid = (actual >= cap * 0.10) & ~np.isnan(actual)
    err = np.abs(forecast[valid] - actual[valid]) / cap
    unit_price = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    ficr = np.sum(actual[valid] * unit_price) / np.sum(actual[valid] * 4.0)
    nmae_term = 1 - err.mean()
    return 0.5 * nmae_term + 0.5 * ficr


def validate_timesplit(pred_train_period, actual_train_period,
                        pred_eval_period, actual_eval_period, cap, **debias_kwargs):
    """
    과거(train_period)로 편향 규칙을 만들어 미래(eval_period)에 적용했을 때
    점수가 개선되는지 확인하는 시간분리 검증 (예: 2023으로 만들어 2024에 적용).
    """
    fc, dl = fit_debias_window(pred_train_period, actual_train_period, cap, **debias_kwargs)
    corrected = apply_bias(pred_eval_period, fc, dl, cap)
    score_before = gscore(actual_eval_period, pred_eval_period, cap)
    score_after = gscore(actual_eval_period, corrected, cap)
    return score_before, score_after
