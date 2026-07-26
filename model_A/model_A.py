# -*- coding: utf-8 -*-
"""
=========================================================================
 BARAM 2026 풍력발전량 예측 — A 모델 (현윤)  현재 최고: 리더보드 0.63625
=========================================================================
 구성: LDAPS/GFS 예보 → 피처(격자별 포함) → LGBM+CatBoost+XGBoost 앙상블
       → group3 라벨 정제(pooled) + g1 편향 보정 → 제출
 사용법:
   1) 아래 CONFIG의 DATA_DIR 를 대회 원본 데이터(train/, test/) 위치로 설정
   2) python model_A.py
 ablation(제거/변경 실험)은 CONFIG 토글만 바꾸면 됨 (README 참고)
 필요 패키지: requirements.txt
=========================================================================
"""
import numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# ========================== CONFIG ==========================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = Path(r"C:\Users\ckhyu")           # ← 대회 원본 데이터(train/, test/, sample_submission.csv) 위치
FEAT_DIR   = SCRIPT_DIR / "features"           # train_features.csv, test_features.csv (레포 포함)
CLEAN_DIR  = SCRIPT_DIR / "cleaned"            # train_labels_clean.csv, scada_unison_clean.csv (레포 포함)
OUT_DIR    = SCRIPT_DIR / "output"             # 결과 저장 위치
OUT_DIR.mkdir(exist_ok=True)

CFG = {
    # --- 앙상블 ---
    "MODELS":        ["lgb", "cat", "xgb"],          # 넣을 모델. 예: ["lgb"] / ["lgb","xgb"]
    "WEIGHTS":       {"lgb": 0.50, "cat": 0.10, "xgb": 0.40},  # 앙상블 비율(선택 모델만 자동 재정규화)
    "N_SEEDS":       5,                               # LGBM 시드 평균 개수
    # --- 피처 ---
    "USE_PERGRID":     True,   # 격자별 풍속 피처(ws10/ws50/ws50³, gfs ws100/80/850/ws100³)
    "USE_PERGRID_DIR": True,   # 격자별 풍향(sin/cos)
    # --- 학습/후처리 기법 ---
    "USE_GEN_WEIGHT":  True,   # 발전량 가중치(0.5 + y/용량)
    "USE_10PCT_CUT":   True,   # 발전량<용량10% 시간 학습 제외
    "USE_G3_DENOISE":  True,   # group3 라벨 정제(unison SCADA와 1000kWh 초과 어긋난 시간 제외)
    "USE_G1_CALIB":    True,   # group1 편향 보정
    "USE_SMOOTH":      True,   # 예보블록 내 3시간 이동중앙값 스무딩
    # --- 실행 옵션 ---
    "RUN_2FOLD":       True,   # 2-fold 검증(2023/2024) 실행
    "MAKE_SUBMISSION": True,   # 최종 제출 파일 생성
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
           "ldaps_l_power_proxy_min", "ldaps_surface_0_lssrate_mean"]

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
    if CFG["USE_PERGRID"]:
        X = X.join(pergrid(DATA_DIR / "train/ldaps_train.csv", "pgl", LP, cube_of=("ws50",))).join(pergrid(DATA_DIR / "train/gfs_train.csv", "pgg", GP, cube_of=("ws100",)))
        Xte = Xte.join(pergrid(DATA_DIR / "test/ldaps_test.csv", "pgl", LP, cube_of=("ws50",))).join(pergrid(DATA_DIR / "test/gfs_test.csv", "pgg", GP, cube_of=("ws100",)))
    if CFG["USE_PERGRID_DIR"]:
        X = X.join(pergrid_dir(DATA_DIR / "train/ldaps_train.csv", "pgld", "heightAboveGround_10_10u", "heightAboveGround_10_10v")).join(pergrid_dir(DATA_DIR / "train/gfs_train.csv", "pggd", "heightAboveGround_100_100u", "heightAboveGround_100_100v"))
        Xte = Xte.join(pergrid_dir(DATA_DIR / "test/ldaps_test.csv", "pgld", "heightAboveGround_10_10u", "heightAboveGround_10_10v")).join(pergrid_dir(DATA_DIR / "test/gfs_test.csv", "pggd", "heightAboveGround_100_100u", "heightAboveGround_100_100v"))
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

def gscore(a, f, cap):
    """대회 지표(그룹별): 0.5*(1-평균오차) + 0.5*발전량가중 밴드점수(≤6%4,≤8%3,>8%0)."""
    v = (a >= cap * 0.10) & ~np.isnan(a); er = np.abs(f[v] - a[v]) / cap
    up = np.select([er <= 0.06, er <= 0.08], [4.0, 3.0], default=0.0)
    return 0.5 * (1 - er.mean()) + 0.5 * (np.sum(a[v] * up) / np.sum(a[v] * 4.0))

def _sw(y, cap):
    return (0.5 + y / cap).to_numpy() if CFG["USE_GEN_WEIGHT"] else np.ones(len(y))

def predict_components(Xtr, Xout, FEATS):
    """그룹별·모델별 예측 {g:{'lgb','cat','xgb'}} (스무딩 전 raw)."""
    out = {g: {} for g in TARGETS}; dm = g3mask(Xtr.index)
    # pooled 스택(모든 그룹 합쳐 학습 — group3용)
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
    """선택 모델을 정규화된 가중치로 섞고 스무딩+클리핑."""
    cap = CAP[g]; ws = {k: CFG["WEIGHTS"][k] for k in CFG["MODELS"]}; s = sum(ws.values())
    raw = sum(ws[k] / s * comp[g][k] for k in CFG["MODELS"])
    return _smooth(raw, Xout, cap)

def fit_debias(f, a, cap, nbin=6):
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
    print(f"피처 {len(FEATS)}개 | 모델 {CFG['MODELS']} | 가중치 {CFG['WEIGHTS']}", flush=True)

    # --- 2-fold 검증 ---
    if CFG["RUN_2FOLD"]:
        print("[2-fold 검증]", flush=True)
        for fold, (vs, ve), grp in [("A", FOLD_A, TARGETS[:2]), ("B", FOLD_B, TARGETS)]:
            Xtr = X[X.index < vs]; va = (X.index >= vs) & (X.index <= ve); Xva = X[va]
            comp = predict_components(Xtr, Xva, FEATS)
            sc = np.nanmean([gscore(labels[g].reindex(Xva.index).to_numpy(float), blend(comp, g, Xva), CAP[g]) for g in grp])
            print(f"  fold {fold}: {sc:.4f}", flush=True)

    if not CFG["MAKE_SUBMISSION"]:
        return

    # --- 전체 학습 → 예측 ---
    print("[전체 학습 → 예측]", flush=True)
    compTe = predict_components(X, Xte, FEATS)
    XtrB = X[X.index < FOLD_B[0]]; vb = (X.index >= FOLD_B[0]) & (X.index <= FOLD_B[1]); XvaB = X[vb]
    if CFG["USE_G1_CALIB"]:
        compB = predict_components(XtrB, XvaB, FEATS)  # g1 보정용 2024 예측

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

    # 앙상블 + g1 보정
    sub = sample[["forecast_id", "forecast_kst_dtm"]].copy()
    for g in TARGETS:
        sub[g] = pd.Series(blend(compTe, g, Xte), index=Xte.index).reindex(sdt.values).to_numpy()
    if CFG["USE_G1_CALIB"]:
        pB1 = blend(compB, "kpx_group_1", XvaB); aB1 = labels["kpx_group_1"].reindex(XvaB.index).to_numpy(float)
        fc, dl = fit_debias(pB1, aB1, CAP["kpx_group_1"])
        sub["kpx_group_1"] = apply_cal(sub["kpx_group_1"].to_numpy(float), fc, dl, CAP["kpx_group_1"])
    for g in TARGETS: sub[g] = np.clip(sub[g], 0, CAP[g])

    # 검증 후 저장
    assert len(sub) == 8760 and not sub[TARGETS].isna().any().any()
    for g in TARGETS: assert (sub[g] >= 0).all() and (sub[g] <= CAP[g]).all()
    sub["forecast_kst_dtm"] = sdt.dt.strftime("%Y-%m-%d %H:%M:%S")
    sub.to_csv(OUT_DIR / "submit_A.csv", index=False, encoding="utf-8-sig")
    print("저장: submit_A.csv | 평균 " + ", ".join(f"{g[-1]}:{sub[g].mean():.0f}" for g in TARGETS), flush=True)

if __name__ == "__main__":
    main()
