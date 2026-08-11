"""
[재학습 없음] 그룹2 전용 - 예측값 기준(fit_debias) 대신 풍향 기준으로 보정했을 때
효과가 있는지 확인. analyze_group2_forensics.py에서 확인된 "풍향별로 편향 방향이
반대(NE-E는 과대예측, S-SW는 과소예측)"라는 발견을 직접 활용.

실행 (model_A_dh 폴더 안에서):
    python test_group2_dir_calib.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA_DIR = Path(r"C:\Users\doon2\Desktop\wind_power\data")  # model_A_dh.py와 동일하게 확인 필요
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import group_score

CAP = 21600
SPLITS = ["2024-03-01", "2024-06-01", "2024-09-01", "2024-11-16"]


def fit_debias_by_dir(dir_deg, a, f, nbin=8):
    """[버그수정] 경계(edges)를 명시적으로 고정해서 반환 - calib에 데이터 없는
    구간이 있어도 apply 시 경계가 어긋나지 않도록 함."""
    edges = np.linspace(0, 360, nbin + 1)
    d = pd.DataFrame({"dir": dir_deg, "a": a, "f": f})
    d["b"] = pd.cut(d["dir"], bins=edges, include_lowest=True)
    gb = d.groupby("b", observed=False).agg(
        dl=("a", lambda s: np.median(s.values - d.loc[s.index, "f"].values) if len(s) > 0 else np.nan)
    )
    dl_values = gb["dl"].fillna(0.0).to_numpy()  # 데이터 없는 구간은 보정 0(안전)
    return edges, dl_values


def apply_debias_by_dir(dir_deg, f, edges, dl_values, cap, shrinkage=1.0):
    """[버그수정] fit에서 받은 동일한 edges로 구간을 매김 (재생성 안 함)."""
    idx = np.asarray(pd.cut(dir_deg, bins=edges, include_lowest=True, labels=False))
    idx = np.where(np.isnan(idx), 0, idx).astype(int)
    idx = np.clip(idx, 0, len(dl_values) - 1)
    offset = dl_values[idx]
    return np.clip(f + offset * shrinkage, 0, cap)


def fit_debias_value(f, a, cap, nbin=6):
    d = pd.DataFrame({"f": f, "a": a})
    d["b"] = pd.qcut(d.f, nbin, labels=False, duplicates="drop")
    gb = d.groupby("b").agg(fc=("f", "median"), dl=("a", lambda s: np.median(s.values - d.loc[s.index, "f"].values)))
    return gb["fc"].to_numpy(), gb["dl"].to_numpy()


def apply_debias_value(f, fc, dl, cap, shrinkage=1.0):
    offset = np.interp(f, fc, dl, left=dl[0], right=dl[-1])
    return np.clip(f + offset * shrinkage, 0, cap)


def main():
    actual = pd.read_csv(DATA_DIR.parent / "model_A" / "output_2024" / "actual_2024.csv")
    lgb = pd.read_csv(DATA_DIR.parent / "model_A" / "output_2024" / "pred_A_lgb_2024.csv")
    cat = pd.read_csv(DATA_DIR.parent / "model_A" / "output_2024" / "pred_A_cat_2024.csv")
    for df in [actual, lgb, cat]:
        df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    m = actual.merge(lgb, on="forecast_kst_dtm", suffixes=("", "_lgb")).merge(cat, on="forecast_kst_dtm", suffixes=("", "_cat"))

    gfs = pd.read_csv(DATA_DIR / "train" / "gfs_train.csv", encoding="utf-8-sig")
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])
    agg = gfs.groupby("forecast_kst_dtm").agg(u=("heightAboveGround_100_100u", "mean"), v=("heightAboveGround_100_100v", "mean"))
    agg["wind_dir_deg"] = (np.degrees(np.arctan2(agg["u"], agg["v"])) + 360) % 360
    m = m.merge(agg[["wind_dir_deg"]], on="forecast_kst_dtm", how="left").sort_values("forecast_kst_dtm").reset_index(drop=True)

    t = "kpx_group_2"
    m2 = m.dropna(subset=[t, "wind_dir_deg"]).copy()
    m2["pred"] = 0.5 * m2[f"{t}_lgb"] + 0.5 * m2[f"{t}_cat"]
    valid = m2[m2[t] >= CAP * 0.10]

    for es in SPLITS:
        es_ts = pd.Timestamp(es)
        calib_start = es_ts - pd.Timedelta(days=45)
        calib = valid[(valid["forecast_kst_dtm"] >= calib_start) & (valid["forecast_kst_dtm"] < es_ts)]
        ev = valid[valid["forecast_kst_dtm"] >= es_ts]
        if len(calib) < 50 or len(ev) < 30:
            continue

        f_e, a_e = ev["pred"].to_numpy(), ev[t].to_numpy()
        base, _, _ = group_score(a_e, f_e, CAP)
        print(f"\n[calib종료={es}, calib={len(calib)}건, 평가={len(ev)}건] 보정없음 Score={base:.4f}")

        # 기존 방식(예측값 기준)
        fc, dl = fit_debias_value(calib["pred"].to_numpy(), calib[t].to_numpy(), CAP, nbin=6)
        for shrink in [0.5, 1.0]:
            f_cal = apply_debias_value(f_e, fc, dl, CAP, shrinkage=shrink)
            sc, _, _ = group_score(a_e, f_cal, CAP)
            print(f"   [기존:예측값기준] shrink={shrink}: Score={sc:.4f} ({sc-base:+.4f})")

        # 신규 방식(풍향 기준)
        edges, dl_values = fit_debias_by_dir(calib["wind_dir_deg"], calib[t], calib["pred"], nbin=8)
        for shrink in [0.5, 1.0]:
            f_cal = apply_debias_by_dir(ev["wind_dir_deg"], f_e, edges, dl_values, CAP, shrinkage=shrink)
            sc, _, _ = group_score(a_e, f_cal, CAP)
            print(f"   [신규:풍향기준]   shrink={shrink}: Score={sc:.4f} ({sc-base:+.4f})")


if __name__ == "__main__":
    main()
