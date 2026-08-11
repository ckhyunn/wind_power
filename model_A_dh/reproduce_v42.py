"""
팀원(현윤) v42(리더보드 0.64410) 정확한 재현. BARAM2026_v42_전체코드.py를
그대로 따르되, 경로만 우리 저장소 구조에 맞춤.

실행 (model_A_dh 폴더 안에서, model_A_dh.py로 submit_A.csv 생성된 후):
    python reproduce_v42.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}


def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return v[np.searchsorted(cw, cw[-1] / 2.0)]


def fit_bias(pred, actual, cap, nbin, weighted=False):
    d = pd.DataFrame({"f": pred, "a": actual})
    d = d[(d.a >= cap * 0.10) & d.f.notna()]
    d["b"] = pd.qcut(d.f, nbin, labels=False, duplicates="drop")
    fc, dl = [], []
    for _, grp in d.groupby("b"):
        diff = (grp.a - grp.f).to_numpy()
        fc.append(grp.f.median())
        dl.append(weighted_median(diff, grp.a.to_numpy()) if weighted else np.median(diff))
    fc, dl = np.array(fc), np.array(dl)
    order = np.argsort(fc)
    return fc[order], dl[order]


def fit_genweight_delta(pred, actual, cap, nbin):
    """[핵심] '발전량가중 편향 - 무가중 편향'의 차이만 계산 (STEP3,4용)."""
    d = pd.DataFrame({"f": pred, "a": actual})
    d = d[(d.a >= cap * 0.10) & d.f.notna()]
    d["b"] = pd.qcut(d.f, nbin, labels=False, duplicates="drop")
    fc, delta = [], []
    for _, grp in d.groupby("b"):
        diff = (grp.a - grp.f).to_numpy()
        fc.append(grp.f.median())
        delta.append(weighted_median(diff, grp.a.to_numpy()) - np.median(diff))
    fc, delta = np.array(fc), np.array(delta)
    order = np.argsort(fc)
    return fc[order], delta[order]


def apply_bias(pred, fc, dl, cap, strength=1.0):
    adj = strength * np.interp(pred, fc, dl, left=dl[0], right=dl[-1])
    return np.clip(pred + adj, 0, cap)


def recent_window(index, days):
    return index > (index.max() - pd.Timedelta(days=days))


def main():
    sub_a = pd.read_csv(REPO_ROOT / "model_A_dh" / "output" / "submit_A.csv", encoding="utf-8-sig")
    v32 = (pd.read_csv(REPO_ROOT / "model_A" / "submit_v32.csv", encoding="utf-8-sig")
             .set_index("forecast_id").reindex(sub_a["forecast_id"]).reset_index())
    oof = pd.read_csv(REPO_ROOT / "model_A" / "output_2024" / "pred_A_blend_2024.csv",
                       parse_dates=["forecast_kst_dtm"]).set_index("forecast_kst_dtm")
    act = pd.read_csv(REPO_ROOT / "model_A" / "output_2024" / "actual_2024.csv",
                       parse_dates=["forecast_kst_dtm"]).set_index("forecast_kst_dtm")

    sub = sub_a.copy()

    # ---------- STEP 1&2. group1 강화 (v35 -> v39), 순차 적용 ----------
    g = "kpx_group_1"; cap = CAP[g]
    win = recent_window(act.index, days=30)
    fc, dl = fit_bias(oof[g][win].to_numpy(), act[g][win].to_numpy(), cap, nbin=6, weighted=False)
    g1 = apply_bias(sub_a[g].to_numpy(), fc, dl, cap, strength=0.5)   # STEP1
    g1 = apply_bias(g1,                  fc, dl, cap, strength=0.3)   # STEP2
    sub[g] = g1
    print(f"STEP1&2 g1: {sub_a[g].mean():.0f} -> {sub[g].mean():.0f} (목표 9166)")

    # ---------- STEP 3. group3 발전량가중 델타 (v41) ----------
    g = "kpx_group_3"; cap = CAP[g]
    win = recent_window(act.index, days=45)
    fc, delta = fit_genweight_delta(oof[g][win].to_numpy(), act[g][win].to_numpy(), cap, nbin=20)
    adj = 1.0 * np.interp(v32[g].to_numpy(), fc, delta, left=delta[0], right=delta[-1])
    sub[g] = np.clip(sub[g].to_numpy() + adj, 0, cap)
    print(f"STEP3   g3: {sub_a[g].mean():.0f} -> {sub[g].mean():.0f} (조정 {adj.mean():+.0f})")

    # ---------- STEP 4. group2 발전량가중 델타, strength=0.5 (v42) ----------
    g = "kpx_group_2"; cap = CAP[g]
    win = recent_window(act.index, days=45)
    fc, delta = fit_genweight_delta(oof[g][win].to_numpy(), act[g][win].to_numpy(), cap, nbin=6)
    adj = 0.5 * np.interp(v32[g].to_numpy(), fc, delta, left=delta[0], right=delta[-1])
    sub[g] = np.clip(sub[g].to_numpy() + adj, 0, cap)
    print(f"STEP4   g2: {sub_a[g].mean():.0f} -> {sub[g].mean():.0f} (조정 {adj.mean():+.0f})")

    for gg in CAP:
        sub[gg] = np.clip(sub[gg], 0, CAP[gg])
    assert len(sub) == 8760, "행수는 8760이어야 함"
    assert not sub[list(CAP)].isna().any().any(), "결측 있음"
    for gg in CAP:
        assert (sub[gg] >= 0).all() and (sub[gg] <= CAP[gg]).all(), f"{gg} 범위 벗어남"

    out = REPO_ROOT / "model_A_dh" / "output" / "submit_v42_reproduced.csv"
    sub.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out}")
    print("목표 리더보드: 0.64410 (1-NMAE 0.86947 / FICR 0.41873)")


if __name__ == "__main__":
    main()
