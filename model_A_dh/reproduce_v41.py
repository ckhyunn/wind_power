"""
팀원(현윤)의 v41(리더보드 0.64404) 재현 - BARAM2026_코드_정리.py의 STEP A/B를
그대로 구현. submit_A.csv(동훈, 0.64129)와 model_A/submit_v32.csv(원본)를 입력으로 함.

실행 (model_A_dh 폴더 안에서, model_A_dh.py로 submit_A.csv 생성된 후):
    python reproduce_v41.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}


def _weighted_median(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return v[np.searchsorted(cw, cw[-1] / 2.0)]


def fit_debias_window(pred, actual, cap, nbin=6, weighted=False):
    d = pd.DataFrame({"f": pred, "a": actual})
    d = d[(d.a >= cap * 0.10) & ~d.f.isna()]
    d["b"] = pd.qcut(d.f, nbin, labels=False, duplicates="drop")
    fc, dl = [], []
    for b, grp in d.groupby("b"):
        diff = (grp.a - grp.f).to_numpy()
        fc.append(grp.f.median())
        dl.append(_weighted_median(diff, grp.a.to_numpy()) if weighted else np.median(diff))
    fc, dl = np.array(fc), np.array(dl)
    order = np.argsort(fc)
    return fc[order], dl[order]


def apply_bias(pred, fc, dl, cap, strength=1.0):
    adjustment = strength * np.interp(pred, fc, dl, left=dl[0], right=dl[-1])
    return np.clip(pred + adjustment, 0, cap)


def main():
    submit_A = pd.read_csv(REPO_ROOT / "model_A_dh" / "output" / "submit_A.csv")
    v32 = pd.read_csv(REPO_ROOT / "model_A" / "submit_v32.csv")
    actual_2024 = pd.read_csv(REPO_ROOT / "model_A" / "output_2024" / "actual_2024.csv")
    pred_A_blend_2024 = pd.read_csv(REPO_ROOT / "model_A" / "output_2024" / "pred_A_blend_2024.csv")

    for df in [submit_A, v32, actual_2024, pred_A_blend_2024]:
        dtcol = "forecast_kst_dtm" if "forecast_kst_dtm" in df.columns else df.columns[0]
        df[dtcol] = pd.to_datetime(df[dtcol])
    actual_2024 = actual_2024.set_index("forecast_kst_dtm")
    pred_A_blend_2024 = pred_A_blend_2024.set_index("forecast_kst_dtm")

    # STEP A: group1 최근 30일 편향, strength=0.3 추가
    recent_g1 = actual_2024.index > (actual_2024.index.max() - pd.Timedelta(days=30))
    fc1, dl1 = fit_debias_window(
        pred_A_blend_2024.loc[recent_g1, "kpx_group_1"].to_numpy(),
        actual_2024.loc[recent_g1, "kpx_group_1"].to_numpy(),
        cap=CAP["kpx_group_1"], nbin=6, weighted=False,
    )
    g1_v39 = apply_bias(submit_A["kpx_group_1"].to_numpy(), fc1, dl1, CAP["kpx_group_1"], strength=0.3)

    # STEP B: group3 최근 45일, 발전량가중 편향, v32(raw) 기준
    recent_g3 = actual_2024.index > (actual_2024.index.max() - pd.Timedelta(days=45))
    fc3, dl3 = fit_debias_window(
        pred_A_blend_2024.loc[recent_g3, "kpx_group_3"].to_numpy(),
        actual_2024.loc[recent_g3, "kpx_group_3"].to_numpy(),
        cap=CAP["kpx_group_3"], nbin=20, weighted=True,
    )
    g3_v41 = apply_bias(v32["kpx_group_3"].to_numpy(), fc3, dl3, CAP["kpx_group_3"], strength=1.0)

    # [동훈추가] group2도 동일 기법(발전량가중, nbin20) 적용 - 오프라인 4개 시점 검증에서
    # 11/16(실전과 가장 유사)은 개선폭 최대(+0.0261, 현재보다 우위), 나머지 3개 시점도
    # 비가중 대비 손해폭이 확실히 줄어드는 일관된 패턴 확인됨 (그룹2 최초의 유망한 신호)
    recent_g2 = actual_2024.index > (actual_2024.index.max() - pd.Timedelta(days=45))
    fc2, dl2 = fit_debias_window(
        pred_A_blend_2024.loc[recent_g2, "kpx_group_2"].to_numpy(),
        actual_2024.loc[recent_g2, "kpx_group_2"].to_numpy(),
        cap=CAP["kpx_group_2"], nbin=20, weighted=True,
    )
    g2_new = apply_bias(v32["kpx_group_2"].to_numpy(), fc2, dl2, CAP["kpx_group_2"], strength=1.0)

    submit_v41 = submit_A.copy()
    submit_v41["kpx_group_1"] = g1_v39
    submit_v41["kpx_group_2"] = g2_new  # [동훈추가]
    submit_v41["kpx_group_3"] = g3_v41
    for g in CAP:
        submit_v41[g] = np.clip(submit_v41[g], 0, CAP[g])

    out_path = REPO_ROOT / "model_A_dh" / "output" / "submit_v41plus_g2.csv"
    submit_v41.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {out_path}")
    print(f"\n그룹별 평균값 비교:")
    for g in CAP:
        print(f"  [{g}] submit_A={submit_A[g].mean():.0f}  v41재현={submit_v41[g].mean():.0f}")


if __name__ == "__main__":
    main()
