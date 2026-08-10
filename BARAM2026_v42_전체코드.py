# -*- coding: utf-8 -*-
"""
===========================================================================
 BARAM 2026 풍력발전량 예측 — v42 전체 코드 (리더보드 0.64410, 현재 최고)
===========================================================================
 팀원 참고용. 이 파일 하나만 실행하면 submit_A.csv -> submit_v42.csv 가 생성됨.

 [전체 구조]
   베이스모델(model_A.py: LGB+Cat+XGB 앙상블)
        |
        v  raw 2025 예측 = submit_v32.csv
   팀원 최근창 편향보정 (3그룹)  =  submit_A.csv     <-- 이 파일의 입력
        |
        +-- STEP 1  g1 += 0.5 x 최근창편향(30일/6구간, 무가중)     -> v35
        +-- STEP 2  g1 += 0.3 x 같은 편향                          -> v39
        +-- STEP 3  g3 += 1.0 x (발전량가중 - 무가중) [45일/20구간] -> v41  (최대 점프)
        +-- STEP 4  g2 += 0.5 x (발전량가중 - 무가중) [45일/6구간]  -> v42  (최종)

 [필요 파일]
   submit_A.csv                 팀원 보정본 (보정의 출발점)
   submit_v32.csv               베이스 raw 2025 예측 (STEP 3,4의 보간 기준)
   output/pred_A_blend_2024.csv 2024 OOF 예측 (2022~23 학습 -> 2024 예측)
   output/actual_2024.csv       2024 실측
   * OOF 파일은 model_A.py 의 predict_components(2022~23 학습, 2024 예측)으로 생성

 [규칙 준수]
   - 편향은 전부 2024년(과거) 데이터로만 계산 -> 2025 예측에 미래정보 없음
   - 외부데이터 미사용, 로컬 추론만
===========================================================================
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ============================ 설정 ============================
BASE = Path(r"C:\Users\ckhyu\OneDrive\바탕 화면\brame2026_팀병합 후")
OOF_DIR = BASE / "baram2026_A" / "output"
V32_PATH = Path(r"C:\Users\ckhyu\OneDrive\바탕 화면\BARAM2026\submit_v32.csv")

CAP = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}


# ============================ 공통 함수 ============================
def weighted_median(values, weights):
    """발전량 가중 중앙값. FICR이 발전량 가중 지표라 이걸 써야 지표와 정합됨."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return v[np.searchsorted(cw, cw[-1] / 2.0)]


def fit_bias(pred, actual, cap, nbin, weighted=False):
    """
    예측값을 크기순 nbin개 구간으로 나눠 구간별 편향(실제-예측)을 계산.
    weighted=True 면 발전량 가중 중앙값 사용.
    반환: (구간 대표 예측값, 구간별 편향)  -> np.interp 로 사용
    """
    d = pd.DataFrame({"f": pred, "a": actual})
    d = d[(d.a >= cap * 0.10) & d.f.notna()]          # 채점 대상(용량 10% 이상)만
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
    """
    '발전량가중 편향 - 무가중 편향' 의 차이만 계산 (STEP 3,4 용).
    팀원 보정(무가중)이 이미 적용된 값 위에 '가중으로 바꿨을 때의 차이'만 얹기 위함.
    """
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
    """구간별 편향을 보간해 적용. strength로 강도 조절 (전 그룹에서 스윕 필요)."""
    adj = strength * np.interp(pred, fc, dl, left=dl[0], right=dl[-1])
    return np.clip(pred + adj, 0, cap)


def recent_window(index, days):
    """최근 N일 마스크. '최근창'이 핵심 — 2025 효율하락의 최신 수준을 반영."""
    return index > (index.max() - pd.Timedelta(days=days))


# ============================ 메인 ============================
def main():
    # --- 입력 로드 ---
    sub_a = pd.read_csv(BASE / "submit_A.csv", encoding="utf-8-sig")
    v32 = (pd.read_csv(V32_PATH, encoding="utf-8-sig")
             .set_index("forecast_id").reindex(sub_a["forecast_id"]).reset_index())
    oof = pd.read_csv(OOF_DIR / "pred_A_blend_2024.csv",
                      parse_dates=["forecast_kst_dtm"]).set_index("forecast_kst_dtm")
    act = pd.read_csv(OOF_DIR / "actual_2024.csv",
                      parse_dates=["forecast_kst_dtm"]).set_index("forecast_kst_dtm")

    sub = sub_a.copy()

    # ---------- STEP 1&2. group1 강화 (v35 -> v39) ----------
    # g1은 구조적으로 계속 과소예측 -> 위로 강화하면 FICR 상승.
    # 팀원은 편향의 약 30%만 적용했었음 -> 0.5 + 0.3 = 0.8 을 더 얹어 최종 약 110% 수준.
    # ★ 실측 결과 g1 평균 9,166 이 정점 (8,861도 9,942도 손해). 이 값을 바꾸지 말 것.
    g = "kpx_group_1"
    cap = CAP[g]
    win = recent_window(act.index, days=30)
    fc, dl = fit_bias(oof[g][win].to_numpy(), act[g][win].to_numpy(), cap, nbin=6, weighted=False)

    g1 = apply_bias(sub_a[g].to_numpy(), fc, dl, cap, strength=0.5)   # STEP 1 -> v35
    g1 = apply_bias(g1,                  fc, dl, cap, strength=0.3)   # STEP 2 -> v39
    sub[g] = g1
    print(f"STEP1&2 g1: {sub_a[g].mean():.0f} -> {sub[g].mean():.0f}  (목표 9166)")

    # ---------- STEP 3. group3 발전량 가중 (v41) — 최대 점프 ----------
    # FICR은 발전량 가중 지표인데 팀원 보정은 무가중 median이라 지표와 어긋나 있었음.
    # '가중 - 무가중' 차이만 계산해 얹음. 보간 기준은 baseline raw(v32) 값.
    # 오프라인 검증(2023창->2024): +0.0294 / 리더보드: +0.00166 (단일 최대)
    g = "kpx_group_3"
    cap = CAP[g]
    win = recent_window(act.index, days=45)
    fc, delta = fit_genweight_delta(oof[g][win].to_numpy(), act[g][win].to_numpy(), cap, nbin=20)
    adj = 1.0 * np.interp(v32[g].to_numpy(), fc, delta, left=delta[0], right=delta[-1])
    sub[g] = np.clip(sub[g].to_numpy() + adj, 0, cap)
    print(f"STEP3   g3: {sub_a[g].mean():.0f} -> {sub[g].mean():.0f}  (조정 {adj.mean():+.0f})")

    # ---------- STEP 4. group2 발전량 가중 50% 강도 (v42) ----------
    # 강도 스윕 결과 100%가 아니라 50~60%가 정점이었음(오프라인 1.0=+0.0039 vs 0.6=+0.0084).
    # 안전하게 0.5 적용. 이동폭도 절반이라 리스크 작음.
    g = "kpx_group_2"
    cap = CAP[g]
    win = recent_window(act.index, days=45)
    fc, delta = fit_genweight_delta(oof[g][win].to_numpy(), act[g][win].to_numpy(), cap, nbin=6)
    adj = 0.5 * np.interp(v32[g].to_numpy(), fc, delta, left=delta[0], right=delta[-1])
    sub[g] = np.clip(sub[g].to_numpy() + adj, 0, cap)
    print(f"STEP4   g2: {sub_a[g].mean():.0f} -> {sub[g].mean():.0f}  (조정 {adj.mean():+.0f})")

    # ---------- 검증 후 저장 ----------
    for gg in CAP:
        sub[gg] = np.clip(sub[gg], 0, CAP[gg])
    assert len(sub) == 8760, "행수는 8760이어야 함"
    assert not sub[list(CAP)].isna().any().any(), "결측 있음"
    for gg in CAP:
        assert (sub[gg] >= 0).all() and (sub[gg] <= CAP[gg]).all(), f"{gg} 범위 벗어남"

    out = BASE / "submit_v42.csv"
    sub.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out}")
    print("  예상 리더보드: 0.64410 (1-NMAE 0.86947 / FICR 0.41873)")


# ===========================================================================
# 참고 1. 오프라인 검증 함수 — 새 아이디어는 반드시 이걸로 먼저 확인할 것
# ===========================================================================
def gscore(actual, forecast, cap):
    """대회 지표(그룹별): 0.5*(1-평균오차율) + 0.5*발전량가중 밴드점수(<=6%:4원, <=8%:3원)."""
    valid = (actual >= cap * 0.10) & ~np.isnan(actual)
    err = np.abs(forecast[valid] - actual[valid]) / cap
    unit_price = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    ficr = np.sum(actual[valid] * unit_price) / np.sum(actual[valid] * 4.0)
    return 0.5 * (1 - err.mean()) + 0.5 * ficr


def validate_forward(pred_2023, act_2023, pred_2024, act_2024, cap,
                     nbin=6, window_days=45, weighted=True, strengths=(0.2, 0.4, 0.6, 0.8, 1.0)):
    """
    2023창으로 보정규칙을 만들어 2024에 적용 -> 점수 개선 확인 (시간순서 준수).
    ★ 강도는 반드시 스윕할 것. 우리 경험상 100%가 최적인 경우가 없었음(50~60%가 정점).
    """
    win = recent_window(act_2023.index, window_days)
    fc, dl = fit_bias(pred_2023[win].to_numpy(), act_2023[win].to_numpy(), cap, nbin, weighted)

    p, a = pred_2024.to_numpy(), act_2024.to_numpy()
    base = gscore(a, p, cap)
    results = {s: gscore(a, apply_bias(p, fc, dl, cap, s), cap) for s in strengths}
    best_s = max(results, key=results.get)
    return {"base": base, "by_strength": results, "best_strength": best_s,
            "best_score": results[best_s], "gain": results[best_s] - base}


# ===========================================================================
# 참고 2. 새 조건축을 시험할 때 쓰는 안정성 판정
#   (계절축은 이 테스트에서 상관 -0.14 로 기각됐고, 일주기축은 +0.77로 통과했으나
#    리더보드에서는 실패함 -> 통과해도 최종 판정은 리더보드)
# ===========================================================================
def axis_stability(bias_2023, bias_2024):
    """
    축별 편향의 연도간 안정성. ★ 반드시 '레벨(평균)을 제거한 변동'으로 비교할 것.
    레벨만 공통이면 상관이 높게 나와도 축 신호가 아님.
    """
    a, b = np.asarray(bias_2023, float), np.asarray(bias_2024, float)
    da, db = a - a.mean(), b - b.mean()
    corr = np.corrcoef(da, db)[0, 1]
    agree = int((np.sign(da) == np.sign(db)).sum())
    return {"corr": corr, "sign_agree": f"{agree}/{len(da)}",
            "verdict": "진짜 축신호" if (corr > 0.5 and agree >= len(da) * 0.7) else "노이즈"}


if __name__ == "__main__":
    main()
