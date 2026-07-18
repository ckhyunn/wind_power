"""
SCADA 실측 데이터로 '컷아웃(cut-out)' 현상이 실제로 존재하는지 진단하는 스크립트.

컷아웃: 풍속이 너무 세지면(보통 20~25m/s 이상) 터빈이 안전을 위해 스스로 발전을 멈추는 현상.
존재한다면 파워커브가 고풍속 구간에서 위로 증가하다가 뚝 떨어지는 모양이 됨.

실행:
    python src/diagnose_cutout.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train"


def diagnose(scada_path: Path, turbine_prefix: str, n_turbines: int):
    df = pd.read_csv(scada_path, encoding="utf-8-sig")
    print(f"\n{'='*60}")
    print(f"[{scada_path.name}] 기간: {df['kst_dtm'].min()} ~ {df['kst_dtm'].max()}, 행수: {len(df)}")

    for i in range(1, n_turbines + 1):
        wtg = f"{turbine_prefix}{i:02d}"
        ws_col, power_col = f"{wtg}_ws", f"{wtg}_power_kw10m"
        if ws_col not in df.columns:
            continue

        ws = df[ws_col].dropna()
        power = df[power_col].dropna()
        max_ws = ws.max()

        # 풍속을 1m/s 단위 구간으로 나눠서 구간별 평균 발전량 계산
        bins = np.arange(0, max_ws + 2, 1)
        df["_ws_bin"] = pd.cut(df[ws_col], bins=bins)
        curve = df.groupby("_ws_bin", observed=True)[power_col].mean()

        print(f"\n[{wtg}] 최대 관측 풍속: {max_ws:.1f} m/s")
        print(curve.to_string())

        # 파워커브가 증가하다가 감소하는 구간이 있는지 자동 체크
        vals = curve.dropna().values
        if len(vals) >= 3:
            peak_idx = np.argmax(vals)
            if peak_idx < len(vals) - 1:
                drop = vals[peak_idx] - vals[-1]
                print(f"  -> 피크({vals[peak_idx]:.0f}) 이후 마지막 구간까지 {drop:.0f} 감소 "
                      f"{'(컷아웃 패턴 의심됨)' if drop > vals[peak_idx]*0.2 else '(뚜렷한 컷아웃 없음)'}")


if __name__ == "__main__":
    diagnose(TRAIN_DIR / "scada_vestas_train.csv", "vestas_wtg", 12)
    diagnose(TRAIN_DIR / "scada_unison_train.csv", "unison_wtg", 5)
