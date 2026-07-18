"""RandomForestRegressor 기반 베이스라인 학습 스켈레톤.

KPX 3개 그룹(vestas, unison, ...) 별로 모델을 따로 학습한다.
시계열 데이터이므로 train_test_split에는 shuffle=False를 사용한다.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from data_loader import load_scada, load_train_labels, load_weather

# TODO: 데이터 병합에 사용할 키 컬럼을 채운다. (예: 날짜/시간 컬럼)
MERGE_KEYS: list[str] = []

# TODO: 모델 입력으로 사용할 피처 컬럼을 채운다.
FEATURE_COLS: list[str] = []

# TODO: 그룹별 예측 대상(발전량) 컬럼을 채운다.
TARGET_COLS: list[str] = []

# TODO: KPX 3개 그룹명을 채운다.
GROUPS: list[str] = []

MODEL_DIR = Path(__file__).resolve().parent.parent / "submissions"


def build_dataset() -> pd.DataFrame:
    """기상 데이터, SCADA 데이터, 라벨을 MERGE_KEYS 기준으로 병합한다."""
    labels = load_train_labels()
    weather = load_weather("train")
    scada = load_scada()

    # TODO: MERGE_KEYS를 기준으로 labels / weather["ldaps"] / weather["gfs"] /
    # scada["vestas"] / scada["unison"]을 병합하는 로직을 작성한다.
    raise NotImplementedError("데이터 병합 로직을 구현해야 합니다.")


def train_group_model(df: pd.DataFrame, group: str, target_col: str) -> RandomForestRegressor:
    X = df[FEATURE_COLS]
    y = df[target_col]

    # 시계열 데이터이므로 순서를 섞지 않는다.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    valid_score = model.score(X_valid, y_valid)
    print(f"[{group}] validation R^2: {valid_score:.4f}")

    return model


def main() -> None:
    df = build_dataset()

    for group, target_col in zip(GROUPS, TARGET_COLS):
        model = train_group_model(df, group, target_col)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / f"model_{group}.pkl")


if __name__ == "__main__":
    main()
