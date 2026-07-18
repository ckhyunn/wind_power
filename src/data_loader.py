"""data/open/ 아래의 대회 데이터를 읽어오는 로더 함수 모음."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "open"


def load_train_labels() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train" / "train_labels.csv")


def load_weather(split: str) -> dict[str, pd.DataFrame]:
    """split: 'train' 또는 'test'. LDAPS, GFS 기상 데이터를 함께 반환한다."""
    if split not in ("train", "test"):
        raise ValueError(f"split은 'train' 또는 'test'여야 합니다: {split}")

    return {
        "ldaps": pd.read_csv(DATA_DIR / split / f"ldaps_{split}.csv"),
        "gfs": pd.read_csv(DATA_DIR / split / f"gfs_{split}.csv"),
    }


def load_scada() -> dict[str, pd.DataFrame]:
    return {
        "vestas": pd.read_csv(DATA_DIR / "train" / "scada_vestas_train.csv"),
        "unison": pd.read_csv(DATA_DIR / "train" / "scada_unison_train.csv"),
    }


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_submission.csv")


def load_info() -> pd.DataFrame:
    return pd.read_excel(DATA_DIR / "info.xlsx")
