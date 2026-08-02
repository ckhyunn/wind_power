"""
LSTM 시계열 모델.

지금까지의 LightGBM/XGBoost는 매 시간을 독립된 행(row)으로 취급함 (lag/lead 피처로
전후 1시간만 겨우 반영). LSTM은 연속된 여러 시간을 하나의 시퀀스로 통째로 학습해서,
트리가 구조적으로 놓치는 '풍속의 흐름/추세' 정보를 담을 수 있을 것으로 기대.

설계:
  - 입력: 과거 SEQ_LEN시간의 예보 피처 시퀀스 (build_group_weather로 만든 피처 그대로 재사용)
  - 출력: 시퀀스 끝 시점의 발전량 (단일 값 회귀)
  - 그룹별로 별도 모델 (트리 파이프라인과 동일 원칙)
  - MAE 손실 (평가지표와 목적함수 일치 원칙 재사용)
  - walk-forward 검증 (그 시점 이전 데이터만 학습에 사용)
  - calib 구간으로 조기종료

트리와 다른 점(주의 필요):
  - 정규화 필수 (StandardScaler, 반드시 train 구간으로만 적합)
  - 시퀀스라 데이터를 섞으면(shuffle) 안 되고, 시퀀스가 holdout/calib 경계를 넘으면 안 됨
  - 결측치는 트리처럼 median 대체가 아니라 시퀀스 내에서 전방채움(ffill) 사용

실행:
    python src/train_lstm.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from features import TARGET_COLS, CAPACITY_KWH, load_turbine_table, compute_group_coords
from train_baseline import build_group_weather, build_features, DATA_DIR, TRAIN_DIR, N_NEAREST_GRIDS
from evaluate import metric, group_score

SEQ_LEN = 24          # 과거 24시간을 하나의 시퀀스로 사용
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 10         # calib 손실 기준 조기종료
LEARNING_RATE = 1e-3

HOLDOUT_DAYS = 90
CALIB_DAYS = 45

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]  # 시퀀스 마지막 시점의 hidden state만 사용
        return self.head(last_hidden).squeeze(-1)


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """연속된 시계열에서 (과거 seq_len시간 -> 현재 시점 타겟) 슬라이딩 윈도우 생성.
    X, y는 이미 시간순 정렬되어 있고 연속적이라고 가정 (호출부에서 보장)."""
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,))
    sequences = np.stack([X[i - seq_len:i] for i in range(seq_len, n + 1)])
    targets = y[seq_len - 1:]
    return sequences, targets


def train_one_group(target: str, X_all: pd.DataFrame, y_all: pd.Series, dt_all: pd.Series):
    cutoff = dt_all.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    calib_start = cutoff - pd.Timedelta(days=CALIB_DAYS)

    is_holdout = dt_all > cutoff
    is_calib = (dt_all > calib_start) & (dt_all <= cutoff)
    is_train = dt_all <= calib_start

    # 결측치: 시퀀스 내 전방채움 + 남은 결측은 0 (스케일링 후이므로 평균 근처)
    X_filled = X_all.ffill().bfill().fillna(0)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_filled[is_train])
    X_cal_scaled = scaler.transform(X_filled[is_calib])
    X_ho_scaled = scaler.transform(X_filled[is_holdout])

    y_tr, y_cal, y_ho = y_all[is_train].values, y_all[is_calib].values, y_all[is_holdout].values

    # 시퀀스 생성 (calib/holdout은 각자 구간 앞의 데이터를 이어붙여 시퀀스 시작점을 확보 -
    # 실제로는 train 뒤에 calib을 붙여서 첫 SEQ_LEN 시간도 시퀀스를 만들 수 있게 함)
    X_tr_seq, y_tr_seq = build_sequences(X_tr_scaled, y_tr, SEQ_LEN)

    X_cal_context = np.concatenate([X_tr_scaled[-SEQ_LEN + 1:], X_cal_scaled])
    y_cal_context = np.concatenate([y_tr[-SEQ_LEN + 1:], y_cal])
    X_cal_seq, y_cal_seq = build_sequences(X_cal_context, y_cal_context, SEQ_LEN)

    X_ho_context = np.concatenate([X_cal_scaled[-SEQ_LEN + 1:], X_ho_scaled])
    y_ho_context = np.concatenate([y_cal[-SEQ_LEN + 1:], y_ho])
    X_ho_seq, y_ho_seq = build_sequences(X_ho_context, y_ho_context, SEQ_LEN)

    print(f"[{target}] 시퀀스 개수 - train={len(X_tr_seq)}, calib={len(X_cal_seq)}, holdout={len(X_ho_seq)}")

    device = DEVICE
    model = LSTMRegressor(n_features=X_tr_seq.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.L1Loss()  # MAE - 지금까지 트리 objective와 동일 원칙

    X_tr_t = torch.tensor(X_tr_seq, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_seq, dtype=torch.float32)
    X_cal_t = torch.tensor(X_cal_seq, dtype=torch.float32).to(device)
    y_cal_t = torch.tensor(y_cal_seq, dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_calib_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            cal_pred = model(X_cal_t)
            cal_loss = loss_fn(cal_pred, y_cal_t).item()

        if cal_loss < best_calib_loss:
            best_calib_loss = cal_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[{target}] epoch {epoch}에서 조기종료 (calib MAE={best_calib_loss:.1f})")
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        X_ho_t = torch.tensor(X_ho_seq, dtype=torch.float32).to(device)
        pred_ho = model(X_ho_t).cpu().numpy()

    pred_ho = np.clip(pred_ho, 0, CAPACITY_KWH[target])
    return pred_ho, y_ho_seq


def main():
    train_labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
    train_labels["kst_dtm"] = pd.to_datetime(train_labels["kst_dtm"])
    ldaps_train = pd.read_csv(TRAIN_DIR / "ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv(TRAIN_DIR / "gfs_train.csv", encoding="utf-8-sig")

    turbine_df = load_turbine_table(DATA_DIR / "info.xlsx")
    group_coords = compute_group_coords(turbine_df)
    train_weather = build_group_weather(ldaps_train, gfs_train, group_coords)

    print(f"디바이스: {DEVICE}")

    holdout_preds, holdout_actuals = {}, {}
    for target in TARGET_COLS:
        weather = train_weather[target]
        X_all = build_features(train_labels.rename(columns={"kst_dtm": "forecast_kst_dtm"}), weather, "forecast_kst_dtm")
        y_all = train_labels[target]
        dt_all = train_labels["kst_dtm"]

        mask_label = y_all.notna()
        X_all, y_all, dt_all = X_all[mask_label].reset_index(drop=True), y_all[mask_label].reset_index(drop=True), dt_all[mask_label].reset_index(drop=True)

        # LSTM은 시퀀스라 정렬 + 연속성이 중요 - 시간순 정렬 명시적으로 보장
        order = dt_all.argsort()
        X_all, y_all, dt_all = X_all.iloc[order].reset_index(drop=True), y_all.iloc[order].reset_index(drop=True), dt_all.iloc[order].reset_index(drop=True)

        # 수치형 피처만 사용 (LSTM은 범주형/텍스트 처리 안 함 - 지금 피처는 전부 수치형이라 문제 없음)
        X_numeric = X_all.select_dtypes(include=[np.number])

        pred_ho, y_ho = train_one_group(target, X_numeric, y_all, dt_all)
        holdout_preds[target] = pred_ho
        holdout_actuals[target] = y_ho

    def to_df(d):
        return pd.concat([pd.Series(v, name=k) for k, v in d.items()], axis=1)

    actual_df = to_df(holdout_actuals)
    pred_df = to_df(holdout_preds)

    score, nmae, ficr = metric(actual_df, pred_df)
    print(f"\n[LSTM 단독 - holdout] Score={score:.4f}  1-NMAE={nmae:.4f}  FICR={ficr:.4f}")
    print("(참고: v29 트리 모델 holdout Score는 raw=0.6225, FICR조정=0.6480)")


if __name__ == "__main__":
    main()
