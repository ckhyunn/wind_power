"""
공용 모델 학습 로직.

train_baseline.py와 backtest.py가 반드시 동일한 모델 학습 방식을 쓰도록 여기 한 곳에
모아둠. (파워커브 피처를 처음 추가했을 때 두 스크립트의 로직이 서로 달라져서
backtest 검증이 무의미해졌던 사고가 있었음 - 같은 실수를 반복하지 않기 위함)

v15: LightGBM 단독 앙상블(v11)에 XGBoost를 더해 '모델 블렌딩'으로 확장.
서로 다른 알고리즘은 오차 패턴도 다르므로, 같은 알고리즘의 시드 앙상블보다
분산 감소 효과가 더 클 것으로 기대.
"""

import numpy as np
import lightgbm as lgb
import xgboost as xgb

LGBM_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="mae",
    n_jobs=-1,
    verbose=-1,
)
LGBM_SEEDS = [42, 123, 2024]

XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="reg:absoluteerror",  # LightGBM 쪽과 마찬가지로 MAE - 평가지표(NMAE/FICR)와 목적함수 일치
    n_jobs=-1,
    verbosity=0,
)
XGB_SEEDS = [7, 77, 777]


def train_blended_ensemble(X_tr, y_tr, X_cal, y_cal,
                            lgbm_seeds=None, xgb_seeds=None, log_target: bool = False) -> list:
    """
    LightGBM + XGBoost를 여러 시드로 각각 학습 (calib을 조기종료 검증셋으로 사용).
    반환: [(kind, seed, fitted_model, best_iteration, log_target), ...] 리스트.
    최종 재학습(refit_final_models)에서 이 정보를 그대로 재사용.

    log_target=True면 y를 log1p로 변환해서 학습 (v20).
    발전량 분포가 0 근처에 몰려있고 고출력 쪽 꼬리가 긴 비대칭 분포라, 원본 스케일
    그대로 MAE로 학습하면 고출력 구간의 절대오차가 손실을 지배할 수 있음. 로그
    변환하면 '상대오차'에 더 가깝게 최적화되어, 저~중출력 구간(전체 시간대의 다수를
    차지)의 정확도가 개선될 가능성이 있음. 예측/평가 시 자동으로 역변환(expm1)됨
    (ensemble_predict/final_predict가 처리하므로 호출부 수정 불필요).
    """
    lgbm_seeds = LGBM_SEEDS if lgbm_seeds is None else lgbm_seeds
    xgb_seeds = XGB_SEEDS if xgb_seeds is None else xgb_seeds

    y_tr_fit = np.log1p(y_tr) if log_target else y_tr
    y_cal_fit = np.log1p(y_cal) if log_target else y_cal

    trained = []

    for seed in lgbm_seeds:
        m = lgb.LGBMRegressor(random_state=seed, **LGBM_PARAMS)
        m.fit(
            X_tr, y_tr_fit,
            eval_set=[(X_cal, y_cal_fit)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        trained.append(("lgbm", seed, m, m.best_iteration_, log_target))

    for seed in xgb_seeds:
        m = xgb.XGBRegressor(random_state=seed, early_stopping_rounds=50, **XGB_PARAMS)
        m.fit(X_tr, y_tr_fit, eval_set=[(X_cal, y_cal_fit)], verbose=False)
        trained.append(("xgb", seed, m, m.best_iteration, log_target))

    return trained


def _model_predict_original_scale(entry, X) -> np.ndarray:
    """trained 항목 하나의 예측을 원래 스케일(kWh)로 반환 (log_target이면 expm1로 역변환)"""
    kind, seed, model, best_iter, log_target = entry
    raw_pred = model.predict(X)
    return np.expm1(raw_pred) if log_target else raw_pred


def ensemble_predict(trained: list, X) -> np.ndarray:
    """trained의 모든 모델(LightGBM+XGBoost 섞여있음) 예측을 평균 (원래 스케일로 역변환 후 평균)"""
    preds = np.column_stack([_model_predict_original_scale(entry, X) for entry in trained])
    return preds.mean(axis=1)


def refit_final_models(trained: list, X_fit, y_fit) -> list:
    """
    train_blended_ensemble()에서 찾은 종류/시드/best_iteration을 그대로 써서
    holdout을 제외한 전체(train+calib) 데이터로 재학습. 별도 검증셋이 없으므로
    조기종료 없이 앞서 찾은 트리 개수를 그대로 사용.
    반환: [(fitted_model, log_target), ...] - final_predict에서 역변환 여부 판단에 사용
    """
    final_models = []
    for kind, seed, _, best_iter, log_target in trained:
        y_fit_use = np.log1p(y_fit) if log_target else y_fit
        if kind == "lgbm":
            m = lgb.LGBMRegressor(random_state=seed, **{**LGBM_PARAMS, "n_estimators": best_iter})
        else:
            m = xgb.XGBRegressor(random_state=seed, **{**XGB_PARAMS, "n_estimators": max(best_iter, 1)})
        m.fit(X_fit, y_fit_use)
        final_models.append((m, log_target))
    return final_models


def final_predict(final_models: list, X) -> np.ndarray:
    preds = np.column_stack([
        (np.expm1(m.predict(X)) if log_target else m.predict(X))
        for m, log_target in final_models
    ])
    return preds.mean(axis=1)
