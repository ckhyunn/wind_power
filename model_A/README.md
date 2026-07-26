# BARAM 2026 — A 모델 (현윤)

> 팀 공유 **베이스 모델**. 현재 리더보드 **최고 점수 0.63625** (NMAE 0.87118 / FICR 0.40133).
> LDAPS/GFS 기상예보로 KPX 3개 그룹의 2025년 시간별 발전량을 예측.

---

## 1. 폴더 구조
```
baram2026_A/
├── model_A.py            # 메인 스크립트 (학습→앙상블→보정→제출)
├── README.md             # 이 문서 (전체 설명)
├── requirements.txt      # 필요 패키지
├── .gitignore            # 원본 데이터·캐시 제외 규칙
├── features/             # (레포 포함) 기본 피처
│   ├── train_features.csv
│   └── test_features.csv
├── cleaned/              # (레포 포함) 정제 데이터
│   ├── train_labels_clean.csv
│   └── scada_unison_clean.csv
└── output/               # 예측 결과 (팀 공유용 — 6절 참고)
    ├── submit_A.csv          # 현재 최종 제출 파일 (앙상블+g1보정, = 0.63625)
    ├── pred_A_lgb.csv        # LightGBM 단독 예측
    ├── pred_A_cat.csv        # CatBoost 단독 예측
    └── pred_A_xgb.csv        # XGBoost 단독 예측
```
> **대회 원본 데이터(train/, test/, sample_submission.csv)는 용량이 커서 레포에 없음.**
> 각자 로컬에 두고 `model_A.py`의 `DATA_DIR` 경로만 맞추면 됨.

## 2. 실행 방법
```bash
pip install -r requirements.txt
# model_A.py 상단 CONFIG의 DATA_DIR 를 대회 원본 데이터 위치로 수정
python model_A.py
```
결과: `output/submit_A.csv` (제출 파일), `output/pred_A_lgb.csv` 등 (모델별 단독 예측)

## 3. 필요 패키지
`numpy, pandas, lightgbm, catboost, xgboost` (requirements.txt 참고)

---

## 4. 모델 구성 (요약)

### 피처
- **기본**: 시간(sin/cos), LDAPS/GFS 풍속·ws³·공기밀도·발전량프록시·풍향·돌풍·구름·강수, lag/lead(ramp), base_means
- **격자별 피처**(핵심): LDAPS 16격자 + GFS 9격자의 **풍속(ws10/ws50/ws50³, ws100/ws80/ws850/ws100³) + 풍향(sin/cos)**
- **가지치기**: gain 중요도 하위 30% 제거 → 최종 약 190개

### 모델 (3종 앙상블) — 튜닝 파라미터는 `model_A.py`에 내장
| 모델 | 핵심 설정 | 앙상블 비율 |
|------|-----------|:---:|
| **LightGBM** | l1, n_est 500, leaves 32, reg_alpha 4.7 (5시드 평균) | **0.50** |
| **CatBoost** | MAE, iter 1100, depth 6 | 0.10 |
| **XGBoost** | MAE, n_est 700, depth 5 | 0.40 |

### 적용 기법 (모두 ON)
| 기법 | 적용 | 설명 |
|------|:---:|------|
| 발전량 가중치 (0.5 + y/용량) | ✅ | 고발전 시간 더 중요하게 |
| 10% 컷 (용량 10% 미만 학습 제외) | ✅ | 채점 제외 구간 |
| group3 통합학습(pooled) | ✅ | g1·2와 합쳐 학습 |
| group3 라벨 정제 | ✅ | unison SCADA와 **1000kWh 초과** 어긋난 시간 제외 |
| g1 편향 보정(calibration) | ✅ | 과소예측을 발전량 구간별 debias |
| 스무딩 (3h 이동중앙값) | ✅ | 예보블록 내 |
| 클리핑 (0~용량) | ✅ | 물리 범위 |

---

## 5. Ablation (제거/변경 실험) — CONFIG만 바꾸면 됨

`model_A.py` 상단 `CFG` 딕셔너리 수정 후 실행. `RUN_2FOLD=True`면 2-fold 점수로 효과 비교 가능.

| 실험 | 바꿀 CONFIG |
|------|-------------|
| LightGBM 단독 | `MODELS=["lgb"]` |
| LGBM + XGB 앙상블 | `MODELS=["lgb","xgb"]` |
| 전체 앙상블(기본) | `MODELS=["lgb","cat","xgb"]` |
| 격자별 피처 제거 | `USE_PERGRID=False, USE_PERGRID_DIR=False` |
| group3 라벨 정제 제거 | `USE_G3_DENOISE=False` |
| Calibration 제거 | `USE_G1_CALIB=False` |
| 앙상블 비율 변경 | `WEIGHTS={"lgb":.., "cat":.., "xgb":..}` |
| 시드 평균 개수 변경 | `N_SEEDS=3` 등 |
| 발전량 가중치 제거 | `USE_GEN_WEIGHT=False` |
| 10% 컷 제거 | `USE_10PCT_CUT=False` |

> 팁: ablation만 볼 땐 `MAKE_SUBMISSION=False`로 두면 2-fold 검증만 빠르게 돌아감.

---

## 6. 출력 파일
- `output/submit_A.csv` — 최종 제출 파일 (앙상블 + g1 보정)
- `output/pred_A_lgb.csv`, `pred_A_cat.csv`, `pred_A_xgb.csv` — **모델별 단독 예측** (동훈 블렌딩/상관 분석용)

---

## 7. 검증 원칙 (팀 공통 권장)
- 예측은 전부 로컬 라이브러리로 수행 (원격 API·외부 비공개 데이터·leakage 금지).
