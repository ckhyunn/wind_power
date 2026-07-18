# wind_power

제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026

- 대회 페이지: https://dacon.io/competitions/official/236727

## 폴더 구조

```
wind_power/
├── data/
│   ├── README.md          # 데이터 다운로드 및 구조 안내 (실제 데이터는 git에 포함되지 않음)
│   └── open/               # 다운로드한 대회 데이터 (gitignore 처리)
├── notebooks/               # 탐색/실험용 노트북
├── src/
│   ├── data_loader.py       # 데이터 로딩 함수
│   ├── evaluate.py          # 평가 산식 (NMAE, FICR, total score)
│   └── train_baseline.py    # RandomForest 베이스라인 학습 스켈레톤
├── submissions/              # 제출 파일 저장
├── requirements.txt
└── README.md
```

## 시작하기

1. 저장소 클론

   ```bash
   git clone https://github.com/ckhyunn/wind_power.git
   cd wind_power
   ```

2. 의존성 설치

   ```bash
   pip install -r requirements.txt
   ```

3. 데이터 다운로드

   [data/README.md](data/README.md)를 참고하여 대회 데이터를 `data/open/` 아래에 내려받는다.

## 평가 산식

```
Score = 0.5 x (1 - NMAE) + 0.5 x FICR
```

- 그룹별 NMAE = mean(|예측 - 실제| / 그룹 설비용량), 실제발전량이 설비용량의 10% 이상인 시간대만 대상
- 1 - NMAE = 1 - (3개 그룹 NMAE 평균)
- FICR = 시간대별 정산금 계단 구간표 기반 (구간값 미공개, TODO)

자세한 내용은 [src/evaluate.py](src/evaluate.py) 참고.

## 참고 링크

- 대회 페이지: https://dacon.io/competitions/official/236727

## 할 일 체크리스트

- [ ] `data/open/`에 대회 데이터 다운로드 및 배치
- [ ] `data_description.md`, `info.xlsx` 확인하여 컬럼/그룹 구조 파악
- [ ] `src/train_baseline.py`의 `MERGE_KEYS`, `FEATURE_COLS`, `TARGET_COLS`, `GROUPS` 채우기
- [ ] `build_dataset()` 데이터 병합 로직 구현
- [ ] FICR 계단 구간표 공개되면 `src/evaluate.py`의 `ficr_score()` 구현
- [ ] 베이스라인 학습 후 검증 스코어 확인
- [ ] 제출 포맷(`sample_submission.csv`)에 맞춰 제출 파일 생성
- [ ] 컷아웃(cut-out) 풍속 구간 처리 - 풍속이 임계값(약 25m/s) 이상일 때 발전량이 급감/0이 되는 구간을
      별도 플래그하거나 다르게 학습시키는 방법 검토
- [ ] SCADA vs 기상예보 비교로 격자별 지형 보정계수 산출 (터빈 실측 풍속과 예보값의 체계적 차이 확인)
- [ ] 태풍 등 이상기상 이력 피처 추가 검토 (기상청 태풍목록 활용 시 외부데이터 사용으로 신고 필요)
- [x] 거리 가중 평균(IDW) 격자 집계 시도 → v7(단순평균, Score 0.6290) 대비
      v8(IDW, Score 0.6284) 근소하게 낮아 채택 안 함. LDAPS 격자가 1.5km로
      이미 충분히 촘촘해서 가중치 차이가 미미했던 것으로 추정.
- [x] 컷아웃 구간 처리 시도 → SCADA 실측(UNISON)에서 22~23m/s 컷아웃 확인했으나,
      예보 풍속(GFS 100m) 기준 '상위 3%'(~11m/s)로는 실제 컷아웃 구간과 스케일이
      맞지 않아 Score 개선 없음(v7: 0.6290 → v9: 0.6237). 예보-실측 풍속 간 스케일
      보정을 먼저 해야 재시도 의미 있음 → 우선순위 낮춰서 보류.
