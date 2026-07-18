# 데이터 안내

## 다운로드

대회 페이지에서 데이터를 다운로드합니다.

- 대회 링크: https://dacon.io/competitions/official/236727 (제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026)
- 다운로드한 `open.zip`(또는 `open` 폴더)의 압축을 이 `data/` 폴더 아래에 풉니다.

압축 해제 후 아래와 같은 구조가 되어야 합니다.

```
data/
└── open/
    ├── train/
    │   ├── ldaps_train.csv
    │   ├── gfs_train.csv
    │   ├── train_labels.csv
    │   ├── scada_vestas_train.csv
    │   └── scada_unison_train.csv
    ├── test/
    │   ├── ldaps_test.csv
    │   └── gfs_test.csv
    ├── sample_submission.csv
    ├── info.xlsx
    └── data_description.md
```

## 파일별 설명

| 파일 | 설명 |
| --- | --- |
| `train/ldaps_train.csv` | LDAPS(국지예보모델) 기반 기상 예측 데이터 (학습 기간) |
| `train/gfs_train.csv` | GFS(전지구예보모델) 기반 기상 예측 데이터 (학습 기간) |
| `train/train_labels.csv` | 학습용 정답(발전량) 라벨 |
| `train/scada_vestas_train.csv` | Vestas 터빈 SCADA 실측 데이터 (학습 기간) |
| `train/scada_unison_train.csv` | Unison 터빈 SCADA 실측 데이터 (학습 기간) |
| `test/ldaps_test.csv` | LDAPS 기반 기상 예측 데이터 (평가 기간) |
| `test/gfs_test.csv` | GFS 기반 기상 예측 데이터 (평가 기간) |
| `sample_submission.csv` | 제출 양식 샘플 |
| `info.xlsx` | 설비/그룹 정보 (설비용량 등) |
| `data_description.md` | 데이터 컬럼 상세 설명 |

## 주의사항

이 데이터는 **대회 목적 외 사용을 금지**합니다. 외부 배포, 재공유, 대회 종료 후 별도 용도로의 활용을 하지 않습니다.
