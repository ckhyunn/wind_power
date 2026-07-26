# daeun_claude_pipeline

`docs/eda_report.md` 근거를 바탕으로 별도로 구성한 그룹별 LightGBM 파이프라인 (TDD, `tests/` 참고). 메인 `src/`, `src/train_baseline.py` 등 기존 파이프라인과는 독립적으로 작성됨 — 비교/병합 여부는 검토 필요.

- 실행: `python3 scripts/make_submission.py` (레포 루트에 `train/`, `test/`, `info.xlsx`, `sample_submission.csv`가 있어야 함 — `.gitignore`로 제외되어 있으니 대회 페이지에서 직접 받을 것)
- CV 결과, 설계 근거, 알려진 단순화/다음 단계: `docs/submission_log.md`
- EDA 근거 원본: `docs/eda_report.md`, `docs/data_explainer.md`, `docs/model_notes.md`
- 전략 문서: `docs/PLAN.md`
