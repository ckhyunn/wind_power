# BARAM 2026 — Wind Power Forecasting: Strategy & Execution Plan

**Competition:** 제3회 풍력발전량 예측 AI 경진대회 (BARAM 2026), DACON #236727
**Today:** 2026-07-19 · **Competition ends:** 2026-08-14 10:00 KST (**~26 days left**) · **Team merge deadline:** 2026-08-07

---

## Phase 1 — Competition Analysis

### 1.1 Prediction task

Predict **hourly wind power generation (kWh)** for three KPX settlement groups (`kpx_group_1`, `kpx_group_2`, `kpx_group_3`) for every hour of the test period **2025-01-01 01:00 → 2026-01-01 00:00** (8,760 hours), using only day-ahead weather forecast data (LDAPS + GFS). No actual generation, SCADA, or reanalysis data is available at inference time for the test period — this is a pure **weather → power** forecasting problem, not a time-series continuation problem (you cannot autoregress on recent actuals for the test set).

Turbine → group mapping (confirmed from `info.xlsx`):
- `kpx_group_1`: VESTAS V126 turbines 1–6 (21.6 MW)
- `kpx_group_2`: VESTAS V126 turbines 7–12 (21.6 MW)
- `kpx_group_3`: UNISON U136 turbines 1–5 (21.0 MW)

SCADA data (`scada_vestas_train.csv`, `scada_unison_train.csv`) is 10-minute turbine-level power/wind speed/wind direction, available **only for the training period**. It cannot be used as a model input for inference (no future SCADA exists), but it is extremely valuable for:
- Fitting an empirical **power curve** (wind speed → power) per turbine/farm, which becomes a physics-informed feature or baseline generated purely from forecast wind speed.
- Cross-checking/denoising `train_labels` (group-level settlement meter) against summed turbine SCADA.
- Understanding curtailment, downtime, and icing/stall behavior that a pure forecast-driven model will miss.

### 1.2 Evaluation metrics

Confirmed from the evaluation page:

```
Total Score = 0.5 × (1 − NMAE) + 0.5 × FICR
```

**NMAE (per group):** `mean( |pred − actual| / group_capacity_kWh )` over included hours, then averaged across the 3 groups. `1 − NMAE` is reported as "평균 예측오차율".

**FICR (Financial Cost Recovery Rate, per group):** `획득 정산금 / 이론상 최대 정산금`, averaged across the 3 groups. This mirrors Korea's actual KPX renewable-energy forecast-incentive settlement system, where the hourly bonus paid is a **step function of the hourly error rate**. ✅ **Confirmed 2026-07-26** by decoding the base64-inlined PNG in the evaluation page's HTML (it doesn't extract via plain-text fetch, but the image is embedded inline in the page source): hourly NMAE ≤6% → 4 won/kWh, 6–8% → 3 won/kWh, >8% → 0 (no settlement). Implemented in `daeun_claude_pipeline/src/metrics.py::ficr()`. Real-leaderboard result on the 1st submission (FICR 0.3402 vs 1-NMAE 0.866) confirms the predicted risk: optimizing NMAE alone does not maximize FICR — see `docs/submission_log.md` "2차 제출" for the calibration fix that closed part of this gap.

**Scoring scope:** only hours where `actual > 10% × group_capacity` are included in both NMAE and FICR. Measured in the data: ~54–61% of historical hours clear this bar per group. This is a critical, non-obvious rule (see 1.4).

**Leaderboard:** Public = 40% of test hours (pre-sampled), Private = remaining 60%, final rank by Private. Max 5 submissions/day.

### 1.3 How the metric should drive model design

- **Optimize a capacity-normalized, masked objective — not plain RMSE/MAE.** Since NMAE divides by a *fixed* group capacity (not by actual output), all three groups should be trained on `error / capacity` scale, not raw kWh, so gradient contributions are comparable across groups. In LightGBM/CatBoost this means either training directly on `target/capacity` (kWh fraction of capacity) or supplying `1/capacity` as a per-row weight if training jointly on all groups.
- **Match the eval mask in training loss/CV where feasible.** Because sub-10%-capacity hours never enter the score, a masked or down-weighted loss (weight ≈ 0 for historically-calm regimes) focuses model capacity on the regime that actually matters. Caution: at inference time you don't know in advance whether a given hour's *actual* will clear 10% — so you can't mask predictions themselves, only use the mask to shape training loss/eval. Also validate that this doesn't quietly degrade calm-hour predictions used for FICR if the *true* mask differs slightly from the training-time proxy.
- **FICR likely rewards getting error into a "good enough" band more than shaving marginal MAE once you're already accurate** (typical of step-function incentive schemes). If a probability-weighted/quantile view of the payout table becomes available, a custom loss (or post-hoc calibration) that pushes predictions to *cross the best band* — even at the cost of slightly higher variance elsewhere — can outperform pure MAE minimization. This is the single highest-leverage modeling decision once the FICR table is confirmed.
- **Because scoring uses Private LB (60%) for final rank, and only 5 submissions/day are allowed, a trustworthy offline CV that correlates with LB is mandatory** — leaderboard probing is not a viable tuning strategy here.
- Per-group **capacity differs only slightly** (21.6/21.6/21.0 MW) but per-group **wind regime, turbine model (VESTAS vs UNISON), and label history length differ substantially** (group_3 labels only start 2023, ~1 year less training signal). Treat as 3 related-but-distinct forecasting problems (shared feature pipeline, group-aware model or per-group models), not one pooled regression blind to group identity.

### 1.4 Hidden pitfalls, constraints, opportunities

- **Weather data leakage window is already handled by the organizers** (`data_available_kst_dtm` = day-ahead 13:00 KST issuance) but *your pipeline must respect it anyway* — never join a forecast row to a training feature using information that would postdate `data_available_kst_dtm` in a live setting, and never use `train_labels`/SCADA of the *same or future* hour as a feature (obvious autoregression leakage, easy to introduce by accident via rolling/lag features computed without a strict causal shift).
- **The <10%-capacity mask is a trap and an opportunity.** Naively trying to hit near-zero output exactly everywhere wastes model capacity; near-threshold hours (just above/below 10%) are the highest-leverage, because misclassifying "will this hour count" direction matters less than getting magnitude right *given* it counts. Do not "give up" on low-wind hours in absolute terms (you don't know ground truth at inference), but do de-emphasize them in loss weighting.
- **External data / pretrained model rules are strict:** only public, license-clear data (no leakage of actual test-period generation/operational data); only open-source model weights released **before 2026-07-06**, local inference only — **no OpenAI/Gemini/HF Inference API/OpenRouter etc.** This rules out any "call an LLM API for weather reasoning" shortcut.
- **Reproducibility is graded, not optional.** Finalists must submit full external-data provenance and preprocessing code; final ranking requires code verification. Build the pipeline as scripted/versioned code from day one — not ad hoc notebook exploration that can't be re-run.
- **Timeline is tight: ~26 days from today to competition close, team-merge deadline in ~19 days.** This should dominate prioritization: cheap, high-ROI wins (solid GBM baseline + good weather-to-farm feature engineering + correct CV) before expensive, uncertain-payoff wins (deep sequence models).
- **16 LDAPS grids + 9 GFS grids vs. 3 KPX groups (effectively 2 farm sites: Taebaek Wind Farm-ish VESTAS site with 2 groups, and a separate UNISON site for group 3)** — the grids must be spatially reduced to farm-relevant features (nearest-grid, IDW/distance-weighted, or full grid-as-features) using turbine lat/long from `info.xlsx`. This spatial reduction is itself a meaningful feature-engineering decision, not a formality.
- **Two independent forecast sources (LDAPS 1.5 km short-range, GFS 0.25° global) give an implicit ensemble/disagreement signal** — spread between LDAPS and GFS wind speed at a given hour is a plausible uncertainty feature.
- **`sample_submission.csv` must keep `forecast_id`/`forecast_kst_dtm` untouched and be generated by code**, not hand-edited in Excel (explicitly warned against — Excel mangles datetime formatting).

### 1.5 Priorities to maximize leaderboard score (given the timeline)

1. Correct, leakage-free data pipeline + an eval-metric implementation that matches the real scoring exactly (confirm FICR table ASAP — this is a blocking unknown).
2. Time-respecting CV that correlates with LB (expanding-window / last-N-months holdout, never random shuffle).
3. Strong tabular baseline (LightGBM/CatBoost) with well-engineered weather→power features (turbine-level power curves, spatial grid reduction, lag/rolling of forecast fields, cyclical time features) — this is where most of the score will come from.
4. Per-group-aware ensembling/blending of a few strong models (GBM variants + a simple physical power-curve model as a floor/blend component).
5. Only after 1–4 are solid and time remains: sequence models (LSTM/TFT/PatchTST) as a stretch track, evaluated against the same CV, blended in only if they demonstrably beat the GBM ensemble out-of-fold.
6. Documentation/reproducibility packaging throughout (not bolted on at the end), since secondary evaluation requires it.

---

## Phase 2 — Existing Project State

**This is a greenfield project.** The directory currently contains **only competition data and docs — no source code, notebooks, or pipeline exist yet**:

```
windcon/
├── data_description.md      (data dictionary, read — see Phase 1)
├── info.xlsx                 (KPX group / turbine metadata, read)
├── sample_submission.csv     (8,760 rows, forecast_id/forecast_kst_dtm/3 group cols, all zero)
├── train/
│   ├── ldaps_train.csv        420,864 rows — 26,304 hours × 16 grids
│   ├── gfs_train.csv          236,736 rows — 26,304 hours × 9 grids
│   ├── train_labels.csv       26,304 hours × 3 groups (group_1/2 NaN ×~104, group_3 NaN 2022 entirely = 8,766 rows)
│   ├── scada_vestas_train.csv 157,819 rows × 12 turbines, 10-min, 2022-01-01→2025-01-01
│   └── scada_unison_train.csv 105,264 rows × 5 turbines, 10-min, 2023-01-01→2025-01-01
└── test/
    ├── ldaps_test.csv         140,160 rows — 8,760 hours × 16 grids
    └── gfs_test.csv           78,840 rows — 8,760 hours × 9 grids
```

Findings from direct inspection (not assumptions):
- All CSVs are `utf-8-sig`; header BOM present — confirmed on `train_labels.csv` (`﻿kst_dtm,...`). Handle with `encoding="utf-8-sig"` everywhere.
- `info.xlsx`'s `info` sheet has 2 header rows and merged cells (group-capacity value only appears once per group, on the first turbine's row) — needs forward-fill logic to parse cleanly, not a flat read.
- LDAPS/GFS are **long format**: one row per `(forecast_kst_dtm, grid_id)`. Must be pivoted/aggregated to one row per hour before joining to labels.
- Wind components are given as **U/V vector components**, not speed/direction directly (`heightAboveGround_10_10u/10v`, etc.) — speed = `sqrt(u²+v²)`, direction = `atan2(v,u)`; this conversion is a required feature-engineering step, not optional.
- `train_labels.kpx_group_3` is entirely missing for all of 2022 (matches the data description: labels start 2023 for group 3) — group_3 model effectively has ~2 years of labeled data vs. ~3 for groups 1/2.

**What works well:** N/A (nothing built yet) — but the data itself is clean, well-documented (`data_description.md` is thorough and accurate against actual files), and consistently keyed on hourly `*_kst_dtm` timestamps, which makes the join logic straightforward.

**Weaknesses / risks to design around:** long-format weather grids at 2 different resolutions/schemas (LDAPS ≠ GFS columns) need a unifying feature layer; group_3's shorter label history increases overfitting risk for that group specifically; FICR formula is unconfirmed (blocking item, not a code weakness).

**Technical debt:** none yet — this is the opportunity to set conventions correctly from the first commit (scripted pipeline, tests for the metric, version-pinned deps) rather than retrofitting later.

---

## Phase 3 — Multi-Agent Development Workflow (Orchestra)

The proposed 9-role split (Planner, Data Analysis, Feature Engineering, Validation, Model, Ensemble, Experiment, Submission) is directionally right but has two structural risks worth fixing before adopting it as-is:

1. **Validation/metric logic must be a shared library, not a pipeline stage owned by one agent.** If "Validation Agent" is the only place the NMAE/FICR/mask logic lives and other agents reimplement scoring ad hoc, you get silent metric drift (e.g., Model Agent using plain MAE while Validation Agent scores NMAE+FICR — they'd optimize different things). Fix: Validation Agent **owns and publishes** `windcon.metrics` and `windcon.cv` as an importable package version-locked early (Phase 4 Task 2); every other agent imports it, never reimplements it.
2. **Experiment Agent's scope overlaps Model/Ensemble Agent's natural workflow.** Splitting "run experiment" from "build model" invites hand-off lag with only ~26 days available. Fix: keep Experiment Agent's *responsibilities* (tracking, logging, reproducibility, ablations) but implement them as infrastructure (a shared experiment-log format + a CLI both Model and Ensemble agents call directly) rather than a gatekeeping agent in the critical path.

With those two fixes, here is the full spec:

| Agent | Responsibilities | Inputs | Outputs | Depends on | Talks to | Deliverables |
|---|---|---|---|---|---|---|
| **Planner** | Task decomposition, milestone tracking, dependency/priority management across the ~26-day timeline | This plan, live status from all agents | Updated task board, re-prioritization calls | none | all | Living milestone doc (Phase 4 §Milestones) |
| **Data Analysis** | EDA, missing-value/outlier audit, turbine↔group mapping validation, SCADA-vs-label cross-check, statistical summaries, candidate-feature list | Raw CSVs, `info.xlsx` | EDA report, cleaned turbine/group mapping config, candidate feature list | none | Feature Eng, Validation | `docs/eda_report.md`, `configs/turbine_group_map.yaml` |
| **Feature Engineering** | Grid→farm spatial reduction, U/V→speed/direction, cyclical time encoding, lag/rolling of forecast fields, power-curve features from SCADA, feature selection/importance | Data Analysis outputs, raw weather CSVs | Versioned feature tables (train+test, identical schema) | Data Analysis | Model, Validation | `src/windcon/features/*.py`, feature parquet cache |
| **Validation** | Owns `windcon.metrics` (exact NMAE/FICR/mask) and `windcon.cv` (time-respecting splitters), leakage audits of every other agent's joins, LB-correlation tracking | Competition metric definition, feature tables, model OOF predictions | Metric library, CV splitter library, leakage-audit checklist results | Data Analysis (for date ranges) | everyone | `src/windcon/metrics.py` (+tests), `src/windcon/cv.py`, `docs/leakage_audit.md` |
| **Model** | Implement/tune candidate models (LightGBM, CatBoost first; XGBoost as diversity; LSTM/TFT/PatchTST only as stretch track), produce OOF + test predictions per model | Feature tables, CV splits, metric library | OOF prediction files, test prediction files, per-model CV scores | Feature Engineering, Validation | Ensemble, Experiment (log) | `src/windcon/models/*.py`, `experiments/oof/*.parquet` |
| **Ensemble** | Weighted blend / stacking of Model Agent outputs, prediction calibration against FICR bands, uncertainty flags for QA | OOF + test predictions from ≥2 models | Final blended OOF + test predictions, blend weights | Model | Submission, Experiment (log) | `src/windcon/ensemble.py`, blend weight config |
| **Experiment** (infra, not gatekeeper) | Experiment log schema, ablation record-keeping, reproducibility check (seed/version pinning), model comparison table | Runs from Model/Ensemble | Append-only experiment log, comparison leaderboard (local) | Model, Ensemble | Planner | `experiments/log.csv`, `docs/model_comparison.md` |
| **Submission** | Build final submission CSV, validate schema (`forecast_id`, `forecast_kst_dtm` untouched; no NaN/negative values; correct encoding), rule-compliance sanity check, submission-count budget tracking (5/day) | Ensemble's final predictions | Submitted CSV, submission log | Ensemble, Validation | Planner | `submissions/*.csv`, `docs/submission_log.md` |

**Parallelizable now:** Data Analysis and (metric-only parts of) Validation can start immediately and simultaneously — they don't depend on each other. Feature Engineering can start once Data Analysis publishes the turbine/group mapping (fast — that mapping is already extracted in Phase 2 above, so Feature Engineering is effectively unblocked today).

**Must be sequential:** Validation's metric library → any Model Agent work that needs to self-score (blocking, do first). Feature Engineering's train+test feature tables → Model Agent. Model Agent's OOF predictions → Ensemble. Ensemble's final predictions → Submission. Experiment Agent logging runs alongside Model/Ensemble continuously (not a gate).

---

## Phase 4 — Execution Plan

### 4.1 System architecture

```
raw CSVs (train/, test/, info.xlsx)
        │
        ▼
 windcon.data          — loaders, utf-8-sig, dtype/parsing, turbine/group map
        │
        ▼
 windcon.features       — U/V→speed/dir, grid spatial reduction, cyclical time,
                           lag/rolling, power-curve features (from SCADA, train-only)
        │
        ├──────────────► windcon.metrics + windcon.cv  (shared, imported everywhere)
        ▼
 windcon.models          — LightGBM / CatBoost / XGBoost (+ stretch: LSTM/TFT/PatchTST)
        │  (per-group OOF + test predictions)
        ▼
 windcon.ensemble        — blend/stack, FICR-aware calibration
        │
        ▼
 windcon.submission      — schema validation, forecast_id/forecast_kst_dtm integrity, write CSV
```

### 4.2 Project folder structure

```
windcon/
├── data/                      # symlink or reference to existing train/, test/, info.xlsx (unchanged)
├── src/windcon/
│   ├── __init__.py
│   ├── config.py              # paths, capacities, group/turbine map, constants
│   ├── data/
│   │   ├── loaders.py         # read_ldaps, read_gfs, read_labels, read_scada
│   │   └── grid_map.py        # nearest-grid / IDW farm-location reduction
│   ├── features/
│   │   ├── wind.py            # uv -> speed/direction
│   │   ├── time_features.py   # cyclical hour/doy, is_daylight, etc.
│   │   ├── lag_rolling.py
│   │   └── power_curve.py     # SCADA-fit empirical power curve per farm
│   ├── metrics.py             # NMAE, FICR, masked total score (SHARED — Validation-owned)
│   ├── cv.py                  # expanding-window / blocked time-series splitters
│   ├── models/
│   │   ├── lgbm.py
│   │   ├── catboost.py
│   │   └── ensemble.py
│   └── submission.py
├── scripts/
│   ├── build_features.py
│   ├── train.py
│   ├── predict.py
│   └── make_submission.py
├── tests/
│   ├── test_metrics.py        # unit tests against hand-computed NMAE/FICR examples
│   ├── test_cv.py             # asserts no future leakage across folds
│   └── test_data_loaders.py
├── experiments/
│   └── log.csv
├── submissions/
├── docs/
│   ├── PLAN.md                 # this file
│   ├── eda_report.md
│   ├── leakage_audit.md
│   └── model_comparison.md
├── configs/
│   └── turbine_group_map.yaml
├── requirements.txt
└── pyproject.toml
```

### 4.3 Milestones & timeline (26 days, today = 2026-07-19)

| Milestone | Window | Exit criteria |
|---|---|---|
| M0 — Foundations | Day 0–1 (Jul 19–20) | `windcon.metrics` implemented + unit-tested; project scaffolding + deps pinned; turbine/group map config committed |
| M1 — Data pipeline & EDA | Day 1–4 (Jul 20–23) | Loaders for all 5 sources; long→wide grid pivot; leakage audit doc; EDA report (missingness, wind distributions, group_3 shortfall) |
| M2 — First working submission | Day 4–6 (Jul 23–25) | Physical power-curve baseline + naive climatology baseline scored via local CV and submitted once (sanity-checks the whole path incl. schema) |
| M3 — Feature engineering + GBM baseline | Day 6–13 (Jul 25–Aug 1) | LightGBM/CatBoost per group with lag/rolling/spatial/power-curve features; CV score tracked in `experiments/log.csv`; beats M2 baseline materially |
| M4 — Ensembling & CV hardening | Day 13–19 (Aug 1–7, ends at team-merge deadline) | ≥2 diverse models blended; CV-vs-LB correlation checked against public LB; FICR-aware calibration applied (pending table confirmation) |
| M5 — Stretch: sequence models | Day 19–23 (Aug 7–11), *only if M3/M4 solid* | LSTM/PatchTST trained on same CV; folded into ensemble only if it beats GBM ensemble OOF |
| M6 — Finalize & document | Day 23–26 (Aug 11–14) | Final submission selected, code frozen/reproducible, external-data provenance doc ready for secondary evaluation |

### 4.4 Task priority (this week)

1. **P0 — blocking:** confirm exact FICR settlement/error-band table (open the evaluation page in-browser or screenshot it; the chart didn't extract via text fetch).
2. **P0:** implement + unit-test `windcon.metrics` (NMAE, FICR placeholder using best-known KPX incentive structure until #1 resolves, masked total score, per-group breakdown).
3. **P0:** implement `windcon.cv` time-respecting splitter; write a leakage test that asserts no fold's train window overlaps its validation `data_available_kst_dtm`.
4. **P1:** data loaders + long→wide pivot for LDAPS/GFS; turbine/group map parser for `info.xlsx`.
5. **P1:** EDA pass (missingness already quantified above; extend to wind-speed distributions per farm, seasonal power curves).
6. **P2:** baseline model + first submission (validates the entire path end-to-end before investing in feature engineering).

### 4.5 Risks & mitigations

| Risk | Mitigation |
|---|---|
| FICR table unknown | Treat as P0 blocking research item; build metrics module with FICR as a swappable/config-driven function so confirming it later is a one-file change, not a rewrite |
| CV doesn't correlate with Private LB | Reserve a held-out "pseudo-private" late-time-window split never touched during feature/model iteration; check correlation before spending submissions |
| Only 5 submissions/day | Never submit to "see what happens" — every submission must follow a specific CV-driven hypothesis, logged in `experiments/log.csv` |
| group_3 has ~1 fewer year of labels | Consider per-group regularization strength / simpler model for group_3, or partial-pooling (train shared model with group as categorical + group-specific fine-tune) |
| Grid-to-farm spatial reduction done naively hurts signal | A/B nearest-grid vs. inverse-distance-weighted vs. all-grids-as-features inside CV before committing |
| Reproducibility failure at secondary evaluation | Pin dependency versions from Day 0; every script runnable via a single documented command; no manual notebook-only steps in the final path |
| Deep learning (M5) burns time with no payoff | Hard time-box; only attempt after M3/M4 exit criteria met, with an explicit go/no-go against GBM ensemble OOF score |

### 4.6 Leaderboard-improvement roadmap (systematic experimentation)

1. Baseline (power curve + climatology) → establishes floor and validates pipeline.
2. Add spatial/temporal weather features → biggest expected jump (GBM baselines on this kind of task are typically feature-quality-bound, not algorithm-bound).
3. Add SCADA-derived power-curve features (train-period only, as a *feature*, not a leak — power curve is a farm characteristic, not future information).
4. Hyperparameter tuning per group (group-aware, not global) once feature set stabilizes.
5. Multi-model ensembling (LightGBM + CatBoost + physical baseline) — diversity typically buys more than marginal tuning at this stage.
6. FICR-aware calibration/post-processing once table confirmed — likely the single highest-ROI change late in the timeline since it directly targets 50% of the score that plain-MAE optimization doesn't directly address.
7. Only then: sequence models, if time remains and CV shows headroom.

---

## Status

**Plan approved 2026-07-19.** Execution paused at planning phase by request. FICR table confirmation demoted from blocking P0 to a task revisited during the EDA phase (Milestone M1), in the new development environment — not required before that.

## Immediate next step

Next phase is a comprehensive EDA (Milestone M1), to be started only on explicit go-ahead. EDA findings will be used to refine this plan (Section 4.3 onward) before any feature engineering or model work begins.
