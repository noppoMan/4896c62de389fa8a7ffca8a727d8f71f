## Project Overview

This repository is the artifact for the our paper. We estimate SVAR models from GitHub commit time series, derive cumulative IRF effects, aggregate them as MIAO scores (AMS_ij), and classify REV vs. non-REV with a decision tree. Reproduction primarily uses `demo.ipynb` and `miao.py`.

## Directory Overview (Key Artifacts)
- `datasets/original/{daily, monthly}`: Daily/monthly original commit counts per OSS project
- `datasets/preprocessed/permute_{0..5}/{period_shift}/{Tm}/`: Stationary series A1, A2, A3 for each Tms after fractional differencing (by group/period)
- `adf_test_results/permute_{i}.csv`: ADF and fractional differencing logs (by period/group/Ai)
- `var_estimation_results/permute_{i}.csv`: SVAR diagnostics (optimal lag, ICs, whiteness test)
- `miao_score_tables/permute_{i}/*.csv`: MIAO score tables (AMS_ij, normalized_AMS_ij)
- `datasets/decisiontree/{normalized, non-normalized}/permute_{i}.csv`: Features for the decision tree
- `prediction/`: Outputs produced by applying the same pipeline in a prediction setting (see below)

## File Overview
- `collect-candidates.sql`: SQL to search for REV candidates and non-REV candidates from ghs
- `dataset.yml`: The dataset used for all groups in the experiment

## Minimal Reproduction
1) Environment
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
2) Run the notebook (`demo.ipynb`)
- Auto-detecting Tm (`miao.auto_detect_tms`)
- Preparation for VAR Analysis (stationarization + export)
- SVAR and IRF-Calculation (`miao.prepare_var`, obtain IRFs)
- MIAO-Score-Calculation (`miao.miao_phase1_with_period_shifts`, `miao.miao_phase2_with_period_shifts`)
- Decision Tree (training and evaluation)

## Prediction
Artifacts under `prediction/` mirror the training pipeline for a prediction setting. For each `permute_{i}` (competitor ordering permutations for non-REV cases):
- `prediction/datasets/preprocessed/permute_{i}/...`: Preprocessed series by period/shift/Tm
- `prediction/var_estimation_results/permute_{i}.csv`: Diagnostics incl. optimal lag and whiteness tests
- `prediction/adf_test_results/permute_{i}.csv`: ADF and fractional differencing logs
- `prediction/miao_score_tables/permute_{i}/*.csv`: MIAO score tables for prediction

## Key Scripts
- `miao.py`
  - `auto_detect_tms`: Detect analysis periods while excluding long dormancy in the target series
  - `prepare_var`: Prepare VAR/SVAR (stationarization, lag selection, whiteness test)
  - `miao_phase1_with_period_shifts`: Collect SCE from IRFs across period shifts
  - `miao_phase2_with_period_shifts`: Aggregate SCE to AMS_ij and produce score tables

## Data/CSV Columns (Selected)
- `adf_test_results/*.csv`: period_shift, Tm, group, repo, statistic, pvalue, fracdiff-n
- `var_estimation_results/*.csv`: period_shift, Tm, group, nobs, lag, aic, bic, hqic, whiteness_test_lag, whiteness_test_statistic, whiteness_test_pvalue
- `datasets/decisiontree/*.csv`: label, group, target, directional AMS_ij features (e.g., `c1 -> t`)
- `classification_results/*.csv`: group, rev (ground truth), predicted (decision tree)