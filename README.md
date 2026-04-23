# Stroke Prediction Project

This repository contains two main stages of the stroke prediction workflow:

- `final_clean.py` prepares the dataset for modelling.
- `final_stroke_ensemble.py` trains and evaluates the final leakage-safe ensemble on the real imbalanced dataset.

The project evolved from an earlier balanced setup into a more realistic medical classification workflow that keeps the original class imbalance and focuses on recall-sensitive evaluation.

## Repository Files

- `healthcare_data.csv` - raw source dataset.
- `healthcare_data_cleaned.csv` - cleaned dataset exported by `final_clean.py`.
- `final_clean.py` - deduplicates records, imputes BMI, adds clinically useful interaction features, and saves the cleaned dataset.
- `final_stroke_ensemble.py` - final ensemble training script with threshold tuning, stacking, and test-set evaluation.
- `visualization_report.py` - standalone script that generates polished EDA figures, statistical-test visuals, and a markdown report.
- `label_shuffle_test.py` - sanity check to confirm the pipeline is not leaking label information.
- `stroke_ensemble_roc_pr_fi.png` - ROC, precision-recall, and XGBoost feature importance plot.
- `stroke_ensemble_confusion.png` - confusion matrices for all tuned models.
- `stroke_ensemble_threshold_sweep.png` - threshold sweep for the stacking model.

## Data Preparation

`final_clean.py` performs the following steps:

- removes the accidental `Unnamed: 0` index column if present;
- deduplicates rows using all clinical columns except `patient_id`;
- imputes missing `bmi_value` values using grouped medians by `gender` and age band;
- adds interaction features such as `age_squared`, `age_x_glucose`, `age_x_hypertension`, `age_x_heart`, `glucose_x_hypertension`, and `age_risk_score`;
- saves the result to `healthcare_data_cleaned.csv`.

## Final Ensemble Workflow

`final_stroke_ensemble.py` runs a full leakage-safe ensemble pipeline on the imbalanced dataset:

- loads `healthcare_data.csv`;
- removes duplicate clinical rows;
- drops leakage-prone or unnecessary columns such as `patient_id`, `age_group`, `bmi_category`, `high_glucose`, `risk_score`, and `lifestyle_risk` when present;
- engineers predictive features including:
  - `age_squared`
  - `age_over_10`
  - `age_x_hypertension`
  - `age_x_heart_disease`
  - `glucose_x_bmi`
  - `glucose_per_bmi`
  - `bmi_deviation`
  - `cvd_count`
  - `is_senior`
  - `log_glucose`
- splits the data into train, validation, and test partitions with stratification;
- handles class imbalance with SMOTE on the training folds only;
- trains three base models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- combines them with:
  - Soft Voting
  - Stacking
- tunes decision thresholds on the validation set for two operating modes:
  - High Sensitivity
  - Balanced
- evaluates the final models on an untouched test set.

## Dataset Summary From The Latest Run

- Original rows: `9722`
- After deduplication: `5110`
- Duplicates removed: `4612`
- Stroke cases: `249`
- Stroke prevalence: `4.87%`

Train/validation/test split from the latest run:

- Train: `3270`
- Validation: `818`
- Test: `1022`
- Train stroke cases: `159` (`4.9%`)
- Test stroke cases: `50` (`4.9%`)
- `scale_pos_weight`: `19.54`

## Final Test Results

### High Sensitivity Mode

| Model | Thr | AUC-ROC | AUC-PR | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.13 | 0.8326 | 0.2102 | 0.9400 | 0.0793 | 0.1462 |
| Random Forest | 0.02 | 0.8183 | 0.1834 | 0.8800 | 0.0841 | 0.1536 |
| XGBoost | 0.56 | 0.8034 | 0.1681 | 0.4600 | 0.1679 | 0.2460 |
| Soft Voting | 0.24 | 0.8282 | 0.1905 | 0.8200 | 0.1059 | 0.1876 |
| Stacking | 0.12 | 0.8383 | 0.2150 | 0.9200 | 0.0819 | 0.1503 |

### Balanced Mode

| Model | Thr | AUC-ROC | AUC-PR | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.78 | 0.8326 | 0.2102 | 0.4200 | 0.2234 | 0.2917 |
| Random Forest | 0.38 | 0.8183 | 0.1834 | 0.4200 | 0.2188 | 0.2877 |
| XGBoost | 0.85 | 0.8034 | 0.1681 | 0.0400 | 0.1538 | 0.0635 |
| Soft Voting | 0.63 | 0.8282 | 0.1905 | 0.2600 | 0.2000 | 0.2261 |
| Stacking | 0.67 | 0.8383 | 0.2150 | 0.7800 | 0.1940 | 0.3108 |

## Best Performing Model

Stacking was the best overall model in the latest run:

- Best AUC-ROC in both modes: `0.8383`
- Best AUC-PR in both modes: `0.2150`
- Best recall in balanced mode: `0.7800`
- Best recall in high-sensitivity mode: `0.9200`

## XGBoost Feature Importance

Top features from the latest run:

1. `age_squared` - `0.155550`
2. `is_senior` - `0.131076`
3. `age` - `0.114347`
4. `cvd_count` - `0.088688`
5. `residence` - `0.071208`
6. `gender` - `0.067930`
7. `has_hypertension` - `0.046056`
8. `smoking_habit` - `0.038471`
9. `age_over_10` - `0.036362`
10. `employment_type` - `0.034493`

## Saved Figures

Running `final_stroke_ensemble.py` produces:

- `stroke_ensemble_roc_pr_fi.png`
- `stroke_ensemble_confusion.png`
- `stroke_ensemble_threshold_sweep.png`

Running `visualization_report.py` produces:

- `visuals/01_class_balance.png`
- `visuals/02_numeric_distributions.png`
- `visuals/03_categorical_counts.png`
- `visuals/04_boxplots.png`
- `visuals/05_correlation_heatmap.png`
- `visuals/06_statistical_tests.png`
- `visuals/07_model_metric_heatmap.png`
- `visuals/08_xgb_feature_importance.png`
- `visuals/visualization_report.md`

## Leakage Check

`label_shuffle_test.py` was used as a sanity check. When labels are shuffled, the models drop to chance-level performance, which supports the claim that the pipeline is not trivially leaking target information.

Observed shuffled-label behavior from the latest check:

- Logistic Regression ROC-AUC: `0.4943`
- Random Forest ROC-AUC: `0.4998`
- XGBoost ROC-AUC: `0.4980`

This is the expected outcome for a leakage-safe pipeline.

## Requirements

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

## How To Run

Run the cleaning step:

```bash
python final_clean.py
```

Run the final ensemble:

```bash
python final_stroke_ensemble.py
```

Run the label shuffle sanity check:

```bash
python label_shuffle_test.py
```

Generate the standalone visuals and report:

```bash
python visualization_report.py
```

## Notes

- The latest ensemble run keeps the natural class imbalance instead of forcing a 50/50 split.
- Threshold tuning is used to make the model more practical for stroke screening, where missing true stroke cases is costly.
- `final_stroke_ensemble.py` is the main reference implementation for the final reported results in this repository.
