# Stroke Visualization Report

This report is generated separately from the training pipeline.

## Dataset Snapshot

| Metric | Value |
| --- | --- |
| Rows | 5,110 |
| Stroke cases | 249 |
| Stroke prevalence | 4.87% |
| Source file | healthcare_data_cleaned.csv |

## Statistical Findings

- `age`, `glucose_level`, and `bmi_value` are compared with Mann-Whitney U tests.
- `gender`, `smoking_habit`, `employment_type`, `residence`, `has_hypertension`, and `has_heart_disease` are compared with chi-square tests.
- The full test table is saved as `statistical_tests_summary.csv`.

## Latest Ensemble Summary

| Mode | Model | AUC-ROC | AUC-PR | Recall | Precision | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| High Sensitivity | Logistic Regression | 0.8326 | 0.2102 | 0.94 | 0.0793 | 0.1462 |
| High Sensitivity | Random Forest | 0.8183 | 0.1834 | 0.88 | 0.0841 | 0.1536 |
| High Sensitivity | XGBoost | 0.8034 | 0.1681 | 0.46 | 0.1679 | 0.246 |
| High Sensitivity | Soft Voting | 0.8282 | 0.1905 | 0.82 | 0.1059 | 0.1876 |
| High Sensitivity | Stacking | 0.8383 | 0.215 | 0.92 | 0.0819 | 0.1503 |
| Balanced | Logistic Regression | 0.8326 | 0.2102 | 0.42 | 0.2234 | 0.2917 |
| Balanced | Random Forest | 0.8183 | 0.1834 | 0.42 | 0.2188 | 0.2877 |
| Balanced | XGBoost | 0.8034 | 0.1681 | 0.04 | 0.1538 | 0.0635 |
| Balanced | Soft Voting | 0.8282 | 0.1905 | 0.26 | 0.2 | 0.2261 |
| Balanced | Stacking | 0.8383 | 0.215 | 0.78 | 0.194 | 0.3108 |

## Generated Figures

- `01_class_balance.png`
- `02_numeric_distributions.png`
- `03_categorical_counts.png`
- `04_boxplots.png`
- `05_correlation_heatmap.png`
- `06_statistical_tests.png`
- `07_model_metric_heatmap.png`
- `08_xgb_feature_importance.png`

## Notes

- The modeling pipeline is unchanged; this script only creates reports and visuals.
- The model summary and feature importance reflect the latest run already captured in the repository output.