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
| High Sensitivity | Logistic Regression | 0.8307 | 0.2324 | 0.96 | 0.0811 | 0.1495 |
| High Sensitivity | Random Forest | 0.821 | 0.1749 | 0.9 | 0.0846 | 0.1546 |
| High Sensitivity | XGBoost | 0.815 | 0.2147 | 0.4 | 0.2041 | 0.2703 |
| High Sensitivity | Soft Voting | 0.8289 | 0.2085 | 0.82 | 0.1099 | 0.1939 |
| High Sensitivity | Stacking | 0.8342 | 0.2278 | 0.9 | 0.0829 | 0.1518 |
| Balanced | Logistic Regression | 0.8307 | 0.2324 | 0.76 | 0.1689 | 0.2764 |
| Balanced | Random Forest | 0.821 | 0.1749 | 0.52 | 0.1926 | 0.2811 |
| Balanced | XGBoost | 0.815 | 0.2147 | 0.18 | 0.3 | 0.225 |
| Balanced | Soft Voting | 0.8289 | 0.2085 | 0.4 | 0.2326 | 0.2941 |
| Balanced | Stacking | 0.8342 | 0.2278 | 0.8 | 0.1633 | 0.2712 |

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