model - XGBOOST
cleaned - using age and gender.(by grouped mean)

use ipynb to clean data if needed. also i included my cleaned one.

# details of the model
---

Features used (10): ['gender', 'age', 'has_hypertension', 'has_heart_disease', 'marital_status', 'employment_type', 'residence', 'glucose_level', 'bmi_value', 'smoking_habit']
Dataset shape: (9722, 10)
Target balance: {0: 4861, 1: 4861}

Train size: 7777 | Test size: 1945

[0]     validation_0-auc:0.88482
[50]    validation_0-auc:0.95595
[100]   validation_0-auc:0.97224
[150]   validation_0-auc:0.98461
[200]   validation_0-auc:0.99118
[250]   validation_0-auc:0.99515
[300]   validation_0-auc:0.99626
[350]   validation_0-auc:0.99733
[383]   validation_0-auc:0.99735

Best iteration: 353

==================================================
         MODEL PERFORMANCE ON TEST SET
==================================================
  Accuracy  : 0.9635  (96.35%)
  ROC-AUC   : 0.9932
==================================================

Classification Report:
              precision    recall  f1-score   support

   No Stroke       1.00      0.93      0.96       973
      Stroke       0.93      1.00      0.96       972

    accuracy                           0.96      1945
   macro avg       0.97      0.96      0.96      1945
weighted avg       0.97      0.96      0.96      1945


5-Fold Cross-Validation AUC:
  Fold 1: 0.9955
  Fold 2: 0.9905
  Fold 3: 0.9956
  Fold 4: 0.9933
  Fold 5: 0.9911
  Mean : 0.9932 (+/- 0.0021)

Plot saved as 'stroke_model_results.png'
Model saved as 'stroke_xgboost_model.json'
