# Stroke Prediction — Competition Roadmap

> **DataXplore Stage 01** | Deadline: 24th April 2026 (11:59 PM)
> A complete action-plan to turn the current repo into a winning submission.

---

## 1. Current State Audit

### What Exists
| File | Status | Notes |
|------|--------|-------|
| `healthcare_data.csv` | Raw data (9722 rows × 17 cols) | 927 missing `bmi_value` |
| `healthcare_data_cleaned.csv` | Cleaned data | BMI imputed via grouped median |
| `clean_data.py` | **Buggy** | References undefined `df_imputed` on line 35 |
| `clean_healthcare_data.ipynb` | Data cleaning notebook | Cells in reverse order, incomplete |
| `model.py` | XGBoost model | 96.35% acc, 0.993 AUC — strong but undocumented |
| `stroke_xgboost_model.json` | Saved model | Works |
| `requirements.txt` | **Empty** | Must be populated |
| `stroke_model_results.png` | Results plot | Confusion matrix, ROC, feature importance |

### Critical Issues to Fix
1. **`clean_data.py`** — `df_imputed` is used before being defined (line 35). The grouped-median imputation writes to `df` but the rest of the script expects `df_imputed`.
2. **`requirements.txt`** — Currently empty. Must list: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `scipy`.
3. **No single competition notebook** — All work is split across `.py` files. Need one cohesive `.ipynb` for the submission.

---

## 2. Data Issues to Investigate

### 2.1 Suspicious Target Balance
- **Current**: `stroke_event` is exactly 4861 vs 4861 (50/50 split)
- **Real-world stroke prevalence** is ~2-5%. This dataset appears pre-balanced or synthetically augmented.
- **Action**: Document this in the report. If the organizers provided it balanced, state that explicitly. If you balanced it yourself, explain the method (SMOTE, oversampling, etc.) and **also show results on the original imbalanced data** for comparison.

### 2.2 Derived / Engineered Columns (Potential Leakage)
These columns are derived from raw features and are already in the dataset:

| Column | Derived From | Risk |
|--------|-------------|------|
| `age_group` | `age` | Redundant binning |
| `bmi_category` | `bmi_value` | Redundant binning |
| `high_glucose` | `glucose_level` | Binary threshold duplicate |
| `risk_score` | Multiple features | **High leakage risk** — composite of target-correlated vars |
| `lifestyle_risk` | `smoking_habit` + others | **High leakage risk** |

- **Current model.py already drops these** — good.
- **Action**: Still analyze them in EDA to show understanding, but clearly justify why they are excluded from modeling.

### 2.3 Missing Values
- Only `bmi_value` has 927 missing rows (9.5%)
- **Current approach**: Grouped median by gender + age_group — reasonable
- **Better approach**: Compare multiple imputation strategies (mean, median, KNN, iterative) and justify the chosen one with RMSE or distribution comparison plots

---

## 3. Detailed Task Checklist

### Phase A: Data Understanding & Exploration (EDA)

- [ ] **A1. Dataset Overview**
  - Shape, dtypes, head/tail, describe()
  - Document each variable with description (use the provided Variable Description table)

- [ ] **A2. Univariate Analysis**
  - Histograms/KDE for continuous: `age`, `glucose_level`, `bmi_value`
  - Bar charts for categorical: `gender`, `employment_type`, `residence`, `smoking_habit`, `marital_status`
  - Box plots to detect outliers in `bmi_value`, `glucose_level`

- [ ] **A3. Bivariate Analysis (vs stroke_event)**
  - Stroke rate by age group, gender, smoking habit, employment type, residence
  - Grouped bar charts / stacked proportions
  - Violin/box plots of `age`, `glucose_level`, `bmi_value` split by stroke_event

- [ ] **A4. Correlation Analysis**
  - Pearson correlation heatmap for numerical features
  - Point-biserial correlation for binary target vs continuous features
  - Cramér's V for categorical × categorical associations

- [ ] **A5. Statistical Tests**
  - **Chi-square tests**: `gender` vs `stroke_event`, `smoking_habit` vs `stroke_event`, etc.
  - **T-tests / Mann-Whitney U**: Compare `age`, `glucose_level`, `bmi_value` distributions between stroke vs no-stroke groups
  - Report p-values and effect sizes
  - This is what separates a good report from a great one (guidelines say "beyond simple visualization")

- [ ] **A6. Class Imbalance Documentation**
  - Show original vs balanced distribution
  - Explain balancing strategy used

- [ ] **A7. Missing Value Analysis**
  - Visualize missingness pattern (e.g., `missingno` library or heatmap)
  - Check if missingness is random (MCAR/MAR/MNAR)

### Phase B: Data Preparation & Visualization

- [ ] **B1. Missing Value Imputation**
  - Compare strategies: mean, median, grouped median, KNN imputer
  - Show distribution before/after imputation
  - Justify final choice

- [ ] **B2. Feature Encoding**
  - Label encoding for ordinal (`smoking_habit`: non_smoker < ex_smoker < current_smoker)
  - One-hot encoding for nominal (`employment_type`, `residence`)
  - Document encoding decisions

- [ ] **B3. Feature Engineering**
  - Interaction terms: `age × glucose_level`, `age × has_hypertension`
  - Polynomial features for key predictors
  - Consider: `glucose_level / bmi_value` ratio
  - Age-based risk flags (age > 60 with hypertension, etc.)

- [ ] **B4. Feature Selection**
  - Mutual information scores
  - Variance Inflation Factor (VIF) for multicollinearity
  - Recursive Feature Elimination (RFE)
  - Compare model performance with different feature sets

- [ ] **B5. Scaling**
  - StandardScaler or MinMaxScaler for models that need it (Logistic Regression, SVM)
  - Not needed for tree-based models but document the decision

- [ ] **B6. Train/Test Split**
  - 80/20 stratified split
  - Document random_state for reproducibility

### Phase C: Model Development

- [ ] **C1. Baseline Model**
  - Logistic Regression as interpretable baseline
  - Report accuracy, precision, recall, F1, AUC

- [ ] **C2. Model Comparison** (compare at least 3-4)
  | Model | Why |
  |-------|-----|
  | Logistic Regression | Interpretable baseline, clinical acceptance |
  | Random Forest | Non-linear, feature importance |
  | XGBoost | Current best performer |
  | LightGBM | Fast, handles imbalance well |
  | SVM (optional) | Good with small-medium datasets |

- [ ] **C3. Hyperparameter Tuning**
  - GridSearchCV or RandomizedSearchCV for top 2 models
  - Document search space and best parameters
  - Current XGBoost params are reasonable but not tuned systematically

- [ ] **C4. Cross-Validation**
  - 5-fold or 10-fold stratified CV
  - Report mean ± std for all metrics
  - Already done for XGBoost (mean AUC 0.9932) — replicate for all models

- [ ] **C5. Model Evaluation Metrics**
  - **Primary**: ROC-AUC (since it's a probability-based prediction task)
  - **Secondary**: Recall/Sensitivity (minimize missed strokes — same reasoning as Gmora notebook)
  - Confusion matrix with annotations
  - ROC curves overlaid for all models
  - Precision-Recall curves

- [ ] **C6. Explainability (SHAP)**
  - SHAP summary plot (beeswarm) — replaces Grad-CAM from Gmora notebook
  - SHAP dependence plots for top 3 features
  - SHAP force plots for individual predictions
  - This is the **single biggest differentiator** for a winning submission

- [ ] **C7. Feature Importance**
  - XGBoost built-in importance (gain, cover, weight)
  - Permutation importance
  - Compare with SHAP importance — do they agree?

### Phase D: Insight Generation

- [ ] **D1. Top Risk Factors** (from model + EDA)
  - Rank features by importance across methods
  - `age` is dominant (corr 0.58) — discuss clinical relevance
  - `glucose_level` and `has_hypertension` are strong predictors

- [ ] **D2. Subgroup Analysis**
  - Stroke rates by demographic segments (age × gender, age × smoking)
  - Identify highest-risk subpopulations
  - Create risk profiles (e.g., "Male, 65+, hypertensive, high glucose → X% stroke probability")

- [ ] **D3. Preventive Healthcare Recommendations**
  - Based on modifiable risk factors: glucose control, smoking cessation, BMI management
  - Actionable insights for healthcare providers
  - Cost-benefit argument for early screening

- [ ] **D4. Limitations & Future Work**
  - Dataset limitations (balanced target, single source, no temporal data)
  - Model limitations (no causal inference, potential confounders)
  - Suggested improvements (longitudinal data, external validation, deep learning)

---

## 4. Report Structure (Matching Competition Guidelines)

The final PDF report must follow this structure:

### Cover Page
- Team Name
- University Name
- Team Member Details (name, student ID, role)

### 1. Introduction (1-2 pages)
- Stroke as a global health challenge (cite WHO statistics)
- Importance of early prediction
- Objective: predict stroke likelihood + identify key risk factors
- Brief overview of approach

### 2. Methodology (2-3 pages)
- Data description (source, size, variables)
- Data cleaning & imputation strategy
- Feature engineering & selection
- Model selection rationale
- Evaluation framework (metrics, cross-validation)
- Tools & libraries used

### 3. Data Analysis (4-6 pages) — **This is the core**
- EDA findings with labeled figures and tables
- Statistical test results
- Correlation analysis
- Missing value analysis
- Feature importance analysis
- Model comparison results
- SHAP explainability visualizations

### 4. Discussion (1-2 pages)
- Interpretation of key findings
- Why certain features matter (clinical reasoning)
- Comparison of model performances
- Strengths and limitations of the analysis
- How the balanced target affects interpretation

### 5. Conclusion (0.5-1 page)
- Summary of findings
- Top 3-5 actionable insights
- Recommendations for healthcare organizations
- Future work suggestions

### Contribution Statement (table)
| Member | Contribution |
|--------|-------------|
| Name 1 | EDA, visualization |
| Name 2 | Model development, tuning |
| ... | ... |

---

## 5. Patterns to Emulate from Gmora Winning Notebook

The reference notebook (`Resources/Copy_of_Gmora_Notebook (1).ipynb`) scored well because of:

| Pattern | How to Apply to Stroke Project |
|---------|-------------------------------|
| **Section numbering** (1-11) | Number all notebook sections clearly |
| **Problem definition** with stats | Open with stroke statistics (WHO: 15M strokes/year, 5M deaths) |
| **Evaluation strategy justification** | Justify why AUC + Recall are primary metrics for stroke |
| **Pretrained model disclosure** | N/A for tabular data, but document XGBoost hyperparameters transparently |
| **Reproducibility** (seed=42) | Already done — keep it |
| **Grad-CAM explainability** | Replace with **SHAP** (the tabular equivalent) |
| **Failure analysis** | Analyze misclassified patients — what makes them hard to predict? |
| **Demo inference** | Create a simple prediction function: input patient data → stroke probability |
| **Training logs + curves** | Plot learning curves, validation metrics per epoch/iteration |
| **Class-wise metrics** | Report precision/recall for both classes separately |

---

## 6. Priority Order (What to Do First)

| Priority | Task | Time Estimate |
|----------|------|--------------|
| **P0** | Fix `clean_data.py` bug + populate `requirements.txt` | 15 min |
| **P1** | Create comprehensive Jupyter notebook with all sections | 3-4 hours |
| **P2** | Complete EDA with statistical tests | 2-3 hours |
| **P3** | Model comparison (Logistic Reg, RF, XGBoost, LightGBM) | 2 hours |
| **P4** | SHAP explainability | 1 hour |
| **P5** | Insight generation + subgroup analysis | 1-2 hours |
| **P6** | Write final PDF report following guidelines structure | 2-3 hours |
| **P7** | Review: figures labeled, tables numbered, no AI flags | 1 hour |

**Total estimated effort: ~13-16 hours**

---

## 7. Technical Notes

### Current Model Performance (for reference)
```
Accuracy  : 96.35%
ROC-AUC   : 0.9932
Precision : 1.00 (No Stroke) / 0.93 (Stroke)
Recall    : 0.93 (No Stroke) / 1.00 (Stroke)
5-Fold CV AUC: 0.9932 ± 0.0021
```

### Key Correlations with Stroke
```
age               0.579
risk_score        0.309  (dropped — leakage)
marital_status    0.277
high_glucose      0.265  (dropped — leakage)
glucose_level     0.256
has_hypertension  0.243
has_heart_disease 0.214
bmi_value         0.113
```

### Important Warning
> The competition guidelines state: **"Use of AI-generated content/tools is strictly prohibited."**
> Use this roadmap for guidance only. All final report text, analysis, and code must be written by team members. Rewrite everything in your own words and style.

---

*Generated as a planning document — not for submission.*
