import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────
df = pd.read_csv('healthcare_data_cleaned.csv')

# Drop leakage, redundant, and ID columns
DROP_COLS = [
    'Unnamed: 0', 'patient_id',   # identifiers
    'age_group', 'bmi_category',  # binned duplicates of age & bmi_value
    'high_glucose',                # binary duplicate of glucose_level
    'risk_score', 'lifestyle_risk' # composite scores — potential leakage
]
df.drop(columns=DROP_COLS, inplace=True)

# Encode categorical columns
cat_cols = ['gender', 'employment_type', 'residence', 'smoking_habit']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop(columns=['stroke_event'])
y = df['stroke_event'].astype(int)

print(f"Features used ({len(X.columns)}): {X.columns.tolist()}")
print(f"Dataset shape: {X.shape}")
print(f"Target balance: {y.value_counts().to_dict()}\n")

# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")

# ─────────────────────────────────────────────
# 3. BUILD XGBOOST MODEL
# ─────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,       # L1 regularization
    reg_lambda=1.0,      # L2 regularization
    use_label_encoder=False,
    eval_metric='auc',
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1
)

# Train with early stopping on a validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=50
)

print(f"\nBest iteration: {model.best_iteration}")

# ─────────────────────────────────────────────
# 4. EVALUATE ON TEST SET
# ─────────────────────────────────────────────
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("         MODEL PERFORMANCE ON TEST SET")
print("="*50)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print("="*50)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))

# ─────────────────────────────────────────────
# 5. CROSS-VALIDATION (5-FOLD)
# ─────────────────────────────────────────────
cv_model = XGBClassifier(
    n_estimators=model.best_iteration,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(cv_model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)

print("\n5-Fold Cross-Validation AUC:")
for i, s in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {s:.4f}")
print(f"  Mean : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Stroke Prediction — XGBoost Results', fontsize=14, fontweight='bold')

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['No Stroke', 'Stroke']).plot(
    ax=axes[0], colorbar=False, cmap='Blues'
)
axes[0].set_title('Confusion Matrix')

# --- ROC Curve ---
RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=axes[1], color='darkorange')
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_title(f'ROC Curve (AUC = {roc_auc:.4f})')

# --- Feature Importance ---
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', ax=axes[2], color='steelblue')
axes[2].set_title('Feature Importance')
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('stroke_model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as 'stroke_model_results.png'")

# ─────────────────────────────────────────────
# 7. SAVE MODEL
# ─────────────────────────────────────────────
model.save_model('stroke_xgboost_model.json')
print("Model saved as 'stroke_xgboost_model.json'")
