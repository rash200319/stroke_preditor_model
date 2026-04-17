import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, precision_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────
df = pd.read_csv('healthcare_data_cleaned.csv')

DROP_COLS = [
    'Unnamed: 0', 'patient_id',
    'age_group', 'bmi_category',
    'high_glucose',
    'risk_score', 'lifestyle_risk'
]
df.drop(columns=DROP_COLS, inplace=True)

cat_cols = ['gender', 'employment_type', 'residence', 'smoking_habit']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['stroke_event'])
y = df['stroke_event'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaled version for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Data ready. Training 3 models...\n")

# ─────────────────────────────────────────────
# 2. DEFINE MODELS
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42, C=0.1),
        "scaled": True
    },
    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        "scaled": False
    },
    "XGBoost": {
        "model": XGBClassifier(
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
    random_state=42,
    n_jobs=-1
),
        "scaled": False
    }
}

# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE ALL MODELS
# ─────────────────────────────────────────────
results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, cfg in models.items():
    model = cfg["model"]
    Xtr = X_train_scaled if cfg["scaled"] else X_train
    Xte = X_test_scaled  if cfg["scaled"] else X_test
    Xcv = scaler.transform(X) if cfg["scaled"] else X

    print(f"Training {name}...")
    model.fit(Xtr, y_train)

    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    cv_auc = cross_val_score(model, Xcv, y, cv=skf, scoring='roc_auc', n_jobs=-1)

    results[name] = {
        "model":     model,
        "scaled":    cfg["scaled"],
        "y_pred":    y_pred,
        "y_proba":   y_proba,
        "accuracy":  accuracy_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "recall":    recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "cv_auc_mean": cv_auc.mean(),
        "cv_auc_std":  cv_auc.std(),
    }
    print(f"  ✓ AUC={results[name]['roc_auc']:.4f}  Recall={results[name]['recall']:.4f}\n")

# ─────────────────────────────────────────────
# 4. PRINT COMPARISON TABLE
# ─────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Model':<22} {'Accuracy':>9} {'ROC-AUC':>9} {'Recall':>8} {'Precision':>10} {'F1':>7} {'CV-AUC':>14}")
print("="*75)
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:>9.4f} {r['roc_auc']:>9.4f} "
          f"{r['recall']:>8.4f} {r['precision']:>10.4f} {r['f1']:>7.4f} "
          f"{r['cv_auc_mean']:>7.4f}±{r['cv_auc_std']:.4f}")
print("="*75)

best = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\n🏆 Best model by ROC-AUC: {best} ({results[best]['roc_auc']:.4f})")

# ─────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Model Comparison — Stroke Prediction', fontsize=16, fontweight='bold', y=1.01)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

colors = ['#4C72B0', '#55A868', '#DD8452']
model_names = list(results.keys())

# --- (A) ROC Curves Overlaid ---
ax_roc = fig.add_subplot(gs[0, 0])
for (name, r), color in zip(results.items(), colors):
    RocCurveDisplay.from_predictions(
        y_test, r['y_proba'], ax=ax_roc,
        name=f"{name} (AUC={r['roc_auc']:.4f})",
        color=color
    )
ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
ax_roc.set_title('ROC Curves — All Models')
ax_roc.legend(fontsize=8)

# --- (B) Metric Bar Chart ---
ax_bar = fig.add_subplot(gs[0, 1])
metrics = ['accuracy', 'roc_auc', 'recall', 'f1']
metric_labels = ['Accuracy', 'ROC-AUC', 'Recall', 'F1']
x = np.arange(len(metrics))
width = 0.25
for i, (name, r) in enumerate(results.items()):
    vals = [r[m] for m in metrics]
    bars = ax_bar.bar(x + i*width, vals, width, label=name, color=colors[i], alpha=0.85)
ax_bar.set_xticks(x + width)
ax_bar.set_xticklabels(metric_labels)
ax_bar.set_ylim(0.75, 1.02)
ax_bar.set_title('Metric Comparison')
ax_bar.legend(fontsize=8)
ax_bar.set_ylabel('Score')

# --- (C) CV AUC with Error Bars ---
ax_cv = fig.add_subplot(gs[0, 2])
cv_means = [results[n]['cv_auc_mean'] for n in model_names]
cv_stds  = [results[n]['cv_auc_std']  for n in model_names]
bars = ax_cv.bar(model_names, cv_means, yerr=cv_stds, capsize=6,
                  color=colors, alpha=0.85, error_kw={'linewidth': 2})
ax_cv.set_ylim(0.75, 1.05)
ax_cv.set_title('5-Fold CV AUC (Mean ± Std)')
ax_cv.set_ylabel('AUC')
for bar, mean in zip(bars, cv_means):
    ax_cv.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax_cv.tick_params(axis='x', labelsize=9)

# --- (D-F) Confusion Matrices ---
for i, (name, r) in enumerate(results.items()):
    ax_cm = fig.add_subplot(gs[1, i])
    cm = confusion_matrix(y_test, r['y_pred'])
    ConfusionMatrixDisplay(cm, display_labels=['No Stroke', 'Stroke']).plot(
        ax=ax_cm, colorbar=False, cmap='Blues'
    )
    ax_cm.set_title(f'{name}\nAcc={r["accuracy"]:.4f} | AUC={r["roc_auc"]:.4f}')

plt.savefig('model_comparison_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as 'model_comparison_results.png'")

# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE COMPARISON
# ─────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Feature Importance — Tree Models', fontsize=13, fontweight='bold')

for ax, name, color in zip(axes, ['Random Forest', 'XGBoost'], ['#55A868', '#DD8452']):
    model = results[name]['model']
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    imp.plot(kind='barh', ax=ax, color=color, alpha=0.85)
    ax.set_title(f'{name} — Feature Importance')
    ax.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Feature importance plot saved as 'feature_importance_comparison.png'")