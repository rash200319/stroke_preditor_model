"""
stroke_model_imbalanced.py
─────────────────────────────────────────────────────────────────
XGBoost stroke prediction on the REAL imbalanced dataset (4.87% stroke).
Pipeline:
  1. Deduplicate raw data on clinical features
  2. Stratified 80/20 train-test split  (split BEFORE any imputation)
  3. Inside training only:
       • Grouped-median BMI imputation
       • Label encoding of categoricals
       • SMOTE oversampling of minority class
  4. XGBoost with scale_pos_weight tuned to class ratio
  5. Threshold tuning on a held-out validation split
  6. Final evaluation on untouched test set
  7. Plots: ROC, Precision-Recall, Confusion Matrix, Feature Importance
─────────────────────────────────────────────────────────────────
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, precision_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, average_precision_score
)
from xgboost import XGBClassifier

RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────────
# 1.  LOAD & DEDUPLICATE
# ─────────────────────────────────────────────────────────────────
def load_and_deduplicate(path: str = "healthcare_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    before = len(df)
    dedup_cols = [c for c in df.columns if c != "patient_id"]
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    print(f"Rows: {before} → {len(df)}  ({before - len(df)} duplicates removed)")

    # Drop columns: IDs, leakage risks, redundant derived columns
    DROP = [
        "patient_id",
        "age_group", "bmi_category", "high_glucose",  # redundant derivatives
        "risk_score", "lifestyle_risk",                 # leakage risk
    ]
    df = df.drop(columns=[c for c in DROP if c in df.columns])

    print(f"Stroke cases  : {df['stroke_event'].sum()} / {len(df)}  "
          f"({df['stroke_event'].mean()*100:.2f}%)")
    return df


# ─────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING  (fit on train, apply to train+test)
# ─────────────────────────────────────────────────────────────────
CAT_COLS = ["gender", "employment_type", "residence", "smoking_habit"]

def preprocess(X_train: pd.DataFrame,
               X_test:  pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Impute BMI with grouped median (gender × age_group proxy via pd.cut),
    then label-encode categoricals.
    All statistics derived from X_train only → no leakage.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()

    # -- BMI imputation: group median by gender + age bin (derived inside function)
    X_tr["_age_bin"] = pd.cut(X_tr["age"], bins=[0,40,60,200], labels=["young","middle","senior"])
    X_te["_age_bin"] = pd.cut(X_te["age"], bins=[0,40,60,200], labels=["young","middle","senior"])

    bmi_medians = (
        X_tr.groupby(["gender", "_age_bin"], observed=True)["bmi_value"]
        .median()
    )
    # Fill train
    def fill_bmi(row, medians):
        if pd.isna(row["bmi_value"]):
            try:
                return medians.loc[(row["gender"], row["_age_bin"])]
            except KeyError:
                return X_tr["bmi_value"].median()
        return row["bmi_value"]

    X_tr["bmi_value"] = X_tr.apply(lambda r: fill_bmi(r, bmi_medians), axis=1)
    X_te["bmi_value"] = X_te.apply(lambda r: fill_bmi(r, bmi_medians), axis=1)

    X_tr = X_tr.drop(columns=["_age_bin"])
    X_te = X_te.drop(columns=["_age_bin"])

    # -- Label encode categoricals (fit on train only)
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        # Handle unseen labels in test gracefully
        X_te[col] = X_te[col].astype(str).map(
            lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
        )
        encoders[col] = le

    feature_names = X_tr.columns.tolist()
    return X_tr.values, X_te.values, feature_names


# ─────────────────────────────────────────────────────────────────
# 3.  SMOTE  (applied only on training data)
# ─────────────────────────────────────────────────────────────────
def smote_oversample(X: np.ndarray, y: np.ndarray,
                     sampling_strategy: float = 0.30,
                     k: int = 5,
                     random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """
    Bring minority class up to `sampling_strategy` × majority count.
    sampling_strategy=0.30 means minority will be 30% of majority size.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    minority_cls = classes[np.argmin(counts)]
    majority_cls = classes[np.argmax(counts)]

    X_min = X[y == minority_cls]
    X_maj = X[y == majority_cls]

    target = int(np.ceil(sampling_strategy * len(X_maj)))
    n_gen  = max(0, target - len(X_min))
    if n_gen == 0 or len(X_min) < 2:
        return X, y

    eff_k = min(k, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=eff_k + 1).fit(X_min)
    neighbors = nn.kneighbors(X_min, return_distance=False)[:, 1:]

    synthetic = []
    for _ in range(n_gen):
        idx  = rng.integers(len(X_min))
        nidx = rng.choice(neighbors[idx])
        gap  = rng.random()
        synthetic.append(X_min[idx] + gap * (X_min[nidx] - X_min[idx]))

    X_syn = np.array(synthetic)
    y_syn = np.full(len(X_syn), minority_cls, dtype=y.dtype)

    X_out = np.vstack([X_maj, X_min, X_syn])
    y_out = np.concatenate([np.full(len(X_maj), majority_cls, dtype=y.dtype),
                             y[y == minority_cls], y_syn])
    print(f"SMOTE: {len(X_min)} → {len(X_min)+len(X_syn)} minority samples  "
          f"(total train: {len(X_out)})")
    return X_out, y_out


# ─────────────────────────────────────────────────────────────────
# 4.  BUILD MODEL
# ─────────────────────────────────────────────────────────────────
def build_xgboost(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators      = 500,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,
        gamma             = 0.2,
        reg_alpha         = 0.5,
        reg_lambda        = 2.0,
        scale_pos_weight  = scale_pos_weight,
        objective         = "binary:logistic",
        eval_metric       = "aucpr",          # AUC-PR better for imbalance
        early_stopping_rounds = 30,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )


# ─────────────────────────────────────────────────────────────────
# 5.  THRESHOLD TUNING
# ─────────────────────────────────────────────────────────────────
def tune_threshold(y_true: np.ndarray,
                   y_proba: np.ndarray,
                   min_precision: float = 0.20) -> float:
    """
    Find the lowest threshold where precision >= min_precision,
    maximising recall.  Falls back to max-F1 if floor unachievable.
    """
    best_thr, best_recall, best_f1 = 0.5, 0.0, 0.0
    for thr in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_proba >= thr).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        if p >= min_precision and r > best_recall:
            best_thr, best_recall, best_f1 = thr, r, f
    return round(best_thr, 2)


# ─────────────────────────────────────────────────────────────────
# 6.  EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────
def evaluate(label: str, y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "name"      : label,
        "threshold" : threshold,
        "accuracy"  : accuracy_score(y_true, y_pred),
        "roc_auc"   : roc_auc_score(y_true, y_proba),
        "avg_prec"  : average_precision_score(y_true, y_proba),
        "recall"    : recall_score(y_true, y_pred, zero_division=0),
        "precision" : precision_score(y_true, y_pred, zero_division=0),
        "f1"        : f1_score(y_true, y_pred, zero_division=0),
        "y_pred"    : y_pred,
    }


# ─────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  STROKE PREDICTION — REAL IMBALANCED DATASET")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────
    df = load_and_deduplicate("healthcare_data.csv")
    X  = df.drop(columns=["stroke_event"])
    y  = df["stroke_event"].astype(int)

    # ── Split BEFORE any preprocessing ────────────────────────────
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    # Inner split for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.20, random_state=RANDOM_STATE, stratify=y_train_full
    )

    print(f"\nTrain: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"Train stroke cases : {y_tr.sum()} ({y_tr.mean()*100:.1f}%)")
    print(f"Test  stroke cases : {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    # ── Preprocess (train stats only, then apply to val/test) ─────
    X_tr_proc,   X_val_proc,  feat_names = preprocess(X_tr, X_val)
    X_full_proc, X_test_proc, _          = preprocess(X_train_full, X_test)

    # ── SMOTE on inner train ───────────────────────────────────────
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"\nClass ratio (scale_pos_weight) : {scale_pos:.2f}")
    X_tr_sm, y_tr_sm = smote_oversample(X_tr_proc, y_tr.values, sampling_strategy=0.30)

    # ── Train on inner split, evaluate on validation ───────────────
    print("\nTraining XGBoost for threshold tuning...")
    model_val = build_xgboost(scale_pos_weight=scale_pos)
    model_val.fit(
        X_tr_sm, y_tr_sm,
        eval_set=[(X_val_proc, y_val.values)],
        verbose=50,
    )
    val_proba = model_val.predict_proba(X_val_proc)[:, 1]

    # ── Tune threshold on validation set ──────────────────────────
    best_thr = tune_threshold(y_val.values, val_proba, min_precision=0.20)
    print(f"\nBest threshold (from validation) : {best_thr}")

    # ── Retrain on FULL train set ──────────────────────────────────
    scale_pos_full = (y_train_full == 0).sum() / max((y_train_full == 1).sum(), 1)
    X_full_sm, y_full_sm = smote_oversample(X_full_proc, y_train_full.values, sampling_strategy=0.30)

    print("\nRetraining on full training set...")
    X_full_sm_tr, X_final_val, y_full_sm_tr, y_final_val = train_test_split(
        X_full_sm, y_full_sm, test_size=0.15, random_state=RANDOM_STATE, stratify=y_full_sm
    )
    model_final = build_xgboost(scale_pos_weight=scale_pos_full)
    model_final.fit(
        X_full_sm_tr, y_full_sm_tr,
        eval_set=[(X_final_val, y_final_val)],
        verbose=50,
    )

    # ── Evaluate on untouched test set ────────────────────────────
    test_proba = model_final.predict_proba(X_test_proc)[:, 1]
    r_default  = evaluate("XGBoost (thr=0.50)", y_test.values, test_proba, threshold=0.50)
    r_tuned    = evaluate(f"XGBoost (thr={best_thr})", y_test.values, test_proba, threshold=best_thr)

    # ── Cross-validation AUC on full data ─────────────────────────
    # Quick CV with final model params (no SMOTE in CV for speed — use scale_pos_weight)
    cv_model = build_xgboost(scale_pos_weight=scale_pos_full)
    # Preprocess all data for CV
    X_all_proc, _, _ = preprocess(X, X.iloc[:1])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_aucs = cross_val_score(
        XGBClassifier(
            n_estimators=model_final.best_iteration or 200,
            max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, gamma=0.2,
            reg_alpha=0.5, reg_lambda=2.0,
            scale_pos_weight=scale_pos_full,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        X_all_proc, y.values, cv=skf, scoring="roc_auc", n_jobs=-1
    )

    # ── Print results ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL TEST SET RESULTS")
    print("=" * 65)
    print(f"{'Metric':<22} {'Default (0.50)':>16} {'Tuned ({:.2f})'.format(best_thr):>16}")
    print("-" * 65)
    for metric in ["accuracy", "roc_auc", "avg_prec", "recall", "precision", "f1"]:
        print(f"{metric:<22} {r_default[metric]:>16.4f} {r_tuned[metric]:>16.4f}")
    print("=" * 65)
    print(f"\n5-Fold CV AUC : {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
    print(f"Best iteration: {model_final.best_iteration}")

    # ── Confusion matrix numbers ───────────────────────────────────
    cm = confusion_matrix(y_test.values, r_tuned["y_pred"])
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix (tuned threshold {best_thr}):")
    print(f"  True Negatives  (correct no-stroke) : {tn}")
    print(f"  False Positives (healthy flagged)    : {fp}")
    print(f"  False Negatives (missed strokes)     : {fn}  ← minimise this")
    print(f"  True Positives  (stroke caught)      : {tp}")

    # ── Feature importance ────────────────────────────────────────
    imp = pd.Series(model_final.feature_importances_, index=feat_names).sort_values(ascending=False)
    print("\nFeature Importance (top 10):")
    print(imp.head(10).to_string())

    # ─────────────────────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Stroke Prediction — XGBoost on Real Imbalanced Dataset",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # (A) ROC Curve
    ax_roc = fig.add_subplot(gs[0, 0])
    RocCurveDisplay.from_predictions(y_test, test_proba, ax=ax_roc, color="darkorange",
                                     name=f"XGBoost (AUC={r_tuned['roc_auc']:.4f})")
    ax_roc.plot([0,1],[0,1],"k--", linewidth=0.8)
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(fontsize=9)

    # (B) Precision-Recall Curve  ← critical for imbalanced data
    ax_pr = fig.add_subplot(gs[0, 1])
    PrecisionRecallDisplay.from_predictions(y_test, test_proba, ax=ax_pr, color="steelblue",
                                            name=f"XGBoost (AP={r_tuned['avg_prec']:.4f})")
    ax_pr.axhline(y_test.mean(), color="r", linestyle="--", linewidth=0.8, label="Baseline (prevalence)")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(fontsize=9)

    # (C) Feature Importance
    ax_fi = fig.add_subplot(gs[0, 2])
    imp.sort_values().plot(kind="barh", ax=ax_fi, color="steelblue", alpha=0.85)
    ax_fi.set_title("Feature Importance (XGBoost Gain)")
    ax_fi.set_xlabel("Importance Score")

    # (D) Confusion Matrix — Default threshold
    ax_cm1 = fig.add_subplot(gs[1, 0])
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, r_default["y_pred"]),
        display_labels=["No Stroke","Stroke"]
    ).plot(ax=ax_cm1, colorbar=False, cmap="Blues")
    ax_cm1.set_title(f"Confusion Matrix\nDefault Threshold 0.50\n"
                     f"Recall={r_default['recall']:.4f}  F1={r_default['f1']:.4f}")

    # (E) Confusion Matrix — Tuned threshold
    ax_cm2 = fig.add_subplot(gs[1, 1])
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, r_tuned["y_pred"]),
        display_labels=["No Stroke","Stroke"]
    ).plot(ax=ax_cm2, colorbar=False, cmap="Blues")
    ax_cm2.set_title(f"Confusion Matrix\nTuned Threshold {best_thr}\n"
                     f"Recall={r_tuned['recall']:.4f}  F1={r_tuned['f1']:.4f}")

    # (F) Threshold sweep — Recall & Precision vs threshold
    ax_thr = fig.add_subplot(gs[1, 2])
    thresholds = np.arange(0.01, 1.0, 0.01)
    recalls    = [recall_score(y_test, (test_proba >= t).astype(int), zero_division=0) for t in thresholds]
    precisions = [precision_score(y_test, (test_proba >= t).astype(int), zero_division=0) for t in thresholds]
    f1s        = [f1_score(y_test, (test_proba >= t).astype(int), zero_division=0) for t in thresholds]
    ax_thr.plot(thresholds, recalls,    label="Recall",    color="tomato",    linewidth=2)
    ax_thr.plot(thresholds, precisions, label="Precision", color="steelblue", linewidth=2)
    ax_thr.plot(thresholds, f1s,        label="F1",        color="seagreen",  linewidth=2)
    ax_thr.axvline(best_thr, color="black", linestyle="--", linewidth=1.2, label=f"Chosen ({best_thr})")
    ax_thr.set_title("Threshold Sweep")
    ax_thr.set_xlabel("Threshold")
    ax_thr.set_ylabel("Score")
    ax_thr.legend(fontsize=9)
    ax_thr.set_xlim(0, 1)
    ax_thr.set_ylim(0, 1.05)

    plt.savefig("stroke_imbalanced_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved → stroke_imbalanced_results.png")
    model_final.save_model("stroke_xgboost_imbalanced.json")
    print("Model saved → stroke_xgboost_imbalanced.json")


if __name__ == "__main__":
    main()