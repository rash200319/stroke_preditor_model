"""
stroke_ensemble.py
══════════════════════════════════════════════════════════════════
Stroke prediction on the REAL imbalanced dataset (4.87% stroke).

Models compared:
  1. Logistic Regression   (interpretable baseline)
  2. Random Forest         (bagging ensemble)
  3. XGBoost               (boosting — primary model)
  4. Soft Voting Ensemble  (avg probabilities of all 3)
  5. Stacking Ensemble     (meta-learner on top of base models)

Pipeline per model:
  • Train-test split BEFORE any preprocessing
  • Grouped-median BMI imputation (fit on train only)
  • Label encoding (fit on train only)
  • SMOTE oversampling on training fold only
  • Threshold tuned on held-out validation set
  • Final evaluation on untouched test set
══════════════════════════════════════════════════════════════════
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, precision_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, average_precision_score
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
CAT_COLS     = ["gender", "employment_type", "residence", "smoking_habit"]


# ══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & DEDUPLICATE
# ══════════════════════════════════════════════════════════════════
def load_data(path="healthcare_data.csv"):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    before = len(df)
    dedup_cols = [c for c in df.columns if c != "patient_id"]
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    print(f"Rows  : {before} → {len(df)}  ({before - len(df)} duplicates removed)")

    DROP = ["patient_id", "age_group", "bmi_category",
            "high_glucose", "risk_score", "lifestyle_risk"]
    df = df.drop(columns=[c for c in DROP if c in df.columns])

    n_stroke = df["stroke_event"].sum()
    print(f"Stroke: {n_stroke} / {len(df)}  ({n_stroke/len(df)*100:.2f}%)")
    return df


# ══════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESSING (no leakage — fit on train, apply to test)
# ══════════════════════════════════════════════════════════════════
def preprocess(X_train, X_test):
    X_tr = X_train.copy()
    X_te = X_test.copy()

    # Grouped-median BMI imputation
    X_tr["_age_bin"] = pd.cut(X_tr["age"], bins=[0,40,60,200],
                               labels=["young","middle","senior"])
    X_te["_age_bin"] = pd.cut(X_te["age"], bins=[0,40,60,200],
                               labels=["young","middle","senior"])
    bmi_med = (X_tr.groupby(["gender","_age_bin"], observed=True)["bmi_value"]
               .median())
    global_med = X_tr["bmi_value"].median()

    def fill_bmi(row):
        if pd.isna(row["bmi_value"]):
            try:   return bmi_med.loc[(row["gender"], row["_age_bin"])]
            except: return global_med
        return row["bmi_value"]

    X_tr["bmi_value"] = X_tr.apply(fill_bmi, axis=1)
    X_te["bmi_value"] = X_te.apply(fill_bmi, axis=1)
    X_tr = X_tr.drop(columns=["_age_bin"])
    X_te = X_te.drop(columns=["_age_bin"])

    # Label encoding — fit on train only
    for col in CAT_COLS:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        X_te[col] = X_te[col].astype(str).map(
            lambda v, le=le: le.transform([v])[0]
            if v in le.classes_ else -1
        )

    feat_names = X_tr.columns.tolist()
    return X_tr.values.astype(float), X_te.values.astype(float), feat_names


# ══════════════════════════════════════════════════════════════════
# STEP 3 — SMOTE (minority oversampling — applied inside train only)
# ══════════════════════════════════════════════════════════════════
def smote(X, y, sampling_strategy=0.35, k=5, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    min_cls = classes[np.argmin(counts)]
    maj_cls = classes[np.argmax(counts)]
    X_min, X_maj = X[y == min_cls], X[y == maj_cls]
    y_min, y_maj = y[y == min_cls], y[y == maj_cls]

    target = int(np.ceil(sampling_strategy * len(X_maj)))
    n_gen  = max(0, target - len(X_min))
    if n_gen == 0 or len(X_min) < 2:
        return X, y

    eff_k    = min(k, len(X_min) - 1)
    nn       = NearestNeighbors(n_neighbors=eff_k + 1).fit(X_min)
    neighbors = nn.kneighbors(X_min, return_distance=False)[:, 1:]

    synthetic = []
    for _ in range(n_gen):
        i   = rng.integers(len(X_min))
        j   = rng.choice(neighbors[i])
        gap = rng.random()
        synthetic.append(X_min[i] + gap * (X_min[j] - X_min[i]))

    X_syn = np.array(synthetic)
    y_syn = np.full(len(X_syn), min_cls, dtype=y.dtype)
    X_out = np.vstack([X_maj, X_min, X_syn])
    y_out = np.concatenate([y_maj, y_min, y_syn])
    return X_out, y_out


# ══════════════════════════════════════════════════════════════════
# STEP 4 — MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════
def make_logistic(spw):
    # spw not used directly — class_weight='balanced' handles it
    return LogisticRegression(
        C=0.5, max_iter=2000, solver="lbfgs",
        class_weight="balanced", random_state=RANDOM_STATE
    )

def make_random_forest(spw):
    return RandomForestClassifier(
        n_estimators=400, max_depth=10,
        min_samples_split=5, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )

def make_xgboost(spw):
    return XGBClassifier(
        n_estimators=500, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5,
        gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
        scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=RANDOM_STATE, n_jobs=-1
    )


# ══════════════════════════════════════════════════════════════════
# STEP 5 — THRESHOLD TUNING
# ══════════════════════════════════════════════════════════════════
def tune_threshold(y_true, y_proba, min_precision=0.08):
    """
    Maximise recall subject to precision >= min_precision.
    min_precision=0.08 means: for every 12 patients flagged,
    at least 1 must be a real stroke — acceptable for screening.
    """
    best_thr, best_recall = 0.5, 0.0
    for thr in np.arange(0.01, 0.99, 0.01):
        y_pred = (y_proba >= thr).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        if p >= min_precision and r > best_recall:
            best_thr, best_recall = round(thr, 2), r
    return best_thr


# ══════════════════════════════════════════════════════════════════
# STEP 6 — STACKING META-FEATURES (out-of-fold — no leakage)
# ══════════════════════════════════════════════════════════════════
def build_oof_meta_features(base_models_fn, X_tr, y_tr, spw,
                             X_te, n_splits=5):
    """
    Generate out-of-fold (OOF) predictions for stacking.
    Each fold trains on K-1 folds and predicts on the held-out fold.
    This prevents the meta-learner from seeing training targets.
    """
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True,
                            random_state=RANDOM_STATE)
    n_models = len(base_models_fn)
    oof_train = np.zeros((len(X_tr), n_models))
    oof_test  = np.zeros((len(X_te), n_models))

    for m_idx, (name, fn) in enumerate(base_models_fn.items()):
        test_preds = np.zeros((len(X_te), n_splits))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            Xf_tr, Xf_val = X_tr[tr_idx], X_tr[val_idx]
            yf_tr, yf_val = y_tr[tr_idx], y_tr[val_idx]

            # SMOTE inside fold only
            Xf_sm, yf_sm = smote(Xf_tr, yf_tr)

            model = fn(spw)
            if name == "XGBoost":
                model.fit(Xf_sm, yf_sm,
                          eval_set=[(Xf_val, yf_val)], verbose=False)
            else:
                model.fit(Xf_sm, yf_sm)

            oof_train[val_idx, m_idx] = model.predict_proba(Xf_val)[:, 1]
            test_preds[:, fold] = model.predict_proba(X_te)[:, 1]

        oof_test[:, m_idx] = test_preds.mean(axis=1)
        print(f"  OOF done: {name}")

    return oof_train, oof_test


# ══════════════════════════════════════════════════════════════════
# STEP 7 — EVALUATE
# ══════════════════════════════════════════════════════════════════
def evaluate(name, y_true, y_proba, thr):
    y_pred = (y_proba >= thr).astype(int)
    return {
        "name"      : name,
        "threshold" : thr,
        "accuracy"  : accuracy_score(y_true, y_pred),
        "roc_auc"   : roc_auc_score(y_true, y_proba),
        "avg_prec"  : average_precision_score(y_true, y_proba),
        "recall"    : recall_score(y_true, y_pred, zero_division=0),
        "precision" : precision_score(y_true, y_pred, zero_division=0),
        "f1"        : f1_score(y_true, y_pred, zero_division=0),
        "y_pred"    : y_pred,
        "y_proba"   : y_proba,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  STROKE ENSEMBLE — REAL IMBALANCED DATASET")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────
    df = load_data("healthcare_data.csv")
    X  = df.drop(columns=["stroke_event"])
    y  = df["stroke_event"].astype(int)

    # ── Stratified split — BEFORE any preprocessing ───────────────
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    # Validation split for threshold tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.20, random_state=RANDOM_STATE, stratify=y_train_full
    )

    spw = (y_train_full == 0).sum() / max((y_train_full == 1).sum(), 1)
    print(f"\nTrain : {len(X_tr)}  Val : {len(X_val)}  Test : {len(X_test)}")
    print(f"Train stroke cases : {y_tr.sum()} ({y_tr.mean()*100:.1f}%)")
    print(f"Test  stroke cases : {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    print(f"scale_pos_weight   : {spw:.2f}")

    # ── Preprocess ────────────────────────────────────────────────
    # For validation threshold tuning
    X_tr_p,   X_val_p,  feat_names = preprocess(X_tr, X_val)
    # For final test evaluation
    X_full_p, X_test_p, _          = preprocess(X_train_full, X_test)

    BASE_FNS = {
        "Logistic Regression": make_logistic,
        "Random Forest"      : make_random_forest,
        "XGBoost"            : make_xgboost,
    }

    # ══════════════════════════════════════════════════════════════
    # PHASE A — Train base models on inner split, tune thresholds
    #           on validation set
    # ══════════════════════════════════════════════════════════════
    print("\n── Phase A: Training base models for threshold tuning ──")
    val_probas = {}
    X_tr_sm, y_tr_sm = smote(X_tr_p, y_tr.values)

    for name, fn in BASE_FNS.items():
        model = fn(spw)
        if name == "XGBoost":
            model.fit(X_tr_sm, y_tr_sm,
                      eval_set=[(X_val_p, y_val.values)], verbose=False)
        else:
            model.fit(X_tr_sm, y_tr_sm)
        val_probas[name] = model.predict_proba(X_val_p)[:, 1]
        print(f"  Trained {name}")

    # Soft voting on validation
    val_probas["Soft Voting"] = np.mean(
        [val_probas[n] for n in BASE_FNS], axis=0
    )
    # Stacking on validation (quick: use base proba as meta-features)
    meta_X_val = np.column_stack([val_probas[n] for n in BASE_FNS])
    # Need OOF on inner train for stacking meta-train
    print("\n── Building OOF meta-features for stacking ──")
    oof_tr, oof_val = build_oof_meta_features(
        BASE_FNS, X_tr_p, y_tr.values, spw, X_val_p
    )
    meta_lr = LogisticRegression(C=1.0, class_weight="balanced",
                                  max_iter=1000, random_state=RANDOM_STATE)
    meta_lr.fit(oof_tr, y_tr.values)
    val_probas["Stacking"] = meta_lr.predict_proba(oof_val)[:, 1]

    # ── Tune thresholds on validation ─────────────────────────────
    print("\n── Tuning thresholds on validation set ──")
    thresholds = {}
    all_names  = list(BASE_FNS.keys()) + ["Soft Voting", "Stacking"]
    for name in all_names:
        thr = tune_threshold(y_val.values, val_probas[name], min_precision=0.08)
        thresholds[name] = thr
        r   = evaluate(name, y_val.values, val_probas[name], thr)
        print(f"  {name:<22} thr={thr:.2f}  "
              f"Recall={r['recall']:.3f}  Precision={r['precision']:.3f}  "
              f"F1={r['f1']:.3f}  AUC={r['roc_auc']:.3f}")

    # ══════════════════════════════════════════════════════════════
    # PHASE B — Retrain on FULL training data, evaluate on test set
    # ══════════════════════════════════════════════════════════════
    print("\n── Phase B: Retraining on full train, evaluating on test ──")
    spw_full  = (y_train_full == 0).sum() / max((y_train_full == 1).sum(), 1)
    X_full_sm, y_full_sm = smote(X_full_p, y_train_full.values)

    # Train base models on full train
    final_models  = {}
    test_probas   = {}

    for name, fn in BASE_FNS.items():
        model = fn(spw_full)
        if name == "XGBoost":
            # Need a small eval set for early stopping
            Xf_tr2, Xf_es, yf_tr2, yf_es = train_test_split(
                X_full_sm, y_full_sm,
                test_size=0.12, random_state=RANDOM_STATE, stratify=y_full_sm
            )
            model.fit(Xf_tr2, yf_tr2,
                      eval_set=[(Xf_es, yf_es)], verbose=50)
        else:
            model.fit(X_full_sm, y_full_sm)
        final_models[name]  = model
        test_probas[name]   = model.predict_proba(X_test_p)[:, 1]
        print(f"  Trained {name}")

    # Soft voting on test
    test_probas["Soft Voting"] = np.mean(
        [test_probas[n] for n in BASE_FNS], axis=0
    )

    # Stacking on test — OOF meta-features on full train
    print("\n── Building OOF meta-features for stacking (full train) ──")
    oof_full_tr, oof_test = build_oof_meta_features(
        BASE_FNS, X_full_p, y_train_full.values, spw_full, X_test_p
    )
    meta_lr_full = LogisticRegression(C=1.0, class_weight="balanced",
                                       max_iter=1000, random_state=RANDOM_STATE)
    meta_lr_full.fit(oof_full_tr, y_train_full.values)
    test_probas["Stacking"] = meta_lr_full.predict_proba(oof_test)[:, 1]

    # ── Final evaluation ──────────────────────────────────────────
    results = {}
    for name in all_names:
        results[name] = evaluate(name, y_test.values,
                                  test_probas[name], thresholds[name])

    # ── Print comparison table ────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL TEST SET RESULTS  (threshold tuned on validation set)")
    print("=" * 80)
    hdr = f"{'Model':<24} {'Thr':>5} {'AUC-ROC':>8} {'AUC-PR':>8} {'Recall':>8} {'Precision':>10} {'F1':>8}"
    print(hdr)
    print("-" * 80)
    for name in all_names:
        r = results[name]
        flag = " ★" if name == "Stacking" else ""
        print(f"{name:<24} {r['threshold']:>5.2f} {r['roc_auc']:>8.4f} "
              f"{r['avg_prec']:>8.4f} {r['recall']:>8.4f} "
              f"{r['precision']:>10.4f} {r['f1']:>8.4f}{flag}")
    print("=" * 80)

    # Confusion matrix breakdown for best model (Stacking)
    best_name = max(results, key=lambda k: results[k]["recall"])
    r_best    = results[best_name]
    cm        = confusion_matrix(y_test.values, r_best["y_pred"])
    tn, fp, fn_count, tp = cm.ravel()
    print(f"\nBest recall model → {best_name} (thr={r_best['threshold']:.2f})")
    print(f"  True Negatives  (healthy correctly cleared) : {tn}")
    print(f"  False Positives (healthy flagged for review) : {fp}")
    print(f"  False Negatives (stroke cases MISSED) ⚠️    : {fn_count}")
    print(f"  True Positives  (stroke cases caught) ✓     : {tp}")

    # ── Feature importance (XGBoost) ─────────────────────────────
    xgb_model = final_models["XGBoost"]
    imp = pd.Series(xgb_model.feature_importances_,
                    index=feat_names).sort_values(ascending=False)
    print("\nXGBoost Feature Importance:")
    print(imp.to_string())

    # ══════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════
    colors = {
        "Logistic Regression": "#4C72B0",
        "Random Forest"      : "#55A868",
        "XGBoost"            : "#DD8452",
        "Soft Voting"        : "#8172B3",
        "Stacking"           : "#C44E52",
    }
    styles = {
        "Logistic Regression": "-",
        "Random Forest"      : "-",
        "XGBoost"            : "-",
        "Soft Voting"        : "--",
        "Stacking"           : "-.",
    }

    # ── Figure 1: ROC + PR + Feature Importance ──────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig1.suptitle("Model Comparison — Stroke Prediction (Imbalanced Dataset)",
                  fontsize=13, fontweight="bold")

    # ROC curves
    ax = axes[0]
    for name in all_names:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test.values, test_probas[name])
        auc = results[name]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=colors[name], linestyle=styles[name], linewidth=2)
    ax.plot([0,1],[0,1],"k--", linewidth=0.8)
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.2)

    # Precision-Recall curves
    ax = axes[1]
    from sklearn.metrics import precision_recall_curve
    for name in all_names:
        prec, rec, _ = precision_recall_curve(y_test.values, test_probas[name])
        ap = results[name]["avg_prec"]
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                color=colors[name], linestyle=styles[name], linewidth=2)
    ax.axhline(y_test.mean(), color="gray", linestyle=":", linewidth=1,
               label=f"Baseline ({y_test.mean()*100:.1f}%)")
    ax.set_title("Precision-Recall Curves\n(More informative for imbalanced data)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.2)

    # Feature importance
    ax = axes[2]
    imp.sort_values().plot(kind="barh", ax=ax, color="steelblue", alpha=0.85)
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig("stroke_ensemble_roc_pr_fi.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure 1 saved → stroke_ensemble_roc_pr_fi.png")

    # ── Figure 2: Confusion Matrices (all 5 models) ───────────────
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
    fig2.suptitle("Confusion Matrices — Tuned Thresholds (Test Set)",
                  fontsize=13, fontweight="bold")
    axes_flat = axes2.flatten()

    for i, name in enumerate(all_names):
        r   = results[name]
        cm  = confusion_matrix(y_test.values, r["y_pred"])
        tn2, fp2, fn2, tp2 = cm.ravel()
        ConfusionMatrixDisplay(cm, display_labels=["No Stroke","Stroke"]).plot(
            ax=axes_flat[i], colorbar=False, cmap="Blues"
        )
        axes_flat[i].set_title(
            f"{name}\nThr={r['threshold']:.2f} | "
            f"Recall={r['recall']:.3f} | F1={r['f1']:.3f}\n"
            f"Missed strokes (FN)={fn2} | False alarms (FP)={fp2}"
        )

    # Hide unused subplot
    axes_flat[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig("stroke_ensemble_confusion.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure 2 saved → stroke_ensemble_confusion.png")

    # ── Figure 3: Threshold sweep for Stacking model ──────────────
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    thrs = np.arange(0.01, 0.99, 0.01)
    stk_proba = test_probas["Stacking"]
    recs  = [recall_score(y_test, (stk_proba >= t).astype(int), zero_division=0) for t in thrs]
    precs = [precision_score(y_test, (stk_proba >= t).astype(int), zero_division=0) for t in thrs]
    f1s   = [f1_score(y_test, (stk_proba >= t).astype(int), zero_division=0) for t in thrs]
    ax3.plot(thrs, recs,  label="Recall",    color="tomato",    linewidth=2.2)
    ax3.plot(thrs, precs, label="Precision", color="steelblue", linewidth=2.2)
    ax3.plot(thrs, f1s,   label="F1",        color="seagreen",  linewidth=2.2)
    ax3.axvline(thresholds["Stacking"], color="black", linestyle="--",
                linewidth=1.5, label=f"Chosen thr={thresholds['Stacking']:.2f}")
    ax3.axhline(0.08, color="orange", linestyle=":", linewidth=1,
                label="Min precision floor (0.08)")
    ax3.set_title("Stacking Ensemble — Threshold Sweep on Test Set")
    ax3.set_xlabel("Threshold")
    ax3.set_ylabel("Score")
    ax3.legend()
    ax3.grid(alpha=0.2)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("stroke_ensemble_threshold_sweep.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure 3 saved → stroke_ensemble_threshold_sweep.png")


if __name__ == "__main__":
    main()
