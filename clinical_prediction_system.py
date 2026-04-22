import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (recall_score, precision_score, f1_score, roc_auc_score, 
                             confusion_matrix, precision_recall_curve, roc_curve)

def design_clinical_system(file_path):
    print("--- 1. Data Preparation & Splitting ---")
    df = pd.read_csv(file_path, index_col=0)
    leakage_features = ['risk_score', 'high_glucose', 'bmi_category', 'age_group', 'patient_id']
    df_clean = df.drop(columns=[col for col in leakage_features if col in df.columns])
    
    X = df_clean.drop('stroke_event', axis=1)
    y = df_clean['stroke_event']
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # 1. Calibrated Base Models
    print("\n--- 2. Building Calibrated Base Models ---")
    # Logistic Regression (Balanced)
    lr = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(class_weight='balanced', C=0.1, random_state=42))])
    
    # Calibrate Random Forest and XGBoost (using Isotonic regression as they are non-linear)
    rf_base = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))])
    rf = CalibratedClassifierCV(rf_base, cv=5, method='isotonic')
    
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_base = Pipeline([('prep', preprocessor), ('clf', XGBClassifier(scale_pos_weight=spw, eval_metric='logloss', random_state=42))])
    xgb = CalibratedClassifierCV(xgb_base, cv=5, method='isotonic')

    # Fit all
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    # 2. Threshold Optimization Function
    def find_optimal_threshold(model, X, y, target_recall=0.85):
        probs = model.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, probs)
        # Find threshold that gives closest recall to target_recall without going too low
        idx = np.where(recalls >= target_recall)[0][-1] 
        return thresholds[idx]

    print("\n--- 3. Optimizing Thresholds (Targeting 85% Recall) ---")
    t_lr = find_optimal_threshold(lr, X_train, y_train)
    t_rf = find_optimal_threshold(rf, X_train, y_train)
    t_xgb = find_optimal_threshold(xgb, X_train, y_train)
    print(f"Optimal Thresholds: LR={t_lr:.2f}, RF={t_rf:.2f}, XGB={t_xgb:.2f}")

    # 3. Ensemble Strategies
    print("\n--- 4. Designing Ensemble Strategies ---")
    
    # A. Weighted Soft Voting
    # Weights prioritize LR for its naturally higher sensitivity in balanced mode
    voting_clf = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
        voting='soft', weights=[2, 1, 1]
    )
    voting_clf.fit(X_train, y_train)
    t_vote = find_optimal_threshold(voting_clf, X_train, y_train)

    # B. Stacking Ensemble
    stacking_clf = StackingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
        final_estimator=LogisticRegression(),
        cv=5
    )
    stacking_clf.fit(X_train, y_train)
    t_stack = find_optimal_threshold(stacking_clf, X_train, y_train)

    # 4. Evaluation Function
    def evaluate_system(name, model, threshold, X, y):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel()
        return {
            'System': name,
            'Recall': recall_score(y, preds),
            'Precision': precision_score(y, preds),
            'F1': f1_score(y, preds),
            'ROC-AUC': roc_auc_score(y, probs),
            'FNR (%)': (fn / (fn + tp)) * 100,
            'FPR (%)': (fp / (fp + tn)) * 100,
            'Threshold': threshold
        }

    systems = [
        ('Calibrated LR', lr, t_lr),
        ('Calibrated RF', rf, t_rf),
        ('Calibrated XGB', xgb, t_xgb),
        ('Weighted Voting', voting_clf, t_vote),
        ('Stacking Ensemble', stacking_clf, t_stack)
    ]

    # C. Two-Stage Pipeline (Custom Logic)
    # Stage 1: LR (Aggressive) -> Stage 2: XGB (Moderator)
    def predict_two_stage(X, model1, t1, model2, t2):
        p1 = model1.predict_proba(X)[:, 1]
        s1_preds = (p1 >= t1).astype(int)
        final_preds = np.zeros(len(X))
        pos_idx = np.where(s1_preds == 1)[0]
        if len(pos_idx) > 0:
            p2 = model2.predict_proba(X.iloc[pos_idx])[:, 1]
            s2_preds = (p2 >= t2).astype(int)
            final_preds[pos_idx] = s2_preds
        return final_preds

    # Evaluate all
    results = []
    for name, model, threshold in systems:
        results.append(evaluate_system(name, model, threshold, X_test, y_test))
    
    # Evaluate Two-Stage separately
    ts_preds = predict_two_stage(X_test, lr, t_lr, xgb, 0.4) # Slightly lower threshold for moderator
    cm_ts = confusion_matrix(y_test, ts_preds)
    tn, fp, fn, tp = cm_ts.ravel()
    results.append({
        'System': 'Two-Stage Pipeline (LR->XGB)',
        'Recall': recall_score(y_test, ts_preds),
        'Precision': precision_score(y_test, ts_preds),
        'F1': f1_score(y_test, ts_preds),
        'ROC-AUC': np.nan, # Hard to define for two-stage logic purely
        'FNR (%)': (fn / (fn + tp)) * 100,
        'FPR (%)': (fp / (fp + tn)) * 100,
        'Threshold': f"S1:{t_lr:.2f}, S2:0.40"
    })

    print("\n--- Final Performance Comparison ---")
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv('clinical_system_comparison.csv', index=False)

    # Generate Precision-Recall Curve for analysis
    plt.figure(figsize=(10, 7))
    for name, model, _ in systems:
        probs = model.predict_proba(X_test)[:, 1]
        p, r, _ = precision_recall_curve(y_test, probs)
        plt.plot(r, p, label=name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve: Clinical Systems Comparison')
    plt.legend()
    plt.savefig('precision_recall_comparison.png')
    plt.close()

if __name__ == "__main__":
    design_clinical_system('cleaned_healthcare_data.csv')
