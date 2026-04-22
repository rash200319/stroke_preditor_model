import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def run_optimized_workflow(file_path):
    print("--- 1. Data Setup ---")
    df = pd.read_csv(file_path, index_col=0)
    leakage_features = ['risk_score', 'high_glucose', 'bmi_category', 'age_group', 'patient_id']
    X = df.drop(columns=[col for col in leakage_features if col in df.columns] + ['stroke_event'])
    y = df['stroke_event']
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # --- 2. Hyperparameter Tuning Definitions ---
    print("\n--- 2. Comprehensive Hyperparameter Tuning (Optimizing for RECALL) ---")
    
    model_params = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=2000, solver='liblinear'),
            'grid': {
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__class_weight': ['balanced', None]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'grid': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [5, 10, None],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__class_weight': ['balanced', 'balanced_subsample']
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'grid': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 5, 7],
                'classifier__subsample': [0.8, 1.0],
                'classifier__scale_pos_weight': [(y_train==0).sum()/(y_train==1).sum(), 1.0] # High weight vs baseline
            }
        }
    }

    best_models = {}
    tuning_report = []

    for name, config in model_params.items():
        print(f"Tuning {name}...")
        pipe = Pipeline([('prep', preprocessor), ('classifier', config['model'])])
        
        # Optimize primarily for recall
        search = GridSearchCV(pipe, config['grid'], cv=cv, scoring='recall', n_jobs=-1)
        search.fit(X_train, y_train)
        
        best_models[name] = search.best_estimator_
        
        # Collect insights
        tuning_report.append({
            'Model': name,
            'Best Recall (CV)': search.best_score_,
            'Best Params': search.best_params_
        })

    # --- 3. Evaluate Tuned Models on Test Set ---
    print("\n--- 3. Performance Comparison After Tuning (Test Set) ---")
    
    test_metrics = []
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        test_metrics.append({
            'Model': name,
            'Recall (Stroke)': recall_score(y_test, y_pred),
            'Precision (Stroke)': precision_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'False Positive Rate (%)': (fp / (fp + tn)) * 100
        })

    comparison_df = pd.DataFrame(test_metrics)
    print(comparison_df)

    # --- 4. Two-Stage System Re-Evaluation ---
    print("\n--- 4. Re-Evaluating Two-Stage System (Tuned LR -> Tuned XGB) ---")
    lr_tuned = best_models['Logistic Regression']
    xgb_tuned = best_models['XGBoost']
    
    s1_preds = lr_tuned.predict(X_test)
    final_preds = np.copy(s1_preds)
    flagged = np.where(s1_preds == 1)[0]
    if len(flagged) > 0:
        X_flagged = X_test.iloc[flagged]
        # Ensure we use the correct model for predicted class
        final_preds[flagged] = xgb_tuned.predict(X_flagged)
    
    ts_metrics = pd.DataFrame([{
        'Model': 'Two-Stage (LR -> XGB)',
        'Recall (Stroke)': recall_score(y_test, final_preds),
        'Precision (Stroke)': precision_score(y_test, final_preds),
        'F1-Score': f1_score(y_test, final_preds),
        'ROC-AUC': np.nan,
        'False Positive Rate (%)': (confusion_matrix(y_test, final_preds)[0,1] / len(y_test[y_test==0])) * 100
    }])
    comparison_df = pd.concat([comparison_df, ts_metrics], ignore_index=True)
    print(comparison_df.iloc[-1:])

    # Save details
    comparison_df.to_csv('optimized_model_comparison.csv', index=False)
    with open('best_hyperparameters.txt', 'w') as f:
        for entry in tuning_report:
            f.write(f"{entry['Model']}:\n{entry['Best Params']}\n\n")

if __name__ == "__main__":
    run_optimized_workflow('cleaned_healthcare_data.csv')
