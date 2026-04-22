import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def calculate_error_rates(file_path):
    # Load data and re-train/get models (same logic as before)
    df = pd.read_csv(file_path, index_col=0)
    leakage_features = ['risk_score', 'high_glucose', 'bmi_category', 'age_group', 'patient_id']
    df_clean = df.drop(columns=[col for col in leakage_features if col in df.columns])
    
    X = df_clean.drop('stroke_event', axis=1)
    y = df_clean['stroke_event']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # 1. Models
    lr = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))])
    rf = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))])
    num_neg = (y_train == 0).sum(); num_pos = (y_train == 1).sum(); spw = num_neg / num_pos
    xgb = Pipeline([('prep', preprocessor), ('clf', XGBClassifier(scale_pos_weight=spw, random_state=42, eval_metric='logloss'))])
    
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    
    # 2. Predictions
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_xgb = xgb.predict(X_test)
    
    # Two-Stage Logic (LR -> XGB)
    y_pred_ts = np.copy(y_pred_lr)
    flagged = np.where(y_pred_lr == 1)[0]
    if len(flagged) > 0:
        y_pred_ts[flagged] = xgb.predict(X_test.iloc[flagged])
        
    models = {
        'Logistic Regression': y_pred_lr,
        'Random Forest': y_pred_rf,
        'XGBoost': y_pred_xgb,
        'Two-Stage (LR -> XGB)': y_pred_ts
    }
    
    error_data = []
    
    for name, preds in models.items():
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        
        # False Positive Rate (FPR): % of healthy people flagged incorrectly
        fpr = (fp / (fp + tn)) * 100
        
        # False Negative Rate (FNR): % of stroke patients missed
        fnr = (fn / (fn + tp)) * 100
        
        error_data.append({
            'Model': name,
            'False Positives (Count)': fp,
            'False Negatives (Count)': fn,
            'False Positive Rate (%)': f"{fpr:.2f}%",
            'False Negative Rate (%)': f"{fnr:.2f}%",
            'Safety Rating': 'High' if fnr < 25 else 'Moderate' if fnr < 50 else 'Low'
        })
        
    error_df = pd.DataFrame(error_data)
    print("\n--- Detailed Error Analysis ---")
    print(error_df)
    error_df.to_csv('detailed_errors.csv', index=False)

if __name__ == "__main__":
    calculate_error_rates('cleaned_healthcare_data.csv')
