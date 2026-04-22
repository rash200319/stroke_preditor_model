import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
import joblib

def evaluate_two_stage_strategy(file_path):
    # 1. Load data and models
    # Note: We'll re-run training briefly to get the tuned models if not saved, 
    # but for simplicity, let's assume they might be in the environment. 
    # Actually, it's safer to re-train or use the logic from train_models.py
    
    # We will use the logic from our previous run
    # I'll re-implement the training part quickly to get the exact models
    from train_models import run_ml_workflow
    
    # To avoid repeating the whole tuning, let's assume we have tuned them.
    # For a robust script, I'll re-import the core logic and fit them with best known params or just fit them.
    
    df = pd.read_csv(file_path, index_col=0)
    leakage_features = ['risk_score', 'high_glucose', 'bmi_category', 'age_group', 'patient_id']
    df_clean = df.drop(columns=[col for col in leakage_features if col in df.columns])
    
    X = df_clean.drop('stroke_event', axis=1)
    y = df_clean['stroke_event']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Re-build pipelines with best parameters found in previous step (approximate or just standard tuned)
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Model 1: Logistic Regression (Screening Specialist)
    lr_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000))
    ])
    
    # Model 2: XGBoost (The Refiner)
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    spw = num_neg / num_pos
    xgb_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(scale_pos_weight=spw, n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42, eval_metric='logloss'))
    ])
    
    print("Training Stage 1 (LR) and Stage 2 (XGB)...")
    lr_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    
    # --- TWO STAGE PREDICTION LOGIC ---
    print("\n--- Applying Two-Stage Prediction Strategy ---")
    
    # Step 1: Logistic Regression Screening
    lr_preds = lr_model.predict(X_test)
    
    # Step 2: Refine only the positive predictions from Stage 1
    final_preds = np.copy(lr_preds)
    
    # Indices where LR predicted a stroke (Class 1)
    flagged_indices = np.where(lr_preds == 1)[0]
    print(f"Total test cases: {len(X_test)}")
    print(f"Cases flagged by Stage 1 (LR): {len(flagged_indices)}")
    
    # Run XGBoost on these flagged cases
    if len(flagged_indices) > 0:
        X_flagged = X_test.iloc[flagged_indices]
        xgb_refine_preds = xgb_model.predict(X_flagged)
        
        # Update the final predictions with the refined XGBoost results
        # final_preds[flagged_indices] will stay 1 only if XGB also says 1
        final_preds[flagged_indices] = xgb_refine_preds
    
    # --- EVALUATION ---
    def get_metrics(y_true, y_pred):
        return {
            'Recall': recall_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'Accuracy': accuracy_score(y_true, y_pred)
        }

    m_lr = get_metrics(y_test, lr_preds)
    m_xgb = get_metrics(y_test, xgb_model.predict(X_test))
    m_combined = get_metrics(y_test, final_preds)
    
    results = pd.DataFrame([m_lr, m_xgb, m_combined], index=['Stage 1 only (LR)', 'XGBoost only', 'Two-Stage Strategy (LR -> XGB)'])
    print("\nComparison Table:")
    print(results)
    
    # Confusion Matrix for Two-Stage
    cm_combined = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix: Two-Stage Strategy (LR -> XGB)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('cm_two_stage.png')
    plt.close()
    
    # Save results for markdown report
    results.to_csv('two_stage_results.csv')

if __name__ == "__main__":
    evaluate_two_stage_strategy('cleaned_healthcare_data.csv')
