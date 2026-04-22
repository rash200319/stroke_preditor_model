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
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, classification_report)
import joblib
import os

def run_ml_workflow(file_path):
    print("--- 1. Data Splitting & Preparation ---")
    df = pd.read_csv(file_path, index_col=0)
    
    leakage_features = ['risk_score', 'high_glucose', 'bmi_category', 'age_group', 'patient_id']
    df_clean = df.drop(columns=[col for col in leakage_features if col in df.columns])
    
    X = df_clean.drop('stroke_event', axis=1)
    y = df_clean['stroke_event']
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Calculate scale_pos_weight for XGBoost to handle imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    spw = num_neg / num_pos
    print(f"Computed scale_pos_weight for XGBoost: {spw:.2f}")

    # 2. Model Development & Tuning
    print("\n--- 2. Model Development & Hyperparameter Tuning ---")
    
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
            'params': {
                'classifier__C': [0.1, 1.0, 10.0]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(scale_pos_weight=spw, random_state=42, eval_metric='logloss'),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5, 7]
            }
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_pipelines = {}
    test_results = []

    plt.figure(figsize=(10, 8)) # ROC comparison

    for name, config in models_config.items():
        print(f"Tuning {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])
        
        search = GridSearchCV(pipeline, config['params'], cv=cv, scoring='f1', n_jobs=-1)
        search.fit(X_train, y_train)
        
        best_pipe = search.best_estimator_
        best_pipelines[name] = best_pipe
        
        y_pred = best_pipe.predict(X_test)
        y_prob = best_pipe.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Model': name,
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'Accuracy': accuracy_score(y_test, y_pred)
        }
        test_results.append(metrics)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics["ROC-AUC"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison - LR vs RF vs XGB')
    plt.legend()
    plt.savefig('all_models_roc_comparison.png')
    plt.close()

    comparison_df = pd.DataFrame(test_results)
    print("\nFinal Model Comparison on Test Set:")
    print(comparison_df)
    
    # Save visualizations and stats as before
    df_melted = comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=df_melted)
    plt.title('Comparison of Model Metrics (LR, RF, XGB)')
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('all_models_metrics_comparison.png')
    plt.close()

    for name, pipe in best_pipelines.items():
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(f'cm_{name.lower().replace(" ", "_")}.png')
        plt.close()
        
        # Feature Importance
        if hasattr(pipe.named_steps['classifier'], 'feature_importances_'):
            importances = pipe.named_steps['classifier'].feature_importances_
        else:
            importances = np.abs(pipe.named_steps['classifier'].coef_[0])
        
        o_cat = pipe.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
        f_names = numerical_cols + list(o_cat)
        feat_imp = pd.Series(importances, index=f_names).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        feat_imp.head(10).plot(kind='barh', color='darkblue')
        plt.title(f'Top 10 Risk Factors - {name}')
        plt.gca().invert_yaxis()
        plt.savefig(f'importance_{name.lower().replace(" ", "_")}.png')
        plt.close()

    comparison_df.to_csv('all_models_comparison.csv', index=False)

if __name__ == "__main__":
    run_ml_workflow('cleaned_healthcare_data.csv')
