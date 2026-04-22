# Healthcare Stroke Prediction: Clinically Optimized ML System

## 🏥 Overview
This project addresses the critical healthcare challenge of predicting stroke events using patient demographic and health data. Unlike generic machine learning models, this system is **clinically optimized** to prioritize patient safety by maximizing the detection of stroke cases (Recall) while managing the "alarm fatigue" caused by false positives.

## 🚀 Key Features
- **Sophisticated Preprocessing**: Automated handling of missing BMI values using age-group mean imputation.
- **Leakage Prevention**: Rigorous removal of derived or risk-scoring features before training.
- **Calibrated Probabilities**: Implemented Isotonic Regression to ensure predicted risk percentages are medically reliable.
- **Two-Stage Screening**: A specialized architecture using Logistic Regression as a high-recall screener followed by XGBoost for refinement.
- **Stacking Ensemble**: Combined the strengths of linear and tree-based models to achieve the highest possible ROC-AUC (0.842).

## 📊 Final Performance Results
Through exhaustive hyperparameter tuning and optimization, the system achieved the following benchmarks:

| Metric | Performance |
| :--- | :--- |
| **Stroke Detection (Recall)** | **84%** |
| **False Negative Rate (Missed Cases)** | **16%** |
| **ROC-AUC** | **0.842** |
| **Accuracy** | **~75-81%** |

## 📂 Project Structure
- `preprocess_data.py`: Data cleaning, imputation, and validation.
- `train_models.py`: Initial model development and baseline evaluation.
- `clinical_prediction_system.py`: Implementation of calibrated probabilities and stacking ensembles.
- `optimized_training_workflow.py`: Comprehensive hyperparameter tuning using GridSearchCV.
- `analysis_assets/`: Visualizations of data distributions and risk factors.

## 🛠️ Installation & Usage
1. **Clone the repository**:
   ```bash
   git clone <your-repo-link>
   cd dataxplore1
   ```
2. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```
3. **Run the clinical prediction pipeline**:
   ```bash
   python clinical_prediction_system.py
   ```

## 🧠 Clinical Insights
- **Top Risk Factors**: Age, Glucose Level, and Hypertension emerged as the most significant biological markers across all models.
- **Safety First**: We specifically optimized for **Recall** because missing a stroke case (False Negative) has a far higher human cost than a false alarm (False Positive).
- **Deployment Recommendation**: The **Tuned Two-Stage Pipeline** is recommended for deployment as it offers the best balance of safety and clinical efficiency.

---
*Created as part of the DataXplore Healthcare ML Initiative.*
