from pathlib import Path
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    fbeta_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.1
F2_THRESHOLD = 0.12
SCALE_POS_WEIGHT = 19.5
TARGET_COLUMN = "stroke_event"
LEAKAGE_COLUMNS = [
    "risk_score",
    "lifestyle_risk",
    "patient_id",
    "id",
    "age_group",
    "bmi_category",
    "high_glucose",
]
DATA_PATH = Path(__file__).with_name("healthcare_data_cleaned.csv")
OUTPUT_PLOT_PATH = Path(__file__).with_name("model_comparison_results.png")


class InteractionFeatureAdder(BaseEstimator, TransformerMixin):
    """Create clinically meaningful interaction features inside the pipeline."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "InteractionFeatureAdder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        if {"age", "has_hypertension"}.issubset(X_out.columns):
            X_out["age_x_has_hypertension"] = X_out["age"] * X_out["has_hypertension"]

        if {"bmi_value", "glucose_level"}.issubset(X_out.columns):
            X_out["bmi_x_glucose_level"] = X_out["bmi_value"] * X_out["glucose_level"]

        return X_out


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a compatible OneHotEncoder across scikit-learn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility fallback
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data() -> pd.DataFrame:
    """Load the stroke dataset and drop columns that would leak target information."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    drop_candidates = [column for column in ["Unnamed: 0", *LEAKAGE_COLUMNS] if column in df.columns]
    if drop_candidates:
        df = df.drop(columns=drop_candidates)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")

    return df


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the predictors from the target column."""

    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y


def identify_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical feature columns from the training data schema."""

    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_features = [column for column in X.columns if column not in categorical_features]
    return numerical_features, categorical_features


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
    use_scaler: bool,
) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes and encodes inside the pipeline."""

    transformers = []

    if numerical_features:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if use_scaler:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numerical_features))

    if categorical_features:
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
        transformers.append(("categorical", Pipeline(categorical_steps), categorical_features))

    if not transformers:
        raise ValueError("No usable feature columns were found after dropping leakage columns.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models(
    numerical_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    """Create the model pipelines used for comparison."""

    engineered_numerical_features = numerical_features.copy()
    if {"age", "has_hypertension"}.issubset(numerical_features):
        engineered_numerical_features.append("age_x_has_hypertension")
    if {"bmi_value", "glucose_level"}.issubset(numerical_features):
        engineered_numerical_features.append("bmi_x_glucose_level")

    logistic_pipeline = Pipeline(
        steps=[
            ("feature_engineering", InteractionFeatureAdder()),
            (
                "preprocessor",
                build_preprocessor(engineered_numerical_features, categorical_features, use_scaler=True),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            ("feature_engineering", InteractionFeatureAdder()),
            (
                "preprocessor",
                build_preprocessor(engineered_numerical_features, categorical_features, use_scaler=False),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgboost_pipeline = Pipeline(
        steps=[
            ("feature_engineering", InteractionFeatureAdder()),
            (
                "preprocessor",
                build_preprocessor(engineered_numerical_features, categorical_features, use_scaler=False),
            ),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    scale_pos_weight=SCALE_POS_WEIGHT,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic_pipeline,
        "Random Forest": random_forest_pipeline,
        "XGBoost": xgboost_pipeline,
    }


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_strategy: StratifiedKFold,
) -> dict[str, float | np.ndarray | Pipeline]:
    """Fit one pipeline, evaluate on the holdout set, and run stratified CV on training data only."""

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= CLASSIFICATION_THRESHOLD).astype(int)
    y_pred_custom = (y_proba >= F2_THRESHOLD).astype(int)
    f2 = fbeta_score(y_test, y_pred_custom, beta=2, zero_division=0)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(pr_recall, pr_precision)

    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv_strategy,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        },
        n_jobs=-1,
        return_train_score=False,
    )

    return {
        "name": name,
        "model": pipeline,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "f2": f2,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": pr_auc,
        "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
        "cv_precision_mean": float(np.mean(cv_scores["test_precision"])),
        "cv_precision_std": float(np.std(cv_scores["test_precision"])),
        "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
        "cv_recall_std": float(np.std(cv_scores["test_recall"])),
        "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
        "cv_f1_std": float(np.std(cv_scores["test_f1"])),
        "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
        "cv_pr_auc_mean": float(np.mean(cv_scores["test_pr_auc"])),
        "cv_pr_auc_std": float(np.std(cv_scores["test_pr_auc"])),
    }


def print_results(results: list[dict[str, float | np.ndarray | Pipeline]]) -> None:
    """Print a concise comparison table for the holdout and cross-validation metrics."""

    summary = pd.DataFrame(
        [
            {
                "Model": result["name"],
                "Accuracy": result["accuracy"],
                "ROC-AUC": result["roc_auc"],
                "PR-AUC": result["pr_auc"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1-score": result["f1"],
                f"F2-score @ {F2_THRESHOLD:.2f}": result["f2"],
                "CV ROC-AUC": f"{result['cv_roc_auc_mean']:.4f} ± {result['cv_roc_auc_std']:.4f}",
                "CV PR-AUC": f"{result['cv_pr_auc_mean']:.4f} ± {result['cv_pr_auc_std']:.4f}",
            }
            for result in results
        ]
    )

    print("\nModel comparison on the 20% holdout test set")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best_result = max(results, key=lambda item: item["roc_auc"])
    print(
        f"\nBest model by holdout ROC-AUC: {best_result['name']} "
        f"({best_result['roc_auc']:.4f}) at threshold {CLASSIFICATION_THRESHOLD:.2f}"
    )


def extract_odds_ratios(logistic_pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """Extract odds ratios from a fitted Logistic Regression pipeline."""

    classifier = logistic_pipeline.named_steps.get("classifier")
    preprocessor = logistic_pipeline.named_steps.get("preprocessor")

    if not isinstance(classifier, LogisticRegression):
        raise TypeError("extract_odds_ratios requires a pipeline with LogisticRegression as classifier.")

    if preprocessor is None or not hasattr(preprocessor, "transform"):
        raise ValueError("The provided pipeline does not include a fitted preprocessor step.")

    if not hasattr(classifier, "coef_"):
        raise ValueError("The Logistic Regression model must be fitted before extracting odds ratios.")

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = np.array([f"feature_{i}" for i in range(classifier.coef_.shape[1])])

    coefficients = classifier.coef_.ravel()
    odds_ratios = np.exp(coefficients)

    odds_df = pd.DataFrame(
        {
            "feature": feature_names,
            "log_odds": coefficients,
            "odds_ratio": odds_ratios,
            "effect_strength": np.abs(coefficients),
        }
    )

    odds_df = odds_df.sort_values("effect_strength", ascending=False).head(top_n)
    return odds_df.drop(columns=["effect_strength"])


def plot_results(
    results: list[dict[str, float | np.ndarray | Pipeline]],
    y_test: pd.Series,
) -> None:
    """Create ROC/PR curves, metric comparison bars, and confusion matrices."""

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Model Comparison - Stroke Prediction", fontsize=16, fontweight="bold", y=1.01)
    grid = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = ["#4C72B0", "#55A868", "#DD8452"]
    model_names = [result["name"] for result in results]

    ax_roc = fig.add_subplot(grid[0, 0])
    for result, color in zip(results, colors):
        RocCurveDisplay.from_predictions(
            y_test,
            result["y_proba"],
            ax=ax_roc,
            name=f"{result['name']} (AUC={result['roc_auc']:.4f})",
            color=color,
        )
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax_roc.set_title("ROC Curves - All Models")
    ax_roc.legend(fontsize=8)

    ax_pr = fig.add_subplot(grid[0, 1])
    baseline_pr = float(y_test.mean())
    for result, color in zip(results, colors):
        PrecisionRecallDisplay.from_predictions(
            y_test,
            result["y_proba"],
            ax=ax_pr,
            name=f"{result['name']} (AUC={result['pr_auc']:.4f})",
            color=color,
        )
    ax_pr.axhline(y=baseline_pr, color="k", linestyle="--", linewidth=0.8, label="Baseline")
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.legend(fontsize=8)

    ax_bar = fig.add_subplot(grid[0, 2])
    metric_keys = ["roc_auc", "pr_auc", "recall", "f2"]
    metric_labels = ["ROC-AUC", "PR-AUC", "Recall", f"F2@{F2_THRESHOLD:.2f}"]
    x_positions = np.arange(len(metric_keys))
    bar_width = 0.25

    for index, (result, color) in enumerate(zip(results, colors)):
        values = [result[key] for key in metric_keys]
        ax_bar.bar(
            x_positions + index * bar_width,
            values,
            bar_width,
            label=result["name"],
            color=color,
            alpha=0.85,
        )
    ax_bar.set_xticks(x_positions + bar_width)
    ax_bar.set_xticklabels(metric_labels)
    ax_bar.set_ylim(0.0, 1.05)
    ax_bar.set_title("Imbalance-Aware Metric Comparison")
    ax_bar.set_ylabel("Score")
    ax_bar.legend(fontsize=8)

    for index, result in enumerate(results):
        ax_cm = fig.add_subplot(grid[1, index])
        cm = confusion_matrix(y_test, result["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["No Stroke", "Stroke"]).plot(
            ax=ax_cm,
            colorbar=False,
            cmap="Blues",
        )
        ax_cm.set_title(
            f"{result['name']}\nAcc={result['accuracy']:.4f} | AUC={result['roc_auc']:.4f}"
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved as '{OUTPUT_PLOT_PATH.name}'")


def main() -> None:
    """Run the leakage-safe model comparison workflow."""

    df = load_data()
    X, y = split_features_and_target(df)
    numerical_features, categorical_features = identify_feature_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_pipelines = build_models(numerical_features, categorical_features)

    print(
        "Data loaded. Training leakage-safe pipelines on the stratified train split...\n"
        f"Using classification threshold: {CLASSIFICATION_THRESHOLD:.2f}\n"
    )

    results = []
    for name, pipeline in model_pipelines.items():
        print(f"Training {name}...")
        result = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test, cv_strategy)
        results.append(result)
        print(
            f"  ROC-AUC={result['roc_auc']:.4f}  PR-AUC={result['pr_auc']:.4f}  "
            f"Recall={result['recall']:.4f}  F1={result['f1']:.4f}  "
            f"F2@{F2_THRESHOLD:.2f}={result['f2']:.4f}\n"
        )

    print_results(results)
    plot_results(results, y_test)

    logistic_result = next((result for result in results if result["name"] == "Logistic Regression"), None)
    if logistic_result is not None:
        odds_df = extract_odds_ratios(logistic_result["model"], top_n=15)
        print("\nTop Logistic Regression odds ratios (by |log-odds|):")
        print(odds_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()