from pathlib import Path
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
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
TARGET_COLUMN = "stroke_event"
LEAKAGE_COLUMNS = [
    "risk_score",
    "lifestyle_risk",
    "patient_id",
    "age_group",
    "bmi_category",
    "high_glucose",
]
DATA_PATH = Path(__file__).with_name("healthcare_data_cleaned.csv")
OUTPUT_PLOT_PATH = Path(__file__).with_name("model_comparison_results.png")


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

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numerical_features, categorical_features, use_scaler=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    tree_preprocessor = build_preprocessor(numerical_features, categorical_features, use_scaler=False)

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgboost_pipeline = Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
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

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

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
        "roc_auc": roc_auc_score(y_test, y_proba),
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
    }


def print_results(results: list[dict[str, float | np.ndarray | Pipeline]]) -> None:
    """Print a concise comparison table for the holdout and cross-validation metrics."""

    summary = pd.DataFrame(
        [
            {
                "Model": result["name"],
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1-score": result["f1"],
                "ROC-AUC": result["roc_auc"],
                "CV ROC-AUC": f"{result['cv_roc_auc_mean']:.4f} ± {result['cv_roc_auc_std']:.4f}",
            }
            for result in results
        ]
    )

    print("\nModel comparison on the 20% holdout test set")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best_result = max(results, key=lambda item: item["roc_auc"])
    print(
        f"\nBest model by holdout ROC-AUC: {best_result['name']} "
        f"({best_result['roc_auc']:.4f})"
    )


def plot_results(
    results: list[dict[str, float | np.ndarray | Pipeline]],
    y_test: pd.Series,
) -> None:
    """Create ROC curves, metric comparison bars, and confusion matrices."""

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

    ax_bar = fig.add_subplot(grid[0, 1])
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
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
    ax_bar.set_title("Holdout Metric Comparison")
    ax_bar.set_ylabel("Score")
    ax_bar.legend(fontsize=8)

    ax_cv = fig.add_subplot(grid[0, 2])
    cv_means = [result["cv_roc_auc_mean"] for result in results]
    cv_stds = [result["cv_roc_auc_std"] for result in results]
    bars = ax_cv.bar(
        model_names,
        cv_means,
        yerr=cv_stds,
        capsize=6,
        color=colors,
        alpha=0.85,
        error_kw={"linewidth": 2},
    )
    ax_cv.set_ylim(0.0, 1.05)
    ax_cv.set_title("5-Fold CV ROC-AUC (Mean ± Std)")
    ax_cv.set_ylabel("ROC-AUC")
    for bar, mean in zip(bars, cv_means):
        ax_cv.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax_cv.tick_params(axis="x", labelsize=9)

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

    print("Data loaded. Training leakage-safe pipelines on the stratified train split...\n")

    results = []
    for name, pipeline in model_pipelines.items():
        print(f"Training {name}...")
        result = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test, cv_strategy)
        results.append(result)
        print(
            f"  ROC-AUC={result['roc_auc']:.4f}  "
            f"Recall={result['recall']:.4f}  F1={result['f1']:.4f}\n"
        )

    print_results(results)
    plot_results(results, y_test)


if __name__ == "__main__":
    main()