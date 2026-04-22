from pathlib import Path
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
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
CLASSIFICATION_THRESHOLD = 0.15
F2_THRESHOLD = 0.15
SCALE_POS_WEIGHT = 19.5
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)
CALIBRATION_METHOD = "sigmoid"  # Platt scaling
HARD_MINING_OVERSAMPLE_FACTOR = 1
HARD_MINING_MAX_ADDED_RATIO = 0.10
HARD_MINING_FP_CONFIDENCE = 0.80
HARD_MINING_FN_CONFIDENCE = 0.20
OVERFIT_GAP_ROC_AUC_WARNING = 0.03
OVERFIT_GAP_PR_AUC_WARNING = 0.05
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
                    n_estimators=200,
                    max_depth=4,
                    min_samples_split=40,
                    min_samples_leaf=20,
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
                    n_estimators=120,
                    max_depth=2,
                    learning_rate=0.03,
                    subsample=0.7,
                    colsample_bytree=0.6,
                    min_child_weight=20,
                    gamma=1.0,
                    reg_alpha=0.5,
                    reg_lambda=5.0,
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
    X_train_full: pd.DataFrame,
    X_train_fit: pd.DataFrame,
    X_calibration: pd.DataFrame,
    X_hard_mining: pd.DataFrame,
    X_threshold: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_full: pd.Series,
    y_train_fit: pd.Series,
    y_calibration: pd.Series,
    y_hard_mining: pd.Series,
    y_threshold: pd.Series,
    y_test: pd.Series,
    cv_strategy: StratifiedKFold,
) -> dict[str, float | np.ndarray | Pipeline]:
    """Fit one pipeline, evaluate on the holdout set, and run stratified CV on training data only."""

    threshold_model = clone(pipeline)
    threshold_model.fit(X_train_fit, y_train_fit)

    inference_model: Pipeline | CalibratedClassifierCV = threshold_model
    calibrated = False

    # Calibrate XGBoost probabilities after fitting, then tune threshold on calibrated probabilities.
    if name == "XGBoost":
        inference_model = CalibratedClassifierCV(
            estimator=threshold_model,
            method=CALIBRATION_METHOD,
            cv="prefit",
        )
        inference_model.fit(X_calibration, y_calibration)
        calibrated = True

    threshold_proba = inference_model.predict_proba(X_threshold)[:, 1]
    tuned_threshold = CLASSIFICATION_THRESHOLD

    threshold_pred = (threshold_proba >= tuned_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_threshold, threshold_pred).ravel()
    threshold_summary = {
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "objective": float((fp / (fp + tn)) + (fn / (fn + tp))) if (fp + tn) > 0 and (fn + tp) > 0 else 0.0,
    }

    hard_mining_proba = inference_model.predict_proba(X_hard_mining)[:, 1]
    hard_mask, hard_fp_count, hard_fn_count = select_hard_examples(
        y_hard_mining,
        hard_mining_proba,
        tuned_threshold,
    )

    X_train_final = X_train_full.copy()
    y_train_final = y_train_full.copy()
    hard_mined_count = int(np.sum(hard_mask))

    if hard_mined_count > 0:
        X_hard_cases = X_hard_mining.loc[hard_mask]
        y_hard_cases = y_hard_mining.loc[hard_mask]

        max_added_rows = max(1, int(len(X_train_full) * HARD_MINING_MAX_ADDED_RATIO))
        max_selected_hard_cases = max(1, max_added_rows // max(1, HARD_MINING_OVERSAMPLE_FACTOR))

        if len(X_hard_cases) > max_selected_hard_cases:
            y_hard_array = y_hard_cases.to_numpy()
            hard_confidence = np.where(y_hard_array == 0, hard_mining_proba[hard_mask], 1.0 - hard_mining_proba[hard_mask])
            keep_idx = np.argsort(-hard_confidence)[:max_selected_hard_cases]
            X_hard_cases = X_hard_cases.iloc[keep_idx]
            y_hard_cases = y_hard_cases.iloc[keep_idx]
            hard_mined_count = len(X_hard_cases)

        X_train_final = pd.concat(
            [X_train_final] + [X_hard_cases] * HARD_MINING_OVERSAMPLE_FACTOR,
            axis=0,
            ignore_index=True,
        )
        y_train_final = pd.concat(
            [y_train_final] + [y_hard_cases] * HARD_MINING_OVERSAMPLE_FACTOR,
            axis=0,
            ignore_index=True,
        )

    if name == "XGBoost":
        # Final calibrated XGBoost model after hard-negative mining retraining.
        full_model_train = clone(pipeline)
        full_model_train.fit(X_train_final, y_train_final)
        inference_model = CalibratedClassifierCV(
            estimator=full_model_train,
            method=CALIBRATION_METHOD,
            cv="prefit",
        )
        inference_model.fit(X_calibration, y_calibration)
    else:
        pipeline.fit(X_train_final, y_train_final)
        inference_model = pipeline

    y_proba = inference_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= tuned_threshold).astype(int)
    y_pred_custom = (y_proba >= F2_THRESHOLD).astype(int)
    f2 = fbeta_score(y_test, y_pred_custom, beta=2, zero_division=0)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(pr_recall, pr_precision)

    cv_scores = cross_validate(
        pipeline,
        X_train_full,
        y_train_full,
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
        return_train_score=True,
    )

    cv_train_roc_auc_mean = float(np.mean(cv_scores["train_roc_auc"]))
    cv_train_pr_auc_mean = float(np.mean(cv_scores["train_pr_auc"]))
    cv_roc_auc_mean = float(np.mean(cv_scores["test_roc_auc"]))
    cv_pr_auc_mean = float(np.mean(cv_scores["test_pr_auc"]))

    overfit_flag, roc_auc_gap, pr_auc_gap = detect_overfitting(
        cv_train_roc_auc_mean,
        cv_roc_auc_mean,
        cv_train_pr_auc_mean,
        cv_pr_auc_mean,
    )

    return {
        "name": name,
        "model": inference_model,
        "calibrated": calibrated,
        "threshold": tuned_threshold,
        "hard_mined_count": hard_mined_count,
        "hard_fp_count": hard_fp_count,
        "hard_fn_count": hard_fn_count,
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
        "cv_roc_auc_mean": cv_roc_auc_mean,
        "cv_roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
        "cv_pr_auc_mean": cv_pr_auc_mean,
        "cv_pr_auc_std": float(np.std(cv_scores["test_pr_auc"])),
        "cv_train_roc_auc_mean": cv_train_roc_auc_mean,
        "cv_train_pr_auc_mean": cv_train_pr_auc_mean,
        "overfit_flag": overfit_flag,
        "roc_auc_gap": roc_auc_gap,
        "pr_auc_gap": pr_auc_gap,
        "threshold_false_positive_rate": threshold_summary["false_positive_rate"],
        "threshold_false_negative_rate": threshold_summary["false_negative_rate"],
        "threshold_objective": threshold_summary["objective"],
    }


def tune_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: np.ndarray = THRESHOLD_GRID,
) -> tuple[float, dict[str, float]]:
    """Find the threshold that minimizes the sum of false positive and false negative rates."""

    best_threshold = 0.5
    best_objective = float("inf")
    best_summary = {
        "false_positive_rate": 1.0,
        "false_negative_rate": 1.0,
        "objective": float("inf"),
    }

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        objective = false_positive_rate + false_negative_rate

        if objective < best_objective:
            best_objective = objective
            best_threshold = float(threshold)
            best_summary = {
                "false_positive_rate": float(false_positive_rate),
                "false_negative_rate": float(false_negative_rate),
                "objective": float(objective),
            }

    return best_threshold, best_summary


def select_hard_examples(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int, int]:
    """Select confidence-gated hard errors to reduce noise amplification."""

    y_array = y_true.to_numpy()
    y_pred = (y_proba >= threshold).astype(int)

    hard_fp_mask = (y_pred == 1) & (y_array == 0) & (y_proba >= HARD_MINING_FP_CONFIDENCE)
    hard_fn_mask = (y_pred == 0) & (y_array == 1) & (y_proba <= HARD_MINING_FN_CONFIDENCE)
    hard_mask = hard_fp_mask | hard_fn_mask

    return hard_mask, int(np.sum(hard_fp_mask)), int(np.sum(hard_fn_mask))


def assert_disjoint_split_indices(split_map: dict[str, pd.DataFrame | pd.Series]) -> None:
    """Ensure evaluation/tuning/mining/calibration splits are disjoint to avoid leakage."""

    items = list(split_map.items())
    for i in range(len(items)):
        left_name, left_data = items[i]
        left_idx = set(left_data.index)
        for j in range(i + 1, len(items)):
            right_name, right_data = items[j]
            overlap = left_idx.intersection(set(right_data.index))
            if overlap:
                raise ValueError(
                    f"Data leakage risk: splits '{left_name}' and '{right_name}' overlap by {len(overlap)} rows."
                )


def detect_overfitting(
    cv_train_roc_auc_mean: float,
    cv_roc_auc_mean: float,
    cv_train_pr_auc_mean: float,
    cv_pr_auc_mean: float,
) -> tuple[bool, float, float]:
    """Detect likely overfitting using train-vs-validation cross-validation gaps."""

    roc_auc_gap = cv_train_roc_auc_mean - cv_roc_auc_mean
    pr_auc_gap = cv_train_pr_auc_mean - cv_pr_auc_mean
    overfit_flag = (roc_auc_gap > OVERFIT_GAP_ROC_AUC_WARNING) or (pr_auc_gap > OVERFIT_GAP_PR_AUC_WARNING)
    return overfit_flag, roc_auc_gap, pr_auc_gap


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
                "Threshold": result["threshold"],
                "Calibrated": result["calibrated"],
                "Hard Mined": result["hard_mined_count"],
                "Hard FP": result["hard_fp_count"],
                "Hard FN": result["hard_fn_count"],
                "Val FP Rate": result["threshold_false_positive_rate"],
                "Val FN Rate": result["threshold_false_negative_rate"],
                "CV Train ROC-AUC": result["cv_train_roc_auc_mean"],
                "CV Train PR-AUC": result["cv_train_pr_auc_mean"],
                "ROC Gap": result["roc_auc_gap"],
                "PR Gap": result["pr_auc_gap"],
                "Overfit Risk": result["overfit_flag"],
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
        f"({best_result['roc_auc']:.4f}) at threshold {best_result['threshold']:.2f}"
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

    n_models = len(results)
    n_cols = max(3, n_models)

    fig = plt.figure(figsize=(6 * n_cols, 12))
    fig.suptitle("Model Comparison - Stroke Prediction", fontsize=16, fontweight="bold", y=1.01)
    grid = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.4, wspace=0.35)

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

    X_train_fit, X_threshold, y_train_fit, y_threshold = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    X_calibration, X_tune_pool, y_calibration, y_tune_pool = train_test_split(
        X_threshold,
        y_threshold,
        test_size=2 / 3,
        random_state=RANDOM_STATE,
        stratify=y_threshold,
    )

    X_hard_mining, X_threshold, y_hard_mining, y_threshold = train_test_split(
        X_tune_pool,
        y_tune_pool,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_tune_pool,
    )

    assert_disjoint_split_indices(
        {
            "train_fit": X_train_fit,
            "calibration": X_calibration,
            "hard_mining": X_hard_mining,
            "threshold_tuning": X_threshold,
            "test": X_test,
        }
    )

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_pipelines = build_models(numerical_features, categorical_features)

    print("Building the Clinical Voting Ensemble...")

    lr_pipeline = model_pipelines["Logistic Regression"]
    rf_pipeline = model_pipelines["Random Forest"]
    xgb_pipeline = model_pipelines["XGBoost"]

    ensemble_pipeline = VotingClassifier(
        estimators=[
            ("LR", lr_pipeline),
            ("RF", rf_pipeline),
            ("XGB", xgb_pipeline),
        ],
        voting="soft",
        weights=[2, 1, 1],
    )

    model_pipelines["Clinical Ensemble"] = ensemble_pipeline

    print(
        "Data loaded. Training leakage-safe pipelines on the stratified train split...\n"
        "XGBoost uses Platt scaling calibration (sigmoid) before fixed-threshold evaluation.\n"
        f"Using a fixed classification threshold for all models: {CLASSIFICATION_THRESHOLD:.2f}.\n"
        "Hard negative mining: only high-confidence wrong cases are used, with mild capped oversampling.\n"
    )

    results = []
    for name, pipeline in model_pipelines.items():
        print(f"Training {name}...")
        result = evaluate_model(
            name,
            pipeline,
            X_train,
            X_train_fit,
            X_calibration,
            X_hard_mining,
            X_threshold,
            X_test,
            y_train,
            y_train_fit,
            y_calibration,
            y_hard_mining,
            y_threshold,
            y_test,
            cv_strategy,
        )
        results.append(result)
        print(
            f"  ROC-AUC={result['roc_auc']:.4f}  PR-AUC={result['pr_auc']:.4f}  "
            f"Recall={result['recall']:.4f}  F1={result['f1']:.4f}  "
            f"F2@{F2_THRESHOLD:.2f}={result['f2']:.4f}  "
            f"thr={result['threshold']:.2f}  hard_mined={result['hard_mined_count']} "
            f"(FP={result['hard_fp_count']}, FN={result['hard_fn_count']})  "
            f"gaps(ROC={result['roc_auc_gap']:.4f}, PR={result['pr_auc_gap']:.4f})\n"
        )

        if result["overfit_flag"]:
            print(
                "  [warning] Potential overfitting detected from train-vs-CV gaps. "
                "Consider stronger regularization or reducing model complexity.\n"
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