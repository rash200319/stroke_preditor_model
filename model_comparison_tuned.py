from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from model_comparison import (
    InteractionFeatureAdder,
    RANDOM_STATE,
    SCALE_POS_WEIGHT as DEFAULT_SCALE_POS_WEIGHT,
    build_preprocessor,
    build_stacking_features,
    evaluate_predictions,
    identify_feature_types,
    load_data,
    rule_based_hybrid_probabilities,
    soft_voting_probabilities,
    split_features_and_target,
)


THRESHOLD_GRID = np.round(np.arange(0.0, 1.01, 0.01), 2)
MIN_PRECISION_FOR_RECALL_OPTIMIZATION = 0.20
THRESHOLD_SELECTION_STRATEGY = "recall_at_precision_floor"

OUTPUT_PLOT_PATH = Path(__file__).with_name("model_comparison_tuned_results.png")
THRESHOLD_TUNING_PLOT_PATH = Path(__file__).with_name("threshold_tuning_results.png")

MODEL_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Soft Voting Ensemble",
    "Stacking Ensemble",
    "Rule-Based Hybrid",
]


def build_weighted_models(
    numerical_features: list[str],
    categorical_features: list[str],
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    """Build class-weighted base models using the current training split."""

    engineered_numerical_features = numerical_features.copy()
    if {"age", "has_hypertension"}.issubset(numerical_features):
        engineered_numerical_features.append("age_x_has_hypertension")
    if {"bmi_value", "glucose_level"}.issubset(numerical_features):
        engineered_numerical_features.append("bmi_x_glucose_level")

    num_positive = float((y_train == 1).sum())
    num_negative = float((y_train == 0).sum())
    scale_pos_weight = num_negative / max(num_positive, 1.0)
    # Keep a sensible fallback if the split is unexpectedly degenerate.
    if not np.isfinite(scale_pos_weight) or scale_pos_weight <= 0:
        scale_pos_weight = DEFAULT_SCALE_POS_WEIGHT

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
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
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
                    scale_pos_weight=scale_pos_weight,
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


def make_threshold_table(y_true: pd.Series, y_proba: np.ndarray) -> pd.DataFrame:
    """Evaluate a probability vector across the full threshold grid."""

    roc_auc = roc_auc_score(y_true, y_proba)
    rows = []

    for threshold in THRESHOLD_GRID:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc,
            }
        )

    return pd.DataFrame(rows)


def choose_best_threshold(threshold_df: pd.DataFrame) -> tuple[float, str]:
    """Select the best threshold using the requested medical objective."""

    if THRESHOLD_SELECTION_STRATEGY == "maximize_f1":
        best_row = threshold_df.sort_values(
            by=["f1", "recall", "threshold"], ascending=[False, False, True]
        ).iloc[0]
        return float(best_row["threshold"]), "maximize_f1"

    eligible = threshold_df[threshold_df["precision"] >= MIN_PRECISION_FOR_RECALL_OPTIMIZATION]
    if eligible.empty:
        eligible = threshold_df
        selection_mode = "fallback_max_f1"
        best_row = eligible.sort_values(
            by=["f1", "recall", "threshold"], ascending=[False, False, True]
        ).iloc[0]
        return float(best_row["threshold"]), selection_mode

    best_row = eligible.sort_values(
        by=["recall", "f1", "threshold"], ascending=[False, False, True]
    ).iloc[0]
    return float(best_row["threshold"]), "maximize_recall_with_precision_floor"


def tune_model_thresholds(
    validation_targets: pd.Series,
    validation_probabilities: dict[str, np.ndarray],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Tune thresholds for every model and ensemble on the validation split only."""

    threshold_rows = []
    tuned_thresholds: dict[str, float] = {}

    for model_name in MODEL_ORDER:
        df = make_threshold_table(validation_targets, validation_probabilities[model_name])
        best_threshold, selection_mode = choose_best_threshold(df)
        best_row = df.loc[np.isclose(df["threshold"], best_threshold)].iloc[0]

        threshold_rows.append(
            {
                "Model": model_name.replace(" Ensemble", "").replace(" Hybrid", ""),
                "Selection": selection_mode,
                "Best Threshold": best_threshold,
                "Accuracy": best_row["accuracy"],
                "Precision": best_row["precision"],
                "Recall": best_row["recall"],
                "F1": best_row["f1"],
                "ROC-AUC": best_row["roc_auc"],
            }
        )
        tuned_thresholds[model_name] = best_threshold

    return tuned_thresholds, pd.DataFrame(threshold_rows)


def build_base_probability_map(
    pipelines: dict[str, object],
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Fit base models and return probability scores for the evaluation split."""

    probabilities = {}
    for model_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        pipeline = pipelines[model_name]
        pipeline.fit(X_fit, y_fit)
        probabilities[model_name] = pipeline.predict_proba(X_eval)[:, 1]
    return probabilities


def build_ensemble_probability_map(
    fitted_pipelines: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    cv_strategy: StratifiedKFold,
) -> dict[str, np.ndarray]:
    """Create ensemble probability scores without leaking the evaluation targets."""

    base_eval_probs = build_base_probability_map(fitted_pipelines, X_train, y_train, X_eval)
    soft_voting_proba = soft_voting_probabilities(base_eval_probs)

    meta_train, meta_eval = build_stacking_features(
        fitted_pipelines,
        X_train,
        y_train,
        cv_strategy,
        X_eval,
    )
    meta_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    meta_model.fit(meta_train, y_train)
    stacking_proba = meta_model.predict_proba(meta_eval)[:, 1]

    rule_based_proba = rule_based_hybrid_probabilities(base_eval_probs)

    return {
        "Logistic Regression": base_eval_probs["Logistic Regression"],
        "Random Forest": base_eval_probs["Random Forest"],
        "XGBoost": base_eval_probs["XGBoost"],
        "Soft Voting Ensemble": soft_voting_proba,
        "Stacking Ensemble": stacking_proba,
        "Rule-Based Hybrid": rule_based_proba,
    }


def evaluate_at_tuned_thresholds(
    y_true: pd.Series,
    probability_map: dict[str, np.ndarray],
    tuned_thresholds: dict[str, float],
) -> list[dict[str, float | np.ndarray]]:
    """Score each model using its tuned threshold."""

    results = []
    for model_name in MODEL_ORDER:
        threshold = tuned_thresholds[model_name]
        result = evaluate_predictions(model_name, y_true, probability_map[model_name], threshold)
        results.append(result)
    return results


def print_threshold_tuning_table(tuning_table: pd.DataFrame) -> None:
    """Print the validation-set tuning summary."""

    display_df = tuning_table.copy()
    display_df["Best Threshold"] = display_df["Best Threshold"].map(lambda value: f"{value:.2f}")
    print("\nThreshold tuning on validation split")
    print(display_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def print_final_comparison_table(results: list[dict[str, float | np.ndarray]]) -> None:
    """Print the final holdout comparison using tuned thresholds."""

    rows = []
    for result in results:
        rows.append(
            {
                "Model": result["name"].replace(" Ensemble", "").replace(" Hybrid", ""),
                "Threshold": f"{result['threshold']:.2f}",
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
                "ROC-AUC": result["roc_auc"],
            }
        )

    summary = pd.DataFrame(rows)
    print("\nFinal holdout comparison using tuned thresholds")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def plot_roc_curves(
    y_true: pd.Series,
    probability_map: dict[str, np.ndarray],
    tuned_thresholds: dict[str, float],
) -> None:
    """Plot ROC curves for all models and ensembles."""

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    style_map = {
        "Logistic Regression": {"color": "#4C72B0", "linestyle": "-", "linewidth": 2.2},
        "Random Forest": {"color": "#55A868", "linestyle": "-", "linewidth": 2.2},
        "XGBoost": {"color": "#DD8452", "linestyle": "-", "linewidth": 2.2},
        "Soft Voting Ensemble": {"color": "#8172B3", "linestyle": "--", "linewidth": 2.4},
        "Stacking Ensemble": {"color": "#64B5CD", "linestyle": "-.", "linewidth": 2.4},
        "Rule-Based Hybrid": {"color": "#C44E52", "linestyle": ":", "linewidth": 2.6},
    }

    for model_name in MODEL_ORDER:
        fpr, tpr, _ = roc_curve(y_true, probability_map[model_name])
        label = f"{model_name} (thr={tuned_thresholds[model_name]:.2f}, AUC={roc_auc_score(y_true, probability_map[model_name]):.4f})"
        ax.plot(fpr, tpr, label=label, **style_map[model_name])

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Chance")
    ax.set_title("ROC Curves - Tuned Base Models and Ensembles")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nROC figure saved as '{OUTPUT_PLOT_PATH.name}'")


def plot_confusion_matrices(
    y_true: pd.Series,
    results: list[dict[str, float | np.ndarray]],
) -> None:
    """Plot confusion matrices for the three ensemble outputs."""

    ensemble_names = ["Soft Voting Ensemble", "Stacking Ensemble", "Rule-Based Hybrid"]
    ensemble_results = [result for result in results if result["name"] in ensemble_names]

    fig = plt.figure(figsize=(18, 5))
    grid = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    for index, result in enumerate(ensemble_results):
        ax = fig.add_subplot(grid[0, index])
        cm = confusion_matrix(y_true, result["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["No Stroke", "Stroke"]).plot(
            ax=ax,
            colorbar=False,
            cmap="Blues",
        )
        ax.set_title(
            f"{result['name']}\nThr={result['threshold']:.2f} | Recall={result['recall']:.4f} | F1={result['f1']:.4f}"
        )

    plt.tight_layout()
    plt.savefig(THRESHOLD_TUNING_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nConfusion-matrix figure saved as '{THRESHOLD_TUNING_PLOT_PATH.name}'")


def main() -> None:
    """Run threshold tuning and improved evaluation without modifying the original script."""

    df = load_data()
    X, y = split_features_and_target(df)
    numerical_features, categorical_features = identify_feature_types(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("Training base models on the inner train split and tuning thresholds on the validation split...")
    validation_pipelines = build_weighted_models(numerical_features, categorical_features, y_train_fit)
    validation_probability_map = build_ensemble_probability_map(
        validation_pipelines,
        X_train_fit,
        y_train_fit,
        X_val,
        cv_strategy,
    )

    tuned_thresholds, tuning_table = tune_model_thresholds(y_val, validation_probability_map)
    print_threshold_tuning_table(tuning_table)

    print("\nRefitting on the full training data and evaluating on the untouched holdout test set...")
    final_pipelines = build_weighted_models(numerical_features, categorical_features, y_train_full)
    final_probability_map = build_ensemble_probability_map(
        final_pipelines,
        X_train_full,
        y_train_full,
        X_test,
        cv_strategy,
    )

    final_results = evaluate_at_tuned_thresholds(y_test, final_probability_map, tuned_thresholds)
    print_final_comparison_table(final_results)

    plot_roc_curves(y_test, final_probability_map, tuned_thresholds)
    plot_confusion_matrices(y_test, final_results)


if __name__ == "__main__":
    main()
