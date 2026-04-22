from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors

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
SMOTE_SAMPLING_STRATEGY = 0.30
SMOTE_K_NEIGHBORS = 5
RANDOM_SEARCH_CV_FOLDS = 3
RANDOM_SEARCH_ITERATIONS = 12

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

BASE_MODEL_SEARCH_SPACES = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"],
        "classifier__class_weight": ["balanced"],
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__class_weight": ["balanced"],
    },
    "XGBoost": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
        "classifier__scale_pos_weight": [3, 5, 7, 10],
    },
}


class SimpleSMOTE:
    """A lightweight SMOTE implementation for dense tabular matrices."""

    def __init__(
        self,
        sampling_strategy: float = SMOTE_SAMPLING_STRATEGY,
        random_state: int = RANDOM_STATE,
        k_neighbors: int = SMOTE_K_NEIGHBORS,
    ) -> None:
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X: np.ndarray, y: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=int)

        classes, counts = np.unique(y_array, return_counts=True)
        if len(classes) != 2:
            return X_array, y_array

        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]

        X_minority = X_array[y_array == minority_class]
        X_majority = X_array[y_array == majority_class]
        y_minority = y_array[y_array == minority_class]
        y_majority = y_array[y_array == majority_class]

        target_minority_count = int(np.ceil(self.sampling_strategy * len(X_majority)))
        if target_minority_count <= len(X_minority) or len(X_minority) < 2:
            return X_array, y_array

        n_samples_to_generate = target_minority_count - len(X_minority)
        effective_k = min(self.k_neighbors, len(X_minority) - 1)
        if effective_k < 1:
            return X_array, y_array

        nn = NearestNeighbors(n_neighbors=effective_k + 1)
        nn.fit(X_minority)
        neighbor_indices = nn.kneighbors(X_minority, return_distance=False)[:, 1:]

        rng = np.random.default_rng(self.random_state)
        synthetic_samples = []

        for _ in range(n_samples_to_generate):
            sample_index = rng.integers(0, len(X_minority))
            sample = X_minority[sample_index]
            neighbor_index = rng.choice(neighbor_indices[sample_index])
            neighbor = X_minority[neighbor_index]
            gap = rng.random()
            synthetic_samples.append(sample + gap * (neighbor - sample))

        X_synthetic = np.asarray(synthetic_samples, dtype=float)
        y_synthetic = np.full(len(X_synthetic), minority_class, dtype=int)

        X_resampled = np.vstack([X_majority, X_minority, X_synthetic])
        y_resampled = np.concatenate([y_majority, y_minority, y_synthetic])
        return X_resampled, y_resampled


class SmoteTabularClassifier(BaseEstimator, ClassifierMixin):
    """Wrap feature engineering, preprocessing, SMOTE, and a classifier in one estimator."""

    def __init__(
        self,
        numerical_features: list[str],
        categorical_features: list[str],
        classifier: object,
        use_scaler: bool,
        sampling_strategy: float = SMOTE_SAMPLING_STRATEGY,
        random_state: int = RANDOM_STATE,
        k_neighbors: int = SMOTE_K_NEIGHBORS,
    ) -> None:
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.classifier = classifier
        self.use_scaler = use_scaler
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def _engineered_numerical_features(self) -> list[str]:
        engineered_numerical_features = self.numerical_features.copy()
        if {"age", "has_hypertension"}.issubset(self.numerical_features):
            engineered_numerical_features.append("age_x_has_hypertension")
        if {"bmi_value", "glucose_level"}.issubset(self.numerical_features):
            engineered_numerical_features.append("bmi_x_glucose_level")
        return engineered_numerical_features

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SmoteTabularClassifier":
        self.feature_engineering_ = InteractionFeatureAdder()
        engineered_numerical_features = self._engineered_numerical_features()
        self.preprocessor_ = build_preprocessor(
            engineered_numerical_features,
            self.categorical_features,
            use_scaler=self.use_scaler,
        )
        self.smote_ = SimpleSMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
        )

        X_engineered = self.feature_engineering_.fit_transform(X, y)
        X_processed = self.preprocessor_.fit_transform(X_engineered, y)
        X_resampled, y_resampled = self.smote_.fit_resample(X_processed, y)

        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X_resampled, y_resampled)
        self.classes_ = np.array(sorted(np.unique(y_resampled)))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_engineered = self.feature_engineering_.transform(X)
        X_processed = self.preprocessor_.transform(X_engineered)
        return self.classifier_.predict_proba(X_processed)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def strip_classifier_prefix(best_params: dict[str, object]) -> dict[str, object]:
    """Convert sklearn search parameter names into classifier constructor kwargs."""

    return {key.removeprefix("classifier__"): value for key, value in best_params.items()}


def build_weighted_models(
    numerical_features: list[str],
    categorical_features: list[str],
    y_train: pd.Series,
    model_params: dict[str, dict[str, object]] | None = None,
) -> dict[str, SmoteTabularClassifier]:
    """Build class-weighted models with SMOTE applied only inside training folds."""

    num_positive = float((y_train == 1).sum())
    num_negative = float((y_train == 0).sum())
    scale_pos_weight = num_negative / max(num_positive, 1.0)
    if not np.isfinite(scale_pos_weight) or scale_pos_weight <= 0:
        scale_pos_weight = DEFAULT_SCALE_POS_WEIGHT

    logistic_classifier_params = {
        "C": 1.0,
        "max_iter": 2000,
        "random_state": RANDOM_STATE,
        "solver": "lbfgs",
        "class_weight": "balanced",
    }
    random_forest_classifier_params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": "balanced",
    }
    xgboost_classifier_params = {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    if model_params:
        logistic_classifier_params.update(model_params.get("Logistic Regression", {}))
        random_forest_classifier_params.update(model_params.get("Random Forest", {}))
        xgboost_classifier_params.update(model_params.get("XGBoost", {}))

    return {
        "Logistic Regression": SmoteTabularClassifier(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            classifier=LogisticRegression(**logistic_classifier_params),
            use_scaler=True,
        ),
        "Random Forest": SmoteTabularClassifier(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            classifier=RandomForestClassifier(**random_forest_classifier_params),
            use_scaler=False,
        ),
        "XGBoost": SmoteTabularClassifier(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            classifier=XGBClassifier(**xgboost_classifier_params),
            use_scaler=False,
        ),
    }


def tune_base_model_hyperparameters(
    numerical_features: list[str],
    categorical_features: list[str],
    X_train_fit: pd.DataFrame,
    y_train_fit: pd.Series,
) -> dict[str, dict[str, object]]:
    """Tune the base learners with randomized search on the inner training split only."""

    print("\nHyperparameter tuning on the inner training split (recall-focused)...")
    tuned_params: dict[str, dict[str, object]] = {}
    base_models = build_weighted_models(numerical_features, categorical_features, y_train_fit)

    for model_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        search_space = BASE_MODEL_SEARCH_SPACES[model_name]
        n_iter = min(
            RANDOM_SEARCH_ITERATIONS,
            int(np.prod([len(values) for values in search_space.values()])),
        )
        search = RandomizedSearchCV(
            estimator=base_models[model_name],
            param_distributions=search_space,
            n_iter=n_iter,
            scoring="recall",
            cv=RANDOM_SEARCH_CV_FOLDS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train_fit, y_train_fit)
        best_params = strip_classifier_prefix(search.best_params_)
        tuned_params[model_name] = best_params
        print(f"Best {model_name} params: {best_params}")

    return tuned_params


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
        best_row = threshold_df.sort_values(
            by=["f1", "recall", "threshold"], ascending=[False, False, True]
        ).iloc[0]
        return float(best_row["threshold"]), "fallback_max_f1"

    best_row = eligible.sort_values(by=["recall", "f1", "threshold"], ascending=[False, False, True]).iloc[0]
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
        auc_value = roc_auc_score(y_true, probability_map[model_name])
        label = f"{model_name} (thr={tuned_thresholds[model_name]:.2f}, AUC={auc_value:.4f})"
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
    """Run hyperparameter tuning, threshold tuning, and final evaluation."""

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

    tuned_model_params = tune_base_model_hyperparameters(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        X_train_fit=X_train_fit,
        y_train_fit=y_train_fit,
    )

    print("Training base models on the inner train split and tuning thresholds on the validation split...")
    validation_pipelines = build_weighted_models(
        numerical_features,
        categorical_features,
        y_train_fit,
        model_params=tuned_model_params,
    )
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
    final_pipelines = build_weighted_models(
        numerical_features,
        categorical_features,
        y_train_full,
        model_params=tuned_model_params,
    )
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
