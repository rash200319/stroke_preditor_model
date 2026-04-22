import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from model_comparison import (
    CLASSIFICATION_THRESHOLD,
    RANDOM_STATE,
    build_models,
    identify_feature_types,
    load_data,
    split_features_and_target,
)


def main() -> None:
    """Run a leakage sanity check by training on shuffled labels."""

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

    rng = np.random.default_rng(RANDOM_STATE)
    y_train_shuffled = pd.Series(
        rng.permutation(y_train.to_numpy()),
        index=y_train.index,
        name=y_train.name,
    )

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = build_models(numerical_features, categorical_features)

    print("Label Shuffle Sanity Check")
    print("=" * 80)
    print("Training labels are shuffled before fitting each model.")
    print(f"Decision threshold: {CLASSIFICATION_THRESHOLD:.2f}\n")

    rows = []
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train_shuffled)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= CLASSIFICATION_THRESHOLD).astype(int)

        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train_shuffled,
            cv=cv_strategy,
            scoring={"roc_auc": "roc_auc"},
            n_jobs=-1,
            return_train_score=False,
        )

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
                "CV ROC-AUC (shuffled)": f"{np.mean(cv_scores['test_roc_auc']):.4f} ± {np.std(cv_scores['test_roc_auc']):.4f}",
            }
        )

    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nExpected behavior: ROC-AUC should be near 0.50 if there is no leakage.")


if __name__ == "__main__":
    main()
