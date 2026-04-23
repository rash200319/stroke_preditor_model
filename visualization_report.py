"""
Standalone visualization and reporting script for the stroke project.

This does not change the modeling pipeline. It only generates:
- publication-style EDA charts
- statistical test summaries
- a markdown report
- model-performance and feature-importance summaries from the latest run
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 14,
        "legend.fontsize": 9,
        "font.size": 10,
    }
)
sns.set_theme(style="whitegrid", context="talk")

RAW_PATH = Path("healthcare_data.csv")
CLEAN_PATH = Path("healthcare_data_cleaned.csv")
OUT_DIR = Path("visuals")


LATEST_ENSEMBLE_RESULTS = {
    "High Sensitivity": {
        "Logistic Regression": {"AUC-ROC": 0.8326, "AUC-PR": 0.2102, "Recall": 0.9400, "Precision": 0.0793, "F1": 0.1462},
        "Random Forest": {"AUC-ROC": 0.8183, "AUC-PR": 0.1834, "Recall": 0.8800, "Precision": 0.0841, "F1": 0.1536},
        "XGBoost": {"AUC-ROC": 0.8034, "AUC-PR": 0.1681, "Recall": 0.4600, "Precision": 0.1679, "F1": 0.2460},
        "Soft Voting": {"AUC-ROC": 0.8282, "AUC-PR": 0.1905, "Recall": 0.8200, "Precision": 0.1059, "F1": 0.1876},
        "Stacking": {"AUC-ROC": 0.8383, "AUC-PR": 0.2150, "Recall": 0.9200, "Precision": 0.0819, "F1": 0.1503},
    },
    "Balanced": {
        "Logistic Regression": {"AUC-ROC": 0.8326, "AUC-PR": 0.2102, "Recall": 0.4200, "Precision": 0.2234, "F1": 0.2917},
        "Random Forest": {"AUC-ROC": 0.8183, "AUC-PR": 0.1834, "Recall": 0.4200, "Precision": 0.2188, "F1": 0.2877},
        "XGBoost": {"AUC-ROC": 0.8034, "AUC-PR": 0.1681, "Recall": 0.0400, "Precision": 0.1538, "F1": 0.0635},
        "Soft Voting": {"AUC-ROC": 0.8282, "AUC-PR": 0.1905, "Recall": 0.2600, "Precision": 0.2000, "F1": 0.2261},
        "Stacking": {"AUC-ROC": 0.8383, "AUC-PR": 0.2150, "Recall": 0.7800, "Precision": 0.1940, "F1": 0.3108},
    },
}

LATEST_XGB_IMPORTANCE = {
    "age_squared": 0.155550,
    "is_senior": 0.131076,
    "age": 0.114347,
    "cvd_count": 0.088688,
    "residence": 0.071208,
    "gender": 0.067930,
    "has_hypertension": 0.046056,
    "smoking_habit": 0.038471,
    "age_over_10": 0.036362,
    "employment_type": 0.034493,
}


def load_dataset() -> tuple[pd.DataFrame, str]:
    if CLEAN_PATH.exists():
        df = pd.read_csv(CLEAN_PATH)
        source = CLEAN_PATH.name
    else:
        df = pd.read_csv(RAW_PATH)
        source = RAW_PATH.name

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df.copy(), source


def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_model_metric_heatmap() -> Path:
    metrics = ["AUC-ROC", "AUC-PR", "Recall", "Precision", "F1"]
    modes = list(LATEST_ENSEMBLE_RESULTS.keys())
    models = list(LATEST_ENSEMBLE_RESULTS[modes[0]].keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Latest Ensemble Performance Summary", fontsize=16, fontweight="bold")

    for ax, mode in zip(axes, modes):
        matrix = pd.DataFrame(
            {model: [LATEST_ENSEMBLE_RESULTS[mode][model][metric] for metric in metrics] for model in models},
            index=metrics,
        )
        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            cbar=False,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(mode)
        ax.set_xlabel("Model")
        ax.set_ylabel("Metric")

    return save_figure(fig, "07_model_metric_heatmap.png")


def save_xgb_importance_plot() -> Path:
    importance = pd.Series(LATEST_XGB_IMPORTANCE).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Latest XGBoost Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    return save_figure(fig, "08_xgb_feature_importance.png")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = df.astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def class_balance_plot(df: pd.DataFrame) -> Path:
    counts = df["stroke_event"].value_counts().sort_index()
    labels = ["No Stroke", "Stroke"]
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title("Class Balance")
    ax.set_ylabel("Number of Records")
    ax.set_xlabel("Target")

    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{count}\n({count / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    return save_figure(fig, "01_class_balance.png")


def numeric_distribution_plots(df: pd.DataFrame) -> Path:
    features = ["age", "glucose_level", "bmi_value"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Numeric Feature Distributions by Stroke Status", fontsize=16, fontweight="bold")

    for ax, feature in zip(axes, features):
        sns.kdeplot(
            data=df,
            x=feature,
            hue="stroke_event",
            common_norm=False,
            fill=True,
            alpha=0.25,
            palette=["#4C72B0", "#DD8452"],
            ax=ax,
        )
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel("Density")

    return save_figure(fig, "02_numeric_distributions.png")


def categorical_count_plots(df: pd.DataFrame) -> Path:
    features = ["gender", "smoking_habit", "employment_type", "residence"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Categorical Feature Counts by Stroke Status", fontsize=16, fontweight="bold")

    for ax, feature in zip(axes.flat, features):
        order = df[feature].value_counts(dropna=False).index
        sns.countplot(
            data=df,
            x=feature,
            hue="stroke_event",
            order=order,
            palette=["#4C72B0", "#DD8452"],
            ax=ax,
        )
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)

    return save_figure(fig, "03_categorical_counts.png")


def boxplot_comparison(df: pd.DataFrame) -> Path:
    features = ["age", "glucose_level", "bmi_value"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Boxplots for Core Risk Features", fontsize=16, fontweight="bold")

    for ax, feature in zip(axes, features):
        sns.boxplot(data=df, x="stroke_event", y=feature, color="#4C72B0", ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Stroke", "Stroke"])
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel(feature.replace("_", " ").title())

    return save_figure(fig, "04_boxplots.png")


def correlation_heatmap(df: pd.DataFrame) -> Path:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    corr = numeric_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold")
    return save_figure(fig, "05_correlation_heatmap.png")


def statistical_tests_plots(df: pd.DataFrame) -> Path:
    numeric_features = ["age", "glucose_level", "bmi_value"]
    categorical_features = ["gender", "smoking_habit", "employment_type", "residence", "has_hypertension", "has_heart_disease"]

    numeric_rows = []
    for feature in numeric_features:
        group_0 = df.loc[df["stroke_event"] == 0, feature].dropna()
        group_1 = df.loc[df["stroke_event"] == 1, feature].dropna()
        stat, p_value = mannwhitneyu(group_0, group_1, alternative="two-sided")
        numeric_rows.append({"Feature": feature, "Test": "Mann-Whitney U", "p_value": p_value, "statistic": stat})

    categorical_rows = []
    for feature in categorical_features:
        contingency = pd.crosstab(df[feature], df["stroke_event"])
        stat, p_value, _, _ = chi2_contingency(contingency)
        categorical_rows.append({"Feature": feature, "Test": "Chi-square", "p_value": p_value, "statistic": stat})

    tests_df = pd.DataFrame(numeric_rows + categorical_rows)
    tests_df["-log10(p)"] = -np.log10(tests_df["p_value"].clip(lower=1e-300))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Statistical Tests: Stroke vs Non-Stroke", fontsize=16, fontweight="bold")

    num_df = tests_df[tests_df["Test"] == "Mann-Whitney U"].sort_values("-log10(p)")
    cat_df = tests_df[tests_df["Test"] == "Chi-square"].sort_values("-log10(p)")

    sns.barplot(data=num_df, x="-log10(p)", y="Feature", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Numeric Features")
    axes[0].set_xlabel("-log10(p-value)")
    axes[0].set_ylabel("")

    sns.barplot(data=cat_df, x="-log10(p)", y="Feature", ax=axes[1], color="#DD8452")
    axes[1].set_title("Categorical Features")
    axes[1].set_xlabel("-log10(p-value)")
    axes[1].set_ylabel("")

    tests_df.to_csv(OUT_DIR / "statistical_tests_summary.csv", index=False)
    return save_figure(fig, "06_statistical_tests.png")


def write_markdown_report(df: pd.DataFrame, source_name: str) -> Path:
    n_rows = len(df)
    stroke_cases = int(df["stroke_event"].sum())
    prevalence = stroke_cases / n_rows if n_rows else 0

    summary_table = pd.DataFrame(
        [
            {"Metric": "Rows", "Value": f"{n_rows:,}"},
            {"Metric": "Stroke cases", "Value": f"{stroke_cases:,}"},
            {"Metric": "Stroke prevalence", "Value": f"{prevalence:.2%}"},
            {"Metric": "Source file", "Value": source_name},
        ]
    )

    model_rows = []
    for mode, models in LATEST_ENSEMBLE_RESULTS.items():
        for model, metrics in models.items():
            model_rows.append(
                {
                    "Mode": mode,
                    "Model": model,
                    **metrics,
                }
            )
    model_df = pd.DataFrame(model_rows)

    lines = []
    lines.append("# Stroke Visualization Report")
    lines.append("")
    lines.append("This report is generated separately from the training pipeline.")
    lines.append("")
    lines.append("## Dataset Snapshot")
    lines.append("")
    lines.append(dataframe_to_markdown(summary_table))
    lines.append("")
    lines.append("## Statistical Findings")
    lines.append("")
    lines.append(
        "- `age`, `glucose_level`, and `bmi_value` are compared with Mann-Whitney U tests."
    )
    lines.append(
        "- `gender`, `smoking_habit`, `employment_type`, `residence`, `has_hypertension`, and `has_heart_disease` are compared with chi-square tests."
    )
    lines.append(
        "- The full test table is saved as `statistical_tests_summary.csv`."
    )
    lines.append("")
    lines.append("## Latest Ensemble Summary")
    lines.append("")
    lines.append(dataframe_to_markdown(model_df.round(4)))
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    lines.append("- `01_class_balance.png`")
    lines.append("- `02_numeric_distributions.png`")
    lines.append("- `03_categorical_counts.png`")
    lines.append("- `04_boxplots.png`")
    lines.append("- `05_correlation_heatmap.png`")
    lines.append("- `06_statistical_tests.png`")
    lines.append("- `07_model_metric_heatmap.png`")
    lines.append("- `08_xgb_feature_importance.png`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- The modeling pipeline is unchanged; this script only creates reports and visuals."
    )
    lines.append(
        "- The model summary and feature importance reflect the latest run already captured in the repository output."
    )

    report_path = OUT_DIR / "visualization_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    df, source_name = load_dataset()
    ensure_output_dir()

    outputs = [
        class_balance_plot(df),
        numeric_distribution_plots(df),
        categorical_count_plots(df),
        boxplot_comparison(df),
        correlation_heatmap(df),
        statistical_tests_plots(df),
        save_model_metric_heatmap(),
        save_xgb_importance_plot(),
        write_markdown_report(df, source_name),
    ]

    print("Visualization assets generated:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
