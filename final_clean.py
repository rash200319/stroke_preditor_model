"""
clean_data_v2.py
Deduplicates on clinical features (excluding patient_id / index),
then saves the true imbalanced dataset for modelling.
"""

import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def impute_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Impute BMI using grouped medians before feature engineering."""

    df_out = df.copy()
    df_out["_age_bin"] = pd.cut(
        df_out["age"],
        bins=[0, 40, 60, 200],
        labels=["young", "middle", "senior"],
    )

    grouped_median = df_out.groupby(["gender", "_age_bin"], observed=True)["bmi_value"].median()
    global_median = df_out["bmi_value"].median()

    def fill_bmi(row: pd.Series) -> float:
        if pd.isna(row["bmi_value"]):
            try:
                return grouped_median.loc[(row["gender"], row["_age_bin"])]
            except KeyError:
                return global_median
        return row["bmi_value"]

    df_out["bmi_value"] = df_out.apply(fill_bmi, axis=1)
    return df_out.drop(columns=["_age_bin"])


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically meaningful interaction features."""

    df_out = df.copy()
    df_out["age_squared"] = df_out["age"] ** 2
    df_out["age_x_hypertension"] = df_out["age"] * df_out["has_hypertension"]
    df_out["age_x_heart"] = df_out["age"] * df_out["has_heart_disease"]
    df_out["glucose_x_hypertension"] = df_out["glucose_level"] * df_out["has_hypertension"]
    df_out["age_risk_score"] = df_out["age"] * (
        df_out["has_hypertension"] + df_out["has_heart_disease"]
    )
    return df_out


def main() -> None:
    df = pd.read_csv("healthcare_data.csv")

    # Drop accidental index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(f"Original shape : {df.shape}")

    # Deduplicate on all columns EXCEPT patient_id (IDs are unique by design)
    dedup_cols = [c for c in df.columns if c != "patient_id"]
    df = df.drop_duplicates(subset=dedup_cols)
    print(f"After dedup    : {df.shape}")

    df = impute_bmi(df)
    df = add_interaction_features(df)
    print(f"After features : {df.shape}")

    print("\nStroke distribution (real imbalanced):")
    print(df["stroke_event"].value_counts())
    print(f"\nStroke prevalence : {df['stroke_event'].mean() * 100:.2f}%")

    print("\nMissing values:")
    mv = df.isnull().sum()
    print(mv[mv > 0])

    df.to_csv("healthcare_data_cleaned.csv", index=False)
    print("\nSaved -> healthcare_data_cleaned.csv")
    print("NOTE: Interaction features were added after BMI imputation and before modeling.")


if __name__ == "__main__":
    main()
