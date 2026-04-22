"""
clean_data_v2.py
Deduplicates on clinical features (excluding patient_id / index),
then saves the true imbalanced dataset for modelling.
"""
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


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

    print("\nStroke distribution (real imbalanced):")
    print(df["stroke_event"].value_counts())
    print(f"\nStroke prevalence : {df['stroke_event'].mean()*100:.2f}%")

    print("\nMissing values:")
    mv = df.isnull().sum()
    print(mv[mv > 0])

    df.to_csv("healthcare_data_cleaned.csv", index=False)
    print("\nSaved → healthcare_data_cleaned2.csv")
    print("NOTE: Imputation is performed inside the model pipeline after train-test split.")


if __name__ == "__main__":
    main()