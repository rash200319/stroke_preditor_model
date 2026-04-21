import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def main() -> None:
    """Prepare dataset without pre-split imputation to avoid leakage."""

    df = pd.read_csv("healthcare_data.csv")

    # Remove accidental index column if present.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("Missing values before modeling")
    print("=" * 60)
    missing_counts = df.isnull().sum()
    print(missing_counts)
    print("\nMissing values percentage per column:")
    print((missing_counts / len(df) * 100).round(2))
    print("\nTotal missing values:", int(missing_counts.sum()))

    output_file = "healthcare_data_cleaned.csv"
    df.to_csv(output_file, index=False)

    print(f"\nSaved dataset to: {output_file}")
    print("No imputation was performed in this step.")
    print("Imputation must be performed inside the model Pipeline after train-test split.")


if __name__ == "__main__":
    main()