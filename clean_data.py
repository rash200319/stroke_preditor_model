import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

## explore missing values 
# Load the healthcare dataset
df = pd.read_csv('healthcare_data.csv', index_col=0)

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nDataset info:")
print(df.info())

# impute missing values using grouped median (age,gender)

# Impute missing BMI values using grouped median (gender + age_group)
df['bmi_value'] = df['bmi_value'].fillna(
    df.groupby(['gender', 'age_group'])['bmi_value'].transform('median')
)

# Verify no missing values remain
print("Missing values per column:")
print(df.isnull().sum())
print("\n" + "="*50)
print("Missing values percentage per column:")
print((df.isnull().sum() / len(df) * 100).round(2))
print("\nTotal missing values:", df.isnull().sum().sum())

print("After imputation Missing values per column:")
print(df_imputed.isnull().sum())
print("\nTotal missing values:", df_imputed.isnull().sum().sum())

# Final verification
print("="*60)
print("FINAL VERIFICATION - Complete Review")
print("="*60)
print("\nOriginal dataset shape:", df.shape)
print("Cleaned dataset shape:", df_imputed.shape)

print("\n" + "="*60)
print("Missing Values Summary:")
print("="*60)
print("Total missing values in original data:", df.isnull().sum().sum())
print("Total missing values in cleaned data:", df_imputed.isnull().sum().sum())

print("\n" + "="*60)
print("Cleaned Dataset Info:")
print("="*60)
print(df_imputed.info())

print("\n" + "="*60)
print("First few rows of cleaned dataset:")
print("="*60)
print(df_imputed.head())

# Save the cleaned dataset
output_file = 'healthcare_data_cleaned.csv'
df_imputed.to_csv(output_file)

print(f"\n✓ Cleaned dataset saved to: {output_file}")
print(f"\nDataset Statistics:")
print(f"- Total rows: {len(df_imputed)}")
print(f"- Total columns: {len(df_imputed.columns)}")
print(f"- No missing values remaining: {df_imputed.isnull().sum().sum() == 0}")