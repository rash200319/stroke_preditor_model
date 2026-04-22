import pandas as pd
import numpy as np

def preprocess_data(file_path):
    # Load data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, index_col=0, na_values=['NA', 'N/A', ''])
    
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")

    # 1. Handle duplicates by deleting duplicate data
    df = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df.shape}")
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicates.")

    # 2. Handle missing data by imputation
    # Use the mean of the missing patient's age group's mean value for that column
    # The column with missing values is 'bmi_value'
    if df['bmi_value'].isnull().any():
        print("Imputing missing values in 'bmi_value'...")
        # Calculate mean per age group
        age_group_means = df.groupby('age_group')['bmi_value'].mean()
        print("Means per age group:")
        print(age_group_means)
        
        # Function to fill missing values
        def fill_bmi(row):
            if pd.isnull(row['bmi_value']):
                return age_group_means[row['age_group']]
            return row['bmi_value']
        
        df['bmi_value'] = df.apply(fill_bmi, axis=1)
        print("Imputation complete.")
    else:
        print("No missing values found in 'bmi_value'.")

    # 3. Validate data by range checks and type checks
    print("\nValidating data...")
    
    # Type checks
    # Ensure numeric columns are numeric
    numeric_cols = ['age', 'has_hypertension', 'has_heart_disease', 'marital_status', 
                    'glucose_level', 'bmi_value', 'stroke_event', 'risk_score', 'high_glucose']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where critical numeric columns couldn't be converted
    df = df.dropna(subset=['age', 'stroke_event'])
    
    # Range checks
    # Define valid ranges
    ranges = {
        'age': (0, 120),
        'glucose_level': (0, 500),
        'bmi_value': (5, 100),
        'has_hypertension': (0, 1),
        'has_heart_disease': (0, 1),
        'stroke_event': (0, 1)
    }
    
    validation_errors = []
    for col, (min_val, max_val) in ranges.items():
        out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
        if not out_of_range.empty:
            validation_errors.append(f"Found {len(out_of_range)} out-of-range values in '{col}' (expected {min_val}-{max_val})")
            # For this task, we'll clip or filter. Let's filter out extreme outliers if they are clearly wrong.
            # However, in medical data, extreme values might be real.
            # For simplicity, let's just report them for now or clip them.
            # As per requirements "validate", usually means checking and reporting or cleaning.
            # I will filter out clearly impossible values (like negative age).
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]

    if validation_errors:
        print("Validation report:")
        for err in validation_errors:
            print(f" - {err}")
    else:
        print("All range checks passed.")

    # Save cleaned data
    output_path = 'cleaned_healthcare_data.csv'
    df.to_csv(output_path)
    print(f"\nCleaned data saved to {output_path}")
    print(f"Final shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    preprocess_data('healthcare_data.csv')
