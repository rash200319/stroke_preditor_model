import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('healthcare_data.csv', index_col=0)

# 1. Check for missing values
missing_info = df.isnull().sum()
print("Missing values per column:")
print(missing_info[missing_info > 0])

# 2. Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

# 3. Check data types
print("\nData types:")
print(df.dtypes)

# 4. Check for 'NA' strings if they are not recognized as NaN
# Looking at the view_file output, it seems NA is used. 
# pandas usually picks up NA but let's be sure.
# Sometimes it might be literal "NA" strings if not handled correctly.
# Re-loading with na_values if necessary.
df = pd.read_csv('healthcare_data.csv', index_col=0, na_values=['NA', 'N/A', ''])
missing_info = df.isnull().sum()
print("\nMissing values per column (after handling NA strings):")
print(missing_info[missing_info > 0])
