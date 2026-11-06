import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_csv(DATA_DIR / "credit_data.csv")

# Fix column names
df.columns = df.iloc[0]
df = df.iloc[1:].reset_index(drop=True)

# Remove ID
df = df.drop(columns=["ID"])

# Convert columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Check missing values
print("Missing values:")
print(df.isnull().sum().sum())

# Check duplicates
print("Duplicate records:", df.duplicated().sum())

# Dataset information
print("\nDataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)

# Target distribution
print("\nTarget distribution:")
print(df["default payment next month"].value_counts())

# Save cleaned dataset
df.to_csv(DATA_DIR / "clean_credit_data.csv", index=False)

print("\nPreprocessing completed!")