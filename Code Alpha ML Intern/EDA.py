
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "eda_results"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Load cleaned dataset
file_path = BASE_DIR / "Credit Score" / "data" / "clean_credit_data.csv"

df = pd.read_csv(file_path)

target = "default payment next month"

# 1. Dataset overview
print("Dataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())

# 2. Statistical analysis
print("\nStatistical Summary:")
print(df.describe())

# 3. Target distribution
print("\nDefault Distribution:")
print(df[target].value_counts())

print("\nDefault Percentage:")
print(df[target].value_counts(normalize=True) * 100)

# 4. Default distribution chart
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x=target)
plt.title("Credit Default Distribution")
plt.xlabel("Default Status (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.savefig(OUTPUT_DIR / "default_distribution.png")
plt.show()

# 5. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="AGE", bins=30, kde=True)
plt.title("Customer Age Distribution")
plt.savefig(OUTPUT_DIR / "age_distribution.png")
plt.show()

# 6. Credit limit analysis
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x=target, y="LIMIT_BAL")
plt.title("Credit Limit vs Default Status")
plt.savefig(OUTPUT_DIR / "credit_limit.png")
plt.show()

# 7. Correlation heatmap
plt.figure(figsize=(16, 12))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.show()

print("\nEDA completed successfully!")
