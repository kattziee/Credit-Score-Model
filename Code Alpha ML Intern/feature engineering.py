
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load cleaned dataset
df = pd.read_csv(BASE_DIR / "data" / "clean_credit_data.csv")

target = "default payment next month"

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Group payment history
payment_columns = [
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

# Create payment delay feature
X["TOTAL_PAYMENT_DELAY"] = (
    X[payment_columns].clip(lower=0).sum(axis=1)
)

# Total bill amount
bill_columns = [
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
]

X["TOTAL_BILL_AMOUNT"] = X[bill_columns].sum(axis=1)

# Total payment amount
pay_columns = [
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

X["TOTAL_PAYMENT_AMOUNT"] = X[pay_columns].sum(axis=1)

# Average bill amount
X["AVERAGE_BILL_AMOUNT"] = X[bill_columns].mean(axis=1)

# Save engineered dataset
engineered_df = pd.concat([X, y], axis=1)

engineered_df.to_csv(
    BASE_DIR / "data" / "engineered_credit_data.csv",
    index=False
)

print("Feature Engineering Completed!")
print("New dataset shape:", engineered_df.shape)
print("\nNew features:")
print([
    "TOTAL_PAYMENT_DELAY",
    "TOTAL_BILL_AMOUNT",
    "TOTAL_PAYMENT_AMOUNT",
    "AVERAGE_BILL_AMOUNT"
])
