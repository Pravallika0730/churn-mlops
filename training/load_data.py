import pandas as pd

# Load dataset
file_path = "data/raw/Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# Basic checks
print("Dataset loaded successfully!")
print("Shape (rows, columns):", df.shape)

print("\nColumn names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nTarget variable distribution:")
print(df["Churn"].value_counts())
