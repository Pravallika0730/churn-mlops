import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Paths
RAW_PATH = "data/raw/Telco-Customer-Churn.csv"
PROCESSED_PATH = "data/processed/processed_churn.csv"

# Load raw data
df = pd.read_csv(RAW_PATH)
print("Raw data loaded:", df.shape)

# Drop customerID (not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric (it has blanks)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical features
cat_cols = df.select_dtypes(include="object").columns
encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

print("After preprocessing:", df.shape)

# Save processed data
df.to_csv(PROCESSED_PATH, index=False)
print("Processed data saved to:", PROCESSED_PATH)
