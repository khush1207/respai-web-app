import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("health_survey_respiratory.csv")

# Keep only asthma and pneumonia
df = df[df["diagnosis"].isin(["Asthma", "Pneumonia"])]

# Drop city column
df = df.drop(columns=["city"])
print(df)
print(df.isnull().sum())

# Fill missing values for numeric columns based on diagnosis mean
numeric_cols = ["age", "cough_severity", "fever_temp_c", "breathlessness_level",
                "chest_pain_level", "fatigue_level", "temperature_c",
                "humidity_percent", "air_quality_index"]

for col in numeric_cols:
    df[col] = df.groupby("diagnosis")[col].transform(lambda x: x.fillna(round(x.mean(), 2)))

# Fill missing binary categorical columns with mode
binary_cols = ["wheezing", "sore_throat", "smoking_history"]
for col in binary_cols:
    df[col] = df.groupby("diagnosis")[col].transform(lambda x: x.fillna(x.mean()))

# Ensure integer columns remain int
int_cols = ["age","cough_severity", "breathlessness_level", "chest_pain_level", "fatigue_level", "wheezing", "sore_throat", "smoking_history"]
df[int_cols] = df[int_cols].astype(int)

# Encode target
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])  # 0: asthma, 1: pneumonia

print(df)
# print(df.info())
print(df.isnull().sum())

# After all preprocessing steps (imputation, encoding, etc.)
df.to_csv("symptoms_dataset_cleaned.csv", index=False)

print("Preprocessed dataset saved as 'symptoms_dataset_cleaned.csv'")
