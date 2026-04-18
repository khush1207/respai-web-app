import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("historical_patient_data_final.csv")

# ===============================
# RENAME COLUMNS (CONSISTENT NAMING)
# ===============================
df.rename(columns={
    "Day2_cough_severity": "Day2_cough",
    "Day2_breathlessness_level": "Day2_breath",
    "Day2_chest_pain_level": "Day2_chest",
    "Day2_fatigue_level": "Day2_fatigue",
    "Day2_fever_temp_c": "Day2_fever",
    "Day2_wheezing": "Day2_wheezing",
    "Day2_sore_throat": "Day2_sore",
    "Day2_air_quality_index" : "Day2_aqi",

    "Day3_cough_severity": "Day3_cough",
    "Day3_breathlessness_level": "Day3_breath",
    "Day3_chest_pain_level": "Day3_chest",
    "Day3_fatigue_level": "Day3_fatigue",
    "Day3_fever_temp_c": "Day3_fever",
    "Day3_wheezing": "Day3_wheezing",
    "Day3_sore_throat": "Day3_sore",
    "Day3_air_quality_index" : "Day3_aqi"
}, inplace=True)

# ... (Previous imports and renaming remain the same) ...

# ===============================
# FEATURE ENGINEERING (LEVELS + DIFF)
# ===============================
symptoms = ["cough", "breath", "chest", "fatigue", "fever", "wheezing", "sore", "aqi"]

X_list = []
for f in symptoms:
    # 1. Current Level (Day 3)
    level = df[f"Day3_{f}"]
    # 2. Absolute Difference (Day 3 - Day 2)
    diff = abs(df[f"Day3_{f}"] - df[f"Day2_{f}"])

    X_list.append(level)
    X_list.append(diff)

# Combine into a matrix: shape (n_samples, 16 features)
X = np.column_stack(X_list)

# Target
y = df["risk_score"].values

# ===============================
# TRAIN & SAVE (Same as before)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
y_pred = model.predict(X_test)

r2 = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)

print("R^2 on test set:", round(r2, 4))
print("MAE:", round(mae, 4))


joblib.dump(model, "models/risk_model.pkl")
print("✅ Improved model saved with 16 features (Levels + Diffs)")
