import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("symptoms_dataset_cleaned.csv")

# Features
symptom_features = ["cough_severity", "breathlessness_level", "chest_pain_level",
                    "fatigue_level", "fever_temp_c", "wheezing", "sore_throat"]
env_features = ["temperature_c", "humidity_percent", "air_quality_index"]

# Scale AQI to 1–5 (Original logic)
min_aqi = df["air_quality_index"].min()
max_aqi = df["air_quality_index"].max()
df["air_quality_index"] = 1 + 4 * (df["air_quality_index"] - min_aqi) / (max_aqi - min_aqi)
df["air_quality_index"] = df["air_quality_index"].round().astype(int)

synthetic_data = []

for _, row in df.iterrows():
    # ---------------------------
    # Day 2
    # ---------------------------
    day2 = {}
    for f in symptom_features:
        day2[f"Day2_{f}"] = round(row[f], 1) if f == "fever_temp_c" else int(row[f])
    for f in env_features:
        day2[f"Day2_{f}"] = round(row[f], 1) if f == "temperature_c" else int(row[f])

    # ---------------------------
    # Day 3 (Slight Variation)
    # ---------------------------
    day3 = {}
    for f in symptom_features:
        if f == "fever_temp_c":
            day3[f"Day3_{f}"] = round(row[f] + np.random.uniform(-0.7, 0.7), 1)
        else:
            day3[f"Day3_{f}"] = int(max(0, min(3, row[f] + np.random.randint(-1, 2))))

    for f in env_features:
        if f == "temperature_c":
            day3[f"Day3_{f}"] = round(row[f] + np.random.uniform(-0.5, 0.5), 1)
        elif f == "air_quality_index":
            day3[f"Day3_{f}"] = int(max(1, min(5, day2[f"Day2_{f}"] + np.random.randint(-1, 2))))
        else:
            day3[f"Day3_{f}"] = int(row[f] + np.random.randint(-1, 2))

    # ---------------------------
    # ✅ IMPROVED RISK CALCULATION (LEVEL + DIFF)
    # ---------------------------
    # Differences (Trends)
    dbreath = abs(day3["Day3_breathlessness_level"] - day2["Day2_breathlessness_level"])
    dfever_val = abs(day3["Day3_fever_temp_c"] - day2["Day2_fever_temp_c"])
    dchest = abs(day3["Day3_chest_pain_level"] - day2["Day2_chest_pain_level"])

    # Levels (Severity) - We use Day 3 as the "Current Status"
    # For fever, we calculate how much it exceeds a "normal" 37.0
    fever_severity = max(0, day3["Day3_fever_temp_c"] - 37.0)
    breath_severity = day3["Day3_breathlessness_level"]
    chest_severity = day3["Day3_chest_pain_level"]

    # Weighted Risk: 50% based on Current Severity, 50% based on Change
    risk_raw = (
        # Trends (Differences)
            (2.0 * dbreath + 1.5 * dfever_val + 1.5 * dchest) +
            # Severity (Current Levels)
            (2.5 * breath_severity + 2.0 * fever_severity + 2.0 * chest_severity) +
            # Minor Symptoms (Mixed)
            (1.0 * day3["Day3_cough_severity"] + 0.5 * abs(day3["Day3_cough_severity"] - day2["Day2_cough_severity"]))
    )

    # Normalize to a 1-10 scale for the model to learn
    risk_score = 1 + (risk_raw / 25) * 9
    risk_score = round(min(max(risk_score, 1), 10), 2)

    # ---------------------------
    # Combine
    # ---------------------------
    combined = {**day2, **day3, "risk_score": risk_score}
    synthetic_data.append(combined)

# Create DataFrame
df_synthetic = pd.DataFrame(synthetic_data)
df_synthetic.to_csv("historical_patient_data_final.csv", index=False)

print(f"✅ 16-Feature Dataset Created! Rows: {len(df_synthetic)}")