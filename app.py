import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Only show Errors
import json
import numpy as np
import requests
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import gdown

# NEW MODERN IMPORTS (Keras 3)
import keras
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
app.secret_key = "super_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===============================
# 1. SETUP MODEL PATHS
# ===============================
MODEL_DIR = "model"
# Make sure your GitHub folder is named "model" (no 's')
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
XRAY_MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# 2. DOWNLOAD FROM GOOGLE DRIVE
# ===============================
if not os.path.exists(XRAY_MODEL_PATH):
    print("Downloading heavy model from Google Drive...")
    file_id = '1OKch6xQ4I-cF8ytPCb1AnQmIpWcsl8bF'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, XRAY_MODEL_PATH, quiet=False)
    print("Download complete!")

# ===============================
# 3. LOAD BOTH MODELS (NUCLEAR OPTION)
# ===============================

# 1. Load Risk Model
try:
    risk_model = joblib.load(RISK_MODEL_PATH)
    print("✅ Risk model loaded successfully!")
except Exception as e:
    print(f"❌ Risk model error: {e}")

# 2. THE NUCLEAR FIX: Manually define the layers to bypass the metadata bug
def build_model_manually():
    from keras import layers, Sequential
    m = Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return m

try:
    # Instead of load_model (which crashes), we build a fresh one and just slide the weights in
    cnn_model = build_model_manually()
    cnn_model.load_weights(XRAY_MODEL_PATH)
    print("✅ X-Ray model loaded successfully via weight-injection!")
except Exception as e:
    print(f"❌ Final attempt error: {e}")

# ===============================
# API & HELPER FUNCTIONS
# ===============================
def get_aqi(city):
    API_KEY = "d308fee30aee063143b15f83368e580c"
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        geo_data = requests.get(geo_url, timeout=5).json()
        if not geo_data: return 3
        lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]
        aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        aqi_data = requests.get(aqi_url, timeout=5).json()
        return aqi_data["list"][0]["main"]["aqi"]
    except:
        return 3


def get_weather(city):
    API_KEY = "d308fee30aee063143b15f83368e580c"
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": API_KEY, "units": "metric"}
        data = requests.get(url, params=params, timeout=5).json()
        if "main" not in data: return 25.0, 60.0
        return data["main"]["temp"], data["main"]["humidity"]
    except:
        return 25.0, 60.0


def get_risk_category(score):
    if score < 4:
        return "Low Risk"
    elif score < 7:
        return "Moderate Risk"
    else:
        return "High Risk"


def get_precautions(score):
    if score < 3:
        return ["Stay hydrated", "Monitor symptoms"]
    elif score < 7:
        return ["Take rest", "Avoid pollution", "Monitor fever"]
    else:
        return ["Consult doctor", "X-ray recommended", "Monitor breathing"]


def predict_xray(path):
    import gc
    from PIL import Image
    try:
        # 1. Open image with Pillow (lighter than Keras image loader)
        with Image.open(path) as img:
            img = img.convert('RGB').resize((224, 224))
            # Convert to float32 immediately to save space
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        # 2. Predict with verbose=0 to stop logging overhead
        prediction = float(cnn_model.predict(img_array, verbose=0)[0][0])

        # 3. NUCLEAR CLEANUP: Delete arrays and force garbage collection
        del img_array
        gc.collect()
        keras.backend.clear_session()

        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.5


def build_features(prev_day, curr_day):
    symptoms = ["breath", "chest", "cough", "fatigue", "wheezing", "sore", "fever", "aqi"]
    feature_list = []
    for s in symptoms:
        feature_list.append(curr_day[s])
        feature_list.append(abs(curr_day[s] - prev_day[s]))
    return np.array([feature_list])


def extract_symptoms(form):
    return {
        "cough": int(form.get("cough", 0)),
        "breath": int(form.get("breath", 0)),
        "chest": int(form.get("chest", 0)),
        "fatigue": int(form.get("fatigue", 0)),
        "fever": float(form.get("fever", 37.0)),
        "wheezing": int(form.get("wheezing", 0)),
        "sore": int(form.get("sore", 0))
    }

# --- NEW HELPER FUNCTION ---
def get_enhanced_precautions(risk, is_final=False, disease="Normal"):
    precautions = []
    if risk < 4:
        precautions = [
            "Get plenty of rest and sleep.",
            "Stay well-hydrated with warm fluids.",
            "Monitor your temperature and symptoms daily.",
            "Maintain good hand hygiene and wear a mask if coughing."
        ]
    elif risk < 7:
        precautions = [
            "Isolate yourself from vulnerable family members.",
            "Use a humidifier or inhale steam to ease breathing.",
            "Monitor your blood oxygen levels using a pulse oximeter.",
            "Avoid cold beverages, dust, and smoking.",
            "Consult a healthcare provider if breathlessness increases."
        ]
    else:
        precautions = [
            "Get a diagnostic chest X-ray as soon as possible.",
            "Do not self-medicate for severe breathlessness.",
            "Keep emergency contact numbers readily available.",
            "Restrict all physical exertion and stay in bed."
        ]

    # Add a strong doctor warning for the final dashboard if Abnormal
    if is_final and disease == "Abnormal":
        precautions.insert(0,
                           "🚨 URGENT: Your diagnostic results indicate an abnormal respiratory condition. Please consult a pulmonologist or visit a medical clinic immediately.")

    return precautions

# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return redirect(url_for('day1'))


@app.route("/day1", methods=["GET", "POST"])
def day1():
    if request.method == "POST":
        city = request.form.get("city")
        temperature, humidity = get_weather(city)
        aqi = get_aqi(city)

        d1 = {
            "age": int(request.form.get("age")),
            "city": city,
            "smoking": int(request.form.get("smoking")),
            "temperature": temperature,
            "humidity": humidity,
            "aqi": aqi,
            **extract_symptoms(request.form)
        }
        json.dump(d1, open("day1.json", "w"))
        flash(f"Data saved. Weather: {temperature}°C, AQI: {aqi}. Proceed to Day 2.", "success")
        return redirect(url_for('day2'))
    return render_template("day1.html")


# --- UPDATED ROUTES ---
@app.route("/day2", methods=["GET", "POST"])
def day2():
    if not os.path.exists("day1.json"):
        flash("Please complete Day 1 first.", "danger")
        return redirect(url_for('day1'))

    d1 = json.load(open("day1.json"))

    if request.method == "POST":
        d2 = {**d1, **extract_symptoms(request.form)}
        with open("day2.json", "w") as f:
            json.dump(d2, f)
        return redirect(url_for('day3'))

    # Basic precautions to show after Day 1
    basic_precautions = [
        "Stay hydrated and get plenty of rest.",
        "Monitor your symptoms closely over the next 24 hours.",
        "Eat a balanced diet to support your immune system.",
        "Avoid exposure to cold air or pollutants."
    ]
    return render_template("day2.html", precautions=basic_precautions)


@app.route("/day3", methods=["GET", "POST"])
def day3():
    if not os.path.exists("day1.json") or not os.path.exists("day2.json"):
        flash("Please complete previous days first.", "danger")
        return redirect(url_for('day1'))

    d1 = json.load(open("day1.json"))
    d2 = json.load(open("day2.json"))

    f2 = build_features(d1, d2)
    risk2 = float(risk_model.predict(f2)[0])
    risk2_rounded = round(min(max(risk2, 1), 10), 2)
    xray_required = risk2_rounded >= 7

    # Get dynamic precautions based on Day 2 risk
    day2_precautions = get_enhanced_precautions(risk2_rounded)

    if request.method == "POST":
        d3 = {**d2, **extract_symptoms(request.form)}
        f3 = build_features(d2, d3)
        risk3 = float(risk_model.predict(f3)[0])

        pneumonia_prob = None
        asthma_prob = None
        disease = "Normal"

        xray_file = request.files.get("xray")
        if xray_file and xray_file.filename != '':
            filename = secure_filename(xray_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            xray_file.save(filepath)

            raw_pneumonia = float(predict_xray(filepath))
            raw_asthma = 1.0 - raw_pneumonia
            risk_multiplier = min(max(risk3 / 10.0, 0.1), 1.0)
            pneumonia_prob = raw_pneumonia * risk_multiplier
            asthma_prob = raw_asthma * risk_multiplier

            if pneumonia_prob >= 0.5 or asthma_prob >= 0.5:
                disease = "Abnormal"
            else:
                disease = "Abnormal" if risk3 >= 7 else "Normal"
        else:
            disease = "Abnormal" if risk3 >= 7 else "Normal"

        dashboard_data = {
            "pneumonia": round(pneumonia_prob, 2) if pneumonia_prob is not None else None,
            "asthma": round(asthma_prob, 2) if asthma_prob is not None else None,
            "disease": disease,
            "risk": round(min(max(risk3, 1), 10), 2),
            # Fetch final detailed precautions
            "precautions": get_enhanced_precautions(risk3, is_final=True, disease=disease)
        }
        return render_template("dashboard.html", data=dashboard_data)

    return render_template("day3.html", xray_required=xray_required, risk=risk2_rounded, precautions=day2_precautions)

if __name__ == "__main__":
    app.run(debug=True)