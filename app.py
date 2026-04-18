import os
import uuid
import numpy as np
import requests
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime,timedelta
import tensorflow.lite as tflite
from PIL import Image

app = Flask(__name__)
app.secret_key = "super_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- DATABASE CONFIG ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///respiratory_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=1)
# Database Models
class Patient(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4())[:8])
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    city = db.Column(db.String(100))
    smoking = db.Column(db.Integer)
    records = db.relationship('MedicalRecord', backref='patient', lazy=True)


class MedicalRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(36), db.ForeignKey('patient.id'), nullable=False)
    assessment_id = db.Column(db.String(36), nullable=False)  # Groups Day 1, 2, 3 together
    day = db.Column(db.Integer)
    symptoms = db.Column(db.JSON)
    env_data = db.Column(db.JSON)
    risk_score = db.Column(db.Float, nullable=True)
    disease_prediction = db.Column(db.String(50), nullable=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)  # Tracks when the record was made


# Create the database tables
with app.app_context():
    db.create_all()

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===============================
# LOAD MODELS - ABSOLUTE PATHS
# ===============================
RISK_MODEL_PATH = "/home/Khushboo12/RespAI/respai/models/risk_model.pkl"
XRAY_MODEL_PATH = "/home/Khushboo12/RespAI/respai/models/pneumonia_model.tflite"

risk_model = None
interpreter = None

try:
    # 1. Load Risk Model
    if os.path.exists(RISK_MODEL_PATH):
        risk_model = joblib.load(RISK_MODEL_PATH)
        print("✅ SUCCESS: Risk model loaded successfully!")
    else:
        print(f"❌ ERROR: Risk model not found at {RISK_MODEL_PATH}")

    # 2. Load TFLite Interpreter for X-Ray
    if os.path.exists(XRAY_MODEL_PATH):
        interpreter = tflite.Interpreter(model_path=XRAY_MODEL_PATH)
        interpreter.allocate_tensors()
        print("✅ SUCCESS: TFLite model loaded successfully!")
    else:
        print(f"❌ ERROR: X-Ray model not found at {XRAY_MODEL_PATH}")

except Exception as e:
    print(f"❌ CRITICAL ERROR during model initialization: {e}")

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
    if interpreter is None:
        print("❌ Interpreter not initialized")
        return None
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        with Image.open(path) as img:
            img = img.convert('RGB').resize((224, 224))
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        return float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return None


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


def get_enhanced_precautions(risk, is_final=False, disease="Normal"):
    precautions = []
    if risk < 4:
        precautions = ["Get plenty of rest and sleep.", "Stay well-hydrated with warm fluids.",
                       "Monitor your temperature and symptoms daily.",
                       "Maintain good hand hygiene and wear a mask if coughing."]
    elif risk < 7:
        precautions = ["Isolate yourself from vulnerable family members.",
                       "Use a humidifier or inhale steam to ease breathing.",
                       "Monitor your blood oxygen levels using a pulse oximeter.",
                       "Avoid cold beverages, dust, and smoking.",
                       "Consult a healthcare provider if breathlessness increases."]
    else:
        precautions = ["Get a diagnostic chest X-ray as soon as possible.",
                       "Do not self-medicate for severe breathlessness.",
                       "Keep emergency contact numbers readily available.",
                       "Restrict all physical exertion and stay in bed."]

    if is_final and disease == "Abnormal":
        precautions.insert(0,
                           "🚨 URGENT: Your diagnostic results indicate an abnormal respiratory condition. Please consult a pulmonologist or visit a medical clinic immediately.")

    return precautions


# ===============================
# STRICT ROUTING & LOGIC
# ===============================
@app.route("/")
def index():
    # Pass hide_sidebar=True so we can hide the instruction panel on the login page
    return render_template("index.html", hide_sidebar=True)


@app.route("/login", methods=["POST"])
def login():
    patient_id = request.form.get("patient_id")
    patient = Patient.query.get(patient_id)

    if not patient:
        flash("Patient ID not found. Please check and try again, or start a new assessment.", "danger")
        return redirect(url_for('index'))

    session['patient_id'] = patient.id
    flash(f"Welcome back, {patient.name}!", "success")

    # Smart Routing: Find their most recent assessment cycle
    latest_record = MedicalRecord.query.filter_by(patient_id=patient.id).order_by(MedicalRecord.date.desc()).first()

    if not latest_record:
        return redirect(url_for('day1'))

    session['assessment_id'] = latest_record.assessment_id

    # Resume exactly where they left off in their current assessment
    if latest_record.day == 3:
        return redirect(url_for('dashboard'))
    elif latest_record.day == 2:
        return redirect(url_for('day3'))
    elif latest_record.day == 1:
        return redirect(url_for('day2'))


@app.route("/day1", methods=["GET", "POST"])
def day1():
    patient_id = session.get('patient_id')
    existing_patient = Patient.query.get(patient_id) if patient_id else None

    if request.method == "POST":
        symptoms = extract_symptoms(request.form)
        city = existing_patient.city if existing_patient else request.form.get("city")

        if not existing_patient:
            # 1. Create NEW Patient Profile
            name = request.form.get("name", "Unknown Patient")
            existing_patient = Patient(
                name=name, age=int(request.form.get("age")),
                city=city, smoking=int(request.form.get("smoking"))
            )
            db.session.add(existing_patient)
            db.session.commit()
            session['patient_id'] = existing_patient.id
            flash(f"Your Patient ID is {existing_patient.id}. PLEASE SAVE THIS ID.", "success")

        # NEW: Create a unique ID for this specific 3-day assessment cycle
        assessment_id = str(uuid.uuid4())
        session['assessment_id'] = assessment_id

        temperature, humidity = get_weather(city)
        aqi = get_aqi(city)

        # 2. Create Day 1 Medical Record attached to this specific assessment
        day1_record = MedicalRecord(
            patient_id=existing_patient.id,
            assessment_id=assessment_id,
            day=1, symptoms=symptoms,
            env_data={"temperature": temperature, "humidity": humidity, "aqi": aqi}
        )
        db.session.add(day1_record)
        db.session.commit()
        return redirect(url_for('daily_complete', day=1))

    return render_template("day1.html", patient=existing_patient)


@app.route("/day2", methods=["GET", "POST"])
def day2():
    # STRICT LOCKING: Must be logged in AND have an active assessment
    patient_id = session.get('patient_id')
    assessment_id = session.get('assessment_id')
    if not patient_id or not assessment_id:
        flash("Unauthorized access. Please log in or start an assessment first.", "danger")
        return redirect(url_for('index'))

    # Prevent skipping Day 1
    day1_record = MedicalRecord.query.filter_by(assessment_id=assessment_id, day=1).first()
    if not day1_record:
        return redirect(url_for('day1'))

    # If Day 2 is already done, safely push them to Day 3
    if MedicalRecord.query.filter_by(assessment_id=assessment_id, day=2).first():
        return redirect(url_for('day3'))

    if request.method == "POST":
        symptoms = extract_symptoms(request.form)

        # NEW: Calculate Risk 2 immediately so we can show it on the logout screen!
        d1 = {**day1_record.env_data, **day1_record.symptoms}
        d2 = {**day1_record.env_data, **symptoms}
        f2 = build_features(d1, d2)

        if risk_model is None:
            flash("AI analysis is temporarily unavailable (Model Load Error).", "danger")
            return redirect(url_for('index'))


        risk2 = float(risk_model.predict(f2)[0])

        # Save the risk_score to the database
        day2_record = MedicalRecord(patient_id=patient_id, assessment_id=assessment_id, day=2, symptoms=symptoms,
                                    risk_score=risk2)
        db.session.add(day2_record)
        db.session.commit()

        return redirect(url_for('daily_complete', day=2))

    return render_template("day2.html",
                           precautions=["Stay hydrated.", "Monitor your symptoms closely.", "Eat a balanced diet."])

@app.route("/day3", methods=["GET", "POST"])
def day3():
    patient_id = session.get('patient_id')
    assessment_id = session.get('assessment_id')
    if not patient_id or not assessment_id:
        flash("Unauthorized access. Please log in first.", "danger")
        return redirect(url_for('index'))

    # Check previous days to prevent tab juggling
    day1_record = MedicalRecord.query.filter_by(assessment_id=assessment_id, day=1).first()
    day2_record = MedicalRecord.query.filter_by(assessment_id=assessment_id, day=2).first()

    if not day1_record: return redirect(url_for('day1'))
    if not day2_record: return redirect(url_for('day2'))

    # Prevent doing Day 3 twice
    if MedicalRecord.query.filter_by(assessment_id=assessment_id, day=3).first():
        return redirect(url_for('dashboard'))

    d1 = {**day1_record.env_data, **day1_record.symptoms}
    d2 = {**day1_record.env_data, **day2_record.symptoms}

    f2 = build_features(d1, d2)
    risk2 = float(risk_model.predict(f2)[0])
    risk2_rounded = round(min(max(risk2, 1), 10), 2)
    xray_required = risk2_rounded >= 7

    if request.method == "POST":
        d3_symptoms = extract_symptoms(request.form)
        d3 = {**day1_record.env_data, **d3_symptoms}

        f3 = build_features(d2, d3)
        if risk_model is None:
            flash("AI analysis is temporarily unavailable.", "danger")
            return redirect(url_for('index'))

        # Calculate final 3-day trend risk
        risk3 = float(risk_model.predict(f3)[0])
        risk3_norm = min(max(risk3, 1), 10) # Keep it in 1-10 range
        symptom_severity = risk3_norm / 10.0 # Convert to 0.0 - 1.0 multiplier

        pneumonia_prob = None
        asthma_prob = None
        disease = "Normal"

        xray_file = request.files.get("xray")

        # --- SCENARIO 1: X-RAY UPLOADED ---
        if xray_file and xray_file.filename != '':
            filename = secure_filename(xray_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            xray_file.save(filepath)

            raw_val = predict_xray(filepath)

            if raw_val is not None:
                p_xray = float(raw_val) # X-Ray AI confidence (0.0 to 1.0)

                # Condition A: X-Ray detects Pneumonia (High Confidence)
                if p_xray >= 0.5:
                    # Pneumonia is high. Weighted 75% on X-ray, 25% on Symptoms
                    pneumonia_prob = (p_xray * 0.75) + (symptom_severity * 0.25)
                    # Asthma is pushed down because the symptoms are explained by pneumonia
                    asthma_prob = symptom_severity * 0.2 * (1.0 - p_xray)

                # Condition B: X-Ray is Normal (Lungs are Clear)
                else:
                    # Pneumonia is forced to stay low
                    pneumonia_prob = (p_xray * 0.8) + (symptom_severity * 0.2)
                    # Asthma shoots up if symptoms are high (because lungs are clear!)
                    asthma_prob = symptom_severity * 0.85

                # Set Disease Status based on combined math
                if pneumonia_prob >= 0.5 or asthma_prob >= 0.5 or risk3_norm >= 7:
                    disease = "Abnormal"
                else:
                    disease = "Normal"
            else:
                # If X-Ray AI crashes, fall back to "No Image" logic
                disease = "Abnormal" if risk3_norm >= 7 else "Normal"

        # --- SCENARIO 2: NO X-RAY UPLOADED ---
        else:
            # Leave probabilities as None so they don't show on the dashboard
            pneumonia_prob = None
            asthma_prob = None

            # Solely rely on Risk Score for status
            disease = "Abnormal" if risk3_norm >= 7 else "Normal"

        # Save to Database
        day3_record = MedicalRecord(
            patient_id=patient_id, assessment_id=assessment_id, day=3,
            symptoms=d3_symptoms, risk_score=risk3_norm, disease_prediction=disease,
            env_data={"pneumonia_prob": pneumonia_prob, "asthma_prob": asthma_prob}
        )
        db.session.add(day3_record)
        db.session.commit()
        return redirect(url_for('dashboard'))

    # CORRECT LINE
    return render_template("day3.html", xray_required=xray_required, risk=risk2_rounded, precautions=get_enhanced_precautions(risk2_rounded))


@app.route("/daily_complete")
def daily_complete():
    day = request.args.get('day', 1, type=int)
    assessment_id = session.get('assessment_id')

    # Default Basic Precautions for Day 1
    precautions = [
        "Stay hydrated and get plenty of rest.",
        "Monitor your symptoms closely.",
        "Eat a balanced diet to support your immune system."
    ]
    risk = None

    # If it's Day 2, fetch the risk score we just calculated and get enhanced precautions
    if assessment_id and day == 2:
        day2_record = MedicalRecord.query.filter_by(assessment_id=assessment_id, day=2).first()
        if day2_record and day2_record.risk_score is not None:
            risk = round(min(max(day2_record.risk_score, 1), 10), 2)
            precautions = get_enhanced_precautions(risk)

    return render_template("daily_complete.html", day=day, precautions=precautions, risk=risk)

@app.route("/dashboard")
def dashboard():
    patient_id = session.get('patient_id')
    assessment_id = session.get('assessment_id')
    if not patient_id or not assessment_id:
        return redirect(url_for('index'))

    patient = Patient.query.get(patient_id)
    day3_record = MedicalRecord.query.filter_by(assessment_id=assessment_id, day=3).first()

    if not day3_record:
        return redirect(url_for('day3'))

    # Safely extract the probabilities from the database JSON
    p_val = day3_record.env_data.get('pneumonia_prob')
    a_val = day3_record.env_data.get('asthma_prob')

    data = {
        "patient_id": patient.id,
        "patient_name": patient.name,

        # Multiply by 100 to make it a percentage (e.g., 0.85 -> 85.0)
        # We only do the math if p_val and a_val actually exist (are not None)
        "pneumonia": round(p_val * 100, 1) if p_val is not None else None,
        "asthma": round(a_val * 100, 1) if a_val is not None else None,

        "disease": day3_record.disease_prediction,
        "risk": round(day3_record.risk_score, 2),
        "precautions": get_enhanced_precautions(day3_record.risk_score, True, day3_record.disease_prediction)
    }
    return render_template("dashboard.html", data=data)


@app.route("/history")
def history():
    # Only show history if logged in
    patient_id = session.get('patient_id')
    if not patient_id:
        return redirect(url_for('index'))

    patient = Patient.query.get(patient_id)
    # Fetch all completed Day 3 records for this patient to show their past assessments
    past_assessments = MedicalRecord.query.filter_by(patient_id=patient_id, day=3).order_by(
        MedicalRecord.date.desc()).all()

    return render_template("history.html", patient=patient, records=past_assessments)


@app.route("/restart")
def restart():
    # WE NO LONGER DELETE RECORDS! We just clear the current assessment ID so they start fresh.
    session.pop('assessment_id', None)
    return redirect(url_for('day1'))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been successfully logged out.", "info")
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)
