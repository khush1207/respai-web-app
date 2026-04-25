# RespAI - Respiratory Disease Detection System

**RespAI** is an AI-powered web application built to help detect respiratory diseases like Pneumonia and Asthma early. It was designed specifically for rural areas where immediate access to X-rays and specialist doctors is difficult. 

Instead of requiring an immediate X-ray, the system tracks a patient's symptoms over 3 days and uses AI to analyze X-ray images only when necessary.

## 🌟 Features
* **3-Day Symptom Tracking:** Monitors how the patient's symptoms change over time.
* **X-Ray AI Analysis:** Uses a lightweight AI model to detect signs of pneumonia.
* **Multi-Language Support:** Easily translates the website into multiple regional languages.
* **Local Weather Data:** Uses live AQI (Air Quality Index) to factor in environmental risks.

## 🛠️ Tech Stack
* **Backend:** Python, Flask
* **AI Models:** TensorFlow Lite (X-rays), Scikit-Learn (Symptoms)
* **Database:** SQLite
* **Frontend:** HTML, Bootstrap
Updated by Khushi