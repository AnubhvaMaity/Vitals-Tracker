# Vitals Tracker

A Streamlit-based personal health analytics app for tracking daily vitals and predicting risk for heart disease, hypertension, obesity, and diabetes.

## 🚀 Features

- Multi-day vitals tracking (systolic/diastolic BP, glucose, cholesterol, BMI, heart rate)
- Guided profile setup with BMI calculation or manual entry
- Daily logging with synced sliders + number inputs
- Risk estimation by rule-based logic + pre-trained ML models (`heart_model.joblib`, `diabetes_model.joblib`, `diabetes_scaler.joblib`)
- Downloadable PDF summary report (requires `reportlab`)
- Import Excel/CSV sessions
- Visual trend charts via Altair

## 🛠️ Prerequisites

- Python 3.8+
- Windows/macOS/Linux

## 📦 Install

```bash
cd "c:\Users\anubh\Downloads\Vitals Tracker"
python -m pip install -r requirements.txt
```

If using PDF export: `pip install reportlab`

## 🧠 Model files

Ensure these files are present in the same folder as `app.py`:

- `heart_model.joblib`
- `diabetes_model.joblib`
- `diabetes_scaler.joblib`

## ▶️ Run the app

```bash
streamlit run app.py
```

## 📁 Data persistence

- User data exported to `user_data/<session_id>_data.csv`

## 🧪 Training ready

Source training scripts include:

- `train_heart_model.py`
- `train_diabetes_model.py`

## 📋 Usage

1. Start a session and enter your name.
2. Set days to track (default 7).
3. Complete profile setup (age, sex, BMI mode).
4. Log daily vital values.
5. Review final risk report and download PDF.

## 🛡️ Notes

- Trained models are used for risk predictions but are not medical advice.
- The app is for personal tracking and should not replace clinical diagnosis.

## ❤️ Improvements

- Add more robust CSV import validation
- Include authentication/persistent user profiles
- Add symptom tracking and context-aware suggestions
