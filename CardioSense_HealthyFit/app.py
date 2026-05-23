# =============================================================
#  app.py  —  Heart Disease Risk Prediction System
#  Flask backend — loads all 4 ensemble models
# =============================================================

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# ── Feature order MUST match training ────────────────────────
FEATURE_ORDER = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
    'smoking', 'diabetes', 'bmi'
]

# ── Load all models + scaler at startup ──────────────────────
MODEL_DIR = 'model'
scaler = None
rf_model = gb_model = ada_model = voting_model = None

try:
    scaler        = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    rf_model      = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
    gb_model      = joblib.load(os.path.join(MODEL_DIR, 'gb_model.pkl'))
    ada_model     = joblib.load(os.path.join(MODEL_DIR, 'ada_model.pkl'))
    voting_model  = joblib.load(os.path.join(MODEL_DIR, 'heart_model.pkl'))
    print("✅ All 4 models and scaler loaded.")
except FileNotFoundError as e:
    print(f"⚠️  Model file missing: {e}. Run train_model.py first.")

# ── Load stored CV/test metrics ───────────────────────────────
METRICS = {}
try:
    with open(os.path.join(MODEL_DIR, 'metrics.json')) as f:
        METRICS = json.load(f)
except FileNotFoundError:
    pass

# ── Model registry (label → model object) ────────────────────
MODELS = {
    "Random Forest":     rf_model,
    "Gradient Boosting": gb_model,
    "AdaBoost":          ada_model,
    "Voting Classifier": voting_model,
}

BEST_MODEL_NAME = "Voting Classifier"   # highest ROC-AUC & Accuracy

# ── Color palette per model (matches CSS classes) ────────────
MODEL_COLORS = {
    "Random Forest":     "cyan",
    "AdaBoost":          "purple",
    "Voting Classifier": "green",
    "Gradient Boosting": "yellow",
}


# ── Personalised recommendations ─────────────────────────────
def get_recommendations(features: dict, risk_high: bool) -> list:
    recs = []
    if risk_high:
        recs.append({"icon": "bi-hospital", "title": "Consult a Cardiologist",
                     "text": "Your risk profile warrants an immediate appointment with a cardiologist for a thorough cardiac evaluation."})
    if features['chol'] > 200:
        recs.append({"icon": "bi-droplet-half", "title": "Manage Cholesterol",
                     "text": f"Your cholesterol ({features['chol']} mg/dl) is elevated. Reduce saturated fats, increase fiber intake, and discuss statins with your doctor."})
    if features['trestbps'] > 120:
        recs.append({"icon": "bi-activity", "title": "Control Blood Pressure",
                     "text": f"Your resting BP ({features['trestbps']} mmHg) is above optimal. Limit sodium, exercise regularly, and monitor BP daily."})
    if features['smoking'] == 1:
        recs.append({"icon": "bi-wind", "title": "Quit Smoking",
                     "text": "Smoking is a major cardiac risk factor. Quitting can reduce your heart disease risk by up to 50% within one year."})
    if features['bmi'] > 25:
        recs.append({"icon": "bi-person-walking", "title": "Healthy Weight Management",
                     "text": f"Your BMI ({features['bmi']:.1f}) is above the healthy range. Aim for 150 min/week of moderate aerobic activity and a balanced diet."})
    if features['diabetes'] == 1:
        recs.append({"icon": "bi-clipboard2-pulse", "title": "Diabetes Management",
                     "text": "Diabetes significantly increases cardiac risk. Maintain HbA1c < 7%, monitor blood glucose, and follow your treatment plan."})
    if features['thalach'] < 120:
        recs.append({"icon": "bi-heart-pulse", "title": "Improve Cardiovascular Fitness",
                     "text": "Your maximum heart rate achieved is low, indicating reduced cardiovascular fitness. Gradually increase aerobic exercise under medical supervision."})
    if not risk_high:
        recs.append({"icon": "bi-shield-check", "title": "Maintain Your Healthy Lifestyle",
                     "text": "Your current risk is low. Continue regular exercise (30 min/day), a heart-healthy diet, and annual health check-ups."})
        recs.append({"icon": "bi-calendar2-heart", "title": "Regular Screenings",
                     "text": "Even at low risk, schedule annual lipid panels, BP checks, and ECGs especially if you are over 40."})
    recs.append({"icon": "bi-moon-stars", "title": "Stress & Sleep",
                 "text": "Chronic stress and poor sleep directly raise cardiac risk. Practice mindfulness, maintain 7–8 hours of sleep, and seek support when needed."})
    return recs[:5]


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    if not all([scaler, rf_model, gb_model, ada_model, voting_model]):
        return render_template('predict.html',
                               error="Models not loaded. Please run train_model.py first.")

    try:
        # ── Validate: check every field is present and non-empty ──
        FIELD_LABELS = {
            'age': 'Age', 'sex': 'Gender', 'cp': 'Chest Pain Type',
            'trestbps': 'Resting Blood Pressure', 'chol': 'Serum Cholesterol',
            'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting ECG',
            'thalach': 'Max Heart Rate', 'exang': 'Exercise-Induced Angina',
            'oldpeak': 'ST Depression', 'slope': 'ST Slope',
            'ca': 'Major Vessels', 'thal': 'Thalassemia',
            'smoking': 'Smoking Status', 'diabetes': 'Diabetes Status', 'bmi': 'BMI',
        }
        missing = [FIELD_LABELS[f] for f in FIELD_LABELS if not request.form.get(f, '').strip()]
        if missing:
            return render_template('predict.html',
                                   error=f"Please fill in all fields. Missing: {', '.join(missing)}.")

        # Safe form parsing
        raw = {}

        for field in FEATURE_ORDER:
            value = request.form.get(field, '').strip()

            print(f"{field} = '{value}'")

            if value == '':
                return render_template(
                    'predict.html',
                    error=f"Field '{field}' is empty. Please select or enter a value."
                )

            try:
                raw[field] = float(value)
            except ValueError:
                return render_template(
                    'predict.html',
                    error=f"Invalid value entered for '{field}'."
                )

        input_df     = pd.DataFrame([raw])[FEATURE_ORDER]
        input_scaled = scaler.transform(input_df)

        # ── Run all 4 models ──────────────────────────────────
        model_results = []
        for name, mdl in MODELS.items():
            pred  = mdl.predict(input_scaled)[0]
            proba = mdl.predict_proba(input_scaled)[0]
            risk  = round(float(proba[1]) * 100, 1)
            safe  = round(float(proba[0]) * 100, 1)

            # Fetch stored test metrics
            stored = METRICS.get(name, {})
            model_results.append({
                "name":       name,
                "color":      MODEL_COLORS.get(name, "cyan"),
                "is_best":    name == BEST_MODEL_NAME,
                "prediction": "High Risk" if pred == 1 else "Low Risk",
                "is_high":    bool(pred == 1),
                "risk_pct":   risk,
                "safe_pct":   safe,
                # Stored test-set metrics (from training)
                "accuracy":   stored.get("accuracy",  0),
                "precision":  stored.get("precision", 0),
                "recall":     stored.get("recall",    0),
                "roc_auc":    stored.get("roc_auc",   0),
                "f1":         stored.get("f1",        0),
                "fit_status": stored.get("fit_status","healthy"),
                "cm_tp":      stored.get("cm_tp",     0),
                "cm_tn":      stored.get("cm_tn",     0),
                "cm_fp":      stored.get("cm_fp",     0),
                "cm_fn":      stored.get("cm_fn",     0),
            })

        # ── Smart voting: majority wins; best model breaks ties ──
        best = next(m for m in model_results if m["is_best"])
        high_votes = sum(1 for m in model_results if m["is_high"])
        low_votes  = len(model_results) - high_votes

        if high_votes != low_votes:
            # Clear majority
            majority_high = high_votes > low_votes
            if majority_high == best["is_high"]:
                # Best model agrees with majority → use best model proba
                risk_pct = best["risk_pct"]
                safe_pct = best["safe_pct"]
            else:
                # Best model is minority; use average proba of majority models
                majority_models = [m for m in model_results if m["is_high"] == majority_high]
                risk_pct = round(sum(m["risk_pct"] for m in majority_models) / len(majority_models), 1)
                safe_pct = round(100 - risk_pct, 1)
            is_high = majority_high
        else:
            # Tie (2 vs 2) → best model is the tiebreaker
            is_high  = best["is_high"]
            risk_pct = best["risk_pct"]
            safe_pct = best["safe_pct"]

        result_label = "High Risk" if is_high else "Low Risk"
        vote_summary = {
            "high_votes": high_votes,
            "low_votes":  low_votes,
            "is_tie":     high_votes == low_votes,
        }

        if risk_pct < 30:
            risk_tier = "low"
        elif risk_pct < 60:
            risk_tier = "moderate"
        else:
            risk_tier = "high"

        recommendations = get_recommendations(raw, is_high)

        patient = {
            'age'     : int(raw['age']),
            'sex'     : 'Male' if raw['sex'] == 1 else 'Female',
            'bmi'     : raw['bmi'],
            'chol'    : int(raw['chol']),
            'trestbps': int(raw['trestbps']),
            'smoking' : 'Yes' if raw['smoking'] == 1 else 'No',
            'diabetes': 'Yes' if raw['diabetes'] == 1 else 'No',
        }

        return render_template(
            'result.html',
            result_label    = result_label,
            risk_pct        = risk_pct,
            safe_pct        = safe_pct,
            is_high         = is_high,
            risk_tier       = risk_tier,
            recommendations = recommendations,
            patient         = patient,
            model_results   = model_results,
            best_model_name = BEST_MODEL_NAME,
            vote_summary    = vote_summary,
        )

    except (ValueError, KeyError) as e:
        return render_template('predict.html',
                               error=f"Invalid input: {str(e)}. Please fill in all fields correctly.")
    except Exception as e:
        return render_template('predict.html', error=f"Prediction error: {str(e)}")


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
