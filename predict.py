# predict.py

import joblib
import numpy as np
from feature_extractor import extract_features
from visualize import risk_meter, plot_url_features
from explain import get_shap_explainer, shap_explanation, shap_to_text


# -------------------------
# Load model & features
# -------------------------

MODEL_PATH = "catboost_phishing_model.pkl"
FEATURES_PATH = "feature_columns.pkl"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# -------------------------
# Prediction function
# -------------------------

def predict_url(url):
    features = extract_features(url)

    # Force correct feature order
    features = features[feature_columns]

    proba = model.predict_proba(features)[0]
    confidence = float(np.max(proba))
    prediction = "PHISHING" if np.argmax(proba) == 1 else "SAFE"

    explanation = generate_explanation(features.iloc[0], prediction)

    return {
        "url": url,
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "explanation": explanation
    }

# -------------------------
def predict_with_explain(url, background_data):
    features = extract_features(url)
    features = features[feature_columns]

    proba = model.predict_proba(features)[0]
    confidence = float(max(proba))
    prediction = "PHISHING" if np.argmax(proba) == 1 else "SAFE"

    # Risk level
    risk = risk_meter(confidence)

    # SHAP
    explainer = get_shap_explainer(model, background_data)
    shap_vals = shap_explanation(explainer, features)
    shap_text = shap_to_text(shap_vals, features)

    # Plot
    plot_url_features(features)

    return {
        "url": url,
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "risk_level": risk,
        "shap_explanation": shap_text
    }

# -------------------------

def generate_explanation(features, prediction):
    reasons = []

    if features['phish_hints'] > 0:
        reasons.append("URL contains suspicious keywords like login or verify")

    if features['nb_dots'] > 3:
        reasons.append("URL has an unusually high number of dots")

    if features['ip'] == 1:
        reasons.append("URL uses an IP address instead of a domain name")

    if features['shortening_service'] == 1:
        reasons.append("URL uses a known URL shortening service")

    if features['ratio_digits_url'] > 0.3:
        reasons.append("URL contains a high number of digits")

    if not reasons:
        reasons.append("URL structure appears normal")

    return reasons
