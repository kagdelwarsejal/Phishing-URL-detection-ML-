# explain.py

import shap
import numpy as np

# -------------------------
# Initialize SHAP
# -------------------------
def get_shap_explainer(model, background_data):
    return shap.TreeExplainer(model, background_data)

# -------------------------
# SHAP Explanation
# -------------------------
def shap_explanation(explainer, features_df):
    shap_values = explainer.shap_values(features_df)

    return shap_values

# -------------------------
# Human-Readable SHAP Text
# -------------------------
def shap_to_text(shap_values, features_df, top_n=3):
    feature_names = features_df.columns
    shap_vals = shap_values[0]

    importance = list(zip(feature_names, shap_vals))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)

    explanations = []

    for feat, val in importance[:top_n]:
        if val > 0:
            explanations.append(f"{feat.replace('_',' ')} increases phishing risk")
        else:
            explanations.append(f"{feat.replace('_',' ')} reduces phishing risk")

    return explanations
