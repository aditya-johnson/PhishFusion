import os
import joblib
import numpy as np
import pandas as pd
from src.data_loader import ALL_FEATURES, MODALITIES
from src.explainability import PhishFusionExplainer

SAVED_MODELS_DIR = "/Users/adityajohnson/Documents/Christian Grey/PhishFusion/saved_models"


class PhishFusionPredictor:
    """Inference engine with Dynamic Attention Weights & SHAP Feature Explanations."""
    def __init__(self, model_filename="sample_adaptive_neural_fusion_proposed.joblib"):
        model_path = os.path.join(SAVED_MODELS_DIR, model_filename)
        scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")

        if not os.path.exists(model_path):
            model_path = os.path.join(SAVED_MODELS_DIR, "lightgbm.joblib")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.explainer = PhishFusionExplainer(model=self.model)

    def predict_sample(self, feature_dict_or_vector, explain=True):
        """
        Accepts a 30-feature vector or dict, scales features, and returns:
        - prediction ('Phishing' vs 'Legitimate')
        - phishing risk percentage
        - sample-adaptive attention weights [g_url, g_domain, g_html]
        - SHAP feature importance attributions & top drivers
        """
        if isinstance(feature_dict_or_vector, dict):
            vector = [feature_dict_or_vector[col] for col in ALL_FEATURES]
        else:
            vector = feature_dict_or_vector

        vector = np.array(vector).reshape(1, -1)
        scaled_vector = self.scaler.transform(vector)

        # Predict probability
        prob_phishing = float(self.model.predict_proba(scaled_vector)[0, 1])
        pred_label = int(prob_phishing >= 0.5)

        # Risk level assessment
        if prob_phishing < 0.25:
            risk_level = "LOW RISK (Legitimate Website)"
        elif prob_phishing < 0.50:
            risk_level = "MODERATE RISK (Caution Advised)"
        elif prob_phishing < 0.80:
            risk_level = "HIGH RISK (Probable Phishing)"
        else:
            risk_level = "CRITICAL RISK (Confirmed Phishing Threat)"

        # Get dynamic sample-adaptive attention weights if supported by model
        if hasattr(self.model, "predict_attention_weights"):
            attn_weights = self.model.predict_attention_weights(scaled_vector)[0]
            dynamic_weights = {
                "url_address_weight": round(float(attn_weights[0]), 4),
                "domain_security_weight": round(float(attn_weights[1]), 4),
                "page_html_weight": round(float(attn_weights[2]), 4)
            }
        else:
            # Equal baseline weighting
            dynamic_weights = {
                "url_address_weight": 0.3333,
                "domain_security_weight": 0.3333,
                "page_html_weight": 0.3333
            }

        # Compute SHAP explanation if requested
        explanation = {}
        if explain:
            explanation = self.explainer.explain_sample(scaled_vector, top_k=5)

        return {
            "prediction": "Phishing" if pred_label == 1 else "Legitimate",
            "prediction_code": pred_label,
            "phishing_probability": round(prob_phishing, 4),
            "risk_percentage": round(prob_phishing * 100, 2),
            "risk_level": risk_level,
            "sample_adaptive_weights": dynamic_weights,
            "explanation": explanation
        }


if __name__ == "__main__":
    predictor = PhishFusionPredictor()
    sample_phishing = [1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1]
    res = predictor.predict_sample(sample_phishing)
    print("Sample Inference Result:")
    print("Prediction:", res["prediction"])
    print("Adaptive Weights:", res["sample_adaptive_weights"])
    print("Top Positive Drivers:", res["explanation"]["top_positive_risk_drivers"])
