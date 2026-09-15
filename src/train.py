import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from src.data_loader import extract_and_load_data, get_prepared_data, ALL_FEATURES
from src.models import get_all_models, PyTorchNeuralClassifier

SAVED_MODELS_DIR = "/Users/adityajohnson/Documents/Christian Grey/PhishFusion/saved_models"
DATA_ZIP_PATH = "/Users/adityajohnson/Downloads/phishing+websites.zip"
DATA_EXTRACT_DIR = "/Users/adityajohnson/Documents/Christian Grey/PhishFusion/data"


def train_and_evaluate_all():
    """Trains all base and multimodal fusion models, runs 5-Fold CV, and saves models."""
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_EXTRACT_DIR, exist_ok=True)

    print("--- [1/4] Extracting & Loading UCI Dataset ---")
    df = extract_and_load_data(DATA_ZIP_PATH, DATA_EXTRACT_DIR)
    data = get_prepared_data(df, test_size=0.2, random_state=42)

    X_train = data["X_train"].values
    X_test = data["X_test"].values
    y_train = data["y_train"].values
    y_test = data["y_test"].values

    # Save scaler
    scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")
    joblib.dump(data["scaler"], scaler_path)
    print(f"Saved scaler to {scaler_path}")

    models = get_all_models()
    results = {}
    fitted_models = {}

    print("\n--- [2/4] Training Models & 5-Fold Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining model: {name} ...")
        
        # Cross-validation accuracy
        cv_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            clone_model = clone(model) if not isinstance(model, PyTorchNeuralClassifier) else get_all_models()[name]
            clone_model.fit(X_train[train_idx], y_train[train_idx])
            preds = clone_model.predict(X_train[val_idx])
            cv_scores.append(np.mean(preds == y_train[val_idx]))
        
        cv_acc_mean = np.mean(cv_scores)
        cv_acc_std = np.std(cv_scores)
        print(f" -> 5-Fold CV Accuracy: {cv_acc_mean:.4f} (+/- {cv_acc_std:.4f})")

        # Fit on full training set
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Test set predictions & probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Save model file
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        model_filename = f"{safe_name}.joblib"
        model_save_path = os.path.join(SAVED_MODELS_DIR, model_filename)
        joblib.dump(model, model_save_path)
        print(f" -> Saved model to {model_save_path}")

        results[name] = {
            "model": model,
            "model_path": model_save_path,
            "cv_acc_mean": cv_acc_mean,
            "cv_acc_std": cv_acc_std,
            "y_pred": y_pred,
            "y_prob": y_prob
        }

    return data, results


if __name__ == "__main__":
    train_and_evaluate_all()
