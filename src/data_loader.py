import os
import io
import zipfile
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Define the 3 Modalities (Total 30 features)
MODALITIES = {
    "url_address": [
        "having_IP_Address",
        "URL_Length",
        "Shortining_Service",
        "having_At_Symbol",
        "double_slash_redirecting",
        "Prefix_Suffix",
        "having_Sub_Domain",
        "SSLfinal_State",
        "Domain_registeration_length",
        "Favicon",
        "port",
        "HTTPS_token"
    ],
    "domain_security": [
        "Request_URL",
        "URL_of_Anchor",
        "Links_in_tags",
        "SFH",
        "Submitting_to_email",
        "Abnormal_URL",
        "age_of_domain",
        "DNSRecord",
        "Google_Index"
    ],
    "page_html_behavior": [
        "Redirect",
        "on_mouseover",
        "RightClick",
        "popUpWidnow",
        "Iframe",
        "web_traffic",
        "Page_Rank",
        "Links_pointing_to_page",
        "Statistical_report"
    ]
}

ALL_FEATURES = (
    MODALITIES["url_address"] +
    MODALITIES["domain_security"] +
    MODALITIES["page_html_behavior"]
)

TARGET_COL = "Result"


def extract_and_load_data(zip_path, extract_dir):
    """Extracts UCI dataset zip if needed and loads clean DataFrame."""
    os.makedirs(extract_dir, exist_ok=True)
    csv_path = os.path.join(extract_dir, "phishing_websites.csv")
    arff_path = os.path.join(extract_dir, "Training Dataset.arff")

    if not os.path.exists(csv_path):
        if not os.path.exists(arff_path):
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            else:
                raise FileNotFoundError(f"Zip file not found at {zip_path}")
        
        # Parse ARFF
        with open(arff_path, 'r', encoding='utf-8') as f:
            data, meta = arff.loadarff(io.StringIO(f.read()))
        
        df = pd.DataFrame(data)
        for col in df.columns:
            df[col] = df[col].astype(int)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Extracted and saved dataset to {csv_path}")
    else:
        df = pd.read_csv(csv_path)

    # Remap target: -1 (Legitimate) -> 0, 1 (Phishing) -> 1
    if df[TARGET_COL].min() == -1:
        df[TARGET_COL] = df[TARGET_COL].map({-1: 0, 1: 1})

    return df


def get_prepared_data(df, test_size=0.2, random_state=42):
    """
    Splits dataset into stratified train and test sets, scales features,
    and returns full features as well as modality-specific splits.
    """
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=ALL_FEATURES, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=ALL_FEATURES, index=X_test.index)

    # Extract modality subsets
    modality_data = {
        "train": {mod: X_train_scaled[cols] for mod, cols in MODALITIES.items()},
        "test": {mod: X_test_scaled[cols] for mod, cols in MODALITIES.items()},
        "train_unscaled": {mod: X_train[cols] for mod, cols in MODALITIES.items()},
        "test_unscaled": {mod: X_test[cols] for mod, cols in MODALITIES.items()}
    }

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_unscaled": X_train,
        "X_test_unscaled": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "modality_data": modality_data
    }
