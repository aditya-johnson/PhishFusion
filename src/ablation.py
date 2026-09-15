import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from src.data_loader import MODALITIES, ALL_FEATURES
from src.models import PyTorchNeuralClassifier

RESULTS_DIR = "/Users/adityajohnson/Documents/Christian Grey/PhishFusion/results"


def run_ablation_study(data):
    """Executes a 7-way Modality Ablation Study comparing single, dual, and sample-adaptive tri-modality fusion."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    url_cols = MODALITIES["url_address"]
    domain_cols = MODALITIES["domain_security"]
    html_cols = MODALITIES["page_html_behavior"]

    ablation_configs = {
        "URL Modality Only": url_cols,
        "Domain Modality Only": domain_cols,
        "Page/HTML Modality Only": html_cols,
        "Dual: URL + Domain": url_cols + domain_cols,
        "Dual: URL + Page/HTML": url_cols + html_cols,
        "Dual: Domain + Page/HTML": domain_cols + html_cols,
        "Tri-Modality Early Fusion": ALL_FEATURES,
        "Sample-Adaptive Fusion (Proposed)": "ADAPTIVE"
    }

    ablation_results = []

    print("\n==========================================================================")
    print("           [Ablation Study] Running 7-Way Modality Evaluation            ")
    print("==========================================================================")

    for config_name, feature_subset in ablation_configs.items():
        if feature_subset == "ADAPTIVE":
            # Train PyTorch Adaptive Neural Classifier
            model = PyTorchNeuralClassifier(is_multimodal=True, is_adaptive=True, epochs=35, batch_size=64, lr=0.001)
            model.fit(X_train.values, y_train.values)
            y_pred = model.predict(X_test.values)
            y_prob = model.predict_proba(X_test.values)[:, 1]
            num_feats = 30
        else:
            X_tr = X_train[feature_subset]
            X_te = X_test[feature_subset]
            model = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
            num_feats = len(feature_subset)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"{config_name:<35} | Feats: {num_feats:2d} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")

        ablation_results.append({
            "Configuration": config_name,
            "Features Count": num_feats,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4)
        })

    df_ablation = pd.DataFrame(ablation_results)
    
    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "ablation_study.csv")
    df_ablation.to_csv(csv_path, index=False)
    print(f"\nSaved ablation study results to {csv_path}")

    # Plot Ablation Chart
    plt.figure(figsize=(13, 6))
    sns.barplot(data=df_ablation, x="Configuration", y="Accuracy", palette="crest")
    plt.title("Ablation Study: Impact of Security Modalities on Phishing Detection Accuracy", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Modality Feature Configuration", fontsize=11, fontweight='bold')
    plt.ylabel("Test Accuracy", fontsize=11, fontweight='bold')
    plt.ylim(0.85, 1.0)
    plt.xticks(rotation=25, ha="right")
    
    for idx, row in df_ablation.iterrows():
        plt.text(idx, row["Accuracy"] + 0.003, f"{row['Accuracy']:.4f}", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "ablation_study.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return df_ablation
