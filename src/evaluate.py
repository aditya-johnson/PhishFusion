import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, roc_curve, precision_recall_curve
)

RESULTS_DIR = "/Users/adityajohnson/Documents/Christian Grey/PhishFusion/results"


def evaluate_and_generate_plots(data, results):
    """Computes all classification metrics and generates clean publication-ready figures."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    y_test = data["y_test"].values

    metrics_list = []

    for name, res in results.items():
        y_pred = res["y_pred"]
        y_prob = res["y_prob"]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        loss = log_loss(y_test, y_prob)
        cv_acc = res["cv_acc_mean"]

        metrics_list.append({
            "Model": name,
            "CV Accuracy": round(cv_acc, 4),
            "Test Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Log Loss": round(loss, 4)
        })

    df_metrics = pd.DataFrame(metrics_list).sort_values(by="Test Accuracy", ascending=False)
    
    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nSaved evaluation metrics summary to {csv_path}")

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Bar Chart Comparison
    plt.figure(figsize=(14, 7))
    metrics_to_plot = ["Test Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    df_plot = df_metrics.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    ax = sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.title("PhishFusion Performance Comparison Across All Models", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Model / Architecture", fontsize=12, fontweight='bold')
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12, fontweight='bold')
    plt.ylim(0.85, 1.0)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.legend(title="Metric", loc="lower right", frameon=True)
    plt.tight_layout()
    bar_chart_path = os.path.join(RESULTS_DIR, "model_comparison_bar.png")
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()

    # 2. ROC & PR Curves Overlay
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, res), color in zip(results.items(), colors):
        y_prob = res["y_prob"]
        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})", color=color, linewidth=2)
        
        # PR
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        axes[1].plot(rec_curve, prec_curve, label=f"{name}", color=color, linewidth=2)

    axes[0].plot([0, 1], [0, 1], 'k--', linestyle='--', alpha=0.5)
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].set_title("Precision-Recall (PR) Curves", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "roc_pr_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    # 3. Confusion Matrix Grid
    n_models = len(results)
    cols = 4
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
                    annot_kws={"size": 14, "weight": "bold"})
        axes[idx].set_title(f"{name}", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xticklabels(["Legitimate (0)", "Phishing (1)"])
        axes[idx].set_yticklabels(["Legitimate (0)", "Phishing (1)"])

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 4. Modality Feature Importance Plot (using XGBoost & Random Forest)
    best_tree_model = results.get("XGBoost", {}).get("model") or results.get("Random Forest", {}).get("model")
    if hasattr(best_tree_model, "feature_importances_"):
        importances = best_tree_model.feature_importances_
        features = data["X_train"].columns
        df_imp = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)
        
        # Modality color coding
        mod_map = {}
        from src.data_loader import MODALITIES
        for mod_name, f_list in MODALITIES.items():
            for f in f_list:
                mod_map[f] = mod_name
        df_imp["Modality"] = df_imp["Feature"].map(mod_map)

        plt.figure(figsize=(12, 8))
        sns.barplot(data=df_imp, x="Importance", y="Feature", hue="Modality", palette="Set2", dodge=False)
        plt.title("Feature Importance by Modality (XGBoost Early Fusion)", fontsize=15, fontweight='bold')
        plt.xlabel("Relative Importance Score", fontsize=12)
        plt.tight_layout()
        imp_path = os.path.join(RESULTS_DIR, "modality_importance.png")
        plt.savefig(imp_path, dpi=300)
        plt.close()

    print("\n--- Summary Performance Table ---")
    print(df_metrics.to_string(index=False))

    return df_metrics
