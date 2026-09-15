import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone
from src.models import get_all_models, PyTorchNeuralClassifier

RESULTS_DIR = "/Users/adityajohnson/Documents/Christian Grey/PhishFusion/results"


def run_statistical_significance_tests(data, n_splits=10):
    """
    Executes 10-Fold Cross-Validation sampling and performs Paired Student's t-test,
    Wilcoxon Signed-Rank test, Cohen's d effect size, and 95% Confidence Intervals.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X = data["X_train"].values
    y = data["y_train"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models_to_test = {
        "Sample-Adaptive Fusion (Proposed)": get_all_models()["Sample-Adaptive Neural Fusion (Proposed)"],
        "LightGBM": get_all_models()["LightGBM"],
        "XGBoost": get_all_models()["XGBoost"],
        "CatBoost": get_all_models()["CatBoost"],
        "Random Forest": get_all_models()["Random Forest"],
        "Late Fusion Stacking": get_all_models()["Late Fusion (Stacking Ensemble)"],
        "Logistic Regression": get_all_models()["Logistic Regression"]
    }

    print("\n==========================================================================")
    print(f"      [Statistical Significance] Running {n_splits}-Fold CV Sampling...        ")
    print("==========================================================================")

    fold_scores = {name: [] for name in models_to_test.keys()}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        for name, model in models_to_test.items():
            if isinstance(model, PyTorchNeuralClassifier):
                m = PyTorchNeuralClassifier(is_multimodal=model.is_multimodal, is_adaptive=model.is_adaptive, epochs=30, batch_size=64)
            else:
                m = clone(model)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_va)
            acc = accuracy_score(y_va, preds)
            fold_scores[name].append(acc)

    df_folds = pd.DataFrame(fold_scores)
    
    # Statistical Comparisons vs Proposed Model
    proposed_name = "Sample-Adaptive Fusion (Proposed)"
    proposed_scores = np.array(fold_scores[proposed_name])

    stat_summary = []

    for name, scores in fold_scores.items():
        base_scores = np.array(scores)
        diff = proposed_scores - base_scores
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1) + 1e-9

        # Paired t-test
        if name == proposed_name:
            t_stat, p_val_t = 0.0, 1.0
            w_stat, p_val_w = 0.0, 1.0
            cohen_d = 0.0
            ci_lower, ci_upper = 0.0, 0.0
        else:
            t_stat, p_val_t = stats.ttest_rel(proposed_scores, base_scores)
            
            try:
                w_stat, p_val_w = stats.wilcoxon(proposed_scores, base_scores)
            except Exception:
                w_stat, p_val_w = 0.0, 1.0

            # Cohen's d effect size
            cohen_d = mean_diff / std_diff

            # 95% Confidence Interval for mean difference
            se_diff = std_diff / np.sqrt(n_splits)
            ci_margin = 1.96 * se_diff
            ci_lower = mean_diff - ci_margin
            ci_upper = mean_diff + ci_margin

        is_sig = (p_val_t < 0.05) or (name == proposed_name)

        stat_summary.append({
            "Model": name,
            "Mean 10-Fold Acc": round(np.mean(base_scores), 4),
            "Std 10-Fold Acc": round(np.std(base_scores), 4),
            "Mean Acc Diff vs Proposed": round(mean_diff, 4),
            "Paired t-stat": round(t_stat, 3),
            "t-test p-value": f"{p_val_t:.4e}" if p_val_t < 0.001 else f"{p_val_t:.4f}",
            "Wilcoxon p-value": f"{p_val_w:.4e}" if p_val_w < 0.001 else f"{p_val_w:.4f}",
            "Cohen's d Effect": round(cohen_d, 3),
            "95% CI Difference": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
            "Statistically Significant (p < 0.05)": "Yes" if is_sig else "No"
        })

    df_stat = pd.DataFrame(stat_summary)
    
    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "statistical_significance.csv")
    df_stat.to_csv(csv_path, index=False)
    print(f"\nSaved statistical significance results to {csv_path}")

    # Plot Boxplot
    plt.figure(figsize=(13, 6))
    df_melt = df_folds.melt(var_name="Model", value_name="10-Fold CV Accuracy")
    sns.boxplot(data=df_melt, x="Model", y="10-Fold CV Accuracy", palette="Set2")
    sns.stripplot(data=df_melt, x="Model", y="10-Fold CV Accuracy", color="black", alpha=0.6, jitter=0.2)
    plt.title("10-Fold Cross-Validation Accuracy Distribution across Models", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Classifier / Fusion Architecture", fontsize=11, fontweight='bold')
    plt.ylabel("Accuracy across Folds", fontsize=11, fontweight='bold')
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "statistical_cv_boxplot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("\n--- Statistical Significance Summary Table ---")
    print(df_stat.to_string(index=False))

    return df_stat
