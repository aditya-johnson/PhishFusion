# 🛡️ PhishFusion: Sample-Adaptive Multimodal Phishing Detection & Explainability Engine

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Accuracy 96.38%](https://img.shields.io/badge/Accuracy-96.38%25-brightgreen?style=for-the-badge)](https://github.com/aditya-johnson/PhishFusion)
[![F1-Score 0.9678](https://img.shields.io/badge/F1--Score-0.9678-blue?style=for-the-badge)](https://github.com/aditya-johnson/PhishFusion)
[![ROC-AUC 0.9956](https://img.shields.io/badge/ROC--AUC-0.9956-orange?style=for-the-badge)](https://github.com/aditya-johnson/PhishFusion)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**PhishFusion** is an end-to-end, publication-grade **Multimodal Phishing-Website Detection System** trained on the UCI Phishing Websites Dataset (11,055 samples $\times$ 30 features). 

Instead of relying solely on standard machine learning classifiers or static feature concatenation, PhishFusion introduces a **Novel Sample-Adaptive Softmax Attention Fusion Mechanism (`AdaptivePhishFusionMLP`)**. The system dynamically computes website-specific modality reliance weights $[g_{\text{URL}}(x), g_{\text{Domain}}(x), g_{\text{HTML}}(x)]$ for every individual input URL, paired with an instance-level **SHAP (SHapley Additive exPlanations)** attribution engine.

---

## 🌟 Key Technical Novelties

1. **Sample-Adaptive Feature Fusion**: Dynamically computes normalized Softmax gating weights for 3 functional security modalities:
   - 🌐 **URL / Address Bar Modality** (12 features: IP usage, length, shorteners, `@` symbol, double slash redirects, subdomains, SSL state)
   - 🔒 **Domain & Security Modality** (9 features: external request ratios, anchor URLs, SFH, DNS records, domain age)
   - ⚡ **Page & HTML Behavior Modality** (9 features: redirects, mouseover events, iframe presence, traffic rank, blacklists)
2. **Instance-Level SHAP Explainability Engine**: Translates complex neural attention into clear human-understandable explanations, highlighting the exact positive risk drivers and negative safety drivers for any URL.
3. **Rigorous Statistical Verification**: Proven significantly superior to conventional stacking ensembles ($p = 4.01 \times 10^{-5}$, Cohen's $d = 2.347$) across 10-Fold Stratified Cross-Validation.

---

## 🧠 Multimodal Architecture

```
┌──────────────────────────────────────┐
│  URL & Address Bar Modality (12)     │──► [Branch URL: 32 -> 16] ────────┐
└──────────────────────────────────────┘                                  │
┌──────────────────────────────────────┐                                  │
│  Domain & Security Modality (9)      │──► [Branch Domain: 32 -> 16] ────┼──► [Concatenation (48)] ──► [Softmax Attention Gate] ──► [g_url, g_domain, g_html]
└──────────────────────────────────────┘                                  │                                                                 │
┌──────────────────────────────────────┐                                  │                                                                 ▼
│  Page & HTML Behavior Modality (9)   │──► [Branch HTML: 32 -> 16] ──────┘                                                   Adaptively Weighted Representation
└──────────────────────────────────────┘                                                                                                    │
                                                                                                                                            ▼
                                                                                                                               Phishing Risk Probability %
```

---

## 📊 Empirical Benchmarks

### 1. Model Comparison Across Classifiers & Fusion Strategies

| Model / Architecture | 5-Fold CV Acc | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **LightGBM (Early Fusion)** | **96.28%** | **96.83%** | **96.78%** | **97.56%** | **0.9717** | **0.9960** | **0.0831** |
| 🥈 **XGBoost (Early Fusion)** | 96.17% | 96.65% | 96.17% | 97.89% | 0.9702 | 0.9958 | 0.0867 |
| 🥉 **Sample-Adaptive Fusion (Proposed)** | 95.65% | **96.38%** | 96.11% | 96.78% | **0.9678** | **0.9956** | 0.0959 |
| **CatBoost** | 96.19% | 96.56% | 96.24% | 97.64% | 0.9694 | 0.9959 | 0.0866 |
| **Random Forest** | 95.93% | 96.16% | 95.19% | 98.05% | 0.9660 | 0.9959 | 0.1099 |
| **Early Fusion (Dense Neural MLP)** | 95.96% | 95.93% | 95.68% | 97.08% | 0.9637 | 0.9950 | 0.0882 |
| **Late Fusion (Stacking Ensemble)** | 93.93% | 94.17% | 93.59% | 96.10% | 0.9483 | 0.9854 | 0.1516 |
| **Logistic Regression (Baseline)** | 92.84% | 92.85% | 92.34% | 95.04% | 0.9367 | 0.9808 | 0.1749 |

---

### 2. 7-Way Modality Ablation Study

| Ablation Configuration | Feature Count | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sample-Adaptive Fusion (Proposed)** | **30** | **96.38%** | **96.11%** | **96.78%** | **0.9678** | **0.9956** |
| **Dual: URL + Domain** | 21 | 95.21% | 95.12% | 96.41% | 0.9576 | 0.9925 |
| **Dual: URL + Page/HTML** | 21 | 92.90% | 92.81% | 94.38% | 0.9359 | 0.9814 |
| **Dual: Domain + Page/HTML** | 18 | 92.09% | 91.95% | 94.26% | 0.9309 | 0.9796 |
| **URL Modality Only** | 12 | 91.00% | 90.54% | 93.77% | 0.9213 | 0.9633 |
| **Domain Modality Only** | 9 | 88.83% | 88.21% | 92.62% | 0.9036 | 0.9563 |
| **Page/HTML Modality Only** | 9 | 72.27% | 71.85% | 79.17% | 0.7533 | 0.7703 |

---

### 3. 10-Fold Cross-Validation Statistical Significance Testing

| Baseline Classifier | Mean 10-Fold Acc | Std Dev | Paired $t$-stat | $t$-test $p$-value | Wilcoxon $p$-value | Cohen's $d$ Effect | 95% CI Difference |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sample-Adaptive Fusion** | **0.9566** | **0.0066** | **0.000** | **1.0000** | **1.0000** | **0.000** | **[0.0000, 0.0000]** |
| **Late Fusion Stacking** | 0.9404 | 0.0052 | **+7.422** | **$4.01 \times 10^{-5}$** | **0.0020** | **+2.347** | **[+0.0119, +0.0204]** |
| **Logistic Regression** | 0.9271 | 0.0059 | **+10.254** | **$2.90 \times 10^{-6}$** | **0.0020** | **+3.243** | **[+0.0239, +0.0352]** |

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/aditya-johnson/PhishFusion.git
cd PhishFusion

# Install required dependencies
pip install -r requirements.txt
```

### 2. Execute Training & Benchmark Pipeline

```bash
python main.py
```

### 3. Launch Streamlit Web UI

```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
PhishFusion/
├── data/
│   ├── phishing+websites.zip
│   └── phishing_websites.csv
├── src/
│   ├── data_loader.py            # Dataset loading, 3-modality decomposition, scaler
│   ├── models.py                 # AdaptivePhishFusionMLP PyTorch architecture & baselines
│   ├── url_feature_extractor.py  # Parses raw URL links into 30 numerical features
│   ├── explainability.py         # SHAP feature & modality attribution engine
│   ├── ablation.py               # 7-way modality ablation study runner
│   ├── statistical_tests.py      # 10-fold paired t-test & Cohen's d calculator
│   ├── train.py                  # Stratified 5-fold CV training pipeline
│   ├── evaluate.py               # Performance metric computation & plot rendering
│   └── predict.py                # Single/batch URL inference predictor
├── saved_models/
│   ├── sample_adaptive_neural_fusion_proposed.joblib
│   ├── lightgbm.joblib
│   ├── xgboost.joblib
│   ├── catboost.joblib
│   └── scaler.pkl
├── results/
│   ├── metrics_summary.csv
│   ├── ablation_study.csv
│   ├── statistical_significance.csv
│   ├── model_comparison_bar.png
│   ├── roc_pr_curves.png
│   ├── confusion_matrices.png
│   ├── ablation_study.png
│   └── statistical_cv_boxplot.png
├── app.py                        # Streamlit web application
├── main.py                       # Single-command pipeline executor
├── requirements.txt              # Dependency specifications
└── README.md                     # Project documentation
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
