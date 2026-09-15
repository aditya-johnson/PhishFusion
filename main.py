import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import train_and_evaluate_all
from src.evaluate import evaluate_and_generate_plots
from src.ablation import run_ablation_study
from src.statistical_tests import run_statistical_significance_tests
from src.predict import PhishFusionPredictor


def main():
    print("==========================================================================")
    print("  PhishFusion V2: Adaptive Multimodal Phishing Detection & Explainability ")
    print("==========================================================================")

    # Step 1: Train all base and adaptive fusion models
    data, results = train_and_evaluate_all()

    # Step 2: Metrics Evaluation & Plotting
    print("\n--- [3/5] Generating Evaluation Metrics & Visualization Plots ---")
    df_metrics = evaluate_and_generate_plots(data, results)

    # Step 3: 7-Way Modality Ablation Study
    print("\n--- [4/5] Executing 7-Way Modality Ablation Study ---")
    df_ablation = run_ablation_study(data)

    # Step 4: 10-Fold Statistical Significance Testing
    print("\n--- [5/5] Performing 10-Fold Statistical Significance Testing ---")
    df_stat = run_statistical_significance_tests(data, n_splits=10)

    # Step 5: Verify Predictor & SHAP Explainability Engine
    print("\n--- Verification: Sample-Adaptive Weights & SHAP Explainability ---")
    predictor = PhishFusionPredictor(model_filename="sample_adaptive_neural_fusion_proposed.joblib")
    test_sample = data["X_test_unscaled"].iloc[0].to_dict()
    true_target = "Phishing" if data["y_test"].iloc[0] == 1 else "Legitimate"

    result = predictor.predict_sample(test_sample, explain=True)
    print(f"\nGroundtruth Label             : {true_target}")
    print(f"Predicted Output              : {result['prediction']}")
    print(f"Phishing Risk Score           : {result['risk_percentage']}%")
    print(f"Sample-Adaptive Weights g(x)  : {result['sample_adaptive_weights']}")
    print(f"Top Positive Risk Drivers     : {result['explanation']['top_positive_risk_drivers']}")

    print("\n==========================================================================")
    print("   PhishFusion V2 Execution Completed Successfully!")
    print("==========================================================================")


if __name__ == "__main__":
    main()
