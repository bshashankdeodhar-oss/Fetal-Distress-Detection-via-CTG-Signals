# Multi-Family Model Benchmark & Held-Out Evaluation Report

## 1. Executive Benchmark Summary
- **Evaluation Split:** 20% Stratified Held-out Test Set (426 unseen patient recordings).
- **Primary Optimization Metric:** **Macro F1 Score** (Equal weight across Normal, Suspect, Pathologic).
- **Top Performing Classifier:** **XGBoost** (Family: *Gradient Boosted Trees*) with **Macro F1 = 0.8998** and **Pathologic Recall = 0.9143**.

## 2. Multi-Family Comparison Table
| Model | Family | Macro F1 | Macro Precision | Macro Recall | F1 (Pathologic) | Recall (Pathologic) | Balanced Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost | Gradient Boosted Trees | **0.8998** | 0.9228 | 0.8817 | 0.9143 | 0.9143 | 0.8817 |
| LightGBM | Gradient Boosted Trees | **0.8949** | 0.9043 | 0.8902 | 0.9041 | 0.9429 | 0.8902 |
| Random Forest | Bagged Tree Ensembles | **0.8810** | 0.8827 | 0.8793 | 0.9143 | 0.9143 | 0.8793 |
| Support Vector Machine (SVC RBF) | Kernel Margin Classifiers | **0.8245** | 0.7918 | 0.8786 | 0.8493 | 0.8857 | 0.8786 |
| Multi-Layer Perceptron (MLP Neural Net) | Deep Neural Networks | **0.7874** | 0.8040 | 0.7756 | 0.7188 | 0.6571 | 0.7756 |
| Cost-Sensitive Logistic Regression | Linear Probabilistic Models | **0.7796** | 0.7404 | 0.8488 | 0.7532 | 0.8286 | 0.8488 |

## 3. Key Clinical Findings & Explainability (SHAP)
1. **Top Predictive Biomarkers:**
   - **`ASTV` (% Abnormal Short-Term Variability):** Strongest non-linear driver of fetal distress. Values $>65\%$ almost exclusively push risk toward *Pathologic*.
   - **`VCR` (Variability Collapse Ratio - Engineered Feature):** Compounding degradation index amplifying acute autonomic dysfunction.
   - **`DP` (Prolonged Decelerations):** Primary acute event biomarker for late-stage fetal hypoxia.
   - **`AC` (Accelerations):** Strongest protective factor indicating healthy somatic neurological reactivity.

2. **Model Family Behaviors:**
   - **Tree Ensembles (LightGBM / XGBoost):** Excelled at handling sharp non-linear thresholds in CTG signals.
   - **Kernel Margin Models (SVM RBF):** Maintained strong regularized boundaries and high Pathologic recall due to cost-sensitive weighting.
   - **Linear Models (Logistic Regression):** Provided a reliable baseline but underperformed on complex deceleration-variability interactions.
