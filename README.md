# Fetal Distress Detection from Cardiotocography (CTG) Signals

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Framework](https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost%20%7C%20Stacking-orange)
![Standard](https://img.shields.io/badge/Clinical%20Standard-FIGO%202015%20%7C%20ACOG-red)
![Metric](https://img.shields.io/badge/Champion%20Macro%20F1-0.9030-brightgreen)

> **A clinically-rigorous, end-to-end machine learning pipeline for automated fetal distress triage from Cardiotocography recordings**, grounded in FIGO/ACOG guidelines and validated via 5-Fold Stratified Cross-Validation with bootstrap confidence intervals.

---

## 🏥 Clinical Problem

Fetal distress — primarily caused by uteroplacental insufficiency and acute umbilical cord compression — manifests as measurable changes in fetal heart rate (FHR) variability and uterine contraction patterns. Missed detection leads to irreversible **hypoxic-ischaemic encephalopathy (HIE)** or intrapartum stillbirth.

**The core challenge:** the dataset is severely imbalanced (8.3% Pathologic, 13.9% Suspect, 77.8% Normal), so standard overall accuracy collapses as a metric and symmetric loss functions are clinically inappropriate.

---

## 🗂️ Pipeline Architecture

```
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  INPUT: UCI CTG Dataset (2,126 recordings × 21 raw morphological features)               │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE A — ADVANCED FEATURE ENGINEERING (src/feature_engineering.py)                     │
  │  21 raw → 39 features (+18 FIGO/ACOG clinical biomarkers)                               │
  │  DSI · VCR · PRI · Autonomic Balance · Variability Entropy · Morphological Complexity    │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE B — OPTUNA BAYESIAN HYPERPARAMETER OPTIMISATION (src/optuna_hpo.py)               │
  │  100-trial TPE study · Optimises 5-Fold CV Macro F1 · Exports best_params_lgb.json       │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE C — 5-FOLD STRATIFIED CV + BOOTSTRAP CI (src/cv_evaluation.py)                   │
  │  Evaluates 5 model families · Mean ± std · 95% bootstrap CI on every metric              │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE D — STACKING ENSEMBLE META-LEARNER (src/stacking_ensemble.py)                    │
  │  LightGBM + XGBoost + Random Forest + SVM → Meta: Cost-Sensitive Logistic Regression     │
  │  5-Fold OOF meta-features + passthrough raw features                                     │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE E — PROBABILITY CALIBRATION + CLINICAL VALIDATION CURVES                          │
  │  Platt Scaling · Reliability Diagrams · ROC-AUC (OvR) · Precision-Recall (OvR)          │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE F — ASYMMETRIC COST CALIBRATION (src/clinical_cost_optimization.py)               │
  │  Cost(FN) = 10 × Cost(FP) · Optimal threshold P* ≥ 0.110 · Pathologic Recall: 94.29%   │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE G — ENRICHED SHAP EXPLAINABILITY (src/shap_suite.py)                             │
  │  Beeswarm · Decision Plot · Multi-class Summary Bar · 3 Case Waterfall Studies           │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  OUTPUT: Blueprint CDSS Dashboard (app.py / frontend/index.html)                         │
  │  Live CTG oscilloscope · Real-time inference · All 7 clinical plots embedded              │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Results — Held-Out Benchmark (426 unseen patients, 20% stratified split)

| Rank | Model | Family | Macro F1 | Pathologic Recall | Balanced Acc |
|:---:|:---|:---|:---:|:---:|:---:|
| 🥇 | **Stacking Ensemble (LGB+XGB+RF+SVM)** | Meta-Learner | **0.9083** | **94.29%** | 89.1% |
| 🥈 | XGBoost | Gradient Boosted Trees | 0.8998 | 91.43% | 88.2% |
| 🥉 | LightGBM (Optuna-Tuned) | Gradient Boosted Trees | 0.8949 | **94.29%** | 89.0% |
| 4 | Random Forest | Bagged Tree Ensembles | 0.8810 | 91.43% | 87.9% |
| 5 | SVM (RBF Kernel) | Kernel Margin | 0.8245 | 88.57% | 87.9% |
| 6 | Logistic Regression | Linear Probabilistic | 0.7796 | 82.86% | 84.9% |

> **Why Macro F1, not Overall Accuracy?** With 77.8% Normal cases, a dummy "predict Normal always" baseline achieves 77.8% accuracy while catching 0% of distress cases. Macro F1 gives equal 33.3% weight to all three classes, penalising failures on the minority Pathologic class.

---

## 🧬 Feature Engineering (18 Clinical Biomarkers Added)

| Feature | Formula | Clinical Meaning |
|:---|:---|:---|
| `DSI` | `(DL + 2DS + 3DP) / (UC + ε)` | Deceleration severity weighted by contraction coupling |
| `VCR` | `(ASTV × ALTV) / (MSTV × MLTV + ε)` | Multi-scale autonomic variability collapse |
| `PRI` | Weighted composite of ASTV, ALTV, DP, DS, AC, MSTV | Pathologic Risk Index — single summary risk score |
| `Autonomic_Balance_Ratio` | `AC / ((ASTV + ALTV)/100 + ε)` | Reactive vs. pathological autonomic balance |
| `Variability_Entropy` | Shannon entropy of ASTV/ALTV distribution | Loss of HRV complexity — marker of hypoxia |
| `Decel_Pattern_Severity` | Weighted sum of DL, DS, DP | Cascade severity of consecutive decelerations |
| `FHR_Instability_Score` | `FHR_Dev × (MLTV + 1) / (MSTV + ε)` | Combined baseline deviation and long-term instability |
| `Morphological_Complexity` | `(Max − Min) × Nmax / (Variance + ε)` | Histogram richness — reduced in pathologic flatline |

---

## 🎯 Asymmetric Clinical Cost Calibration

Standard ML uses symmetric loss: `Cost(FN) = Cost(FP) = 1`.

In obstetrics, a missed fetal distress case risks **irreversible hypoxic brain injury**, while a false alarm results in precautionary C-section — a far lower clinical harm.

We applied **Bayesian decision-theoretic threshold optimisation**:

```
Clinical Loss = 10 × N(FN) + 2 × N(FP)
Optimal Threshold P* = 0.110
Pathologic Recall: 94.29% (33 / 35 distress cases caught)
```

---

## 🚀 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/bshashankdeodhar-oss/Fetal-Distress-Detection-via-CTG-Signals.git
cd "Fetal Distress Detection from CTG Signals"

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (in order)
python src/feature_engineering.py
python src/optuna_hpo.py          # ~5 min — 100-trial Optuna search
python src/cv_evaluation.py
python src/stacking_ensemble.py
python src/calibration_analysis.py
python src/shap_suite.py

# Launch the Streamlit dashboard
streamlit run app.py
```

---

## 📁 Repository Structure

```
├── datasets/uci_ctg/
│   ├── CTG_cleaned.csv                        # Cleaned base dataset (2126 × 21)
│   └── CTG_features_engineered.csv            # v2 dataset with 39 features
├── src/
│   ├── feature_engineering.py                 # Phase A — 18 clinical biomarkers
│   ├── optuna_hpo.py                          # Phase B — Bayesian HPO
│   ├── cv_evaluation.py                       # Phase C — 5-Fold CV + Bootstrap CI
│   ├── stacking_ensemble.py                   # Phase D — Meta-Learner
│   ├── calibration_analysis.py                # Phase E — ROC / PR / Calibration
│   ├── shap_suite.py                          # Phase F — Full SHAP explainability
│   └── clinical_cost_optimization.py          # Asymmetric cost threshold sweep
├── outputs/
│   ├── figures/                               # All 18+ plots
│   ├── reports/                               # Markdown clinical reports
│   ├── model_benchmark_comparison.csv         # Full leaderboard
│   ├── cv_results_with_ci.csv                 # CV results with 95% CIs
│   └── best_params_lgb.json                   # Optuna champion parameters
├── frontend/                                  # Pure HTML/JS Blueprint dashboard
├── app.py                                     # Streamlit CDSS application
└── DEFENSE_AND_NOVELTY_GUIDE.md               # Hackathon defense script
```

---

## 📚 Clinical References

1. Ayres-de-Campos, D. et al. (2015). *FIGO Consensus Guidelines on Intrapartum Fetal Monitoring*. IJGO.
2. American College of Obstetricians and Gynecologists (2009). *ACOG Practice Bulletin 116*.
3. Goldberger, A. et al. *PhysioBank, PhysioToolkit, and PhysioNet*. Circulation (2000).
4. Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
5. Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD.
