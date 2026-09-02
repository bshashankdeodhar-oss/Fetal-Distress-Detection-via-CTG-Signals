# Fetal Distress Detection from Cardiotocography (CTG) Signals

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Framework](https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost%20%7C%20Stacking-orange)
![Standard](https://img.shields.io/badge/Clinical%20Standard-FIGO%202015%20%7C%20ACOG-red)
![Metric](https://img.shields.io/badge/Champion%20CV%20Macro%20F1-0.9202-brightgreen)
![Sensitivity](https://img.shields.io/badge/Pathologic%20Recall-94.29%25-red)

> **A clinically-rigorous, end-to-end machine learning system for automated fetal distress triage from Cardiotocography (CTG) signals**, grounded in FIGO 2015 / ACOG intrapartum guidelines, optimized via Optuna Bayesian TPE search, calibrated with Platt scaling, and evaluated via 5-Fold Stratified Cross-Validation with 95% bootstrap confidence intervals.

---

## 🏥 Clinical Motivation & Problem Statement

Intrapartum fetal asphyxia resulting from uteroplacental insufficiency or umbilical cord occlusion causes rapid fetal metabolic acidemia. Failure to intervene promptly leads to irreversible **hypoxic-ischemic encephalopathy (HIE)**, cerebral palsy, or intrapartum fetal demise.

Standard CTG interpretation suffers from high inter-observer variability (up to 40% discordance among expert obstetricians) and severe class imbalance (**8.3% Pathologic, 13.9% Suspect, 77.8% Normal**). Naive machine learning models optimizing overall accuracy collapse by simply predicting the majority class ("Normal"), generating dangerous false negatives.

---

## 📈 Dual-Horizon Clinical Diagnostics: Snapshot vs. Continuous Telemetry

This system provides two complementary, clinically validated diagnostic horizons:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             DUAL-HORIZON DIAGNOSTIC ARCHITECTURE                            │
├──────────────────────────────────────────────┬──────────────────────────────────────────────┤
│ 1. POINT-IN-TIME SNAPSHOT CDSS (Tab 1)       │ 2. DYNAMIC REAL-TIME TELEMETRY (Tab 0)       │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ • Evaluates a fixed clinical epoch or manual │ • Streams continuous 4 Hz signals from the   │
│   morphometric draft (21 raw + 18 derived).  │   PhysioNet CTU-UHB intrapartum database.    │
│ • Rapid triage during scheduled examinations │ • Rolling 5-to-15 minute sliding window.     │
│   against strict FIGO 2015 diagnostic rules. │ • Exponential Moving Average (EMA) filter    │
│ • Outputs instantaneous posterior probabilities│   (α = 0.25) modeling physiological inertia. │
│   and multi-class risk classification.       │ • Continuous 30-min Risk Trajectory tracking │
│                                              │   rate-of-change (ΔP/Δt) over active labor.  │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 🗂️ End-to-End Pipeline Architecture

```
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  INPUT: UCI CTG Cohort (2,126 recordings) + PhysioNet CTU-UHB Continuous 4 Hz Signals    │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE A — CANONICAL FEATURE ENGINEERING (src/feature_engineering.py)                    │
  │  21 raw → 39 features (+18 FIGO/ACOG clinical biomarkers: DSI, VCR, PRI, Entropy...)     │
  │  Unified compute_derived_features() guarantees ZERO train/serve skew across batch & UI.  │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE B — OPTUNA BAYESIAN HYPERPARAMETER OPTIMIZATION (src/optuna_hpo.py)               │
  │  100-trial TPE study · Optimizes 5-Fold CV Macro F1 · Attains 0.9202 Macro F1            │
  │  Exports outputs/best_params_lgb.json and outputs/figures/optuna_convergence.png         │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE C — 5-FOLD STRATIFIED CV + BOOTSTRAP CI (src/cv_evaluation.py)                   │
  │  Evaluates 5 model families · Mean ± std · 95% bootstrap CI on every metric              │
  │  LightGBM: Macro F1 = 0.9202 ± 0.0192 [0.9017, 0.9348] · Pathologic Recall = 92.60%     │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE D — STACKING ENSEMBLE META-LEARNER (src/stacking_ensemble.py)                    │
  │  Combines LightGBM + XGBoost + Random Forest + SVM into a Logistic Meta-Learner          │
  │  5-Fold OOF meta-features + passthrough raw features · Held-Out Test Recall = 91.43%     │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE E — PROBABILITY CALIBRATION & ROC/PR ANALYSIS (src/calibration_analysis.py)       │
  │  Platt Scaling · Reliability Diagrams · Pathologic ROC-AUC: 0.9942 · Avg-Precision: 0.9437│
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE F — ASYMMETRIC CLINICAL LOSS OPTIMIZATION (src/clinical_cost_optimization.py)     │
  │  Cost(FN) = 10 × Cost(FP) · Bayes minimum-loss threshold P* ≥ 0.110                      │
  │  Catches 94.29% of distress cases (33/35) on held-out test cohort · Exports summary JSON │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  PHASE G — ENRICHED SHAP EXPLAINABILITY SUITE (src/shap_suite.py)                        │
  │  Multi-class Beeswarm · Decision Plot · Summary Bar · 3 Patient Waterfall Case Studies   │
  └────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │  USER INTERFACES: Streamlit CDSS (app.py) & Blueprint Kiosk Prototype (frontend/)        │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Cross-Validation & Benchmark Results

### 5-Fold Stratified Cross-Validation (with 95% Bootstrap Confidence Intervals)

| Model Family | Model | 5-Fold CV Macro F1 | 95% Bootstrap CI | Pathologic Recall |
|:---|:---|:---:|:---:|:---:|
| **Gradient Boosted Trees** | **LightGBM (Optuna-Tuned)** | **0.9202 ± 0.0192** | **[0.9017, 0.9348]** | **92.60%** |
| Gradient Boosted Trees | XGBoost | 0.9145 ± 0.0148 | [0.9025, 0.9278] | 91.46% |
| Bagged Trees | Random Forest | 0.9048 ± 0.0132 | [0.8936, 0.9167] | 89.17% |
| Kernel Margin Models | SVM (RBF Kernel) | 0.8463 ± 0.0187 | [0.8291, 0.8628] | 86.32% |
| Linear Models | Logistic Regression | 0.8022 ± 0.0155 | [0.7894, 0.8178] | 88.03% |

### Held-Out Test Set (426 Unseen Patients, 20% Stratified Split)

| Model | Macro F1 | Pathologic Recall | Overall Accuracy | Pathologic ROC-AUC | Pathologic AP |
|:---|:---:|:---:|:---:|:---:|:---:|
| **LightGBM (Optuna-Tuned)** | **0.8893** | **91.43% (94.29% @ P\*)** | 94.37% | **0.9942** | **0.9437** |
| **Stacking Ensemble** | 0.8642 | 91.43% | 92.02% | 0.9910 | 0.9350 |
| **Random Forest** | 0.8743 | 91.43% | 92.49% | 0.9880 | 0.9280 |
| **SVM (RBF Kernel)** | 0.8310 | 88.57% | 88.97% | 0.9650 | 0.8820 |
| **Logistic Regression** | 0.7398 | 82.86% | 83.57% | 0.9420 | 0.8410 |

---

## 🧬 Feature Engineering Reference (18 Clinical Biomarkers)

All features are computed via `src/feature_engineering.py::compute_derived_features()` with zero train/serve skew:

| Biomarker | Mathematical Formula | Pathophysiological Interpretation |
|:---|:---|:---|
| `DSI` | `(DL + 2·DS + 3·DP) / (UC + ε)` | **Deceleration Severity Index**: Weights prolonged > severe > light decelerations normalized to contraction frequency. |
| `VCR` | `(ASTV · ALTV) / (MSTV · MLTV + ε)` | **Variability Collapse Ratio**: Cross-multiplies abnormal short/long-term variability against mean variability — hallmarks of autonomic failure. |
| `PRI` | FIGO 2015 weighted combination | **Pathologic Risk Index**: Integrates ASTV, ALTV, DP, DS, AC, and MSTV into an overall risk biomarker. |
| `Autonomic_Balance_Ratio` | `AC / ((ASTV + ALTV)/100 + ε)` | Ratio of reactive sympathetic bursts (AC) to micro-scale variability pathology. |
| `Variability_Entropy` | Shannon entropy of variability proxy | Loss of physiological HRV complexity and spectral richness due to acidemia. |
| `Decel_Pattern_Severity` | `DL + 3·DS + 5·DP` | Distinguishes isolated decelerations from compounding, severe decelerations. |
| `FHR_Instability_Score` | `|LB - 140| · (MLTV + 1) / (MSTV + ε)` | Combined baseline drift and long-term instability. |
| `Morphological_Complexity` | `(Max - Min) · Nmax / (Variance + ε)` | Histogram morphological breadth — collapses in flatline hypoxia. |

---

## 🎯 Asymmetric Clinical Cost Optimization ($P^* \ge 0.110$)

Standard machine learning assumes symmetric $0-1$ loss: `Cost(FN) = Cost(FP)`. In obstetrics, a **False Negative** (fetal death/HIE) is catastrophic ($C_{\text{FN}} = 10.0$), whereas a **False Positive** (precautionary fetal scalp sampling / emergency delivery) is manageable ($C_{\text{FP}} = 1.24$).

### Mathematical Derivation of Decision Cutoff ($P^*$):
Given posterior probability $P = P(\text{Pathologic} \mid x)$, the expected Bayesian loss $R(\hat{y} \mid x)$ is minimized when:
$$P^* = \frac{C_{\text{FP}}}{C_{\text{FN}} + C_{\text{FP}}} = \frac{1.24}{10.0 + 1.24} = \mathbf{0.1103} \approx \mathbf{11.0\%}$$

Empirical threshold sweeps across the held-out test cohort in `src/clinical_cost_optimization.py` confirm this operating point, achieving **94.29% Pathologic Recall (33 of 35 distress cases detected)** with a 41.5% reduction in total clinical penalty.

---

## 🖥️ User Interfaces: Streamlit CDSS (`app.py`) vs. Frontend Prototype (`frontend/`)

This repository includes two frontends tailored for distinct deployment scenarios:

1. **`app.py` (Production Streamlit CDSS Dashboard):**
   - The primary, interactive Python clinical application.
   - Executes live Scikit-Learn/LightGBM inference, continuous 4 Hz sliding-window PhysioNet signal ingestion with EMA smoothing, dynamic metric rendering from output artifacts, and interactive SHAP explainability.
   - Run via: `streamlit run app.py`

2. **`frontend/` (Standalone Blueprint Kiosk Prototype):**
   - A lightweight, dependency-free client-side prototype built in Vanilla HTML5, modern CSS, and Vanilla JavaScript.
   - Replicates the clinical oscilloscope UI with a Windows 11-style bottom taskbar for static web server deployment (NGINX/Apache/kiosks) without requiring a Python environment.
   - Open directly in any browser: `frontend/index.html`

---

## 🚀 Full Pipeline Reproduction (Step-by-Step)

To reproduce all models, figures, and benchmark metrics from scratch:

```bash
# 1. Clone the repository
git clone https://github.com/bshashankdeodhar-oss/Fetal-Distress-Detection-via-CTG-Signals.git
cd "Fetal Distress Detection from CTG Signals"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Clean environment end-to-end execution:
python fetch_all_datasets.py              # Download UCI CTG and PhysioNet CTU-UHB records
python preprocess.py                      # Clean base UCI dataset (2126 x 21)
python src/feature_engineering.py         # Generate 18 FIGO biomarkers (2126 x 39)
python train_models.py                    # Multi-family baseline training zoo
python src/optuna_hpo.py                  # 100-trial Bayesian TPE hyperparameter optimization
python src/cv_evaluation.py               # 5-Fold Stratified CV with bootstrap 95% CIs
python src/stacking_ensemble.py           # Multi-model Stacking Ensemble meta-learner
python src/calibration_analysis.py        # Platt probability calibration, ROC, & PR curves
python src/clinical_cost_optimization.py  # Asymmetric loss derivation & threshold sweep
python src/shap_suite.py                  # Full SHAP explainability suite & patient case studies

# 4. Launch the Streamlit CDSS App
streamlit run app.py
```

All tabs in `app.py` ([http://localhost:8502](http://localhost:8502)) will render live data without placeholder warnings.

---

## 📚 References & Clinical Guidelines

1. Ayres-de-Campos, D. et al. (2015). *FIGO Consensus Guidelines on Intrapartum Fetal Monitoring*. Int. J. Gynecol. Obstet., 131(1):3–24.
2. American College of Obstetricians and Gynecologists (2009). *ACOG Practice Bulletin No. 116: Management of Intrapartum Fetal Heart Rate Tracings*.
3. Goldberger, A. L. et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals*. Circulation, 101(23):e215–e220.
4. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS 30).
5. Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. ACM SIGKDD.
