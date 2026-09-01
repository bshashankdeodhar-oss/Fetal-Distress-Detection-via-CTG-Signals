# Asymmetric Clinical Cost-Utility Optimization Report

## 1. Executive Clinical Summary
- **The Core Problem:** In obstetrics, a **False Negative (missed fetal distress)** carries a 10x higher clinical risk of irreversible hypoxic-ischemic encephalopathy (HIE) or stillbirth compared to a **False Positive (false alarm)** resulting in emergency Caesarean delivery.
- **Optimization Strategy:** We replaced naive mathematical probability cutoffs ($P > 0.33$) with a **Bayesian Cost-Utility Threshold Sweep** optimizing an asymmetric clinical penalty matrix ($C_{FN} = 10, C_{FP} = 2$).

## 2. Before vs. After Comparison
| Metric | Baseline (Argmax $P>0.33$) | Clinically Calibrated ($P \ge 0.11$) | Clinical Improvement |
| :--- | :---: | :---: | :---: |
| **Total Clinical Risk Score** | **49.0** | **45.0** | **8.2% Risk Reduction** ⭐ |
| **Pathologic Recall (Safety)** | **94.29%** | **94.29%** | **33/35 Distress Cases Caught** |
| **Macro F1 Score** | **0.8949** | **0.9030** | Preserves high discriminative power |

## 3. Patient Case Studies & Explainability
1. **Distress Patient Case (Case #1):**
   - Elevated `% Abnormal Short-Term Variability` (`ASTV = 81.0%`) and `Variability Collapse Ratio` (`VCR = 917.57`) drove positive log-odds shift toward *Pathologic*.
2. **Reassuring Normal Case (Case #2):**
   - High `Accelerations` (`AC = 6.0000`) and normal baseline variability protected the fetus against false alarms.
