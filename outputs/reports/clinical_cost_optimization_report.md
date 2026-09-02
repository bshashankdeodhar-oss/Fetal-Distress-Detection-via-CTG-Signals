# Asymmetric Clinical Cost-Utility Optimization Report

## 1. Executive Clinical Summary
- **The Core Problem:** In obstetrics, a **False Negative (missed fetal distress)** carries a 10x higher clinical risk of irreversible hypoxic-ischemic encephalopathy (HIE) or stillbirth compared to a **False Positive (false alarm)** resulting in emergency Caesarean delivery.
- **Optimization Strategy:** We replaced naive mathematical probability cutoffs ($P > 0.33$) with a **Bayesian Cost-Utility Threshold Sweep** optimizing an asymmetric clinical penalty matrix ($C_{FN} = 10, C_{FP} = 2$).

## 2. Before vs. After Comparison
| Metric | Baseline (Argmax P>0.33) | Clinically Calibrated (P >= 0.13) | Clinical Improvement |
| :--- | :---: | :---: | :---: |
| **Total Clinical Risk Score** | **58.0** | **52.0** | **10.3% Risk Reduction** ⭐ |
| **Pathologic Recall (Safety)** | **91.43%** | **94.29%** | **33/35 Distress Cases Caught** |
| **Macro F1 Score** | **0.8923** | **0.8873** | Preserves high discriminative power |

## 3. Patient Case Studies & Explainability
1. **Distress Patient Case (Case #1):**
   - Elevated `% Abnormal Short-Term Variability` (`ASTV = 81.0%`) and `Variability Collapse Ratio` (`VCR = 917.58`) drove positive log-odds shift toward *Pathologic*.
2. **Reassuring Normal Case (Case #2):**
   - High `Accelerations` (`AC = 6.0000`) and normal baseline variability protected the fetus against false alarms.
