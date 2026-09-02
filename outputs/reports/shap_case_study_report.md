# SHAP Case Study Report — Enriched Explainability Suite

## Representative Patient Analysis

### Normal Patient (test index 0)
- **True Label:** Normal | **Predicted:** Normal
- **Top 3 SHAP drivers:**
  - `PRI` = -9.500 → SHAP +1.2797 (↑ RISK)
  - `Autonomic_Reactivity_Index` = 0.240 → SHAP +0.5787 (↑ RISK)
  - `ASTV` = 25.000 → SHAP +0.5501 (↑ RISK)

### Suspect Patient (test index 21)
- **True Label:** Suspect | **Predicted:** Suspect
- **Top 3 SHAP drivers:**
  - `ASTV` = 61.000 → SHAP +1.5268 (↑ RISK)
  - `PRI` = 1.910 → SHAP +1.2356 (↑ RISK)
  - `STV_LTV_Ratio` = 0.052 → SHAP +0.9305 (↑ RISK)

### Pathologic Patient (test index 34)
- **True Label:** Pathologic | **Predicted:** Pathologic
- **Top 3 SHAP drivers:**
  - `ASTV` = 81.000 → SHAP +5.4617 (↑ RISK)
  - `FHR_Instability_Score` = 203.499 → SHAP +1.4897 (↑ RISK)
  - `PRI` = 3.010 → SHAP +1.0182 (↑ RISK)
