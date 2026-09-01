# Phase 1: Clinical Exploratory Data Analysis & Biological Insights

## 1. Executive Summary & Imbalance Profile
- **Total Patient Recordings:** 2126 CTG examinations
- **Class 1 (Normal):** 1655 cases (77.85%)
- **Class 2 (Suspect):** 295 cases (13.88%)
- **Class 3 (Pathologic):** 176 cases (8.28%)
- **Imbalance Ratio:** ~9.4 : 1.7 : 1.0 (Normal : Suspect : Pathologic)

## 2. Key Clinical Biomarker Findings
1. **Abnormal Short-Term Variability (`ASTV`):**
   - Normal Median: ~41.0%
   - Suspect Median: ~63.0%
   - Pathologic Median: ~65.0%
   - *Clinical Insight:* High ASTV directly reflects loss of autonomic heart-rate modulation due to cerebral hypoxia.

2. **Prolonged Decelerations (`DP`):**
   - Almost entirely absent in Normal cases. Spikes dramatically in Pathological cases as the fetal heart fails to recover post-contraction.

3. **Accelerations (`AC`):**
   - Accelerations indicate fetal somatic movement and intact autonomic nervous system reactivity. High AC strongly correlates with normal fetal health (protective factor $r = -0.46$).

## 3. Multi-collinearity Insights
- High correlation observed between histogram statistics (`Mode`, `Mean`, `Median`).
- Tree-based models naturally handle collinearity, while regularized models (L2 Ridge / Logistic) and PCA/StandardScaler are necessary for linear/margin classifiers.
