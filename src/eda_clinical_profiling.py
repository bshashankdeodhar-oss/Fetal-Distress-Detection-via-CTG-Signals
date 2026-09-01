import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure outputs directories exist
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

def run_clinical_eda():
    print("==================================================")
    print("  PHASE 1: CLINICAL EXPLORATORY DATA ANALYSIS (EDA)")
    print("==================================================")
    
    csv_path = os.path.join('datasets', 'uci_ctg', 'CTG_cleaned.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {csv_path}. Please run preprocess.py first.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset successfully: {df.shape[0]} records, {df.shape[1]} columns.")
    
    target_col = 'NSP'
    feature_cols = [c for c in df.columns if c != target_col]
    
    class_map = {1: 'Normal', 2: 'Suspect', 3: 'Pathologic'}
    df['State'] = df[target_col].map(class_map)
    palette = {'Normal': '#10b981', 'Suspect': '#f59e0b', 'Pathologic': '#ef4444'}
    
    # -------------------------------------------------------------
    # 1. Class Distribution Analysis
    # -------------------------------------------------------------
    print("\n--- 1. Class Distribution & Imbalance Audit ---")
    class_counts = df['State'].value_counts()
    class_pcts = df['State'].value_counts(normalize=True) * 100
    
    for cls in ['Normal', 'Suspect', 'Pathologic']:
        print(f"  {cls:<12}: {class_counts[cls]:>5} samples ({class_pcts[cls]:>5.2f}%)")
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(class_counts.index, class_counts.values, color=[palette[c] for c in class_counts.index], width=0.55, edgecolor='black', linewidth=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}\n({height/len(df)*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('CTG Target Class Distribution (Severe Imbalance)', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Number of Patient Recordings', fontsize=10)
    ax.set_ylim(0, max(class_counts.values) * 1.18)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('outputs/figures/eda_class_distribution.png', dpi=300)
    plt.close()
    print("  -> Saved outputs/figures/eda_class_distribution.png")
    
    # -------------------------------------------------------------
    # 2. Correlation Matrix with Clinical Distress
    # -------------------------------------------------------------
    print("\n--- 2. Feature Correlation with Fetal Distress (NSP) ---")
    corr_matrix = df[feature_cols + [target_col]].corr(method='spearman')
    nsp_corrs = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    
    print("Top Positively Correlated with Distress (Risk Factors):")
    for feat, val in nsp_corrs.head(6).items():
        print(f"  + {feat:<10}: {val:+.4f}")
    print("\nTop Negatively Correlated with Distress (Protective Factors):")
    for feat, val in nsp_corrs.tail(4).items():
        print(f"  - {feat:<10}: {val:+.4f}")
        
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.2f', 
                annot_kws={'size': 7}, cbar_kws={'shrink': 0.8}, linewidths=0.5)
    plt.title('Spearman Correlation Matrix (Clinical CTG Morphometrics)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('outputs/figures/eda_correlation_matrix.png', dpi=300)
    plt.close()
    print("  -> Saved outputs/figures/eda_correlation_matrix.png")
    
    # -------------------------------------------------------------
    # 3. Key Clinical Biomarkers Comparison Boxplots
    # -------------------------------------------------------------
    print("\n--- 3. Generating Clinical Biomarker Distributions ---")
    key_features = ['ASTV', 'ALTV', 'DP', 'AC', 'UC', 'MSTV']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    feature_labels = {
        'ASTV': '% Abnormal Short-Term Var (ASTV)',
        'ALTV': '% Abnormal Long-Term Var (ALTV)',
        'DP': 'Prolonged Decelerations / sec (DP)',
        'AC': 'Accelerations / sec (AC)',
        'UC': 'Uterine Contractions / sec (UC)',
        'MSTV': 'Mean Short-Term Var (MSTV)'
    }
    
    for i, feat in enumerate(key_features):
        sns.boxplot(x='State', y=feat, data=df, ax=axes[i], order=['Normal', 'Suspect', 'Pathologic'],
                    palette=palette, width=0.5, fliersize=2)
        axes[i].set_title(feature_labels[feat], fontsize=11, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        
    plt.suptitle('Clinical Biomarker Stratification across Fetal States', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('outputs/figures/eda_clinical_feature_boxplots.png', dpi=300)
    plt.close()
    print("  -> Saved outputs/figures/eda_clinical_feature_boxplots.png")
    
    # -------------------------------------------------------------
    # 4. Deceleration vs Variability Interaction
    # -------------------------------------------------------------
    print("\n--- 4. Generating 2D Clinical Distress Trajectory (ASTV vs Prolonged Decelerations) ---")
    fig, ax = plt.subplots(figsize=(8, 6))
    for state, color in palette.items():
        sub = df[df['State'] == state]
        ax.scatter(sub['ASTV'], sub['DP'] * 1000, label=state, alpha=0.65, s=35, color=color, edgecolors='none')
        
    ax.set_title('Hypoxia Trajectory: Short-Term Variability vs. Prolonged Decelerations', fontsize=12, fontweight='bold')
    ax.set_xlabel('Percentage of Abnormal Short Term Variability (ASTV %)', fontsize=10)
    ax.set_ylabel('Prolonged Decelerations (DP per 1,000s)', fontsize=10)
    ax.legend(title='Fetal State', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('outputs/figures/eda_variability_vs_decelerations.png', dpi=300)
    plt.close()
    print("  -> Saved outputs/figures/eda_variability_vs_decelerations.png")
    
    # -------------------------------------------------------------
    # 5. Export Summary Markdown Report
    # -------------------------------------------------------------
    report_md = f"""# Phase 1: Clinical Exploratory Data Analysis & Biological Insights

## 1. Executive Summary & Imbalance Profile
- **Total Patient Recordings:** {len(df)} CTG examinations
- **Class 1 (Normal):** {class_counts['Normal']} cases ({class_pcts['Normal']:.2f}%)
- **Class 2 (Suspect):** {class_counts['Suspect']} cases ({class_pcts['Suspect']:.2f}%)
- **Class 3 (Pathologic):** {class_counts['Pathologic']} cases ({class_pcts['Pathologic']:.2f}%)
- **Imbalance Ratio:** ~9.4 : 1.7 : 1.0 (Normal : Suspect : Pathologic)

## 2. Key Clinical Biomarker Findings
1. **Abnormal Short-Term Variability (`ASTV`):**
   - Normal Median: ~{df[df['NSP']==1]['ASTV'].median():.1f}%
   - Suspect Median: ~{df[df['NSP']==2]['ASTV'].median():.1f}%
   - Pathologic Median: ~{df[df['NSP']==3]['ASTV'].median():.1f}%
   - *Clinical Insight:* High ASTV directly reflects loss of autonomic heart-rate modulation due to cerebral hypoxia.

2. **Prolonged Decelerations (`DP`):**
   - Almost entirely absent in Normal cases. Spikes dramatically in Pathological cases as the fetal heart fails to recover post-contraction.

3. **Accelerations (`AC`):**
   - Accelerations indicate fetal somatic movement and intact autonomic nervous system reactivity. High AC strongly correlates with normal fetal health (protective factor $r = {nsp_corrs['AC']:.2f}$).

## 3. Multi-collinearity Insights
- High correlation observed between histogram statistics (`Mode`, `Mean`, `Median`).
- Tree-based models naturally handle collinearity, while regularized models (L2 Ridge / Logistic) and PCA/StandardScaler are necessary for linear/margin classifiers.
"""
    with open('outputs/reports/eda_clinical_summary.md', 'w') as f:
        f.write(report_md)
    print("  -> Saved outputs/reports/eda_clinical_summary.md")
    
    print("\n==================================================")
    print("  PHASE 1 EDA COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == '__main__':
    run_clinical_eda()
