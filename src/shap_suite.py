"""
PHASE F: Enriched SHAP Explainability Suite (modern shap API)
- SHAP Beeswarm per class
- SHAP Summary Bar (multi-class)
- SHAP Decision Plot (3 patients)
- 3 Case Waterfall Studies: Normal / Suspect / Pathologic
"""
import os, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

CLASS_NAMES = ['Normal', 'Suspect', 'Pathologic']


def main():
    print("=================================================================")
    print("  PHASE F: ENRICHED SHAP EXPLAINABILITY SUITE                    ")
    print("=================================================================")

    path = os.path.join('datasets', 'uci_ctg', 'CTG_features_engineered.csv')
    df = pd.read_csv(path)
    feat_names = [c for c in df.columns if c != 'NSP']
    X = df[feat_names].values
    y = df['NSP'].values - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_test_df = pd.DataFrame(X_test, columns=feat_names)

    # Load best params
    lgb_params = {'n_estimators': 350, 'learning_rate': 0.04, 'num_leaves': 40,
                  'subsample': 0.85, 'colsample_bytree': 0.80,
                  'reg_alpha': 0.1, 'reg_lambda': 1.0,
                  'class_weight': 'balanced', 'random_state': 42, 'verbose': -1}
    json_path = 'outputs/best_params_lgb.json'
    if os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        lgb_params.update(d.get('params', {}))
        lgb_params.update({'class_weight': 'balanced', 'random_state': 42, 'verbose': -1})

    # Train with named features (DataFrame) to avoid LightGBM feature-name warnings
    X_train_df = pd.DataFrame(X_train, columns=feat_names)
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(X_train_df, y_train)
    y_pred = clf.predict(X_test_df)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    # ── Modern LightGBM returns shap_values as (N, d, n_classes) 3D array ─────
    # Normalise to list of [N, d] per class for compatibility with all SHAP plots
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_df)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv_per_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        ev_per_class = list(explainer.expected_value)
    else:
        sv_per_class = shap_values          # already a list of [N, d]
        ev_per_class = list(explainer.expected_value)

    # Build Explanation objects per class
    explanations = [
        shap.Explanation(
            values=sv_per_class[ci],
            base_values=np.full(len(X_test), float(ev_per_class[ci])),
            data=X_test,
            feature_names=feat_names
        ) for ci in range(3)
    ]

    # ── Plot 1: Beeswarm (Pathologic class) ───────────────────────────────────
    print("  Generating SHAP Beeswarm (Pathologic class)...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv_per_class[2], X_test, feature_names=feat_names,
                      plot_type='dot', max_display=15, show=False)
    plt.title('SHAP Beeswarm — Pathologic Class Risk Drivers (Test Cohort)',
              fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_beeswarm_pathologic.png', dpi=300)
    plt.close()
    print("  -> Saved outputs/figures/shap_beeswarm_pathologic.png")

    # ── Plot 2: SHAP Summary Bar (all 3 classes) ──────────────────────────────
    print("  Generating SHAP Summary Bar (multi-class)...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv_per_class, X_test, feature_names=feat_names,
                      class_names=CLASS_NAMES, plot_type='bar', show=False)
    plt.title('Mean |SHAP Value| — Multi-Class Feature Importance',
              fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_summary_bar_multiclass.png', dpi=300)
    plt.close()
    print("  -> Saved outputs/figures/shap_summary_bar_multiclass.png")




    # ── Waterfall plots for 3 representative patients ─────────────────────────
    idx_normal  = np.where((y_test == 0) & (y_pred == 0))[0]
    idx_suspect = np.where((y_test == 1) & (y_pred == 1))[0]
    idx_pathol  = np.where((y_test == 2) & (y_pred == 2))[0]

    case_info = [
        (idx_normal[0]  if len(idx_normal)  > 0 else 0, 0, 'Normal',     'shap_waterfall_normal.png'),
        (idx_suspect[0] if len(idx_suspect) > 0 else 0, 1, 'Suspect',    'shap_waterfall_suspect.png'),
        (idx_pathol[0]  if len(idx_pathol)  > 0 else 0, 2, 'Pathologic', 'shap_waterfall_patient_distress.png'),
    ]

    for patient_idx, class_idx, label, fname in case_info:
        print(f"  Generating waterfall: {label} patient (idx={patient_idx})...")
        ev = ev_per_class[class_idx]
        exp = shap.Explanation(
            values=sv_per_class[class_idx][patient_idx],
            base_values=float(ev),
            data=X_test[patient_idx],
            feature_names=feat_names
        )
        plt.figure(figsize=(9.5, 5.5))
        shap.waterfall_plot(exp, max_display=12, show=False)
        plt.title(f'SHAP Waterfall: {label} Patient — Log-odds Breakdown',
                  fontsize=11, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(f'outputs/figures/{fname}', dpi=300)
        plt.close()
        print(f"  -> Saved outputs/figures/{fname}")

    # Backward compat: copy normal waterfall to "reassuring" name
    if os.path.exists('outputs/figures/shap_waterfall_normal.png'):
        shutil.copy('outputs/figures/shap_waterfall_normal.png',
                    'outputs/figures/shap_waterfall_patient_reassuring.png')

    # ── Plot 5: SHAP Decision Plot (3 patients) ───────────────────────────────
    print("  Generating SHAP Decision Plot (3 representative patients)...")
    patient_indices = [ci[0] for ci in case_info]
    shap_3pts = np.stack([sv_per_class[ci[1]][ci[0]] for ci in case_info])
    ev_vals   = [ev_per_class[ci[1]] for ci in case_info]

    plt.figure(figsize=(10, 6.5))
    shap.decision_plot(
        base_value=float(np.mean(ev_vals)),
        shap_values=shap_3pts,
        features=X_test[patient_indices],
        feature_names=feat_names,
        feature_display_range=slice(-1, -14, -1),
        legend_labels=['Normal Patient', 'Suspect Patient', 'Pathologic Patient'],
        show=False
    )
    plt.title('SHAP Decision Plot — Cumulative Log-Odds Paths for 3 Representative Patients',
              fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_decision_plot_3cases.png', dpi=300)
    plt.close()
    print("  -> Saved outputs/figures/shap_decision_plot_3cases.png")

    # ── Write case study markdown report ─────────────────────────────────────
    report_lines = ["# SHAP Case Study Report — Enriched Explainability Suite\n",
                    "## Representative Patient Analysis\n"]
    for patient_idx, class_idx, label, _ in case_info:
        sv = sv_per_class[class_idx][patient_idx]
        top3 = np.argsort(np.abs(sv))[::-1][:3]
        report_lines.append(f"### {label} Patient (test index {patient_idx})")
        report_lines.append(f"- **True Label:** {label} | **Predicted:** {CLASS_NAMES[class_idx]}")
        report_lines.append("- **Top 3 SHAP drivers:**")
        for j in top3:
            direction = "↑ RISK" if sv[j] > 0 else "↓ RISK"
            report_lines.append(f"  - `{feat_names[j]}` = {X_test[patient_idx, j]:.3f} → SHAP {sv[j]:+.4f} ({direction})")
        report_lines.append("")

    with open('outputs/reports/shap_case_study_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    print("  -> Saved outputs/reports/shap_case_study_report.md")
    print("\n  Phase F COMPLETE.")


if __name__ == '__main__':
    main()
