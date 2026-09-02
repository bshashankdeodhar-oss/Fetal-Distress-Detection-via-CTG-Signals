"""
PHASE E: Probability Calibration + ROC/PR Curves + Reliability Diagrams
Applies Platt Scaling to the champion LightGBM and plots all clinical
validation curves required for a credible clinical ML evaluation.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss
)

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)

CLASS_NAMES  = ['Normal', 'Suspect', 'Pathologic']
CLASS_COLORS = ['#16a34a', '#d97706', '#dc2626']


def main():
    print("=================================================================")
    print("  PHASE E: CALIBRATION + ROC + PR CURVES                         ")
    print("=================================================================")

    # Load data
    path = os.path.join('datasets', 'uci_ctg', 'CTG_features_engineered.csv')
    df = pd.read_csv(path)
    feat_names = [c for c in df.columns if c != 'NSP']
    X = df[feat_names].values
    y = df['NSP'].values - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

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

    # Raw (uncalibrated) model
    raw_clf = lgb.LGBMClassifier(**lgb_params)
    raw_clf.fit(X_train, y_train)
    raw_probs = raw_clf.predict_proba(X_test)

    # Platt-Scaling calibrated model (sigmoid)
    cal_clf = CalibratedClassifierCV(
        lgb.LGBMClassifier(**lgb_params),
        method='sigmoid', cv=5
    )
    cal_clf.fit(X_train, y_train)
    cal_probs = cal_clf.predict_proba(X_test)

    print("  Raw vs Calibrated Brier Scores (lower = better calibrated):")
    for i, cls in enumerate(CLASS_NAMES):
        y_bin = (y_test == i).astype(int)
        bs_raw = brier_score_loss(y_bin, raw_probs[:, i])
        bs_cal = brier_score_loss(y_bin, cal_probs[:, i])
        print(f"    {cls:12s}: Raw={bs_raw:.4f}  Calibrated={bs_cal:.4f}")

    # ── Figure 1: Reliability (Calibration) Diagrams ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        y_bin = (y_test == i).astype(int)
        ax = axes[i]
        CalibrationDisplay.from_predictions(
            y_bin, raw_probs[:, i], n_bins=8,
            ax=ax, name='Uncalibrated LightGBM',
            color='#6b7280', linestyle='--'
        )
        CalibrationDisplay.from_predictions(
            y_bin, cal_probs[:, i], n_bins=8,
            ax=ax, name='Platt Calibrated',
            color=color
        )
        ax.set_title(f'Reliability Diagram — {cls}', fontweight='bold')
        ax.legend(fontsize=8)
    plt.suptitle('Probability Calibration: Raw vs Platt-Scaled LightGBM',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('outputs/figures/calibration_reliability_diagrams.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print("  -> Saved outputs/figures/calibration_reliability_diagrams.png")

    # ── Figure 2: ROC Curves (One-vs-Rest) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, cal_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{cls} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    ax.set_title('ROC Curves — One-vs-Rest (Calibrated LightGBM)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig('outputs/figures/roc_curves_ovr.png', dpi=300)
    plt.close()
    print("  -> Saved outputs/figures/roc_curves_ovr.png")

    # ── Figure 3: Precision-Recall Curves ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        y_bin = (y_test == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, cal_probs[:, i])
        ap = average_precision_score(y_bin, cal_probs[:, i])
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f'{cls} (AP = {ap:.3f})')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax.set_ylabel('Precision (PPV)', fontsize=11)
    ax.set_title('Precision-Recall Curves — One-vs-Rest (Calibrated LightGBM)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig('outputs/figures/pr_curves_ovr.png', dpi=300)
    plt.close()
    print("  -> Saved outputs/figures/pr_curves_ovr.png")

    # Save AUC summary
    auc_summary = {}
    for i, cls in enumerate(CLASS_NAMES):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, cal_probs[:, i])
        auc_summary[cls] = {'roc_auc': round(auc(fpr, tpr), 4),
                             'avg_precision': round(average_precision_score(y_bin, cal_probs[:, i]), 4)}
    with open('outputs/calibration_auc_summary.json', 'w') as f:
        json.dump(auc_summary, f, indent=2)
    print("  -> Saved outputs/calibration_auc_summary.json")
    for cls, vals in auc_summary.items():
        print(f"    {cls:12s}: ROC-AUC = {vals['roc_auc']:.4f}  Avg-Precision = {vals['avg_precision']:.4f}")

    print("\n  Phase E COMPLETE.")


if __name__ == '__main__':
    main()
