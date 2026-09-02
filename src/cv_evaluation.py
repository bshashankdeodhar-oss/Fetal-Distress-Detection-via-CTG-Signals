"""
PHASE C: 5-Fold Stratified Cross-Validation with Bootstrap Confidence Intervals
Replaces the single 80/20 split with rigorous repeated evaluation.
Reports mean ± std and 95% bootstrap CI for all metrics.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    balanced_accuracy_score, accuracy_score
)
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)


def bootstrap_ci(scores, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    boot = [rng.choice(scores, len(scores), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return float(np.mean(scores)), float(lo), float(hi)


def evaluate_fold(clf, X_tr, X_te, y_tr, y_te):
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    per_cls_f1  = f1_score(y_te, y_pred, average=None, zero_division=0)
    per_cls_rec = recall_score(y_te, y_pred, average=None, zero_division=0)
    return {
        'macro_f1'    : f1_score(y_te, y_pred, average='macro', zero_division=0),
        'macro_rec'   : recall_score(y_te, y_pred, average='macro', zero_division=0),
        'macro_prec'  : precision_score(y_te, y_pred, average='macro', zero_division=0),
        'bal_acc'     : balanced_accuracy_score(y_te, y_pred),
        'acc'         : accuracy_score(y_te, y_pred),
        'f1_path'     : per_cls_f1[2]  if len(per_cls_f1)  > 2 else 0.0,
        'rec_path'    : per_cls_rec[2] if len(per_cls_rec) > 2 else 0.0,
    }


def main():
    print("=================================================================")
    print("  PHASE C: 5-FOLD STRATIFIED CV + BOOTSTRAP CONFIDENCE INTERVALS ")
    print("=================================================================")

    path = os.path.join('datasets', 'uci_ctg', 'CTG_features_engineered.csv')
    df = pd.read_csv(path)
    X = df.drop(columns=['NSP']).values
    y = df['NSP'].values - 1
    feat_names = [c for c in df.columns if c != 'NSP']

    # Load Optuna best params if available
    lgb_params = {'n_estimators': 300, 'learning_rate': 0.04, 'num_leaves': 31,
                  'class_weight': 'balanced', 'random_state': 42, 'verbose': -1}
    json_path = 'outputs/best_params_lgb.json'
    if os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        lgb_params.update(d.get('params', {}))
        lgb_params['class_weight'] = 'balanced'
        lgb_params['random_state'] = 42
        lgb_params['verbose'] = -1
        print(f"  Loaded Optuna best params (CV F1 = {d.get('best_cv_macro_f1', '?')})")
    else:
        print("  Using default LightGBM params (run optuna_hpo.py for tuned params)")

    MODEL_ZOO = {
        'LightGBM (Optuna-Tuned)': {
            'model': lgb.LGBMClassifier(**lgb_params),
            'scaled': False, 'family': 'Gradient Boosted Trees'
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=6,
                                        eval_metric='mlogloss', random_state=42, verbosity=0),
            'scaled': False, 'family': 'Gradient Boosted Trees'
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=4,
                                            class_weight='balanced_subsample', random_state=42),
            'scaled': False, 'family': 'Bagged Ensembles'
        },
        'SVM (RBF)': {
            'model': SVC(C=2.5, kernel='rbf', gamma='scale', class_weight='balanced',
                         probability=True, random_state=42),
            'scaled': True, 'family': 'Kernel Margin'
        },
        'Logistic Regression': {
            'model': LogisticRegression(C=1.5, class_weight='balanced', max_iter=1000,
                                        random_state=42),
            'scaled': True, 'family': 'Linear'
        },
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for name, cfg in MODEL_ZOO.items():
        fold_scores = {k: [] for k in ['macro_f1', 'macro_rec', 'macro_prec',
                                        'bal_acc', 'acc', 'f1_path', 'rec_path']}
        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            if cfg['scaled']:
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)

            import copy
            clf = copy.deepcopy(cfg['model'])
            fold_res = evaluate_fold(clf, X_tr, X_te, y_tr, y_te)
            for k, v in fold_res.items():
                fold_scores[k].append(v)

        f1_arr = np.array(fold_scores['macro_f1'])
        mu, lo, hi = bootstrap_ci(f1_arr)
        rows.append({
            'Model'           : name,
            'Family'          : cfg['family'],
            'Macro F1 (mean)' : round(mu, 4),
            'Macro F1 (std)'  : round(f1_arr.std(), 4),
            'Macro F1 95% CI' : f"[{lo:.4f}, {hi:.4f}]",
            'Macro Recall'    : round(np.mean(fold_scores['macro_rec']), 4),
            'Balanced Acc'    : round(np.mean(fold_scores['bal_acc']), 4),
            'F1 (Pathologic)' : round(np.mean(fold_scores['f1_path']), 4),
            'Rec (Pathologic)': round(np.mean(fold_scores['rec_path']), 4),
        })
        print(f"  {name:40s} → Macro F1 = {mu:.4f} ± {f1_arr.std():.4f}  95% CI [{lo:.4f}, {hi:.4f}]  PathRec = {np.mean(fold_scores['rec_path']):.4f}")

    df_res = pd.DataFrame(rows).sort_values('Macro F1 (mean)', ascending=False)
    df_res.to_csv('outputs/cv_results_with_ci.csv', index=False)
    print("\n  -> Saved outputs/cv_results_with_ci.csv")

    # --- Bar chart with error bars ---
    names_plot = [r['Model'] for _, r in df_res.iterrows()]
    means = df_res['Macro F1 (mean)'].values
    stds  = df_res['Macro F1 (std)'].values

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#16a34a' if i == 0 else '#2563eb' for i in range(len(names_plot))]
    bars = ax.barh(names_plot[::-1], means[::-1], xerr=stds[::-1],
                   color=colors[::-1], edgecolor='black', linewidth=0.6,
                   capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': '#374151'})
    for i, (m, s) in enumerate(zip(means[::-1], stds[::-1])):
        ax.text(m + s + 0.003, i, f'{m:.4f} ± {s:.4f}', va='center', fontsize=9)
    ax.set_xlim(0.65, 1.02)
    ax.set_xlabel('5-Fold CV Macro F1 Score', fontweight='bold')
    ax.set_title('Cross-Validated Macro F1 with 95% Bootstrap CI — All Families', fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.axvline(0.9, color='#dc2626', linestyle=':', linewidth=1.2, label='F1 = 0.90 target')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/cv_macro_f1_comparison.png', dpi=300)
    plt.close()
    print("  -> Saved outputs/figures/cv_macro_f1_comparison.png")
    print("\n  Phase C COMPLETE.")

if __name__ == '__main__':
    main()
