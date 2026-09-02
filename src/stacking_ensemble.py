"""
PHASE D: Stacking Ensemble Meta-Learner
Base: Optuna-tuned LightGBM + XGBoost + Random Forest + SVM (RBF)
Meta-learner: Cost-sensitive Logistic Regression on out-of-fold predictions
"""
import os, json, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    balanced_accuracy_score, accuracy_score, confusion_matrix
)

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)


def plot_cm(cm, class_names, title, path, champion=False):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cmap = 'Greens' if champion else 'Blues'
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, linewidths=0.5, ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('True Clinical State', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted Clinical State', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    print("=================================================================")
    print("  PHASE D: STACKING ENSEMBLE META-LEARNER                        ")
    print("=================================================================")

    # Load data
    path = os.path.join('datasets', 'uci_ctg', 'CTG_features_engineered.csv')
    df = pd.read_csv(path)
    feat_names = [c for c in df.columns if c != 'NSP']
    X = df[feat_names].values
    y = df['NSP'].values - 1
    class_names = ['Normal', 'Suspect', 'Pathologic']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(y_train)} | Test: {len(y_test)} "
          f"(N={np.sum(y_test==0)} S={np.sum(y_test==1)} P={np.sum(y_test==2)})")

    # Load Optuna params if available
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
        print(f"  Using Optuna-tuned params (CV F1={d.get('best_cv_macro_f1','?')})")

    # Define base estimators (SVM needs scaling via Pipeline)
    estimators = [
        ('lgb', lgb.LGBMClassifier(**lgb_params)),
        ('xgb', xgb.XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=6,
                                   eval_metric='mlogloss', random_state=42, verbosity=0)),
        ('rf',  RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=4,
                                       class_weight='balanced_subsample', random_state=42)),
        ('svm', Pipeline([('sc', StandardScaler()),
                          ('clf', SVC(C=2.5, kernel='rbf', gamma='scale',
                                      class_weight='balanced', probability=True, random_state=42))])),
    ]

    # Meta-learner: scaled + cost-sensitive Logistic Regression (saga handles multi-class well)
    meta = Pipeline([
        ('scale', StandardScaler()),
        ('lr', LogisticRegression(C=1.0, solver='saga', class_weight='balanced',
                                  max_iter=5000, random_state=42, tol=1e-4))
    ])

    # Stacking classifier with 5-fold out-of-fold meta-features (no passthrough — cleaner for LogReg)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1
    )

    print("\n  Fitting Stacking Ensemble (5-fold OOF meta-features + passthrough)...")
    stack.fit(X_train, y_train)
    y_pred = stack.predict(X_test)

    # Metrics
    macro_f1  = f1_score(y_test, y_pred, average='macro')
    macro_rec = recall_score(y_test, y_pred, average='macro')
    bal_acc   = balanced_accuracy_score(y_test, y_pred)
    acc       = accuracy_score(y_test, y_pred)
    f1_pc     = f1_score(y_test, y_pred, average=None)
    rec_pc    = recall_score(y_test, y_pred, average=None)
    cm        = confusion_matrix(y_test, y_pred)

    print(f"\n  STACKING ENSEMBLE RESULTS (Held-Out Test Set, N={len(y_test)}):")
    print(f"  Macro F1         : {macro_f1:.4f}")
    print(f"  Macro Recall     : {macro_rec:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Overall Accuracy : {acc:.4f}")
    print(f"  F1  [N / S / P]  : {f1_pc[0]:.4f} / {f1_pc[1]:.4f} / {f1_pc[2]:.4f}")
    print(f"  Rec [N / S / P]  : {rec_pc[0]:.4f} / {rec_pc[1]:.4f} / {rec_pc[2]:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    plot_cm(cm, class_names,
            f'Stacking Ensemble Confusion Matrix (Macro F1 = {macro_f1:.4f})',
            'outputs/figures/confusion_matrix_stacking_ensemble.png', champion=True)
    print("  -> Saved outputs/figures/confusion_matrix_stacking_ensemble.png")

    # Compare against individual models from existing benchmark CSV
    bench_path = 'outputs/model_benchmark_comparison.csv'
    if os.path.exists(bench_path):
        bench = pd.read_csv(bench_path)
        # Add stacking row
        stack_row = {
            'Model': 'Stacking Ensemble (LGB+XGB+RF+SVM)',
            'Family': 'Meta-Learner (Stacking)',
            'Macro F1': round(macro_f1, 4),
            'Macro Precision': round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
            'Macro Recall': round(macro_rec, 4),
            'F1 (Normal)': round(f1_pc[0], 4),
            'F1 (Suspect)': round(f1_pc[1], 4),
            'F1 (Pathologic)': round(f1_pc[2], 4),
            'Recall (Pathologic)': round(rec_pc[2], 4),
            'Balanced Accuracy': round(bal_acc, 4),
            'Overall Accuracy': round(acc, 4),
        }
        bench = pd.concat([bench, pd.DataFrame([stack_row])], ignore_index=True)
        bench = bench.sort_values('Macro F1', ascending=False).reset_index(drop=True)
        bench.to_csv(bench_path, index=False)
        print(f"\n  Updated {bench_path} with stacking row.")
        print(f"\n  TOP-3 MODEL LEADERBOARD (Held-Out):")
        print(bench[['Model', 'Macro F1', 'Recall (Pathologic)']].head(3).to_string(index=False))

    # Save ensemble Macro F1 for downstream comparison
    with open('outputs/stacking_ensemble_result.json', 'w') as f:
        json.dump({'macro_f1': round(macro_f1, 4),
                   'pathologic_recall': round(rec_pc[2], 4),
                   'balanced_accuracy': round(bal_acc, 4)}, f, indent=2)

    print("\n  Phase D COMPLETE.")


if __name__ == '__main__':
    main()
