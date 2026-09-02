"""
PHASE B: Optuna Bayesian Hyperparameter Optimization
100-trial TPE study on LightGBM optimising 5-Fold Stratified CV Macro F1.
Exports best_params_lgb.json and a convergence/importance plot.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

def load_data():
    path = os.path.join('datasets', 'uci_ctg', 'CTG_features_engineered.csv')
    df = pd.read_csv(path)
    X = df.drop(columns=['NSP']).values
    y = df['NSP'].values - 1
    return X, y

def objective(trial, X, y):
    params = {
        'n_estimators'      : trial.suggest_int('n_estimators', 100, 600),
        'learning_rate'     : trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'num_leaves'        : trial.suggest_int('num_leaves', 16, 96),
        'max_depth'         : trial.suggest_int('max_depth', 4, 12),
        'min_child_samples' : trial.suggest_int('min_child_samples', 5, 50),
        'subsample'         : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree'  : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha'         : trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
        'reg_lambda'        : trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
        'class_weight'      : 'balanced',
        'random_state'      : 42,
        'verbose'           : -1,
    }
    clf = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    return scores.mean()

def main():
    print("=================================================================")
    print("  PHASE B: OPTUNA BAYESIAN HYPERPARAMETER OPTIMISATION           ")
    print("=================================================================")

    X, y = load_data()
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print("  Running 100-trial Optuna TPE study (this takes ~3-5 min)...\n")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
    )
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100, show_progress_bar=False)

    best = study.best_params
    best_val = study.best_value
    print(f"\n  Best 5-Fold CV Macro F1: {best_val:.4f}")
    print(f"  Best hyperparameters:")
    for k, v in best.items():
        print(f"    {k}: {v}")

    # Save best params
    out_json = 'outputs/best_params_lgb.json'
    with open(out_json, 'w') as f:
        json.dump({'best_cv_macro_f1': round(best_val, 4), 'params': best}, f, indent=2)
    print(f"\n  -> Saved {out_json}")

    # Plot optimization history
    trials_df = study.trials_dataframe()
    rolling_best = trials_df['value'].cummax()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(trials_df['number'], trials_df['value'], alpha=0.4, color='#6b7280', linewidth=0.8, label='Trial Score')
    axes[0].plot(trials_df['number'], rolling_best, color='#2563eb', linewidth=2.0, label='Cumulative Best')
    axes[0].axhline(best_val, color='#dc2626', linestyle='--', linewidth=1.2, label=f'Best = {best_val:.4f}')
    axes[0].set_xlabel('Trial Number')
    axes[0].set_ylabel('5-Fold CV Macro F1')
    axes[0].set_title('Optuna Convergence History (LightGBM)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # Hyperparameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        names = list(importances.keys())
        vals  = list(importances.values())
        axes[1].barh(names[::-1], vals[::-1], color='#2563eb', edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel('Importance Score')
        axes[1].set_title('Hyperparameter Importance (fANOVA)', fontweight='bold')
        axes[1].grid(axis='x', alpha=0.4)
    except Exception:
        axes[1].text(0.5, 0.5, 'Importance not available', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('outputs/figures/optuna_convergence.png', dpi=300)
    plt.close()
    print("  -> Saved outputs/figures/optuna_convergence.png")
    print("\n  Phase B COMPLETE.")

if __name__ == '__main__':
    main()
