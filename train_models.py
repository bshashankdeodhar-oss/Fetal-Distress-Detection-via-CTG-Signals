import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import shap

from preprocess import load_and_preprocess_data

def plot_confusion_matrix(cm, class_names, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.ylabel('True Class', fontsize=10)
    plt.xlabel('Predicted Class', fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_feature_importance(importances, feature_names, title, filename, top_n=15):
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.barh(range(len(indices)), importances[indices][::-1], color='#3b82f6', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1], fontsize=9)
    plt.xlabel('Relative Importance / Weight', fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    print("==================================================")
    print("  FETAL DISTRESS DETECTION FROM CTG SIGNALS PIPELINE")
    print("==================================================")
    
    # 1. Load Data
    df, feature_names, target_name = load_and_preprocess_data()
    X = df[feature_names].values
    y = df[target_name].values - 1  # Map 1, 2, 3 -> 0 (Normal), 1 (Suspect), 2 (Pathologic)
    
    class_names = ['Normal', 'Suspect', 'Pathologic']
    
    # 2. Stratified Train / Held-out Test Split (80% Train, 20% Held-Out Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"\nDataset Splitting:")
    print(f"  Training Set: {X_train.shape[0]} samples")
    print(f"  Held-out Test Set: {X_test.shape[0]} samples")
    print(f"  Class breakdown in Test: Normal={np.sum(y_test==0)}, Suspect={np.sum(y_test==1)}, Pathologic={np.sum(y_test==2)}")
    
    # 3. Preprocessing (Standard Scaling for linear/SVM/neural, raw/scaled for trees)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Model Zoo Definition (Multiple Distinct Model Families)
    models = {
        'LightGBM (Ensemble/Tree)': {
            'model': lgb.LGBMClassifier(
                n_estimators=200, 
                learning_rate=0.05, 
                class_weight='balanced', 
                random_state=42, 
                verbose=-1
            ),
            'scaled': False,
            'family': 'Gradient Boosted Decision Trees'
        },
        'Random Forest (Ensemble/Tree)': {
            'model': RandomForestClassifier(
                n_estimators=300, 
                class_weight='balanced_subsample', 
                random_state=42,
                max_depth=12,
                min_samples_split=4
            ),
            'scaled': False,
            'family': 'Bagged Decision Trees'
        },
        'Cost-Sensitive Logistic Regression (Linear)': {
            'model': LogisticRegression(
                class_weight='balanced', 
                max_iter=1000, 
                C=1.0, 
                random_state=42
            ),
            'scaled': True,
            'family': 'Linear Margin Models'
        },
        'Support Vector Machine (Kernel/Margin)': {
            'model': SVC(
                C=2.0, 
                kernel='rbf', 
                class_weight='balanced', 
                probability=True, 
                random_state=42
            ),
            'scaled': True,
            'family': 'Kernel Margin Models'
        },
        'Multi-Layer Perceptron (Neural)': {
            'model': MLPClassifier(
                hidden_layer_sizes=(128, 64), 
                activation='relu', 
                max_iter=500, 
                early_stopping=True, 
                random_state=42
            ),
            'scaled': True,
            'family': 'Deep Neural Networks'
        }
    }
    
    results = []
    trained_models = {}
    
    print("\n--- Training and Evaluating Models on Held-out Split ---")
    for name, config in models.items():
        clf = config['model']
        is_scaled = config['scaled']
        family = config['family']
        
        X_tr = X_train_scaled if is_scaled else X_train
        X_te = X_test_scaled if is_scaled else X_test
        
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        
        f1_per_class = f1_score(y_test, y_pred, average=None)
        rec_per_class = recall_score(y_test, y_pred, average=None)
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Save Confusion Matrix Plot
        clean_slug = name.split()[0].lower().replace('-', '_')
        cm_filename = f"outputs/figures/confusion_matrix_{clean_slug}.png"
        plot_confusion_matrix(cm, class_names, f"Confusion Matrix: {name}", cm_filename)
        
        results.append({
            'Model': name,
            'Family': family,
            'Macro F1': macro_f1,
            'Macro Precision': macro_prec,
            'Macro Recall': macro_rec,
            'F1 (Normal)': f1_per_class[0],
            'F1 (Suspect)': f1_per_class[1],
            'F1 (Pathologic)': f1_per_class[2],
            'Recall (Pathologic)': rec_per_class[2],
            'Overall Accuracy': acc,
            'Balanced Accuracy': bal_acc,
            'Confusion Matrix': cm.tolist()
        })
        
        trained_models[name] = {
            'clf': clf,
            'scaled': is_scaled,
            'cm': cm,
            'y_pred': y_pred
        }
        
        print(f"\nModel: {name} [{family}]")
        print(f"  Macro F1: {macro_f1:.4f} | Balanced Acc: {bal_acc:.4f} | Acc: {acc:.4f}")
        print(f"  Pathologic F1: {f1_per_class[2]:.4f} | Pathologic Recall: {rec_per_class[2]:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        
    results_df = pd.DataFrame(results).sort_values(by='Macro F1', ascending=False)
    results_df.to_csv('outputs/model_benchmark_comparison.csv', index=False)
    
    print("\n--- Summary Benchmark Comparison Table ---")
    print(results_df[['Model', 'Family', 'Macro F1', 'Macro Precision', 'Macro Recall', 'F1 (Pathologic)', 'Recall (Pathologic)']].to_string(index=False))
    
    # 5. Feature Importance & SHAP Analysis
    print("\n--- Conducting Feature Importance & SHAP Analysis ---")
    
    # (A) Tree Importance for LightGBM & Random Forest
    lgb_model = trained_models['LightGBM (Ensemble/Tree)']['clf']
    lgb_importances = lgb_model.feature_importances_ / np.sum(lgb_model.feature_importances_)
    plot_feature_importance(
        lgb_importances, 
        feature_names, 
        "Feature Importance: LightGBM (Gain/Split)", 
        "outputs/figures/feature_importance_lightgbm.png"
    )
    
    rf_model = trained_models['Random Forest (Ensemble/Tree)']['clf']
    rf_importances = rf_model.feature_importances_
    plot_feature_importance(
        rf_importances, 
        feature_names, 
        "Feature Importance: Random Forest (MDI)", 
        "outputs/figures/feature_importance_random_forest.png"
    )
    
    # (B) Permutation Importance for SVM & Logistic Regression
    svm_model = trained_models['Support Vector Machine (Kernel/Margin)']['clf']
    perm_svm = permutation_importance(svm_model, X_test_scaled, y_test, n_repeats=10, random_state=42, scoring='f1_macro')
    plot_feature_importance(
        perm_svm.importances_mean, 
        feature_names, 
        "Permutation Importance (Held-out Macro F1): SVM (RBF)", 
        "outputs/figures/feature_importance_svm.png"
    )
    
    lr_model = trained_models['Cost-Sensitive Logistic Regression (Linear)']['clf']
    lr_weights = np.mean(np.abs(lr_model.coef_), axis=0)
    plot_feature_importance(
        lr_weights, 
        feature_names, 
        "Mean Absolute Coefficients: Logistic Regression", 
        "outputs/figures/feature_importance_logistic.png"
    )
    
    # (C) SHAP Analysis for LightGBM (TreeExplainer)
    try:
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_test)
        
        # Save SHAP Summary Plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values, 
            X_test, 
            feature_names=feature_names, 
            class_names=class_names, 
            show=False
        )
        plt.title("SHAP Multi-Class Summary Plot: LightGBM", fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig("outputs/figures/shap_summary_lightgbm.png", dpi=300)
        plt.close()
        print("SHAP multi-class summary plot generated successfully.")
    except Exception as e:
        print("SHAP TreeExplainer exception:", e)
        
    print("\n==================================================")
    print("  EXPERIMENTS COMPLETE. ALL ARTIFACTS GENERATED.")
    print("==================================================")

if __name__ == '__main__':
    main()
