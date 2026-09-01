import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import shap

# Create output directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

def load_and_preprocess_data():
    excel_path = 'data/CTG.xls'
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"{excel_path} not found.")
    
    # Read the 'Raw Data' or 'Data' sheet
    xl = pd.ExcelFile(excel_path)
    sheet_to_use = 'Raw Data' if 'Raw Data' in xl.sheet_names else xl.sheet_names[0]
    df = xl.parse(sheet_to_use)
    
    print(f"Loaded sheet '{sheet_to_use}' with shape: {df.shape}")
    
    # The 21 clinical feature names
    feature_cols = [
        'LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP',
        'ASTV', 'MSTV', 'ALTV', 'MLTV',
        'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
        'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
    ]
    target_col = 'NSP'
    
    # Check if column names match or need cleaning
    # Find matching columns
    cols_found = {}
    for req in feature_cols + [target_col]:
        matched = [c for c in df.columns if str(c).strip().upper() == req.upper()]
        if matched:
            cols_found[req] = matched[0]
        else:
            print(f"Warning: column {req} not found exactly. Searching...")
            
    print("Found columns:", cols_found)
    
    # Extract clean dataframe
    df_clean = df[[cols_found[c] for c in feature_cols + [target_col]]].copy()
    df_clean.columns = feature_cols + [target_col]
    
    # Drop rows where target is NaN or non-numeric
    df_clean = df_clean.dropna(subset=[target_col])
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[target_col])
    df_clean[target_col] = df_clean[target_col].astype(int)
    
    # Ensure all feature columns are numeric
    for c in feature_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
    df_clean = df_clean.dropna()
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print("Target distribution:")
    class_mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathologic'}
    counts = df_clean[target_col].value_counts().sort_index()
    for k, v in counts.items():
        print(f"  Class {k} ({class_mapping.get(k, 'Unknown')}): {v} samples ({v/len(df_clean)*100:.2f}%)")
        
    df_clean.to_csv('data/CTG_cleaned.csv', index=False)
    return df_clean, feature_cols, target_col

if __name__ == '__main__':
    df, features, target = load_and_preprocess_data()
