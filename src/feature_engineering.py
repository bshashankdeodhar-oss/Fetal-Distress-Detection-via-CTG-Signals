import os
import sys
import numpy as np
import pandas as pd

def engineer_clinical_features(df_input=None):
    print("==================================================")
    print("  PHASE 2: BIOMEDICAL FEATURE ENGINEERING         ")
    print("==================================================")
    
    if df_input is None:
        csv_path = os.path.join('datasets', 'uci_ctg', 'CTG_cleaned.csv')
        df = pd.read_csv(csv_path)
    else:
        df = df_input.copy()
        
    print(f"Original Feature Matrix: {df.shape[1] - 1} features across {df.shape[0]} records.")
    
    eps = 1e-5
    
    # 1. Deceleration Severity Index (DSI)
    # Severe and prolonged decelerations are given much higher clinical weights relative to uterine contractions
    df['DSI'] = (df['DL'] + 2.0 * df['DS'] + 3.0 * df['DP']) / (df['UC'] + eps)
    
    # 2. Variability Collapse Ratio (VCR)
    # Severe hypoxia causes ASTV and ALTV to surge while MSTV and MLTV collapse
    df['VCR'] = (df['ASTV'] * df['ALTV']) / ((df['MSTV'] * df['MLTV']) + eps)
    
    # 3. Baseline FHR Deviation (Tachycardia/Bradycardia distance from 140 bpm)
    df['FHR_Dev'] = (df['LB'] - 140.0).abs()
    
    # 4. Contraction-Deceleration Coupling (Uterine stress response)
    df['Contraction_Decel_Coupling'] = df['UC'] * (df['DL'] + 2.0 * df['DS'] + 3.0 * df['DP'])
    
    # 5. Autonomic Reactivity Index (Accelerations vs Abnormal Short Term Variability)
    df['Autonomic_Reactivity_Index'] = df['AC'] / (df['ASTV'] + eps)
    
    # 6. Histogram Skew Dynamic Ratio
    df['Hist_Spread_Ratio'] = df['Width'] / (df['Max'] + eps)
    
    print(f"Engineered 6 Domain Features. New Matrix: {df.shape[1] - 1} features across {df.shape[0]} records.")
    print("New Engineered Columns:", ['DSI', 'VCR', 'FHR_Dev', 'Contraction_Decel_Coupling', 'Autonomic_Reactivity_Index', 'Hist_Spread_Ratio'])
    
    out_csv = os.path.join('datasets', 'uci_ctg', 'CTG_features_engineered.csv')
    df.to_csv(out_csv, index=False)
    print(f"  -> Saved engineered dataset to {out_csv}")
    
    print("\n==================================================")
    print("  PHASE 2 COMPLETED SUCCESSFULLY!                ")
    print("==================================================")
    return df

if __name__ == '__main__':
    engineer_clinical_features()
