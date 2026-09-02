"""
PHASE A: Advanced Clinical Feature Engineering
Unified canonical feature engineering module for training and serving.
All 18 engineered features are grounded in FIGO 2015 / ACOG clinical cardiotocography.

Zero train/serve skew: used identically in:
  - Batch dataset preparation (DataFrame)
  - Real-time sliding-window telemetry (src/live_stream_engine.py)
  - Interactive clinical drafting (app.py)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

BASE_FEATURE_NAMES = [
    'LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP',
    'ASTV', 'MSTV', 'ALTV', 'MLTV',
    'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
    'Mode', 'Mean', 'Median', 'Variance', 'Tendency'
]

DERIVED_FEATURE_NAMES = [
    'DSI', 'VCR', 'FHR_Dev', 'Contraction_Decel_Coupling',
    'Autonomic_Reactivity_Index', 'Hist_Spread_Ratio',
    'PRI', 'Decel_Pattern_Severity', 'Autonomic_Balance_Ratio',
    'FHR_Instability_Score', 'UC_AC_Coupling', 'Morphological_Complexity',
    'Contraction_Load_Index', 'Basal_Reactivity_Score',
    'STV_LTV_Ratio', 'Hist_Skew_Proxy', 'Zero_Crossing_Density',
    'Variability_Entropy'
]

ALL_FEATURE_NAMES = BASE_FEATURE_NAMES + DERIVED_FEATURE_NAMES


def compute_derived_features(data):
    """
    Computes all 18 derived clinical biomarkers for cardiotocography.
    Accepts either:
      - pd.DataFrame: vectorized operations across all rows (training/batch).
      - dict or pd.Series: scalar operations for a single patient (serving/UI).
    Returns the updated data with derived features added.
    """
    eps = 1e-6

    if isinstance(data, pd.DataFrame):
        df = data.copy()

        # 1. Deceleration Severity Index (DSI)
        df['DSI'] = (df['DL'] + 2.0 * df['DS'] + 3.0 * df['DP']) / (df['UC'] + eps)

        # 2. Variability Collapse Ratio (VCR)
        df['VCR'] = (df['ASTV'] * df['ALTV']) / (df['MSTV'] * df['MLTV'] + eps)

        # 3. FHR baseline deviation from optimal 140 bpm
        df['FHR_Dev'] = np.abs(df['LB'] - 140.0)

        # 4. Contraction-Deceleration Coupling
        df['Contraction_Decel_Coupling'] = df['UC'] * (df['DL'] + 2.0 * df['DS'] + 3.0 * df['DP'])

        # 5. Autonomic Reactivity Index
        df['Autonomic_Reactivity_Index'] = df['AC'] / (df['ASTV'] + eps)

        # 6. Histogram Spread Ratio
        df['Hist_Spread_Ratio'] = df['Width'] / (df['Max'] + eps)

        # 7. Pathologic Risk Index (PRI) - FIGO weighted composite
        df['PRI'] = (
            (df['ASTV'] / 100.0) * 3.0 +
            (df['ALTV'] / 100.0) * 2.0 +
            np.clip(df['DP'] * 1000.0, 0, 5) * 2.5 +
            np.clip(df['DS'] * 500.0, 0, 3) * 1.5 -
            np.clip(df['AC'] * 500.0, 0, 5) * 2.0 -
            np.clip((df['MSTV'] - 1.0), 0, 5) * 0.5
        )

        # 8. Decel Pattern Severity
        df['Decel_Pattern_Severity'] = (
            df['DL'] +
            np.where(df['DS'] > 0, df['DS'] * 3.0, 0) +
            np.where(df['DP'] > 0, df['DP'] * 5.0, 0)
        )

        # 9. Autonomic Balance Ratio
        df['Autonomic_Balance_Ratio'] = df['AC'] / ((df['ASTV'] + df['ALTV']) / 100.0 + eps)

        # 10. FHR Instability Score
        df['FHR_Instability_Score'] = df['FHR_Dev'] * (df['MLTV'] + 1.0) / (df['MSTV'] + eps)

        # 11. UC-AC Coupling
        df['UC_AC_Coupling'] = df['AC'] / (df['UC'] + eps)

        # 12. Morphological Complexity
        df['Morphological_Complexity'] = (df['Max'] - df['Min']) * df['Nmax'] / (df['Variance'] + eps)

        # 13. Contraction Load Index
        df['Contraction_Load_Index'] = df['UC'] * df['Width']

        # 14. Basal Reactivity Score
        df['Basal_Reactivity_Score'] = (df['AC'] / (df['UC'] + eps)) * (1.0 / (df['FHR_Dev'] + 1.0))

        # 15. STV/LTV Ratio
        df['STV_LTV_Ratio'] = df['MSTV'] / (df['MLTV'] + eps)

        # 16. Histogram Skew Proxy
        df['Hist_Skew_Proxy'] = (df['Mode'] - df['Mean']) / (df['Width'] + eps)

        # 17. Zero Crossing Density
        df['Zero_Crossing_Density'] = df['Nzeros'] / (df['Width'] + eps)

        # 18. Variability Entropy (HRV spectral complexity proxy)
        def _row_entropy(row):
            p = np.array([
                max(float(row['ASTV']), 0.01),
                max(float(row['ALTV']), 0.01),
                max(100.0 - float(row['ASTV']) - float(row['ALTV']), 0.01)
            ])
            p = p / p.sum()
            return float(scipy_entropy(p, base=2))

        df['Variability_Entropy'] = df.apply(_row_entropy, axis=1)
        return df

    else:
        # Scalar dictionary or Series
        res = dict(data)

        lb = float(res.get('LB', 140.0))
        ac = float(res.get('AC', 0.0))
        uc = float(res.get('UC', 0.005))
        dl = float(res.get('DL', 0.0))
        ds = float(res.get('DS', 0.0))
        dp = float(res.get('DP', 0.0))
        astv = float(res.get('ASTV', 25.0))
        mstv = float(res.get('MSTV', 1.5))
        altv = float(res.get('ALTV', 0.0))
        mltv = float(res.get('MLTV', 10.0))
        width = float(res.get('Width', 60.0))
        f_min = float(res.get('Min', 110.0))
        f_max = float(res.get('Max', 170.0))
        nmax = float(res.get('Nmax', 3.0))
        nzeros = float(res.get('Nzeros', 0.0))
        mode = float(res.get('Mode', 140.0))
        mean = float(res.get('Mean', 140.0))
        variance = float(res.get('Variance', 8.0))

        # 1. DSI
        res['DSI'] = (dl + 2.0 * ds + 3.0 * dp) / (uc + eps)

        # 2. VCR
        res['VCR'] = (astv * altv) / (mstv * mltv + eps)

        # 3. FHR_Dev
        res['FHR_Dev'] = abs(lb - 140.0)

        # 4. Contraction_Decel_Coupling
        res['Contraction_Decel_Coupling'] = uc * (dl + 2.0 * ds + 3.0 * dp)

        # 5. Autonomic_Reactivity_Index
        res['Autonomic_Reactivity_Index'] = ac / (astv + eps)

        # 6. Hist_Spread_Ratio
        res['Hist_Spread_Ratio'] = width / (f_max + eps)

        # 7. PRI
        res['PRI'] = (
            (astv / 100.0) * 3.0 +
            (altv / 100.0) * 2.0 +
            min(dp * 1000.0, 5.0) * 2.5 +
            min(ds * 500.0, 3.0) * 1.5 -
            min(ac * 500.0, 5.0) * 2.0 -
            min(max(mstv - 1.0, 0.0), 5.0) * 0.5
        )

        # 8. Decel_Pattern_Severity
        res['Decel_Pattern_Severity'] = dl + (ds * 3.0 if ds > 0 else 0.0) + (dp * 5.0 if dp > 0 else 0.0)

        # 9. Autonomic_Balance_Ratio
        res['Autonomic_Balance_Ratio'] = ac / ((astv + altv) / 100.0 + eps)

        # 10. FHR_Instability_Score
        res['FHR_Instability_Score'] = res['FHR_Dev'] * (mltv + 1.0) / (mstv + eps)

        # 11. UC_AC_Coupling
        res['UC_AC_Coupling'] = ac / (uc + eps)

        # 12. Morphological_Complexity
        res['Morphological_Complexity'] = (f_max - f_min) * nmax / (variance + eps)

        # 13. Contraction_Load_Index
        res['Contraction_Load_Index'] = uc * width

        # 14. Basal_Reactivity_Score
        res['Basal_Reactivity_Score'] = (ac / (uc + eps)) * (1.0 / (res['FHR_Dev'] + 1.0))

        # 15. STV_LTV_Ratio
        res['STV_LTV_Ratio'] = mstv / (mltv + eps)

        # 16. Hist_Skew_Proxy
        res['Hist_Skew_Proxy'] = (mode - mean) / (width + eps)

        # 17. Zero_Crossing_Density
        res['Zero_Crossing_Density'] = nzeros / (width + eps)

        # 18. Variability_Entropy
        p = np.array([
            max(astv, 0.01),
            max(altv, 0.01),
            max(100.0 - astv - altv, 0.01)
        ])
        p = p / p.sum()
        res['Variability_Entropy'] = float(scipy_entropy(p, base=2))

        return res


def engineer_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper calling the canonical compute_derived_features implementation."""
    return compute_derived_features(df)


def main():
    print("=================================================================")
    print("  PHASE A: ADVANCED CLINICAL FEATURE ENGINEERING                 ")
    print("=================================================================")

    in_path = os.path.join('datasets', 'uci_ctg', 'CTG_cleaned.csv')
    if not os.path.exists(in_path):
        from preprocess import load_and_preprocess_data
        print(f"Generating cleaned dataset via preprocess.py...")
        df_clean, _, _ = load_and_preprocess_data()
    else:
        df_clean = pd.read_csv(in_path)

    print(f"Input Cleaned Dataset: {df_clean.shape} ({df_clean.shape[1]-1} base features)")

    df_eng = compute_derived_features(df_clean)
    out_path = os.path.join('datasets', 'uci_ctg', 'CTG_features_engineered.csv')
    df_eng.to_csv(out_path, index=False)

    print(f"Engineered Dataset Saved: {out_path}")
    print(f"Shape: {df_eng.shape} ({df_eng.shape[1]-1} features total, +{len(DERIVED_FEATURE_NAMES)} engineered)")
    print(f"Derived Features: {', '.join(DERIVED_FEATURE_NAMES)}")
    print("Feature engineering complete with zero train/serve skew.")


if __name__ == '__main__':
    main()
