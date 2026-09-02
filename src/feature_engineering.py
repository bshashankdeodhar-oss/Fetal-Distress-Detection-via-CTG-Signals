"""
PHASE A: Advanced Clinical Feature Engineering (v2)
Expands feature space from 6 engineered features to 15+
All features grounded in FIGO/ACOG cardiotocography pathophysiology.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

def engineer_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full v2 feature engineering pipeline to a raw CTG dataframe.
    Input columns expected: LB, AC, FM, UC, DL, DS, DP, ASTV, MSTV, ALTV, MLTV,
                            Width, Min, Max, Nmax, Nzeros, Mode, Mean, Median, Variance, Tendency
    """
    df = df.copy()
    eps = 1e-6

    # --- ORIGINAL v1 FEATURES ---
    # Deceleration Severity Index: weights prolonged > severe > light decelerations
    # Normalised by contraction frequency to measure coupling intensity
    df['DSI'] = (df['DL'] + 2.0 * df['DS'] + 3.0 * df['DP']) / (df['UC'] + eps)

    # Variability Collapse Ratio: cross-multiplies short and long term abnormal
    # variability against their mean counterparts — acute autonomic failure signal
    df['VCR'] = (df['ASTV'] * df['ALTV']) / (df['MSTV'] * df['MLTV'] + eps)

    # FHR deviation from the physiological optimum of 140 bpm
    df['FHR_Dev'] = np.abs(df['LB'] - 140.0)

    # Contraction-Deceleration Coupling: uterine load × total deceleration burden
    df['Contraction_Decel_Coupling'] = df['UC'] * (df['DL'] + 2.0 * df['DS'] + 3.0 * df['DP'])

    # Autonomic Reactivity Index: accelerations suppress pathological variance
    df['Autonomic_Reactivity_Index'] = df['AC'] / (df['ASTV'] + eps)

    # Histogram Spread Ratio: width of FHR histogram vs max value
    df['Hist_Spread_Ratio'] = df['Width'] / (df['Max'] + eps)

    # --- NEW v2 FEATURES ---

    # 1. Pathologic Risk Index (PRI): composite weighted risk score
    # Derived from the FIGO 2015 classification thresholds
    df['PRI'] = (
        (df['ASTV'] / 100.0) * 3.0 +
        (df['ALTV'] / 100.0) * 2.0 +
        np.clip(df['DP'] * 1000, 0, 5) * 2.5 +
        np.clip(df['DS'] * 500, 0, 3) * 1.5 -
        np.clip(df['AC'] * 500, 0, 5) * 2.0 -
        np.clip((df['MSTV'] - 1.0), 0, 5) * 0.5
    )

    # 2. Deceleration Pattern Severity: detects whether decelerations are
    # isolated (light) or cascading (severe + prolonged together)
    df['Decel_Pattern_Severity'] = (
        df['DL'] +
        np.where(df['DS'] > 0, df['DS'] * 3.0, 0) +
        np.where(df['DP'] > 0, df['DP'] * 5.0, 0)
    )

    # 3. Autonomic Balance Ratio: ratio of reactive (AC) to pathological (ASTV+ALTV)
    # High ratio = healthy parasympathetic tone; Low ratio = autonomic failure
    df['Autonomic_Balance_Ratio'] = df['AC'] / ((df['ASTV'] + df['ALTV']) / 100.0 + eps)

    # 4. FHR Instability Score: combines baseline deviation with long-term variability
    df['FHR_Instability_Score'] = df['FHR_Dev'] * (df['MLTV'] + 1.0) / (df['MSTV'] + eps)

    # 5. UC-AC Coupling: do accelerations occur in proportion to contractions?
    # A healthy fetal nervous system produces accelerations during contractions.
    df['UC_AC_Coupling'] = df['AC'] / (df['UC'] + eps)

    # 6. Morphological Complexity: breadth of the FHR histogram
    # A narrow, peaked histogram = reduced variability = pathological
    df['Morphological_Complexity'] = (df['Max'] - df['Min']) * df['Nmax'] / (df['Variance'] + eps)

    # 7. Contraction Load Index: total contraction burden per minute equivalent
    df['Contraction_Load_Index'] = df['UC'] * df['Width']

    # 8. Basal Reactivity Score: accelerations per contraction normalised to baseline
    df['Basal_Reactivity_Score'] = (df['AC'] / (df['UC'] + eps)) * (1.0 / (df['FHR_Dev'] + 1.0))

    # 9. Short-vs-Long Term Variability Ratio: differential between micro and macro HRV
    # In acute hypoxia, MSTV drops sharply while MLTV may briefly spike
    df['STV_LTV_Ratio'] = df['MSTV'] / (df['MLTV'] + eps)

    # 10. Histogram Skew Proxy: distance between mode and mean relative to width
    # Negative skew = more values above modal FHR = tachycardia compensation
    df['Hist_Skew_Proxy'] = (df['Mode'] - df['Mean']) / (df['Width'] + eps)

    # 11. Zero-crossing Density: Nzeros relative to histogram breadth
    # High zero crossing = flat, non-reactive trace
    df['Zero_Crossing_Density'] = df['Nzeros'] / (df['Width'] + eps)

    # 12. Variability Entropy (proxy): using ASTV + ALTV distribution spread
    # Mimics spectral entropy of HRV — low entropy = loss of complexity = hypoxia
    def _variability_entropy(row):
        probs = np.array([
            max(row['ASTV'], 0.01),
            max(row['ALTV'], 0.01),
            max(100 - row['ASTV'] - row['ALTV'], 0.01)
        ])
        probs = probs / probs.sum()
        return float(scipy_entropy(probs, base=2))

    df['Variability_Entropy'] = df.apply(_variability_entropy, axis=1)

    return df


def main():
    print("=================================================================")
    print("  PHASE A: ADVANCED CLINICAL FEATURE ENGINEERING v2              ")
    print("=================================================================")

    # Load cleaned baseline dataset
    in_path = os.path.join('datasets', 'uci_ctg', 'CTG_cleaned.csv')
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Could not find: {in_path}")

    df = pd.read_csv(in_path)
    target = df['NSP'].copy()

    # Drop any previously engineered v1 columns to avoid duplication
    v1_cols = ['DSI', 'VCR', 'FHR_Dev', 'Contraction_Decel_Coupling',
               'Autonomic_Reactivity_Index', 'Hist_Spread_Ratio']
    df_base = df.drop(columns=[c for c in v1_cols if c in df.columns])

    # Remove target before engineering, then re-attach
    df_features = df_base.drop(columns=['NSP'])
    df_engineered = engineer_features_v2(df_features)
    df_engineered['NSP'] = target.values

    out_path = os.path.join('datasets', 'uci_ctg', 'CTG_features_engineered.csv')
    df_engineered.to_csv(out_path, index=False)

    n_orig = df_features.shape[1]
    n_new = df_engineered.shape[1] - 1  # exclude NSP
    print(f"  Original features : {n_orig}")
    print(f"  Engineered features: {n_new}  (+{n_new - n_orig} new)")
    print(f"  Dataset shape      : {df_engineered.shape}")
    print(f"  Saved to           : {out_path}")

    # Print new feature names
    orig_set = set(df_features.columns)
    new_feats = [c for c in df_engineered.columns if c not in orig_set and c != 'NSP']
    print(f"\n  New features added ({len(new_feats)}):")
    for f in new_feats:
        print(f"    + {f}")

    print("\n  Phase A COMPLETE.")


if __name__ == '__main__':
    main()
