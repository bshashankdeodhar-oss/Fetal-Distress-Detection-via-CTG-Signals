import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import shap

st.set_page_config(
    page_title="CTG-ENGINEERING // FETAL DISTRESS MASTER BLUEPRINT",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Architectural Blueprint Custom CSS Injection (Desktop Taskbar Mode)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Architects+Daughter&family=Roboto+Mono:wght@300;400;600;700&display=swap');
    
    /* Hide Streamlit default sidebar & header decoration */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }

    /* Master Blueprint Background & Grid */
    .stApp {
        background-color: #002b4e !important;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.05) 1px, transparent 1px),
            linear-gradient(rgba(0, 255, 255, 0.12) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.12) 1px, transparent 1px) !important;
        background-size: 20px 20px, 20px 20px, 100px 100px, 100px 100px !important;
        color: rgba(255, 255, 255, 0.88) !important;
        font-family: 'Roboto Mono', monospace !important;
    }

    /* Desktop Taskbar / Command Dock */
    .desktop-taskbar {
        background: rgba(0, 20, 40, 0.92);
        border: 1px solid #00ffff;
        box-shadow: 0 4px 20px rgba(0, 255, 255, 0.15);
        padding: 8px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        position: relative;
    }
    
    .taskbar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .taskbar-icon {
        font-size: 20px;
    }
    .taskbar-title {
        color: #ffffff;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }
    .taskbar-spec {
        color: #00ffff;
        font-size: 10px;
        letter-spacing: 1px;
    }
    
    .taskbar-status {
        border: 1px solid #00ff88;
        background: rgba(0, 255, 136, 0.1);
        color: #00ff88;
        font-size: 10px;
        font-weight: 700;
        padding: 3px 8px;
        letter-spacing: 1px;
    }

    /* Dimension Line */
    .dimension-line {
        border-top: 1px dashed rgba(0, 255, 255, 0.3);
        text-align: center;
        color: #00ffff;
        font-size: 10px;
        letter-spacing: 1px;
        padding-top: 4px;
        margin-bottom: 14px;
    }
    
    /* Redline Note */
    .redline-box {
        background-color: rgba(255, 51, 51, 0.12);
        border: 1px dashed #ff3333;
        padding: 8px 14px;
        color: #ff9999;
        font-family: 'Architects Daughter', cursive;
        font-size: 14px;
        margin-bottom: 14px;
    }
    
    /* Stamp Badges */
    .stamp-normal {
        border: 2px solid #00ff88;
        background: rgba(0, 255, 136, 0.08);
        color: #00ff88;
        padding: 8px 14px;
        text-align: center;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .stamp-suspect {
        border: 2px solid #ffcc00;
        background: rgba(255, 204, 0, 0.1);
        color: #ffcc00;
        padding: 8px 14px;
        text-align: center;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .stamp-pathologic {
        border: 2px solid #ff3333;
        background: rgba(255, 51, 51, 0.15);
        color: #ff3333;
        padding: 8px 14px;
        text-align: center;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train_models():
    csv_path = os.path.join('datasets', 'uci_ctg', 'CTG_features_engineered.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join('datasets', 'uci_ctg', 'CTG_cleaned.csv')
        
    df = pd.read_csv(csv_path)
    target_col = 'NSP'
    feature_names = [c for c in df.columns if c != target_col]
    
    X = df[feature_names].values
    y = df[target_col].values - 1  # 0: Normal, 1: Suspect, 2: Pathologic
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Champion LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=250, learning_rate=0.04, num_leaves=31, class_weight='balanced', random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight='balanced_subsample', random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Train SVM
    svm_model = SVC(
        C=2.5, kernel='rbf', class_weight='balanced', probability=True, random_state=42
    )
    svm_model.fit(X_train_scaled, y_train)
    
    return {
        'df': df,
        'feature_names': feature_names,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'lgb': lgb_model,
        'rf': rf_model,
        'svm': svm_model
    }

data_bundle = load_and_train_models()
feature_names = data_bundle['feature_names']
class_names = ['Normal', 'Suspect', 'Pathologic']

# ----------------- DESKTOP COMMAND TASKBAR -----------------
st.markdown("""
<div class="desktop-taskbar">
    <div class="taskbar-brand">
        <span class="taskbar-icon">📐</span>
        <div>
            <div class="taskbar-title">CTG-ENGINEERING // FETAL MONITORING DESKTOP</div>
            <div class="taskbar-spec">SPEC: CTG-2126 // SHEET NO. 01 // REV. 2026.09</div>
        </div>
    </div>
    <div class="taskbar-status">STATUS: ONLINE // COST_CALIBRATED P&ge;0.11</div>
</div>
""", unsafe_allow_html=True)

# Horizontal Command Controls Dock (Replacing bulky vertical sidebar)
taskbar_col1, taskbar_col2, taskbar_col3 = st.columns([1.2, 1.2, 1.0])

with taskbar_col1:
    selected_model_name = st.selectbox(
        "⚙️ CLASSIFIER INFERENCE FAMILY:",
        ["LightGBM (Gradient Boosted Trees)", "Random Forest (Bagged Ensembles)", "Support Vector Machine (SVC RBF)"]
    )

with taskbar_col2:
    scenario = st.selectbox(
        "📋 DRAFTING PRESET SCENARIO:",
        ["Custom Manual Drafting", "Preset A: Normal Reassuring", "Preset B: Borderline Reduced Variability", "Preset C: Acute Pathologic Distress"]
    )

with taskbar_col3:
    st.markdown("<div style='font-size:11px; color:#00ffff; margin-bottom:4px;'>📐 HELD-OUT COHORT:</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px; font-weight:700; color:#ffffff;'>426 PATIENTS (332N / 59S / 35P)<br><span style='color:#00ff88;'>MACRO F1: 0.9030</span></div>", unsafe_allow_html=True)

st.markdown('<div class="dimension-line">&larr; MASTER WORKSPACE DIMENSION: 100% FLUID RESPONSIVE GRID &rarr;</div>', unsafe_allow_html=True)

st.markdown("""
<div class="redline-box">
    <strong>[ENGINEER'S REDLINE // ASYMMETRIC LOSS ENFORCED]</strong>
    Clinical safety calibration: <strong>Cost(Missed Distress) = 10 &times; Cost(False Alarm)</strong>. 
    Pathologic trigger point set at <strong>P &ge; 0.110</strong> (Recall: 94.29%).
</div>
""", unsafe_allow_html=True)

default_values = {
    'LB': 135.0, 'AC': 0.003, 'FM': 0.0, 'UC': 0.005, 'DL': 0.0, 'DS': 0.0, 'DP': 0.0,
    'ASTV': 28.0, 'MSTV': 1.8, 'ALTV': 0.0, 'MLTV': 8.5, 'Width': 60.0, 'Min': 110.0,
    'Max': 170.0, 'Nmax': 3.0, 'Nzeros': 0.0, 'Mode': 138.0, 'Mean': 136.0, 'Median': 137.0,
    'Variance': 8.0, 'Tendency': 0.0
}

if scenario == "Preset A: Normal Reassuring":
    default_values.update({'LB': 135.0, 'AC': 0.005, 'ASTV': 22.0, 'ALTV': 0.0, 'DP': 0.0, 'MSTV': 2.1})
elif scenario == "Preset B: Borderline Reduced Variability":
    default_values.update({'LB': 152.0, 'AC': 0.001, 'ASTV': 58.0, 'ALTV': 18.0, 'DP': 0.0, 'MSTV': 0.8})
elif scenario == "Preset C: Acute Pathologic Distress":
    default_values.update({'LB': 168.0, 'AC': 0.000, 'ASTV': 78.0, 'ALTV': 45.0, 'DP': 0.003, 'DS': 0.001, 'MSTV': 0.4})

tabs = st.tabs(["📐 [01.0] Live Parameter Drafting", "📊 [02.0] Multi-Family Benchmark", "📈 [03.0] SHAP Impact Vectors"])

# TAB 1: PARAMETER DRAFTING
with tabs[0]:
    col1, col2 = st.columns([1.1, 1.3])
    
    inputs = {}
    with col1:
        st.markdown("### [ SEC 01.1 // MORPHOMETRIC DRAFTING ]")
        inputs['LB'] = st.slider("BASELINE FHR (LB bpm)", 80.0, 200.0, float(default_values['LB']), 1.0)
        inputs['AC'] = st.slider("ACCELERATIONS / SEC (AC)", 0.0, 0.02, float(default_values['AC']), 0.001, format="%.3f")
        inputs['UC'] = st.slider("UTERINE CONTRACTIONS / SEC (UC)", 0.0, 0.02, float(default_values['UC']), 0.001, format="%.3f")
        inputs['ASTV'] = st.slider("% ABNORMAL SHORT-TERM VAR (ASTV)", 0.0, 100.0, float(default_values['ASTV']), 1.0)
        inputs['DP'] = st.slider("PROLONGED DECELERATIONS / SEC (DP)", 0.0, 0.01, float(default_values['DP']), 0.0005, format="%.4f")
        inputs['ALTV'] = st.slider("% ABNORMAL LONG-TERM VAR (ALTV)", 0.0, 100.0, float(default_values['ALTV']), 1.0)
        
        # Other defaults
        for k in ['FM', 'DL', 'DS', 'MSTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']:
            inputs[k] = float(default_values[k])
            
        # Engineered Domain Features
        eps = 1e-5
        if 'DSI' in feature_names:
            inputs['DSI'] = (inputs['DL'] + 2.0 * inputs['DS'] + 3.0 * inputs['DP']) / (inputs['UC'] + eps)
        if 'VCR' in feature_names:
            inputs['VCR'] = (inputs['ASTV'] * inputs['ALTV']) / ((inputs['MSTV'] * inputs['MLTV']) + eps)
        if 'FHR_Dev' in feature_names:
            inputs['FHR_Dev'] = abs(inputs['LB'] - 140.0)
        if 'Contraction_Decel_Coupling' in feature_names:
            inputs['Contraction_Decel_Coupling'] = inputs['UC'] * (inputs['DL'] + 2.0 * inputs['DS'] + 3.0 * inputs['DP'])
        if 'Autonomic_Reactivity_Index' in feature_names:
            inputs['Autonomic_Reactivity_Index'] = inputs['AC'] / (inputs['ASTV'] + eps)
        if 'Hist_Spread_Ratio' in feature_names:
            inputs['Hist_Spread_Ratio'] = inputs['Width'] / (inputs['Max'] + eps)

    with col2:
        st.markdown("### [ SEC 01.2 // BLUEPRINT TRIAGE VERDICT ]")
        input_vec = np.array([inputs[fn] for fn in feature_names]).reshape(1, -1)
        
        if "LightGBM" in selected_model_name:
            probs = data_bundle['lgb'].predict_proba(input_vec)[0]
        elif "Random Forest" in selected_model_name:
            probs = data_bundle['rf'].predict_proba(input_vec)[0]
        else:
            input_scaled = data_bundle['scaler'].transform(input_vec)
            probs = data_bundle['svm'].predict_proba(input_scaled)[0]
            
        p_norm, p_susp, p_path = probs[0], probs[1], probs[2]
        
        # Clinical Triage Stamp
        if p_path >= 0.110:
            st.markdown('<div class="stamp-pathologic">[ CLASS 3: PATHOLOGICAL (FETAL DISTRESS) ]<br><span style="font-size:11px; color:#ffffff;">CRITICAL REDLINE: SEVERE HYPOXIA RISK &ge; 11% (SAFETY THRESHOLD)</span></div>', unsafe_allow_html=True)
        elif p_susp >= 0.350:
            st.markdown('<div class="stamp-suspect">[ CLASS 2: SUSPECT (EQUIVOCAL) ]<br><span style="font-size:11px; color:#ffffff;">CAUTION: BORDERLINE VARIABILITY. ESCALATE SURVEILLANCE.</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="stamp-normal">[ CLASS 1: NORMAL (REASSURING) ]<br><span style="font-size:11px; color:#ffffff;">HOMEOSTASIS INTACT. PHYSIOLOGICAL BASELINE & ACCELERATIONS.</span></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**CALIBRATED POSTERIOR PROBABILITY VECTORS:**")
        
        prob_df = pd.DataFrame({
            'State': ['NORMAL (P1)', 'SUSPECT (P2)', 'PATHOLOGIC (P3)'],
            'Probability': [p_norm, p_susp, p_path]
        })
        
        fig, ax = plt.subplots(figsize=(6.5, 2.4))
        fig.patch.set_facecolor('#002646')
        ax.set_facecolor('#001a33')
        palette = ['#00ff88', '#ffcc00', '#ff3333']
        sns.barplot(x='Probability', y='State', data=prob_df, palette=palette, ax=ax, edgecolor='#00ffff', linewidth=1)
        for i, v in enumerate([p_norm, p_susp, p_path]):
            ax.text(v + 0.02, i, f"{v*100:.1f}%", va='center', fontweight='bold', color='#ffffff', fontfamily='monospace', fontsize=10)
        ax.set_xlim(0, 1.18)
        ax.set_xlabel('POSTERIOR PROBABILITY', color='#00ffff', fontsize=9, fontfamily='monospace')
        ax.set_ylabel('')
        ax.tick_params(colors='#00ffff', labelsize=9)
        ax.grid(axis='x', linestyle='--', color='#00ffff', alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Mini metrics
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("DSI Severity", f"{inputs.get('DSI', 0.0):.3f}")
        with mcol2:
            st.metric("VCR Collapse", f"{inputs.get('VCR', 0.0):.1f}")
        with mcol3:
            st.metric("FHR Dev", f"{inputs.get('FHR_Dev', 0.0):.1f} BPM")

# TAB 2: BENCHMARK
with tabs[1]:
    st.subheader("[ SEC 02.0 // MULTI-FAMILY MODEL LEADERBOARD ]")
    if os.path.exists('outputs/model_benchmark_comparison.csv'):
        bench_df = pd.read_csv('outputs/model_benchmark_comparison.csv')
        st.dataframe(bench_df, use_container_width=True)
    if os.path.exists('outputs/figures/cost_optimized_confusion_matrix.png'):
        st.image('outputs/figures/cost_optimized_confusion_matrix.png', caption="Held-Out Decision Matrix: Baseline vs. Clinically Calibrated Risk", use_container_width=True)

# TAB 3: SHAP
with tabs[2]:
    st.subheader("[ SEC 03.0 // SHAP XAI EXPLAINABILITY SCHEMATICS ]")
    sc1, sc2 = st.columns(2)
    with sc1:
        if os.path.exists('outputs/figures/shap_waterfall_patient_distress.png'):
            st.image('outputs/figures/shap_waterfall_patient_distress.png', caption="Case Study 1: Distress Patient Waterfall", use_container_width=True)
    with sc2:
        if os.path.exists('outputs/figures/shap_waterfall_patient_reassuring.png'):
            st.image('outputs/figures/shap_waterfall_patient_reassuring.png', caption="Case Study 2: Reassuring Patient Waterfall", use_container_width=True)
