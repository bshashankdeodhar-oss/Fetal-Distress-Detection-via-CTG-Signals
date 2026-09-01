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
    page_title="Fetal Distress AI - Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .badge-normal {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-suspect {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-pathologic {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
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
    
    # Train LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Train SVM
    svm_model = SVC(
        C=2.0, kernel='rbf', class_weight='balanced', probability=True, random_state=42
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

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=64)
    st.title("Settings & Model")
    
    selected_model_name = st.selectbox(
        "Classifier Family:",
        ["LightGBM (Gradient Boosted Trees)", "Random Forest (Tree Ensemble)", "Support Vector Machine (RBF Kernel)"]
    )
    
    st.markdown("---")
    st.subheader("Preset Patient Scenarios")
    scenario = st.selectbox(
        "Load Clinical Scenario:",
        ["Custom Patient Input", "Scenario A: Healthy Reactive Fetus", "Scenario B: Borderline Reduced Variability", "Scenario C: Severe Distress / Late Decelerations"]
    )

# Scenario presets
default_values = {
    'LB': 135.0, 'AC': 0.003, 'FM': 0.0, 'UC': 0.005, 'DL': 0.0, 'DS': 0.0, 'DP': 0.0,
    'ASTV': 28.0, 'MSTV': 1.8, 'ALTV': 0.0, 'MLTV': 8.5, 'Width': 60.0, 'Min': 110.0,
    'Max': 170.0, 'Nmax': 3.0, 'Nzeros': 0.0, 'Mode': 138.0, 'Mean': 136.0, 'Median': 137.0,
    'Variance': 8.0, 'Tendency': 0.0
}

if scenario == "Scenario A: Healthy Reactive Fetus":
    default_values.update({'LB': 132.0, 'AC': 0.006, 'ASTV': 22.0, 'ALTV': 0.0, 'DP': 0.0, 'MSTV': 2.1})
elif scenario == "Scenario B: Borderline Reduced Variability":
    default_values.update({'LB': 148.0, 'AC': 0.000, 'ASTV': 58.0, 'ALTV': 18.0, 'DP': 0.0, 'MSTV': 0.8})
elif scenario == "Scenario C: Severe Distress / Late Decelerations":
    default_values.update({'LB': 165.0, 'AC': 0.000, 'ASTV': 78.0, 'ALTV': 45.0, 'DP': 0.003, 'DS': 0.001, 'MSTV': 0.4})

# ----------------- MAIN UI -----------------
st.markdown('<div class="main-header">🩺 Cardiotocography (CTG) Fetal Distress Decision Support</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Family AI Triage System with Cost-Sensitive Risk Calibration & SHAP Explainability</div>', unsafe_allow_html=True)

tabs = st.tabs(["🔬 Patient Risk Evaluation", "📊 Model Zoo Benchmark", "📈 Explainability & SHAP"])

# TAB 1: PATIENT EVALUATION
with tabs[0]:
    st.subheader("Patient CTG Parameter Inputs")
    col1, col2, col3 = st.columns(3)
    
    inputs = {}
    with col1:
        st.markdown("**1. Baseline & Rhythm**")
        inputs['LB'] = st.slider("Baseline FHR (LB bpm)", 80.0, 200.0, float(default_values['LB']), 1.0)
        inputs['AC'] = st.slider("Accelerations / sec (AC)", 0.0, 0.02, float(default_values['AC']), 0.001, format="%.3f")
        inputs['FM'] = st.slider("Fetal Movements / sec (FM)", 0.0, 0.5, float(default_values['FM']), 0.01)
        inputs['UC'] = st.slider("Uterine Contractions / sec (UC)", 0.0, 0.02, float(default_values['UC']), 0.001, format="%.3f")
        
    with col2:
        st.markdown("**2. Decelerations & Variability**")
        inputs['DL'] = st.slider("Light Decelerations / sec (DL)", 0.0, 0.02, float(default_values['DL']), 0.001, format="%.3f")
        inputs['DS'] = st.slider("Severe Decelerations / sec (DS)", 0.0, 0.01, float(default_values['DS']), 0.0005, format="%.4f")
        inputs['DP'] = st.slider("Prolonged Decelerations / sec (DP)", 0.0, 0.01, float(default_values['DP']), 0.0005, format="%.4f")
        inputs['ASTV'] = st.slider("% Abnormal Short Term Var (ASTV)", 0.0, 100.0, float(default_values['ASTV']), 1.0)
        inputs['MSTV'] = st.slider("Mean Short Term Var (MSTV)", 0.0, 10.0, float(default_values['MSTV']), 0.1)
        inputs['ALTV'] = st.slider("% Abnormal Long Term Var (ALTV)", 0.0, 100.0, float(default_values['ALTV']), 1.0)
        inputs['MLTV'] = st.slider("Mean Long Term Var (MLTV)", 0.0, 50.0, float(default_values['MLTV']), 0.5)

    with col3:
        st.markdown("**3. Histogram Metrics**")
        inputs['Width'] = st.number_input("Histogram Width", 0.0, 200.0, float(default_values['Width']), 1.0)
        inputs['Min'] = st.number_input("Histogram Min", 50.0, 180.0, float(default_values['Min']), 1.0)
        inputs['Max'] = st.number_input("Histogram Max", 100.0, 250.0, float(default_values['Max']), 1.0)
        inputs['Nmax'] = st.number_input("Histogram Peaks (Nmax)", 0.0, 20.0, float(default_values['Nmax']), 1.0)
        inputs['Nzeros'] = st.number_input("Histogram Zeros (Nzeros)", 0.0, 10.0, float(default_values['Nzeros']), 1.0)
        inputs['Mode'] = st.number_input("Histogram Mode", 50.0, 200.0, float(default_values['Mode']), 1.0)
        inputs['Mean'] = st.number_input("Histogram Mean", 50.0, 200.0, float(default_values['Mean']), 1.0)
        inputs['Median'] = st.number_input("Histogram Median", 50.0, 200.0, float(default_values['Median']), 1.0)
        inputs['Variance'] = st.number_input("Histogram Variance", 0.0, 300.0, float(default_values['Variance']), 1.0)
        inputs['Tendency'] = st.selectbox("Histogram Tendency", [-1.0, 0.0, 1.0], index=1)

    # Compute engineered features if present in feature_names
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

    # Build input array
    input_vector = np.array([inputs[fn] for fn in feature_names]).reshape(1, -1)
    
    st.markdown("---")
    
    # Predict
    if "LightGBM" in selected_model_name:
        clf = data_bundle['lgb']
        probs = clf.predict_proba(input_vector)[0]
    elif "Random Forest" in selected_model_name:
        clf = data_bundle['rf']
        probs = clf.predict_proba(input_vector)[0]
    else:
        clf = data_bundle['svm']
        input_scaled = data_bundle['scaler'].transform(input_vector)
        probs = clf.predict_proba(input_scaled)[0]
        
    pred_idx = np.argmax(probs)
    pred_class = class_names[pred_idx]
    
    res_col1, res_col2 = st.columns([1, 1.4])
    
    with res_col1:
        st.markdown("### Clinical Triage Output")
        if pred_class == "Normal":
            st.markdown('<span class="badge-normal">🟢 Class 1: NORMAL (Reassuring)</span>', unsafe_allow_html=True)
            st.success("Physiological autonomic regulation intact. Continue standard intrapartum monitoring.")
        elif pred_class == "Suspect":
            st.markdown('<span class="badge-suspect">🟡 Class 2: SUSPECT (Equivocal)</span>', unsafe_allow_html=True)
            st.warning("Borderline variability or decelerations detected. Initiate conservative intrauterine resuscitation and close surveillance.")
        else:
            st.markdown('<span class="badge-pathologic">🔴 Class 3: PATHOLOGICAL (Fetal Distress)</span>', unsafe_allow_html=True)
            st.error("Severe fetal hypoxia / acidosis risk detected. Immediate obstetric review and preparation for delivery recommended.")
            
    with res_col2:
        st.markdown("### Calibrated Probability Distribution")
        prob_df = pd.DataFrame({
            'State': class_names,
            'Probability': probs
        })
        fig, ax = plt.subplots(figsize=(6, 2.5))
        palette = ['#10b981', '#f59e0b', '#ef4444']
        sns.barplot(x='Probability', y='State', data=prob_df, palette=palette, ax=ax, edgecolor='black', linewidth=0.5)
        for i, v in enumerate(probs):
            ax.text(v + 0.02, i, f"{v*100:.1f}%", va='center', fontweight='bold', fontsize=10)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Predicted Probability', fontsize=9)
        ax.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig)

# TAB 2: MODEL BENCHMARK
with tabs[1]:
    st.subheader("Held-Out Test Set Performance Benchmark")
    if os.path.exists('outputs/model_benchmark_comparison.csv'):
        bench_df = pd.read_csv('outputs/model_benchmark_comparison.csv')
        st.dataframe(bench_df.style.highlight_max(subset=['Macro F1', 'Recall (Pathologic)', 'Balanced Accuracy'], color='#dbeafe'), use_container_width=True)
    else:
        st.info("Run `python src/model_zoo_benchmark.py` to generate the complete comparative table.")
        
    st.markdown("### Held-out Confusion Matrices")
    cm_cols = st.columns(3)
    cm_files = [
        ("LightGBM", "outputs/figures/confusion_matrix_lightgbm.png"),
        ("XGBoost", "outputs/figures/confusion_matrix_xgboost.png"),
        ("SVM (RBF)", "outputs/figures/confusion_matrix_support_vector_machine_svc_rbf.png")
    ]
    for i, (name, path) in enumerate(cm_files):
        if os.path.exists(path):
            with cm_cols[i]:
                st.image(path, caption=f"{name} 3x3 Confusion Matrix", use_container_width=True)

# TAB 3: XAI & SHAP
with tabs[2]:
    st.subheader("Global & Local Interpretability (SHAP Analysis)")
    shap_col1, shap_col2 = st.columns(2)
    with shap_col1:
        if os.path.exists('outputs/figures/feature_importance_lightgbm.png'):
            st.image('outputs/figures/feature_importance_lightgbm.png', caption="LightGBM Feature Importance (Gain)", use_container_width=True)
    with shap_col2:
        if os.path.exists('outputs/figures/shap_summary_multiclass.png'):
            st.image('outputs/figures/shap_summary_multiclass.png', caption="SHAP Multi-Class Feature Summary", use_container_width=True)
