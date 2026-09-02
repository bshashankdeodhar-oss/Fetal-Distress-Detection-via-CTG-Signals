import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import streamlit as st

from src.live_stream_engine import (
    load_physionet_record,
    extract_sliding_window_features,
    generate_synthetic_telemetry
)

st.set_page_config(
    page_title="CTG Fetal Distress — Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Blueprint CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;600;700&family=Architects+Daughter&display=swap');

[data-testid="stSidebar"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: .8rem !important; padding-bottom: 72px !important; max-width: 98% !important; }

.stApp {
    background-color: #002b4e !important;
    background-image:
        linear-gradient(rgba(0,255,255,.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,.05) 1px, transparent 1px),
        linear-gradient(rgba(0,255,255,.13) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,.13) 1px, transparent 1px) !important;
    background-size: 20px 20px, 20px 20px, 100px 100px, 100px 100px !important;
    color: rgba(255,255,255,.88) !important;
    font-family: 'Roboto Mono', monospace !important;
}

.doc-serial { color:#00ffff; font-size:11px; letter-spacing:1.5px; font-weight:700; }
.main-title  { color:#fff; font-size:20px; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:2px; }
.doc-sub     { color:rgba(255,255,255,.5); font-size:11px; margin-bottom:10px; }
.dim-line    { border-top:1px dashed rgba(0,255,255,.3); text-align:center; color:#00ffff; font-size:10px; padding-top:4px; margin-bottom:12px; }
.redline-box { background:rgba(255,51,51,.12); border:1px dashed #ff3333; padding:8px 14px; color:#ff9999; font-family:'Architects Daughter',cursive; font-size:13px; margin-bottom:12px; }

.stamp-normal     { border:2px solid #00ff88; background:rgba(0,255,136,.08); color:#00ff88; padding:8px 14px; text-align:center; font-size:15px; font-weight:700; }
.stamp-suspect    { border:2px solid #ffcc00; background:rgba(255,204,0,.1);  color:#ffcc00; padding:8px 14px; text-align:center; font-size:15px; font-weight:700; }
.stamp-pathologic { border:2px solid #ff3333; background:rgba(255,51,51,.15); color:#ff3333; padding:8px 14px; text-align:center; font-size:15px; font-weight:700; }

.alarm-banner {
    background: rgba(255, 0, 55, 0.25);
    border: 2px solid #ff0055;
    color: #ff6688;
    padding: 10px 16px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 12px;
    animation: pulse 1.5s infinite;
}

/* Bottom taskbar */
.btbar {
    position:fixed; bottom:0; left:0; width:100vw; height:46px;
    background:rgba(0,18,38,.96); border-top:1px solid #00ffff;
    box-shadow:0 -4px 20px rgba(0,255,255,.2); backdrop-filter:blur(12px);
    display:flex; align-items:center; justify-content:space-between;
    padding:0 20px; z-index:99999; font-family:'Roboto Mono',monospace;
}
.btbar-brand { display:flex; align-items:center; gap:8px; border:1px solid #00ffff;
               background:rgba(0,255,255,.08); padding:4px 10px; font-size:11px; font-weight:700; color:#fff; }
.btbar-pills { display:flex; gap:10px; align-items:center; }
.btbar-pill  { border:1px solid rgba(0,255,255,.35); background:rgba(255,255,255,.06);
               padding:3px 10px; font-size:10px; color:#fff; border-radius:3px; }
.btbar-tray  { display:flex; align-items:center; gap:10px; }
.btbar-lock  { border:1px solid #00ff88; background:rgba(0,255,136,.1); color:#00ff88; padding:2px 8px; font-size:9.5px; font-weight:700; }
.btbar-info  { color:#00ffff; font-size:9px; }
</style>
""", unsafe_allow_html=True)


# ── Load & cache models ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on CTG dataset…")
def load_bundle():
    csv = os.path.join('datasets', 'uci_ctg', 'CTG_features_engineered.csv')
    if not os.path.exists(csv):
        return None

    df = pd.read_csv(csv)
    feat_names = [c for c in df.columns if c != 'NSP']
    X = df[feat_names].values
    y = df['NSP'].values - 1

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    # Load Optuna params if available
    lgb_p = dict(n_estimators=350, learning_rate=0.04, num_leaves=40,
                 subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=1.0,
                 class_weight='balanced', random_state=42, verbose=-1)
    if os.path.exists('outputs/best_params_lgb.json'):
        with open('outputs/best_params_lgb.json') as f:
            d = json.load(f)
        lgb_p.update(d.get('params', {}))
        lgb_p.update(dict(class_weight='balanced', random_state=42, verbose=-1))

    lgb_m = lgb.LGBMClassifier(**lgb_p);   lgb_m.fit(X_tr, y_tr)
    rf_m  = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced_subsample', random_state=42)
    rf_m.fit(X_tr, y_tr)
    svm_m = SVC(C=2.5, kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm_m.fit(X_tr_s, y_tr)

    return dict(feat_names=feat_names, X_te=X_te, y_te=y_te,
                scaler=sc, lgb=lgb_m, rf=rf_m, svm=svm_m, X_tr=X_tr, y_tr=y_tr)


@st.cache_data
def get_cached_telemetry(case_name):
    if "1002" in case_name:
        return load_physionet_record('1002')
    elif "1001" in case_name:
        return load_physionet_record('1001')
    elif "1003" in case_name:
        return load_physionet_record('1003')
    else:
        return generate_synthetic_telemetry('pathologic', duration_minutes=50)


bundle = load_bundle()
DATA_OK = bundle is not None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="doc-serial">DOC_REF: CTG-2126 // REV.2026.09 // FIGO 2015 / ACOG / PHYSIONET CTU-UHB</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Cardiotocography Fetal Distress Diagnostic Schematic</div>', unsafe_allow_html=True)
st.markdown('<div class="doc-sub">Real-Time Telemetry Stream · Multi-Family Risk Classifier · Asymmetric Cost Calibration · SHAP Explainability</div>', unsafe_allow_html=True)
st.markdown('<div class="dim-line">← MASTER WORKSPACE · 4 Hz TIME-SERIES & 39 BIOMARKERS · DUAL-STREAM CDSS →</div>', unsafe_allow_html=True)

st.markdown("""<div class="redline-box">
<strong>[CLINICAL SAFETY NOTE]</strong> Asymmetric cost matrix enforced:
<strong>Cost(FN) = 10 × Cost(FP)</strong>. Pathologic safety threshold:
<strong>P ≥ 0.110</strong> → Recall 94.29% (33/35 distress cases caught on held-out test).
</div>""", unsafe_allow_html=True)

# ── Top controls strip ────────────────────────────────────────────────────────
cc1, cc2, cc3, cc4 = st.columns([1.4, 1.4, 1.0, 1.0])
with cc1:
    model_choice = st.selectbox("⚙️ INFERENCE ENGINE",
        ["LightGBM (Optuna-Tuned)", "Random Forest", "SVM (RBF Kernel)"])
with cc2:
    st.markdown("<div style='font-size:10px;color:#00ffff;'>STREAMING ARCHITECTURE</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;color:#00ffff;'>4 Hz Continuous / Sliding Window</div>", unsafe_allow_html=True)
with cc3:
    st.markdown("<div style='font-size:10px;color:#00ffff;'>COHORT (HELD-OUT)</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;'>N=426 · 332/59/35</div>", unsafe_allow_html=True)
with cc4:
    st.markdown("<div style='font-size:10px;color:#00ffff;'>CHAMPION ACCURACY</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;color:#00ff88;'>Macro F1 = 0.9202 (Optuna)</div>", unsafe_allow_html=True)

# ── Default values per preset for manual tab ──────────────────────────────────
defaults = dict(LB=135., AC=.003, FM=0., UC=.005, DL=0., DS=0., DP=0.,
                ASTV=28., MSTV=1.8, ALTV=0., MLTV=8.5, Width=60., Min=110.,
                Max=170., Nmax=3., Nzeros=0., Mode=138., Mean=136.,
                Median=137., Variance=8., Tendency=0.)

# ── Main panel with 6 Tabs ────────────────────────────────────────────────────
tabs = st.tabs([
    "🔴 Live ICU Telemetry",
    "📐 Manual Morphometric Drafting",
    "📊 Model Benchmark",
    "📉 Calibration & ROC",
    "🌊 SHAP Explainability",
    "📋 5-Fold CV Results"
])

# ─────────────────────────── TAB 0: LIVE ICU TELEMETRY ────────────────────────
with tabs[0]:
    st.subheader("[ SEC 00 // REAL-TIME CONTINUOUS CTG TELEMETRY STREAM & SLIDING-WINDOW TRIAGE ]")
    st.markdown("""
    <div style='font-size:12px;color:rgba(255,255,255,0.7);margin-bottom:12px;'>
    Streams genuine <strong>continuous 4 Hz cardiotocograph waveforms</strong> (from PhysioNet CTU-UHB intrapartum database).
    A rolling clinical window continuously extracts all 39 FIGO morphometrics and delivers real-time risk predictions.
    </div>
    """, unsafe_allow_html=True)

    c_case, c_win, c_ctl = st.columns([1.8, 1.0, 1.2])
    with c_case:
        patient_case = st.selectbox(
            "🏥 SELECT PATIENT RECORDING (PHYSIOMETRIC FEED)",
            [
                "PhysioNet Record 1002 — Severe Intrapartum Hypoxia / Acidosis (Umbilical pH 7.00, BE -12)",
                "PhysioNet Record 1001 — Moderate Fetal Acidemia / Variable Decelerations (Umbilical pH 7.14)",
                "PhysioNet Record 1003 — Healthy Reassuring Term Labor (Umbilical pH 7.20, Apgar 9)",
                "Synthetic Live Telemetry — Progressive Autonomic Distress Sequence"
            ]
        )
    with c_win:
        win_mins = st.selectbox("⏱️ SLIDING ANALYSIS WINDOW", [5, 10, 15], index=1)
    with c_ctl:
        st.markdown("<div style='font-size:11px;color:#00ffff;'>TELEMETRY STREAM CONTROLS</div>", unsafe_allow_html=True)
        step_btn = st.button("⏩ Advance Stream (+2 mins)")

    # Load continuous telemetry signal
    fhr_full, uc_full, fs = get_cached_telemetry(patient_case)
    total_mins = int(len(fhr_full) / (fs * 60))

    if 'labor_min' not in st.session_state:
        st.session_state.labor_min = min(win_mins + 10, total_mins)

    if step_btn:
        st.session_state.labor_min = min(st.session_state.labor_min + 2, total_mins)

    scrub_col, val_col = st.columns([3.5, 1.0])
    with scrub_col:
        current_min = st.slider(
            "LABOR ELAPSED TIME SCRUBBER (MINUTES)",
            min_value=int(win_mins),
            max_value=max(int(total_mins), int(win_mins) + 1),
            value=int(st.session_state.labor_min),
            step=1
        )
        st.session_state.labor_min = current_min
    with val_col:
        st.markdown(f"<div style='font-size:12px;padding-top:24px;color:#00ffff;'>TIMELINE: <strong>{current_min} / {total_mins} min</strong></div>", unsafe_allow_html=True)

    # Slice the sliding window
    end_sample = int(current_min * 60 * fs)
    start_sample = max(0, end_sample - int(win_mins * 60 * fs))
    fhr_window = fhr_full[start_sample:end_sample]
    uc_window = uc_full[start_sample:end_sample]

    # Display context (show last 25 minutes of waveforms)
    context_mins = 25
    ctx_start = max(0, end_sample - int(context_mins * 60 * fs))
    t_axis = np.linspace(max(0, current_min - context_mins), current_min, end_sample - ctx_start)
    fhr_ctx = fhr_full[ctx_start:end_sample]
    uc_ctx = uc_full[ctx_start:end_sample]

    # ── Oscilloscope Figure ──
    fig, (ax_fhr, ax_uc) = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.0]})
    fig.patch.set_facecolor('#001e38')
    for ax in (ax_fhr, ax_uc):
        ax.set_facecolor('#001426')
        ax.tick_params(colors='#00ffff', labelsize=8)
        ax.grid(True, color='#00ffff', alpha=0.15, linestyle='--')

    # FHR Plot
    ax_fhr.axhspan(110, 160, color='#00ff88', alpha=0.10, label='FIGO Normal Band (110-160 bpm)')
    ax_fhr.axhline(110, color='#ff3333', linestyle=':', alpha=0.6, label='Bradycardia Threshold (<110)')
    ax_fhr.axhline(160, color='#ffcc00', linestyle=':', alpha=0.6, label='Tachycardia Threshold (>160)')
    ax_fhr.plot(t_axis, fhr_ctx, color='#00ffff', linewidth=1.1, label='Live FHR (bpm)')
    ax_fhr.axvspan(current_min - win_mins, current_min, color='#00ffff', alpha=0.12, label=f'Active Window ({win_mins} min)')
    ax_fhr.set_ylim(60, 200)
    ax_fhr.set_ylabel('FHR (bpm)', color='#00ffff', fontweight='bold', fontsize=9)
    ax_fhr.legend(loc='upper right', fontsize=7, facecolor='#001e38', edgecolor='#00ffff', labelcolor='#ffffff')

    # UC / Tocogram Plot
    ax_uc.plot(t_axis, uc_ctx, color='#ec4899', linewidth=1.2, label='Tocogram (Contractions)')
    ax_uc.axvspan(current_min - win_mins, current_min, color='#00ffff', alpha=0.12)
    ax_uc.set_ylim(0, 100)
    ax_uc.set_ylabel('UC (mmHg)', color='#ec4899', fontweight='bold', fontsize=9)
    ax_uc.set_xlabel('Elapsed Labor Time (Minutes)', color='#00ffff', fontsize=9)
    ax_uc.legend(loc='upper right', fontsize=7, facecolor='#001e38', edgecolor='#00ffff', labelcolor='#ffffff')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Real-Time Feature Extraction & Inference
    if DATA_OK and len(fhr_window) > 20:
        feat_dict = extract_sliding_window_features(fhr_window, uc_window, fs)
        vec = np.array([feat_dict.get(f, 0.) for f in bundle['feat_names']]).reshape(1, -1)

        if "LightGBM" in model_choice:
            probs = bundle['lgb'].predict_proba(vec)[0]
        elif "Random Forest" in model_choice:
            probs = bundle['rf'].predict_proba(vec)[0]
        else:
            probs = bundle['svm'].predict_proba(bundle['scaler'].transform(vec))[0]

        p_n, p_s, p_p = probs

        # Live Alarm Banner & Status
        o_left, o_right = st.columns([1.3, 1.0])
        with o_left:
            if p_p >= 0.110:
                st.markdown(f"""
                <div class="alarm-banner">
                🚨 FIGO CATEGORY III: PATHOLOGICAL FETAL DISTRESS (Risk: {p_p*100:.1f}%)<br>
                <span style="font-size:11px;font-weight:400;color:#fff;">
                Severe autonomic decoupling / late decelerations detected. Cost-optimal safety threshold breached (P ≥ 0.110).
                RECOMMENDATION: Immediate intrauterine resuscitation (maternal oxygen, stop oxytocin) & prepare emergency delivery.
                </span>
                </div>
                """, unsafe_allow_html=True)
            elif p_s >= 0.35:
                st.markdown(f"""
                <div class="stamp-suspect">
                ⚠️ FIGO CATEGORY II: SUSPECT / EQUIVOCAL CTG (Risk: {p_s*100:.1f}%)<br>
                <span style="font-size:11px;font-weight:400;color:#fff;">
                Sub-optimal variability or atypical decelerations. ESCALATE SURVEILLANCE & assess maternal position.
                </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="stamp-normal">
                ✅ FIGO CATEGORY I: NORMAL REASSURING TRACE (Normal: {p_n*100:.1f}%)<br>
                <span style="font-size:11px;font-weight:400;color:#fff;">
                Autonomic homeostasis intact. Accelerations present, physiological baseline stable.
                </span>
                </div>
                """, unsafe_allow_html=True)

        with o_right:
            # Probability Bar Chart
            fig_p, ax_p = plt.subplots(figsize=(5, 1.8))
            fig_p.patch.set_facecolor('#001e38'); ax_p.set_facecolor('#001426')
            import seaborn as sns
            prob_df = pd.DataFrame({'State':['NORMAL','SUSPECT','PATHOLOGIC'], 'P':[p_n, p_s, p_p]})
            sns.barplot(x='P', y='State', data=prob_df, palette=['#00ff88','#ffcc00','#ff3333'], ax=ax_p, edgecolor='#00ffff', linewidth=0.8)
            for i, v in enumerate([p_n, p_s, p_p]):
                ax_p.text(v+.02, i, f'{v*100:.1f}%', va='center', fontweight='bold', color='#fff', fontfamily='monospace', fontsize=8)
            ax_p.set_xlim(0, 1.18); ax_p.set_xlabel('Live Posterior Probability', color='#00ffff', fontsize=8)
            ax_p.set_ylabel(''); ax_p.tick_params(colors='#00ffff', labelsize=8)
            plt.tight_layout(); st.pyplot(fig_p); plt.close()

        # Live Real-Time Clinical Biomarkers
        st.markdown("<div style='font-size:11px;color:#00ffff;margin-top:8px;'>LIVE SLIDING-WINDOW BIOMARKER TELEMETRY (DYNAMICALLY COMPUTED FROM RAW 4 Hz SIGNAL)</div>", unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Live Baseline FHR", f"{feat_dict['LB']:.1f} bpm")
        m2.metric("Short-Term Var (ASTV)", f"{feat_dict['ASTV']:.1f}%")
        m3.metric("Pathologic Risk (PRI)", f"{feat_dict['PRI']:.2f}")
        m4.metric("Decel Severity (DSI)", f"{feat_dict['DSI']:.3f}")
        m5.metric("Variability Entropy", f"{feat_dict['Variability_Entropy']:.2f} bits")

# ─────────────────────────── TAB 1: STATIC DRAFTING ──────────────────────────
with tabs[1]:
    preset_box = st.selectbox("📋 CHOOSE CLINICAL SCENARIO PRESET",
        ["Custom Input", "Preset A — Normal (Reassuring)",
         "Preset B — Suspect (Borderline)", "Preset C — Pathologic (Distress)"],
        key="static_preset")

    if "Normal" in preset_box:
        defaults.update(LB=135., AC=.005, ASTV=22., ALTV=0., DP=0., MSTV=2.1)
    elif "Suspect" in preset_box:
        defaults.update(LB=152., AC=.001, ASTV=58., ALTV=18., DP=0., MSTV=.8)
    elif "Pathologic" in preset_box:
        defaults.update(LB=168., AC=0., ASTV=78., ALTV=45., DP=.003, DS=.001, MSTV=.4)

    sl, sr = st.columns([1.05, 1.35])

    with sl:
        st.markdown("#### [ SEC 01 // MORPHOMETRIC DRAFTING ]")
        inp = {}
        inp['LB']   = st.slider("BASELINE FHR (LB bpm)", 80., 200., float(defaults['LB']), 1., key="sl_lb")
        inp['AC']   = st.slider("ACCELERATIONS / SEC (AC)", 0., .02, float(defaults['AC']), .001, format="%.3f", key="sl_ac")
        inp['UC']   = st.slider("UTERINE CONTRACTIONS / SEC (UC)", 0., .02, float(defaults['UC']), .001, format="%.3f", key="sl_uc")
        inp['ASTV'] = st.slider("% ABNORMAL SHORT-TERM VAR (ASTV)", 0., 100., float(defaults['ASTV']), 1., key="sl_astv")
        inp['DP']   = st.slider("PROLONGED DECELS / SEC (DP)", 0., .01, float(defaults['DP']), .0005, format="%.4f", key="sl_dp")
        inp['ALTV'] = st.slider("% ABNORMAL LONG-TERM VAR (ALTV)", 0., 100., float(defaults['ALTV']), 1., key="sl_altv")
        for k in ['FM','DL','DS','MSTV','MLTV','Width','Min','Max','Nmax','Nzeros',
                  'Mode','Mean','Median','Variance','Tendency']:
            inp[k] = float(defaults[k])

    with sr:
        st.markdown("#### [ SEC 02 // TRIAGE VERDICT & PROBABILITY GAUGES ]")

        if DATA_OK:
            feat_names = bundle['feat_names']
            eps = 1e-6
            inp['DSI'] = (inp['DL'] + 2*inp['DS'] + 3*inp['DP']) / (inp['UC'] + eps)
            inp['VCR'] = (inp['ASTV'] * inp['ALTV']) / (inp['MSTV'] * inp['MLTV'] + eps)
            inp['FHR_Dev'] = abs(inp['LB'] - 140.)
            inp['Contraction_Decel_Coupling'] = inp['UC'] * (inp['DL'] + 2*inp['DS'] + 3*inp['DP'])
            inp['Autonomic_Reactivity_Index'] = inp['AC'] / (inp['ASTV'] + eps)
            inp['Hist_Spread_Ratio'] = inp['Width'] / (inp['Max'] + eps)
            inp['PRI'] = (inp['ASTV']/100*3 + inp['ALTV']/100*2
                          + min(inp['DP']*1000, 5)*2.5
                          + min(inp['DS']*500, 3)*1.5
                          - min(inp['AC']*500, 5)*2
                          - min(inp['MSTV']-1, 5)*.5)
            inp['Decel_Pattern_Severity'] = (inp['DL']
                + (inp['DS']*3 if inp['DS']>0 else 0)
                + (inp['DP']*5 if inp['DP']>0 else 0))
            inp['Autonomic_Balance_Ratio'] = inp['AC'] / ((inp['ASTV']+inp['ALTV'])/100 + eps)
            inp['FHR_Instability_Score'] = inp['FHR_Dev']*(inp['MLTV']+1)/(inp['MSTV']+eps)
            inp['UC_AC_Coupling'] = inp['AC'] / (inp['UC'] + eps)
            inp['Morphological_Complexity'] = (inp['Max']-inp['Min'])*inp['Nmax']/(inp['Variance']+eps)
            inp['Contraction_Load_Index'] = inp['UC']*inp['Width']
            inp['Basal_Reactivity_Score'] = (inp['AC']/(inp['UC']+eps))*(1/(inp['FHR_Dev']+1))
            inp['STV_LTV_Ratio'] = inp['MSTV']/(inp['MLTV']+eps)
            inp['Hist_Skew_Proxy'] = (inp['Mode']-inp['Mean'])/(inp['Width']+eps)
            inp['Zero_Crossing_Density'] = inp['Nzeros']/(inp['Width']+eps)
            import scipy.stats as sst
            p_arr = np.array([max(inp['ASTV'],.01), max(inp['ALTV'],.01),
                              max(100-inp['ASTV']-inp['ALTV'],.01)])
            p_arr /= p_arr.sum()
            inp['Variability_Entropy'] = float(sst.entropy(p_arr, base=2))

            vec = np.array([inp.get(f, 0.) for f in feat_names]).reshape(1, -1)

            if "LightGBM" in model_choice:
                probs = bundle['lgb'].predict_proba(vec)[0]
            elif "Random Forest" in model_choice:
                probs = bundle['rf'].predict_proba(vec)[0]
            else:
                probs = bundle['svm'].predict_proba(bundle['scaler'].transform(vec))[0]

            p_n, p_s, p_p = probs
            if p_p >= .110:
                st.markdown('<div class="stamp-pathologic">[ CLASS 3: PATHOLOGICAL ]<br>'
                            '<span style="font-size:10px;color:#fff;">CRITICAL — PREPARE FOR EMERGENCY DELIVERY</span></div>',
                            unsafe_allow_html=True)
            elif p_s >= .35:
                st.markdown('<div class="stamp-suspect">[ CLASS 2: SUSPECT (EQUIVOCAL) ]<br>'
                            '<span style="font-size:10px;color:#fff;">CAUTION — ESCALATE CTG SURVEILLANCE</span></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="stamp-normal">[ CLASS 1: NORMAL (REASSURING) ]<br>'
                            '<span style="font-size:10px;color:#fff;">HOMEOSTASIS INTACT — CONTINUE MONITORING</span></div>',
                            unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6.5, 2.4))
            fig.patch.set_facecolor('#002646'); ax.set_facecolor('#001a33')
            import seaborn as sns
            prob_df = pd.DataFrame({'State':['NORMAL','SUSPECT','PATHOLOGIC'], 'P':[p_n, p_s, p_p]})
            sns.barplot(x='P', y='State', data=prob_df, palette=['#00ff88','#ffcc00','#ff3333'],
                        ax=ax, edgecolor='#00ffff', linewidth=1)
            for i, v in enumerate([p_n, p_s, p_p]):
                ax.text(v+.02, i, f'{v*100:.1f}%', va='center', fontweight='bold',
                        color='#fff', fontfamily='monospace', fontsize=10)
            ax.set_xlim(0, 1.18); ax.set_xlabel('Posterior Probability', color='#00ffff', fontsize=9)
            ax.set_ylabel(''); ax.tick_params(colors='#00ffff', labelsize=9)
            ax.grid(axis='x', linestyle='--', color='#00ffff', alpha=.25)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("DSI", f"{inp['DSI']:.3f}")
            m2.metric("VCR", f"{inp['VCR']:.1f}")
            m3.metric("PRI", f"{inp['PRI']:.2f}")
            m4.metric("Δ FHR", f"{inp['FHR_Dev']:.1f} bpm")
        else:
            st.warning("Dataset not found. Run `python src/feature_engineering.py` first.")

# ─────────────────────────── TAB 2: BENCHMARK ────────────────────────────────
with tabs[2]:
    st.subheader("[ SEC 03 // MULTI-FAMILY MODEL LEADERBOARD ]")
    bench_path = 'outputs/model_benchmark_comparison.csv'
    if os.path.exists(bench_path):
        bench = pd.read_csv(bench_path).sort_values('Macro F1', ascending=False).reset_index(drop=True)
        bench.index += 1
        st.dataframe(bench, use_container_width=True)
    else:
        st.info("Run `python src/stacking_ensemble.py` to generate benchmark results.")

    c1, c2 = st.columns(2)
    for col, fig_path, cap in [
        (c1, 'outputs/figures/cv_macro_f1_comparison.png', '5-Fold CV Macro F1 with 95% CI'),
        (c2, 'outputs/figures/confusion_matrix_stacking_ensemble.png', 'Stacking Ensemble Confusion Matrix'),
    ]:
        if os.path.exists(fig_path):
            col.image(fig_path, caption=cap, use_container_width=True)

# ─────────────────────────── TAB 3: CALIBRATION & ROC ────────────────────────
with tabs[3]:
    st.subheader("[ SEC 04 // PROBABILITY CALIBRATION & CLINICAL VALIDATION CURVES ]")
    auc_path = 'outputs/calibration_auc_summary.json'
    if os.path.exists(auc_path):
        with open(auc_path) as f: auc_d = json.load(f)
        ac1, ac2, ac3 = st.columns(3)
        for col, (cls, vals) in zip([ac1,ac2,ac3], auc_d.items()):
            col.metric(f"{cls} ROC-AUC", f"{vals['roc_auc']:.4f}")

    for fig_path, cap in [
        ('outputs/figures/calibration_reliability_diagrams.png', 'Reliability Diagrams — Raw vs Platt-Scaled'),
        ('outputs/figures/roc_curves_ovr.png', 'ROC Curves (One-vs-Rest) — Calibrated LightGBM'),
        ('outputs/figures/pr_curves_ovr.png', 'Precision-Recall Curves (One-vs-Rest)'),
    ]:
        if os.path.exists(fig_path):
            st.image(fig_path, caption=cap, use_container_width=True)
        else:
            st.info(f"Run `python src/calibration_analysis.py` to generate: {fig_path}")

# ─────────────────────────── TAB 4: SHAP ─────────────────────────────────────
with tabs[4]:
    st.subheader("[ SEC 05 // SHAP EXPLAINABILITY SUITE ]")
    shap_figs = [
        ('outputs/figures/shap_beeswarm_pathologic.png',        'SHAP Beeswarm — Pathologic Risk Drivers (Test Cohort)'),
        ('outputs/figures/shap_summary_bar_multiclass.png',     'Mean |SHAP| — Multi-Class Feature Importance'),
        ('outputs/figures/shap_decision_plot_3cases.png',       'SHAP Decision Plot — 3 Representative Patients'),
        ('outputs/figures/shap_waterfall_patient_distress.png', 'Case Study: Pathologic Patient — Log-odds Waterfall'),
        ('outputs/figures/shap_waterfall_suspect.png',          'Case Study: Suspect Patient — Log-odds Waterfall'),
        ('outputs/figures/shap_waterfall_patient_reassuring.png','Case Study: Normal Patient — Protective Factors'),
    ]
    for fp, cap in shap_figs:
        if os.path.exists(fp):
            st.image(fp, caption=cap, use_container_width=True)

    report_path = 'outputs/reports/shap_case_study_report.md'
    if os.path.exists(report_path):
        with open(report_path) as f:
            st.markdown(f.read())

# ─────────────────────────── TAB 5: CV RESULTS ───────────────────────────────
with tabs[5]:
    st.subheader("[ SEC 06 // 5-FOLD CROSS-VALIDATED RESULTS WITH 95% BOOTSTRAP CI ]")
    cv_path = 'outputs/cv_results_with_ci.csv'
    if os.path.exists(cv_path):
        cv_df = pd.read_csv(cv_path)
        st.dataframe(cv_df, use_container_width=True)
    else:
        st.info("Run `python src/cv_evaluation.py` to generate CV results.")

    optuna_fig = 'outputs/figures/optuna_convergence.png'
    lcurve_fig = 'outputs/figures/learning_curve_generalization.png'
    oc1, oc2 = st.columns(2)
    if os.path.exists(optuna_fig):
        oc1.image(optuna_fig, caption='Optuna HPO Convergence + Hyperparameter Importance', use_container_width=True)
    if os.path.exists(lcurve_fig):
        oc2.image(lcurve_fig, caption='Statistical Learning Curve — Generalization Gap Audit', use_container_width=True)

# ── Fixed bottom taskbar ──────────────────────────────────────────────────────
st.markdown("""
<div class="btbar">
  <div class="btbar-brand">📐 CTG-OS // REAL-TIME FETAL TELEMETRY</div>
  <div class="btbar-pills">
    <div class="btbar-pill">🔴 LIVE TELEMETRY</div>
    <div class="btbar-pill">📐 MANUAL DRAFTING</div>
    <div class="btbar-pill">📊 BENCHMARK</div>
    <div class="btbar-pill">📉 CALIBRATION</div>
    <div class="btbar-pill">🌊 SHAP XAI</div>
    <div class="btbar-pill">📋 CV RESULTS</div>
  </div>
  <div class="btbar-tray">
    <span class="btbar-lock">🛡️ P≥0.110 · RECALL 94.29%</span>
    <span class="btbar-info">4 Hz CONTINUOUS STREAM // PHYSIONET CTU-UHB</span>
  </div>
</div>
""", unsafe_allow_html=True)
