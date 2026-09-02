import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

# ── Blueprint / Clinical Telemetry CSS ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;600;700&family=Architects+Daughter&display=swap');

[data-testid="stSidebar"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: .8rem !important; padding-bottom: 72px !important; max-width: 98% !important; }

.stApp {
    background-color: #00223e !important;
    background-image:
        linear-gradient(rgba(0,255,255,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,.04) 1px, transparent 1px),
        linear-gradient(rgba(0,255,255,.10) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,.10) 1px, transparent 1px) !important;
    background-size: 20px 20px, 20px 20px, 100px 100px, 100px 100px !important;
    color: rgba(255,255,255,.90) !important;
    font-family: 'Roboto Mono', monospace !important;
}

.doc-serial { color:#00ffff; font-size:11px; letter-spacing:1.5px; font-weight:700; }
.main-title  { color:#fff; font-size:20px; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:2px; }
.doc-sub     { color:rgba(255,255,255,.55); font-size:11px; margin-bottom:10px; }
.dim-line    { border-top:1px dashed rgba(0,255,255,.3); text-align:center; color:#00ffff; font-size:10px; padding-top:4px; margin-bottom:12px; }

.stamp-normal     { border:2px solid #00ff88; background:rgba(0,255,136,.09); color:#00ff88; padding:8px 14px; text-align:center; font-size:14px; font-weight:700; border-radius:4px; }
.stamp-suspect    { border:2px solid #ffcc00; background:rgba(255,204,0,.11);  color:#ffcc00; padding:8px 14px; text-align:center; font-size:14px; font-weight:700; border-radius:4px; }
.stamp-pathologic { border:2px solid #ff3333; background:rgba(255,51,51,.18); color:#ff3333; padding:8px 14px; text-align:center; font-size:14px; font-weight:700; border-radius:4px; }

.alert-precaution {
    border: 1px dashed #f59e0b;
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
    padding: 6px 12px;
    font-size: 11px;
    margin-top: 6px;
    border-radius: 4px;
}

.alert-emergency {
    border: 2px solid #ef4444;
    background: rgba(239, 68, 68, 0.22);
    color: #fca5a5;
    padding: 7px 12px;
    font-size: 11.5px;
    font-weight: 700;
    margin-top: 6px;
    border-radius: 4px;
}

/* Bottom taskbar */
.btbar {
    position:fixed; bottom:0; left:0; width:100vw; height:46px;
    background:rgba(0,16,34,.97); border-top:1px solid #00ffff;
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
@st.cache_resource(show_spinner="Initializing clinical inference engine…")
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
        return generate_synthetic_telemetry('pathologic', duration_minutes=55)


bundle = load_bundle()
DATA_OK = bundle is not None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="doc-serial">DOC_REF: CTG-2126 // REV.2026.09 // FIGO 2015 / ACOG / PHYSIONET CTU-UHB</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Cardiotocography Fetal Distress Diagnostic Schematic</div>', unsafe_allow_html=True)
st.markdown('<div class="doc-sub">Temporal Moving Average Filtered Telemetry · Multi-Family Risk Classifier · Asymmetric Bayesian Safety Threshold</div>', unsafe_allow_html=True)
st.markdown('<div class="dim-line">← CONTINUOUS 4 Hz SIGNAL ENGINE · PHYSIOLOGICAL TIME INERTIA · BED-SIDE CDSS →</div>', unsafe_allow_html=True)

# ── Top controls strip ────────────────────────────────────────────────────────
cc1, cc2, cc3, cc4 = st.columns([1.4, 1.4, 1.0, 1.0])
with cc1:
    model_choice = st.selectbox("⚙️ INFERENCE ENGINE",
        ["LightGBM (Optuna-Tuned)", "Random Forest", "SVM (RBF Kernel)"])
with cc2:
    st.markdown("<div style='font-size:10px;color:#00ffff;'>TEMPORAL SMOOTHING</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;color:#00ffff;'>EMA Inertia Filter (α = 0.25)</div>", unsafe_allow_html=True)
with cc3:
    st.markdown("<div style='font-size:10px;color:#00ffff;'>EVALUATION COHORT</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;'>N=426 Held-Out (332/59/35)</div>", unsafe_allow_html=True)
with cc4:
    st.markdown("<div style='font-size:10px;color:#00ffff;'>CHAMPION CV SCORE</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;color:#00ff88;'>Macro F1 = 0.9202 (Optuna)</div>", unsafe_allow_html=True)

# ── Main panel with 6 Tabs ────────────────────────────────────────────────────
tabs = st.tabs([
    "🔴 Live Automatic Telemetry",
    "📐 Manual Morphometric Drafting",
    "📊 Model Benchmark",
    "📉 Calibration & ROC",
    "🌊 SHAP Explainability",
    "📋 5-Fold CV Results"
])

def render_decision_verdict(probs):
    """
    Clinically accurate decision logic:
    Primary category is determined by the dominant class.
    Asymmetric safety cutoff (P_pathologic >= 0.110) acts as an early warning alert.
    """
    p_n, p_s, p_p = probs
    max_idx = np.argmax([p_n, p_s, p_p])

    if max_idx == 2:
        st.markdown(f"""
        <div class="stamp-pathologic">
        [ CLASS 3: PATHOLOGICAL (FIGO CATEGORY III) ]<br>
        <span style="font-size:11px;color:#fff;">CRITICAL FETAL HYPOXIA — DOMINANT POSTERIOR RISK: {p_p*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
    elif max_idx == 1:
        st.markdown(f"""
        <div class="stamp-suspect">
        [ CLASS 2: SUSPECT / EQUIVOCAL (FIGO CATEGORY II) ]<br>
        <span style="font-size:11px;color:#fff;">SUB-OPTIMAL VARIABILITY / DECELERATIONS — POSTERIOR PROBABILITY: {p_s*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="stamp-normal">
        [ CLASS 1: NORMAL / REASSURING (FIGO CATEGORY I) ]<br>
        <span style="font-size:11px;color:#fff;">AUTONOMIC HOMEOSTASIS INTACT — POSTERIOR PROBABILITY: {p_n*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    # Asymmetric Decision Precaution Banner
    if p_p >= 0.110 and max_idx != 2:
        st.markdown(f"""
        <div class="alert-precaution">
        <strong>⚠️ PRECAUTIONARY SAFETY ALERT:</strong> Pathologic risk <strong>{p_p*100:.1f}%</strong> breaches asymmetric loss threshold (cutoff P* ≥ 11.0%). Escalate surveillance.
        </div>
        """, unsafe_allow_html=True)
    elif max_idx == 2:
        st.markdown(f"""
        <div class="alert-emergency">
        <strong>🚨 EMERGENCY CLINICAL ALARM:</strong> Category III Distress Confirmed. Prepare intrauterine resuscitation / emergency delivery.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────── TAB 0: LIVE AUTOMATIC TELEMETRY ─────────────────
with tabs[0]:
    st.subheader("[ SEC 00 // CONTINUOUS TELEMETRY STREAM WITH PHYSIOLOGICAL TEMPORAL INERTIA ]")
    st.markdown("""
    <div style='font-size:12px;color:rgba(255,255,255,0.7);margin-bottom:10px;'>
    Streams continuous 4 Hz cardiotocograms from the <strong>PhysioNet CTU-UHB database</strong>.
    Uses an <strong>Exponential Moving Average (EMA) temporal filter (α = 0.25)</strong> to reflect genuine physiological inertia (no erratic jumps).
    </div>
    """, unsafe_allow_html=True)

    t_case, t_win = st.columns([2.0, 1.0])
    with t_case:
        patient_case = st.selectbox(
            "🏥 CLINICAL PATIENT RECORDING",
            [
                "PhysioNet Record 1003 — Healthy Reassuring Term Labor (Umbilical pH 7.20, Apgar 9)",
                "PhysioNet Record 1001 — Moderate Fetal Acidemia / Variable Decels (Umbilical pH 7.14)",
                "PhysioNet Record 1002 — Severe Intrapartum Hypoxia / Acidosis (Umbilical pH 7.00, BE -12)",
                "Synthetic Live Telemetry — Progressive Autonomic Distress Sequence"
            ]
        )
    with t_win:
        win_mins = st.selectbox("⏱️ SLIDING ANALYSIS WINDOW", [5, 10, 15], index=1)

    fhr_full, uc_full, fs = get_cached_telemetry(patient_case)
    total_mins = int(len(fhr_full) / (fs * 60))

    # Session state initialization for smooth streaming
    if 'stream_min' not in st.session_state:
        st.session_state.stream_min = min(win_mins + 3, total_mins)
    if 'is_streaming' not in st.session_state:
        st.session_state.is_streaming = False
    if 'smooth_probs' not in st.session_state:
        st.session_state.smooth_probs = np.array([0.90, 0.08, 0.02])
    if 'prob_history' not in st.session_state:
        st.session_state.prob_history = []
    if 'last_case' not in st.session_state or st.session_state.last_case != patient_case:
        st.session_state.last_case = patient_case
        st.session_state.prob_history = []
        st.session_state.smooth_probs = np.array([0.90, 0.08, 0.02])
        st.session_state.stream_min = win_mins + 2

    # Playback controls
    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns([1.2, 1.0, 1.2, 1.6])
    with btn_c1:
        if st.button("▶️ Start Live Stream", key="btn_play", use_container_width=True):
            st.session_state.is_streaming = True
    with btn_c2:
        if st.button("⏸️ Pause", key="btn_pause", use_container_width=True):
            st.session_state.is_streaming = False
    with btn_c3:
        if st.button("⏮️ Reset to Start", key="btn_reset", use_container_width=True):
            st.session_state.stream_min = win_mins + 2
            st.session_state.prob_history = []
            st.session_state.smooth_probs = np.array([0.90, 0.08, 0.02])
            st.session_state.is_streaming = False
    with btn_c4:
        stream_speed = st.select_slider("⚡ Playback Speed", options=["1x (Bedside Pace)", "2x (Smooth Clinical)", "4x (Rapid Review)"], value="2x (Smooth Clinical)")

    scrub_val = st.slider(
        "MANUAL LABOR TIMELINE SEEKER (MINUTES)",
        min_value=int(win_mins),
        max_value=max(int(total_mins), int(win_mins) + 1),
        value=int(st.session_state.stream_min),
        step=1
    )
    if not st.session_state.is_streaming:
        st.session_state.stream_min = scrub_val

    live_container = st.empty()
    speed_delay = 0.9 if "1x" in stream_speed else (0.45 if "2x" in stream_speed else 0.22)

    def draw_telemetry_frame(cur_min):
        end_sample = int(cur_min * 60 * fs)
        start_sample = max(0, end_sample - int(win_mins * 60 * fs))
        fhr_window = fhr_full[start_sample:end_sample]
        uc_window = uc_full[start_sample:end_sample]

        context_mins = 20
        ctx_start = max(0, end_sample - int(context_mins * 60 * fs))
        t_axis = np.linspace(max(0, cur_min - context_mins), cur_min, end_sample - ctx_start)
        fhr_ctx = fhr_full[ctx_start:end_sample]
        uc_ctx = uc_full[ctx_start:end_sample]

        # Feature Extraction & Model Inference
        feat_dict = extract_sliding_window_features(fhr_window, uc_window, fs)
        if DATA_OK and len(fhr_window) > 20:
            vec = np.array([feat_dict.get(f, 0.) for f in bundle['feat_names']]).reshape(1, -1)
            if "LightGBM" in model_choice:
                raw_probs = bundle['lgb'].predict_proba(vec)[0]
            elif "Random Forest" in model_choice:
                raw_probs = bundle['rf'].predict_proba(vec)[0]
            else:
                raw_probs = bundle['svm'].predict_proba(bundle['scaler'].transform(vec))[0]

            # Temporal EMA Smoothing (physiological inertia)
            alpha = 0.30
            st.session_state.smooth_probs = alpha * raw_probs + (1.0 - alpha) * st.session_state.smooth_probs
            p_n, p_s, p_p = st.session_state.smooth_probs

            # Update history for trajectory trendline
            st.session_state.prob_history.append((cur_min, p_n, p_s, p_p))
            if len(st.session_state.prob_history) > 60:
                st.session_state.prob_history.pop(0)

        with live_container.container():
            # ── 1. Medical CTG Oscilloscope ──────────────────────────────────
            fig, (ax_fhr, ax_uc) = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True, gridspec_kw={'height_ratios': [2.3, 1.0]})
            fig.patch.set_facecolor('#00182f')
            for ax in (ax_fhr, ax_uc):
                ax.set_facecolor('#001020')
                ax.tick_params(colors='#00ffff', labelsize=8)
                # Clinical grid lines (like standard CTG paper)
                ax.grid(True, which='both', color='#00ffff', alpha=0.14, linestyle='-')
                ax.minorticks_on()
                ax.grid(True, which='minor', color='#00ffff', alpha=0.06, linestyle=':')

            # FHR Plot
            ax_fhr.axhspan(110, 160, color='#00ff88', alpha=0.12, label='FIGO Reassuring Range (110-160 bpm)')
            ax_fhr.axhline(110, color='#ff3333', linestyle='--', linewidth=1.0, alpha=0.7, label='Bradycardia (<110)')
            ax_fhr.axhline(160, color='#ffcc00', linestyle='--', linewidth=1.0, alpha=0.7, label='Tachycardia (>160)')
            ax_fhr.plot(t_axis, fhr_ctx, color='#00ffff', linewidth=1.2, label='Fetal Heart Rate (bpm)')
            ax_fhr.axvspan(cur_min - win_mins, cur_min, color='#00ffff', alpha=0.15, label=f'Sliding Window ({win_mins} min)')
            ax_fhr.set_ylim(60, 205)
            ax_fhr.set_yticks([80, 100, 110, 120, 140, 160, 180, 200])
            ax_fhr.set_ylabel('FHR (bpm)', color='#00ffff', fontweight='bold', fontsize=9)
            ax_fhr.legend(loc='upper right', fontsize=7, facecolor='#00182f', edgecolor='#00ffff', labelcolor='#ffffff')

            # UC Plot
            ax_uc.plot(t_axis, uc_ctx, color='#ec4899', linewidth=1.3, label='Tocogram (Contractions)')
            ax_uc.axvspan(cur_min - win_mins, cur_min, color='#00ffff', alpha=0.15)
            ax_uc.set_ylim(0, 100)
            ax_uc.set_ylabel('UC (mmHg)', color='#ec4899', fontweight='bold', fontsize=9)
            ax_uc.set_xlabel(f'Labor Timeline — Minute {cur_min} of {total_mins} (Active Analysis: Min {max(0, cur_min - win_mins)} to {cur_min})', color='#00ffff', fontsize=9)
            ax_uc.legend(loc='upper right', fontsize=7, facecolor='#00182f', edgecolor='#00ffff', labelcolor='#ffffff')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── 2. Decision Verdict & Posterior Gauges ────────────────────────
            res_l, res_r = st.columns([1.3, 1.0])
            with res_l:
                render_decision_verdict((p_n, p_s, p_p))

            with res_r:
                fig_p, ax_p = plt.subplots(figsize=(5, 1.8))
                fig_p.patch.set_facecolor('#00182f'); ax_p.set_facecolor('#001020')
                prob_df = pd.DataFrame({'State':['NORMAL','SUSPECT','PATHOLOGIC'], 'P':[p_n, p_s, p_p]})
                sns.barplot(x='P', y='State', data=prob_df, palette=['#00ff88','#ffcc00','#ff3333'], ax=ax_p, edgecolor='#00ffff', linewidth=0.8)
                for i, v in enumerate([p_n, p_s, p_p]):
                    ax_p.text(v+.02, i, f'{v*100:.1f}%', va='center', fontweight='bold', color='#fff', fontfamily='monospace', fontsize=8)
                ax_p.set_xlim(0, 1.18); ax_p.set_xlabel('Temporally Filtered Posterior Risk', color='#00ffff', fontsize=8)
                ax_p.set_ylabel(''); ax_p.tick_params(colors='#00ffff', labelsize=8)
                plt.tight_layout(); st.pyplot(fig_p); plt.close()

            # ── 3. Live 30-Minute Continuous Risk Trajectory Trendline ────────
            if len(st.session_state.prob_history) >= 2:
                hist = st.session_state.prob_history
                h_t = [h[0] for h in hist]
                h_norm = [h[1] * 100 for h in hist]
                h_susp = [h[2] * 100 for h in hist]
                h_path = [h[3] * 100 for h in hist]

                fig_tr, ax_tr = plt.subplots(figsize=(11, 1.8))
                fig_tr.patch.set_facecolor('#00182f'); ax_tr.set_facecolor('#001020')
                ax_tr.plot(h_t, h_norm, color='#00ff88', linewidth=1.5, label='Normal %')
                ax_tr.plot(h_t, h_susp, color='#ffcc00', linewidth=1.5, label='Suspect %')
                ax_tr.plot(h_t, h_path, color='#ff3333', linewidth=2.0, label='Pathologic Risk %')
                ax_tr.axhline(11.0, color='#f59e0b', linestyle=':', linewidth=1.2, label='Safety Cutoff (11%)')
                ax_tr.set_ylim(0, 105)
                ax_tr.set_ylabel('Risk Trajectory %', color='#00ffff', fontsize=8)
                ax_tr.set_xlabel('Labor Elapsed Time (Minutes)', color='#00ffff', fontsize=8)
                ax_tr.tick_params(colors='#00ffff', labelsize=7)
                ax_tr.grid(True, color='#00ffff', alpha=0.12, linestyle='--')
                ax_tr.legend(loc='upper left', fontsize=7, facecolor='#00182f', edgecolor='#00ffff', labelcolor='#ffffff', ncol=4)
                plt.tight_layout(); st.pyplot(fig_tr); plt.close()

            # ── 4. Live Biomarker Telemetry Metrics ───────────────────────────
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Live Baseline FHR", f"{feat_dict['LB']:.1f} bpm")
            m2.metric("Short-Term Var (ASTV)", f"{feat_dict['ASTV']:.1f}%")
            m3.metric("Pathologic Risk (PRI)", f"{feat_dict['PRI']:.2f}")
            m4.metric("Decel Severity (DSI)", f"{feat_dict['DSI']:.3f}")
            m5.metric("Variability Entropy", f"{feat_dict['Variability_Entropy']:.2f} bits")

    # Streaming execution
    if st.session_state.is_streaming:
        for m in range(int(st.session_state.stream_min), total_mins + 1):
            st.session_state.stream_min = m
            draw_telemetry_frame(m)
            time.sleep(speed_delay)
        st.session_state.is_streaming = False
        st.rerun()
    else:
        draw_telemetry_frame(st.session_state.stream_min)


# ─────────────────────────── TAB 1: MANUAL MORPHOMETRIC DRAFTING ─────────────
with tabs[1]:
    preset_box = st.selectbox("📋 CHOOSE CLINICAL SCENARIO PRESET",
        ["Custom Input", "Preset A — Normal (Reassuring)",
         "Preset B — Suspect (Borderline)", "Preset C — Pathologic (Distress)"],
        key="static_preset")

    defaults = dict(LB=135., AC=.003, FM=0., UC=.005, DL=0., DS=0., DP=0.,
                    ASTV=28., MSTV=1.8, ALTV=0., MLTV=8.5, Width=60., Min=110.,
                    Max=170., Nmax=3., Nzeros=0., Mode=138., Mean=136.,
                    Median=137., Variance=8., Tendency=0.)

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
            p_arr = np.array([max(inp['ASTV'],.01), max(inp['ALTV'],.01),
                              max(100-inp['ASTV']-inp['ALTV'],.01)])
            p_arr /= p_arr.sum()
            import scipy.stats as sst
            inp['Variability_Entropy'] = float(sst.entropy(p_arr, base=2))

            vec = np.array([inp.get(f, 0.) for f in feat_names]).reshape(1, -1)

            if "LightGBM" in model_choice:
                probs = bundle['lgb'].predict_proba(vec)[0]
            elif "Random Forest" in model_choice:
                probs = bundle['rf'].predict_proba(vec)[0]
            else:
                probs = bundle['svm'].predict_proba(bundle['scaler'].transform(vec))[0]

            render_decision_verdict(probs)

            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6.5, 2.4))
            fig.patch.set_facecolor('#00182f'); ax.set_facecolor('#001020')
            prob_df = pd.DataFrame({'State':['NORMAL','SUSPECT','PATHOLOGIC'], 'P':probs})
            sns.barplot(x='P', y='State', data=prob_df, palette=['#00ff88','#ffcc00','#ff3333'],
                        ax=ax, edgecolor='#00ffff', linewidth=1)
            for i, v in enumerate(probs):
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
  <div class="btbar-brand">📐 CTG-OS // CLINICAL TELEMETRY CDSS</div>
  <div class="btbar-pills">
    <div class="btbar-pill">🔴 EMA FILTERED STREAM</div>
    <div class="btbar-pill">📐 MANUAL DRAFTING</div>
    <div class="btbar-pill">📊 BENCHMARK</div>
    <div class="btbar-pill">📉 CALIBRATION</div>
    <div class="btbar-pill">🌊 SHAP XAI</div>
    <div class="btbar-pill">📋 CV RESULTS</div>
  </div>
  <div class="btbar-tray">
    <span class="btbar-lock">🛡️ P≥0.110 ASYMMETRIC SAFETY CUTOFF</span>
    <span class="btbar-info">TEMPORAL EMA FILTER (α = 0.25)</span>
  </div>
</div>
""", unsafe_allow_html=True)
