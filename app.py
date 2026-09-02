import os
import json
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


bundle = load_bundle()
DATA_OK = bundle is not None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="doc-serial">DOC_REF: CTG-2126 // REV.2026.09 // FIGO 2015 / ACOG</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Cardiotocography Fetal Distress Diagnostic Schematic</div>', unsafe_allow_html=True)
st.markdown('<div class="doc-sub">Multi-Family Clinical Risk Classifier · Asymmetric Cost Calibration · SHAP Explainability</div>', unsafe_allow_html=True)
st.markdown('<div class="dim-line">← MASTER WORKSPACE · 39 FEATURES · 2,126 PATIENTS · STACKING ENSEMBLE CHAMPION →</div>', unsafe_allow_html=True)

st.markdown("""<div class="redline-box">
<strong>[CLINICAL SAFETY NOTE]</strong> Asymmetric cost matrix enforced:
<strong>Cost(FN) = 10 × Cost(FP)</strong>. Pathologic safety threshold:
<strong>P ≥ 0.110</strong> → Recall 94.29% (33/35 distress cases caught on held-out test).
</div>""", unsafe_allow_html=True)

# ── Top controls strip ────────────────────────────────────────────────────────
cc1, cc2, cc3, cc4 = st.columns([1.4, 1.4, 1.0, 1.0])
with cc1:
    model_choice = st.selectbox("⚙️ CLASSIFIER FAMILY",
        ["LightGBM (Optuna-Tuned)", "Random Forest", "SVM (RBF Kernel)"])
with cc2:
    preset = st.selectbox("📋 CLINICAL PRESET",
        ["Custom Input", "Preset A — Normal (Reassuring)",
         "Preset B — Suspect (Borderline)", "Preset C — Pathologic (Distress)"])
with cc3:
    st.markdown("<div style='font-size:10px;color:#00ffff;'>COHORT (HELD-OUT)</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;'>N=426 · 332/59/35</div>", unsafe_allow_html=True)
with cc4:
    st.markdown("<div style='font-size:10px;color:#00ffff;'>CHAMPION MODEL</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;color:#00ff88;'>Macro F1 = 0.9030+</div>", unsafe_allow_html=True)

# ── Default values per preset ─────────────────────────────────────────────────
defaults = dict(LB=135., AC=.003, FM=0., UC=.005, DL=0., DS=0., DP=0.,
                ASTV=28., MSTV=1.8, ALTV=0., MLTV=8.5, Width=60., Min=110.,
                Max=170., Nmax=3., Nzeros=0., Mode=138., Mean=136.,
                Median=137., Variance=8., Tendency=0.)
if "Normal" in preset:
    defaults.update(LB=135., AC=.005, ASTV=22., ALTV=0., DP=0., MSTV=2.1)
elif "Suspect" in preset:
    defaults.update(LB=152., AC=.001, ASTV=58., ALTV=18., DP=0., MSTV=.8)
elif "Pathologic" in preset:
    defaults.update(LB=168., AC=0., ASTV=78., ALTV=45., DP=.003, DS=.001, MSTV=.4)

# ── Main panel ────────────────────────────────────────────────────────────────
tabs = st.tabs(["📐 Live Drafting", "📊 Benchmark", "📉 Calibration & ROC",
                "🌊 SHAP Explainability", "📋 CV Results"])

# ─────────────────────────── TAB 1: LIVE DRAFTING ────────────────────────────
with tabs[0]:
    sl, sr = st.columns([1.05, 1.35])

    with sl:
        st.markdown("#### [ SEC 01 // MORPHOMETRIC DRAFTING ]")
        inp = {}
        inp['LB']   = st.slider("BASELINE FHR (LB bpm)", 80., 200., float(defaults['LB']), 1.)
        inp['AC']   = st.slider("ACCELERATIONS / SEC (AC)", 0., .02, float(defaults['AC']), .001, format="%.3f")
        inp['UC']   = st.slider("UTERINE CONTRACTIONS / SEC (UC)", 0., .02, float(defaults['UC']), .001, format="%.3f")
        inp['ASTV'] = st.slider("% ABNORMAL SHORT-TERM VAR (ASTV)", 0., 100., float(defaults['ASTV']), 1.)
        inp['DP']   = st.slider("PROLONGED DECELS / SEC (DP)", 0., .01, float(defaults['DP']), .0005, format="%.4f")
        inp['ALTV'] = st.slider("% ABNORMAL LONG-TERM VAR (ALTV)", 0., 100., float(defaults['ALTV']), 1.)
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
with tabs[1]:
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
with tabs[2]:
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
with tabs[3]:
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
with tabs[4]:
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
  <div class="btbar-brand">📐 CTG-OS // FETAL MONITORING</div>
  <div class="btbar-pills">
    <div class="btbar-pill">🎛️ LIVE DRAFTING</div>
    <div class="btbar-pill">📊 BENCHMARK</div>
    <div class="btbar-pill">📉 CALIBRATION</div>
    <div class="btbar-pill">🌊 SHAP XAI</div>
    <div class="btbar-pill">📋 CV RESULTS</div>
  </div>
  <div class="btbar-tray">
    <span class="btbar-lock">🛡️ P≥0.110 · RECALL 94.29%</span>
    <span class="btbar-info">DOC_REF: CTG-2126 // FIGO 2015</span>
  </div>
</div>
""", unsafe_allow_html=True)
