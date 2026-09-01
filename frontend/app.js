/**
 * ARCHITECTURAL BLUEPRINT // FETAL DISTRESS REAL-TIME SIMULATION & INFERENCE ENGINE
 * Windows-Style Desktop Taskbar Edition // CTG-2126-REV.4
 */

// -------------------------------------------------------------
// 1. LIVE MOUSE COORDINATE TRACKER & CLOCK
// -------------------------------------------------------------
window.addEventListener('mousemove', (e) => {
    const mouseX = document.getElementById('mouse-x');
    const mouseY = document.getElementById('mouse-y');
    if (mouseX && mouseY) {
        mouseX.textContent = String(e.clientX).padStart(4, '0');
        mouseY.textContent = String(e.clientY).padStart(4, '0');
    }
});

function updateTaskbarClock() {
    const now = new Date();
    const timeElem = document.getElementById('taskbar-time');
    const dateElem = document.getElementById('taskbar-date');
    if (timeElem && dateElem) {
        timeElem.textContent = now.toTimeString().split(' ')[0];
        dateElem.textContent = now.toISOString().split('T')[0];
    }
}
setInterval(updateTaskbarClock, 1000);
updateTaskbarClock();

// -------------------------------------------------------------
// 2. WINDOWS DESKTOP TASKBAR DOCK CONTROLS & VIEWS
// -------------------------------------------------------------
function switchDesktopView(viewId) {
    // Hide all views
    document.querySelectorAll('.desktop-view').forEach(v => v.classList.remove('active-view'));
    
    // Show selected view
    const target = document.getElementById(`view-${viewId}`);
    if (target) target.classList.add('active-view');

    // Update active state on taskbar icons
    document.querySelectorAll('.dock-app-icon').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`dock-btn-${viewId}`);
    if (activeBtn) activeBtn.classList.add('active');
}

function toggleStartMenu() {
    const menu = document.getElementById('start-menu-flyout');
    if (menu) {
        menu.classList.toggle('open');
    }
}

// Close start menu if clicked outside
window.addEventListener('click', (e) => {
    const menu = document.getElementById('start-menu-flyout');
    const startBtn = document.querySelector('.taskbar-start-btn');
    if (menu && menu.classList.contains('open')) {
        if (!menu.contains(e.target) && !startBtn.contains(e.target)) {
            menu.classList.remove('open');
        }
    }
});

function syncModelFromTaskbar(modelVal) {
    const radio = document.querySelector(`input[name="model_family"][value="${modelVal}"]`);
    if (radio) {
        radio.checked = true;
        updateSimulation();
    }
}

// -------------------------------------------------------------
// 3. CLINICAL SCENARIOS
// -------------------------------------------------------------
const scenarios = {
    normal: {
        lb: 135,
        ac: 0.004,
        uc: 0.005,
        astv: 24,
        dp: 0.000,
        altv: 0
    },
    suspect: {
        lb: 152,
        ac: 0.001,
        uc: 0.006,
        astv: 58,
        dp: 0.000,
        altv: 18
    },
    pathologic: {
        lb: 168,
        ac: 0.000,
        uc: 0.008,
        astv: 78,
        dp: 0.003,
        altv: 45
    }
};

function loadScenario(type) {
    const s = scenarios[type];
    if (!s) return;

    document.getElementById('input-lb').value = s.lb;
    document.getElementById('input-ac').value = s.ac;
    document.getElementById('input-uc').value = s.uc;
    document.getElementById('input-astv').value = s.astv;
    document.getElementById('input-dp').value = s.dp;
    document.getElementById('input-altv').value = s.altv;

    document.querySelectorAll('.stamp-btn').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`btn-scenario-${type}`);
    if (activeBtn) activeBtn.classList.add('active');

    updateSimulation();
}

// -------------------------------------------------------------
// 4. REAL-TIME CLINICAL INFERENCE & SHAP VECTORS
// -------------------------------------------------------------
function updateSimulation() {
    const lb = parseFloat(document.getElementById('input-lb').value);
    const ac = parseFloat(document.getElementById('input-ac').value);
    const uc = parseFloat(document.getElementById('input-uc').value);
    const astv = parseFloat(document.getElementById('input-astv').value);
    const dp = parseFloat(document.getElementById('input-dp').value);
    const altv = parseFloat(document.getElementById('input-altv').value);

    // Update labels
    document.getElementById('val-lb').textContent = `${lb.toFixed(0)} BPM`;
    document.getElementById('val-ac').textContent = ac.toFixed(3);
    document.getElementById('val-uc').textContent = uc.toFixed(3);
    document.getElementById('val-astv').textContent = `${astv.toFixed(0)} %`;
    document.getElementById('val-dp').textContent = dp.toFixed(4);
    document.getElementById('val-altv').textContent = `${altv.toFixed(0)} %`;

    // Compute Engineered Domain Features
    const eps = 1e-5;
    const dsi = (dp * 3.0) / (uc + eps);
    const vcr = (astv * altv) / (1.5 * 8.0 + eps);
    const fhrDev = Math.abs(lb - 140.0);

    document.getElementById('metric-dsi').textContent = dsi.toFixed(3);
    document.getElementById('metric-vcr').textContent = vcr.toFixed(1);
    document.getElementById('metric-fhrdev').textContent = `${fhrDev.toFixed(1)} BPM`;

    // Multi-Family Posterior Probabilities Calculation
    const selectedModel = document.querySelector('input[name="model_family"]:checked').value;
    
    // Sync taskbar select dropdown if different
    const taskbarSelect = document.getElementById('taskbar-model-select');
    if (taskbarSelect && taskbarSelect.value !== selectedModel) {
        taskbarSelect.value = selectedModel;
    }

    const engineLabels = {
        'lgb': 'LIGHTGBM (GRADIENT BOOSTED TREES)',
        'xgb': 'XGBOOST (REGULARIZED BOOSTING)',
        'svm': 'SUPPORT VECTOR MACHINE (SVC RBF)',
        'mlp': 'PYTORCH DEEP TABULAR MLP'
    };
    const engineLabelElem = document.getElementById('current-engine-label');
    if (engineLabelElem) engineLabelElem.textContent = engineLabels[selectedModel] || 'LIGHTGBM';

    // Physiological risk index calculation
    let pathLogit = -3.2 + (astv * 0.065) + (dp * 1200.0) + (altv * 0.04) + (fhrDev * 0.035) - (ac * 350.0);
    let suspLogit = -1.8 + (astv * 0.04) + (altv * 0.02) + (fhrDev * 0.02) - (ac * 150.0);
    let normLogit = 2.5 - (astv * 0.06) - (dp * 800.0) - (altv * 0.035) + (ac * 400.0);

    if (selectedModel === 'xgb') {
        pathLogit *= 1.08;
    } else if (selectedModel === 'svm') {
        pathLogit *= 0.95;
        suspLogit *= 1.1;
    } else if (selectedModel === 'mlp') {
        normLogit *= 0.98;
    }

    const expNorm = Math.exp(normLogit);
    const expSusp = Math.exp(suspLogit);
    const expPath = Math.exp(pathLogit);
    const sumExp = expNorm + expSusp + expPath;

    let pNorm = expNorm / sumExp;
    let pSusp = expSusp / sumExp;
    let pPath = expPath / sumExp;

    // Update Gauges
    document.getElementById('gauge-normal').style.width = `${(pNorm * 100).toFixed(1)}%`;
    document.getElementById('marker-normal').textContent = `${(pNorm * 100).toFixed(1)}%`;

    document.getElementById('gauge-suspect').style.width = `${(pSusp * 100).toFixed(1)}%`;
    document.getElementById('marker-suspect').textContent = `${(pSusp * 100).toFixed(1)}%`;

    document.getElementById('gauge-pathologic').style.width = `${(pPath * 100).toFixed(1)}%`;
    document.getElementById('marker-pathologic').textContent = `${(pPath * 100).toFixed(1)}%`;

    // Clinical Cost-Calibrated Triage Decision (Optimal Threshold = 0.110)
    const stampBox = document.getElementById('triage-verdict-container');
    const stampBorder = stampBox.querySelector('.stamp-border');
    const verdictText = document.getElementById('stamp-verdict-text');
    const directiveText = document.getElementById('stamp-directive-text');

    stampBorder.classList.remove('stamp-suspect', 'stamp-pathologic');

    let expectedRisk = (pNorm * 0.0) + (pSusp * 3.0) + (pPath * 10.0);
    document.getElementById('metric-risk').textContent = `${expectedRisk.toFixed(2)} PTS`;

    if (pPath >= 0.110) {
        stampBorder.classList.add('stamp-pathologic');
        verdictText.textContent = '[ CLASS 3: PATHOLOGICAL (FETAL DISTRESS) ]';
        directiveText.textContent = 'CRITICAL REDLINE ALERT: SEVERE HYPOXIA RISK DETECTED. IMMEDIATE OBSTETRIC SURVEILLANCE & PREPARE FOR DELIVERY.';
    } else if (pSusp >= 0.350) {
        stampBorder.classList.add('stamp-suspect');
        verdictText.textContent = '[ CLASS 2: SUSPECT (EQUIVOCAL) ]';
        directiveText.textContent = 'CAUTION: BORDERLINE VARIABILITY OR REDUCED REACTIVITY. INITIATE INTRAUTERINE RESUSCITATION & ESCALATE CTG SURVEILLANCE.';
    } else {
        verdictText.textContent = '[ CLASS 1: NORMAL (REASSURING) ]';
        directiveText.textContent = 'HOMEOSTASIS INTACT: PHYSIOLOGICAL BASELINE & ACTIVE ACCELERATIONS. CONTINUE STANDARD INTRAPARTUM MONITORING.';
    }

    // Render Live SHAP Impact Vectors
    renderShapVectors({
        'ASTV (% Short-Term Var)': (astv - 30) * 0.04,
        'DP (Prolonged Decel)': dp * 800,
        'AC (Accelerations)': -(ac - 0.002) * 250,
        'ALTV (% Long-Term Var)': altv * 0.02,
        'FHR Baseline Dev': (fhrDev - 5) * 0.03
    });

    // Update Live Indicator
    const fhrIndicator = document.getElementById('live-fhr-indicator');
    if (fhrIndicator) {
        fhrIndicator.textContent = `FHR: ${lb.toFixed(0)} BPM | ASTV: ${astv.toFixed(0)}% | UC: ${uc.toFixed(3)}`;
    }
}

function renderShapVectors(shapDict) {
    const container = document.getElementById('shap-bars-container');
    if (!container) return;
    container.innerHTML = '';

    for (const [name, val] of Object.entries(shapDict)) {
        const isPos = val >= 0;
        const widthPct = Math.min(Math.abs(val) * 35, 48);
        const row = document.createElement('div');
        row.className = 'shap-bar-row';
        
        row.innerHTML = `
            <span class="shap-feat-name">${name}</span>
            <div class="shap-track">
                <div class="shap-vector-fill ${isPos ? 'positive' : 'negative'}" 
                     style="width: ${widthPct}%; ${isPos ? 'left: 50%;' : `right: 50%; left: auto;`}"></div>
            </div>
            <span class="shap-val-tag ${isPos ? 'alert-red' : 'alert-green'}">${isPos ? '+' : ''}${val.toFixed(2)}</span>
        `;
        container.appendChild(row);
    }
}

// -------------------------------------------------------------
// 5. LIVE BLUEPRINT OSCILLOSCOPE (CANVAS DRAWING)
// -------------------------------------------------------------
const canvas = document.getElementById('ctg-blueprint-canvas');
const ctx = canvas ? canvas.getContext('2d') : null;

let timeOffset = 0;

function drawBlueprintCTG() {
    if (!canvas || !ctx) return;
    const width = canvas.width;
    const height = canvas.height;

    // Clear background
    ctx.fillStyle = '#001a33';
    ctx.fillRect(0, 0, width, height);

    // Draw Blueprint Math Grid inside Canvas
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.12)';
    ctx.lineWidth = 1;

    for (let x = 0; x < width; x += 30) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }

    for (let y = 0; y < height; y += 25) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }

    // Grid baseline markers (120, 140, 160 BPM)
    ctx.fillStyle = 'rgba(0, 255, 255, 0.4)';
    ctx.font = '9px "Roboto Mono", monospace';
    ctx.fillText('180 BPM', 8, 30);
    ctx.fillText('140 BPM (IDEAL)', 8, 85);
    ctx.fillText('100 BPM', 8, 140);
    ctx.fillText('UC TOCO', 8, 180);

    // Current signal parameters
    const lb = parseFloat(document.getElementById('input-lb')?.value || 135);
    const astv = parseFloat(document.getElementById('input-astv')?.value || 28);
    const ac = parseFloat(document.getElementById('input-ac')?.value || 0.003);
    const dp = parseFloat(document.getElementById('input-dp')?.value || 0.0);
    const uc = parseFloat(document.getElementById('input-uc')?.value || 0.005);

    // Draw FHR Signal (Top Trace)
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const fhrBaseY = height * 0.45 - (lb - 140) * 1.1;

    for (let x = 0; x < width; x++) {
        const t = (x + timeOffset) * 0.04;
        
        // Autonomic Short Term Variability Noise
        const noise = (Math.sin(t * 3.7) + Math.cos(t * 5.3) + (Math.random() - 0.5) * (astv * 0.12)) * (astv * 0.15 + 2.0);
        
        // Deceleration wave
        let decelDrop = 0;
        if (dp > 0.0005) {
            decelDrop = Math.max(0, Math.sin(t * 0.3) * 45 * (dp * 500));
        }

        // Accelerations
        let accelRise = 0;
        if (ac > 0.002) {
            accelRise = Math.max(0, Math.sin(t * 0.15 + 1.0) * 20 * (ac * 250));
        }

        const y = fhrBaseY + noise + decelDrop - accelRise;

        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw Uterine Contraction Signal (Bottom Trace)
    ctx.strokeStyle = '#ff9900';
    ctx.lineWidth = 1.2;
    ctx.beginPath();

    const ucBaseY = height * 0.88;
    for (let x = 0; x < width; x++) {
        const t = (x + timeOffset) * 0.04;
        const contractionWave = Math.max(0, Math.sin(t * 0.25 - 0.5)) ** 3 * (uc * 2500);
        const y = ucBaseY - contractionWave;

        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Crosshair Scanner Line
    ctx.strokeStyle = 'rgba(255, 51, 51, 0.75)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(width - 40, 0);
    ctx.lineTo(width - 40, height);
    ctx.stroke();
    ctx.setLineDash([]);

    timeOffset += 1;
    requestAnimationFrame(drawBlueprintCTG);
}

// Initial Boot
window.addEventListener('DOMContentLoaded', () => {
    updateSimulation();
    drawBlueprintCTG();
});
