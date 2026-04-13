/**
 * app.js
 * ──────
 * EnergyIQ frontend application.
 * Handles API calls, Chart.js rendering, real-time simulation,
 * and all user interactions.
 *
 * No build step required – plain ES2022 modules-ish style.
 */

// ── Config ────────────────────────────────────────────────────────────────
const API_BASE = window.location.origin;

// Chart.js global defaults (dark theme)
Chart.defaults.color = '#7a8499';
Chart.defaults.borderColor = '#1f2430';
Chart.defaults.font.family = "'Space Mono', monospace";
Chart.defaults.font.size = 11;

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  selectedModel: 'linear',
  anomalyMethod: 'rolling_zscore',
  simRunning: false,
  simInterval: null,
  simData: [],
  simLabels: [],
  historySeries: [],
};

// ── Chart Instances ────────────────────────────────────────────────────────
const charts = {};

// ── API Helpers ────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.error(`API error [${path}]:`, e);
    return null;
  }
}

// ── Chart Factory Helpers ──────────────────────────────────────────────────
function accentGradient(ctx, alpha1 = 0.35, alpha2 = 0.0) {
  const g = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height);
  g.addColorStop(0, `rgba(0,229,160,${alpha1})`);
  g.addColorStop(1, `rgba(0,229,160,${alpha2})`);
  return g;
}

function destroyChart(key) {
  if (charts[key]) { charts[key].destroy(); charts[key] = null; }
}

// ── Health Check ───────────────────────────────────────────────────────────
async function checkHealth() {
  const dot   = document.getElementById('apiStatus');
  const label = document.getElementById('apiLabel');
  const data  = await apiFetch('/api/health');
  if (data && data.status === 'ok') {
    dot.classList.add('online');
    label.textContent = 'API Online';
  } else {
    dot.classList.add('offline');
    label.textContent = 'API Offline';
  }
}

// ── Historical Chart ───────────────────────────────────────────────────────
async function loadHistory(days = 7) {
  const loader = document.getElementById('historyLoader');
  loader.classList.remove('hidden');

  const data = await apiFetch(`/api/history?days=${days}`);
  if (!data) return;

  state.historySeries = data.values;
  updateKPIs(data.values);

  // Downsample for chart (max 200 points)
  const step = Math.max(1, Math.floor(data.values.length / 200));
  const vals = data.values.filter((_, i) => i % step === 0);
  const lbls = data.timestamps.filter((_, i) => i % step === 0)
    .map(t => t.slice(11, 16)); // HH:MM

  loader.classList.add('hidden');
  destroyChart('history');

  const ctx = document.getElementById('historyChart').getContext('2d');
  charts.history = new Chart(ctx, {
    type: 'line',
    data: {
      labels: lbls,
      datasets: [{
        label: 'Energy (kW)',
        data: vals,
        borderColor: '#00e5a0',
        borderWidth: 1.5,
        backgroundColor: accentGradient(ctx),
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 4,
      }],
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#11141a',
          borderColor: '#1f2430',
          borderWidth: 1,
          titleColor: '#00e5a0',
          bodyColor: '#e2e8f0',
          callbacks: { label: ctx => `${ctx.parsed.y.toFixed(3)} kW` },
        },
      },
      scales: {
        x: { grid: { color: '#1f2430' }, ticks: { maxTicksLimit: 10 } },
        y: { grid: { color: '#1f2430' }, title: { display: true, text: 'kW' } },
      },
    },
  });
}

function updateKPIs(series) {
  const mean = (series.reduce((a, b) => a + b, 0) / series.length).toFixed(2);
  const peak = Math.max(...series).toFixed(2);
  setKPI('kpi-mean', `${mean} kW`);
  setKPI('kpi-peak', `${peak} kW`);
}

function setKPI(id, val) {
  const el = document.getElementById(id);
  if (el) el.querySelector('.kpi-val').textContent = val;
}

// ── Prediction ──────────────────────────────────────────────────────────────
async function runPrediction() {
  const btn      = document.getElementById('runPredictBtn');
  const btnText  = document.getElementById('predictBtnText');
  const spinner  = document.getElementById('predictSpinner');
  const loader   = document.getElementById('predictLoader');
  const statsRow = document.getElementById('forecastStats');

  btnText.textContent = 'Forecasting…';
  spinner.classList.remove('hidden');
  btn.disabled = true;
  loader.classList.remove('hidden');
  loader.textContent = 'Running model…';

  const nSteps = parseInt(document.getElementById('nSteps').value);
  const rawInput = document.getElementById('pastUsage').value.trim();
  const pastUsage = rawInput
    ? rawInput.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
    : [];

  const payload = {
    model: state.selectedModel,
    n_steps: nSteps,
    past_usage: pastUsage,
  };

  const data = await apiFetch('/api/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  });

  btnText.textContent = 'Run Forecast';
  spinner.classList.add('hidden');
  btn.disabled = false;

  if (!data) {
    loader.textContent = 'Prediction failed – check API connection.';
    return;
  }

  loader.classList.add('hidden');

  // Render chart
  destroyChart('predict');
  const ctx = document.getElementById('predictChart').getContext('2d');
  const labels = data.timestamps.map(t => t.slice(11, 16));

  const datasets = [
    {
      label: 'Predicted',
      data: data.predicted,
      borderColor: '#00e5a0',
      borderWidth: 2,
      tension: 0.35,
      pointRadius: 0,
      fill: false,
    },
    {
      label: 'Upper CI',
      data: data.confidence_upper,
      borderColor: 'rgba(0,229,160,0.15)',
      borderWidth: 1,
      tension: 0.35,
      pointRadius: 0,
      fill: '+1',
      backgroundColor: 'rgba(0,229,160,0.07)',
    },
    {
      label: 'Lower CI',
      data: data.confidence_lower,
      borderColor: 'rgba(0,229,160,0.15)',
      borderWidth: 1,
      tension: 0.35,
      pointRadius: 0,
      fill: false,
    },
  ];

  charts.predict = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#11141a',
          borderColor: '#1f2430',
          borderWidth: 1,
          callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(3)} kW` },
        },
      },
      scales: {
        x: { grid: { color: '#1f2430' }, ticks: { maxTicksLimit: 8 } },
        y: { grid: { color: '#1f2430' } },
      },
    },
  });

  // Update summary stats
  const pred = data.predicted;
  const avg  = (pred.reduce((a, b) => a + b, 0) / pred.length).toFixed(3);
  const peak = Math.max(...pred).toFixed(3);
  const min  = Math.min(...pred).toFixed(3);
  document.getElementById('fMean').textContent = `${avg} kW`;
  document.getElementById('fPeak').textContent = `${peak} kW`;
  document.getElementById('fMin').textContent  = `${min} kW`;
  statsRow.style.display = 'grid';

  // Load suggestions too
  loadSuggestions(pred);
}

// ── Suggestions ─────────────────────────────────────────────────────────────
async function loadSuggestions(series = null) {
  const grid = document.getElementById('suggestionsGrid');
  grid.innerHTML = `
    <div class="suggestion-skeleton"></div>
    <div class="suggestion-skeleton"></div>
    <div class="suggestion-skeleton"></div>
  `;

  const payload = series ? { series } : {};
  const data = await apiFetch('/api/suggestions', {
    method: 'POST',
    body: JSON.stringify(payload),
  });

  if (!data) { grid.innerHTML = '<p style="color:var(--text-dim)">Failed to load suggestions.</p>'; return; }

  // Update saving KPI
  const maxSaving = Math.max(...data.suggestions.map(s => s.saving_pct || 0));
  setKPI('kpi-saving', maxSaving > 0 ? `~${maxSaving}%` : 'Good!');

  grid.innerHTML = data.suggestions.map(s => `
    <div class="suggestion-card impact-${s.impact}">
      <div class="suggestion-category">${s.category}</div>
      <div class="suggestion-title">${s.title}</div>
      <div class="suggestion-detail">${s.detail}</div>
      <div class="suggestion-footer">
        <span class="impact-pill ${s.impact}">${s.impact} impact</span>
        ${s.saving_pct > 0 ? `<span class="saving-text">Save up to ${s.saving_pct}%</span>` : ''}
      </div>
    </div>
  `).join('');
}

// ── Anomaly Detection ────────────────────────────────────────────────────────
async function runAnomalyDetection() {
  const series = state.historySeries.length > 0
    ? state.historySeries
    : Array.from({ length: 168 }, (_, i) =>
        1.5 + Math.sin(2 * Math.PI * i / 24) * 0.5 + (Math.random() - 0.5) * 0.2
      );

  // Inject a few obvious spikes for demo
  const anomSeries = [...series];
  [10, 40, 90, 130].forEach(i => { if (anomSeries[i] !== undefined) anomSeries[i] *= 2.8; });

  const data = await apiFetch('/api/anomalies', {
    method: 'POST',
    body: JSON.stringify({ series: anomSeries, method: state.anomalyMethod }),
  });

  if (!data) return;

  // Update stat panel
  document.getElementById('anomCount').textContent = data.count;
  const pct = Math.min(data.pct, 100);
  document.getElementById('anomPctBar').style.width = `${pct}%`;
  document.getElementById('anomPctLabel').textContent = `${pct.toFixed(1)}%`;
  const badge = document.getElementById('severityBadge');
  badge.textContent = data.severity;
  badge.className = `severity-badge ${data.severity}`;
  setKPI('kpi-anomaly', data.count);

  // Render anomaly chart
  destroyChart('anomaly');
  const ctx = document.getElementById('anomChart').getContext('2d');
  const step = Math.max(1, Math.floor(anomSeries.length / 200));
  const vals = anomSeries.filter((_, i) => i % step === 0);
  const labels = vals.map((_, i) => `${i * step}h`);

  // Anomaly points
  const anomVals = anomSeries.map((v, i) =>
    data.indices.includes(i) ? v : null
  ).filter((_, i) => i % step === 0);

  charts.anomaly = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Energy (kW)',
          data: vals,
          borderColor: '#4da8ff',
          borderWidth: 1.5,
          fill: false,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Anomaly',
          data: anomVals,
          type: 'scatter',
          backgroundColor: '#ff4d6d',
          borderColor: '#ff4d6d',
          pointRadius: 5,
          pointHoverRadius: 7,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: true,
          labels: { color: '#7a8499', boxWidth: 12 },
        },
        tooltip: {
          backgroundColor: '#11141a',
          borderColor: '#1f2430',
          borderWidth: 1,
        },
      },
      scales: {
        x: { grid: { color: '#1f2430' }, ticks: { maxTicksLimit: 10 } },
        y: { grid: { color: '#1f2430' } },
      },
    },
  });
}

// ── Model Metrics ────────────────────────────────────────────────────────────
async function loadModelMetrics() {
  const grid = document.getElementById('metricsGrid');
  const data = await apiFetch('/api/model-metrics');
  if (!data) return;

  grid.innerHTML = Object.entries(data).map(([name, m]) => `
    <div class="metric-card">
      <div class="metric-model">${name}</div>
      ${Object.entries(m).map(([k, v]) => `
        <div class="metric-row">
          <span class="metric-name">${k.toUpperCase()}</span>
          <span class="metric-val">${typeof v === 'number' ? v.toFixed(4) : v}</span>
        </div>
      `).join('')}
    </div>
  `).join('');

  renderMetricsBarChart(data);
}

function renderMetricsBarChart(data) {
  destroyChart('metrics');
  const ctx = document.getElementById('metricsChart').getContext('2d');
  const models = Object.keys(data);
  const colors = ['#00e5a0', '#4da8ff', '#ffb347'];

  const metricKeys = ['mae', 'rmse'];

  charts.metrics = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: metricKeys.map(k => k.toUpperCase()),
      datasets: models.map((m, i) => ({
        label: m,
        data: metricKeys.map(k => data[m][k]),
        backgroundColor: `${colors[i]}33`,
        borderColor: colors[i],
        borderWidth: 2,
        borderRadius: 4,
      })),
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: true,
          labels: { color: '#7a8499', boxWidth: 14 },
        },
        title: { display: true, text: 'MAE & RMSE Comparison (lower = better)', color: '#7a8499' },
        tooltip: {
          backgroundColor: '#11141a',
          borderColor: '#1f2430',
          borderWidth: 1,
        },
      },
      scales: {
        x: { grid: { color: '#1f2430' } },
        y: { grid: { color: '#1f2430' }, beginAtZero: true },
      },
    },
  });
}

// ── Real-Time Simulation ──────────────────────────────────────────────────────
function initSimChart() {
  const ctx = document.getElementById('simChart').getContext('2d');
  state.simData   = Array(60).fill(null);
  state.simLabels = Array(60).fill('');

  charts.sim = new Chart(ctx, {
    type: 'line',
    data: {
      labels: state.simLabels,
      datasets: [{
        label: 'Live kW',
        data: state.simData,
        borderColor: '#ffb347',
        borderWidth: 2,
        fill: true,
        backgroundColor: (context) => {
          const g = context.chart.ctx.createLinearGradient(0, 0, 0, context.chart.height);
          g.addColorStop(0, 'rgba(255,179,71,0.25)');
          g.addColorStop(1, 'rgba(255,179,71,0)');
          return g;
        },
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 4,
        spanGaps: true,
      }],
    },
    options: {
      animation: false,
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#11141a',
          borderColor: '#1f2430',
          borderWidth: 1,
          callbacks: { label: c => `${c.parsed.y.toFixed(3)} kW` },
        },
      },
      scales: {
        x: { grid: { color: '#1f2430' }, ticks: { maxTicksLimit: 8 } },
        y: { grid: { color: '#1f2430' }, min: 0 },
      },
    },
  });
}

async function simTick() {
  const lastVals = state.simData.filter(v => v !== null).slice(-6);
  const data = await apiFetch('/api/simulate', {
    method: 'POST',
    body: JSON.stringify({ last_values: lastVals }),
  });
  if (!data) return;

  state.simData.shift();
  state.simData.push(data.next_value);
  state.simLabels.shift();
  state.simLabels.push(data.timestamp.slice(11, 19));

  charts.sim.data.datasets[0].data  = state.simData;
  charts.sim.data.labels            = state.simLabels;
  charts.sim.update('none');
}

function toggleSimulation() {
  const btn    = document.getElementById('simStartBtn');
  const status = document.getElementById('simStatus');

  if (state.simRunning) {
    clearInterval(state.simInterval);
    state.simRunning = false;
    btn.textContent = '▶ Start';
    btn.classList.remove('btn-primary');
    btn.classList.add('btn-ghost');
    status.textContent = 'Stopped';
  } else {
    state.simRunning = true;
    btn.textContent = '⏹ Stop';
    btn.classList.remove('btn-ghost');
    btn.classList.add('btn-primary');
    status.textContent = 'Running';
    simTick();
    state.simInterval = setInterval(simTick, 2000);
  }
}

// ── UI Control Wiring ──────────────────────────────────────────────────────────
function wireControls() {
  // Slider
  const slider = document.getElementById('nSteps');
  const sliderVal = document.getElementById('nStepsVal');
  slider.addEventListener('input', () => { sliderVal.textContent = `${slider.value}h`; });

  // Model selector
  document.querySelectorAll('.seg[data-model]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.seg[data-model]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.selectedModel = btn.dataset.model;
    });
  });

  // Timeframe tabs
  document.querySelectorAll('.tab[data-days]').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab[data-days]').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      loadHistory(parseInt(tab.dataset.days));
    });
  });

  // Anomaly method selector
  document.querySelectorAll('.seg[data-anom-method]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.seg[data-anom-method]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.anomalyMethod = btn.dataset.anomMethod;
      runAnomalyDetection();
    });
  });

  // Nav scroll
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const sec = document.getElementById(link.dataset.section);
      if (sec) sec.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

function scrollTo(sectionId) {
  const el = document.getElementById(sectionId);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Initialise ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  wireControls();
  initSimChart();

  // Staggered initialisation
  await checkHealth();
  await loadHistory(7);
  await loadSuggestions();
  await runAnomalyDetection();
  await loadModelMetrics();
});
