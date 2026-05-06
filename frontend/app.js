/* ═══════════════════════════════════════════════════════
   Data Forge — Frontend Logic
   Handles file upload, task config, training, and download
   ═══════════════════════════════════════════════════════ */

const API = "http://127.0.0.1:5001";

// ── State ──────────────────────────────────────────
let sessionId = null;
let columns = [];
let columnTypes = {};
let modelId = null;

// ── DOM refs ───────────────────────────────────────
const dropZone        = document.getElementById("drop-zone");
const fileInput       = document.getElementById("file-input");
const fileInfo        = document.getElementById("file-info");
const fileName        = document.getElementById("file-name");
const fileShape       = document.getElementById("file-shape");
const previewSection  = document.getElementById("preview-section");
const previewTable    = document.getElementById("preview-table");
const configSection   = document.getElementById("config-section");
const targetGroup     = document.getElementById("target-group");
const targetSelect    = document.getElementById("target-select");
const trainBtn        = document.getElementById("train-btn");
const trainSpinner    = document.getElementById("train-spinner");
const resultsSection  = document.getElementById("results-section");
const resultsMeta     = document.getElementById("results-meta");
const resultsBody     = document.getElementById("results-body");
const downloadBtn     = document.getElementById("download-btn");

// ── File Upload ────────────────────────────────────

// Drag & drop
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drop-zone--active"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drop-zone--active"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drop-zone--active");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

async function handleFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["csv", "xlsx"].includes(ext)) {
    alert("Please upload a .csv or .xlsx file.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  // Reset state
  resetResults();

  try {
    const res = await fetch(`${API}/api/upload`, { method: "POST", body: formData });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || "Upload failed"); }
    const data = await res.json();

    sessionId = data.session_id;
    columns = data.columns;
    columnTypes = data.column_types || {};

    // Show file info
    fileName.textContent = file.name;
    fileShape.textContent = `${data.shape[0]} rows × ${data.shape[1]} columns`;
    fileInfo.classList.remove("hidden");

    // Render preview table
    renderPreview(data.columns, data.preview);
    previewSection.classList.remove("hidden");

    // Show config section
    populateTargetSelect(data.columns);
    configSection.classList.remove("hidden");

    // Scroll smoothly
    previewSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    console.error("Upload Error Details:", err);
    alert("Upload error: " + err.message);
  }
}

// ── Preview Table ──────────────────────────────────

function renderPreview(cols, rows) {
  let html = "<thead><tr>";
  cols.forEach(c => html += `<th>${escapeHtml(c)}</th>`);
  html += "</tr></thead><tbody>";
  rows.forEach(row => {
    html += "<tr>";
    cols.forEach(c => html += `<td>${escapeHtml(String(row[c] ?? ""))}</td>`);
    html += "</tr>";
  });
  html += "</tbody>";
  previewTable.innerHTML = html;
}

// ── Task Selection ─────────────────────────────────

document.querySelectorAll('input[name="task"]').forEach(radio => {
  radio.addEventListener("change", () => {
    const task = radio.value;
    if (task === "clustering") {
      targetGroup.classList.add("hidden");
    } else {
      targetGroup.classList.remove("hidden");
    }
    trainBtn.disabled = false;
    updateTargetSelectValidation();
  });
});

function updateTargetSelectValidation() {
  if (Object.keys(columnTypes).length === 0) return;
  const taskRadio = document.querySelector('input[name="task"]:checked');
  if (!taskRadio) return;
  const task = taskRadio.value;
  
  Array.from(targetSelect.options).forEach(opt => {
    const cType = columnTypes[opt.value];
    if (task === "classification" && cType === "continuous") {
      opt.disabled = true;
      opt.textContent = `${opt.value} (Invalid - Continuous)`;
    } else if (task === "regression" && cType === "categorical") {
      opt.disabled = true;
      opt.textContent = `${opt.value} (Invalid - Categorical)`;
    } else {
      opt.disabled = false;
      opt.textContent = opt.value;
    }
  });

  if (targetSelect.selectedOptions.length > 0 && targetSelect.selectedOptions[0].disabled) {
    const firstEnabled = Array.from(targetSelect.options).find(o => !o.disabled);
    if (firstEnabled) targetSelect.value = firstEnabled.value;
  }
}

function populateTargetSelect(cols) {
  targetSelect.innerHTML = cols.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
  updateTargetSelectValidation();
}

// ── Training ───────────────────────────────────────

trainBtn.addEventListener("click", startTraining);

async function startTraining() {
  const taskRadio = document.querySelector('input[name="task"]:checked');
  if (!taskRadio) { alert("Please select an ML task."); return; }

  const task = taskRadio.value;
  const target = task === "clustering" ? null : targetSelect.value;

  trainBtn.disabled = true;
  trainBtn.querySelector(".btn__text").textContent = "Training…";
  trainSpinner.classList.remove("hidden");
  document.getElementById("train-error").classList.add("hidden");
  resetResults();

  try {
    const res = await fetch(`${API}/api/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, task, target_column: target }),
    });

    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || "Training failed"); }
    const data = await res.json();

    modelId = data.model_id;
    renderResults(data);

    resultsSection.classList.remove("hidden");
    downloadBtn.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    console.error("Fetch Error Details:", err);
    document.getElementById("train-error-text").textContent = err.message;
    document.getElementById("train-error").classList.remove("hidden");
  } finally {
    trainBtn.disabled = false;
    trainBtn.querySelector(".btn__text").textContent = "Train Model";
    trainSpinner.classList.add("hidden");
  }
}

// ── Render Results ─────────────────────────────────

function renderResults(data) {
  // Meta badges
  let metaHtml = `<span class="results-meta__badge results-meta__badge--task">${data.task}</span>`;
  metaHtml += `<span class="results-meta__badge results-meta__badge--algo">Best: ${data.best_algorithm}</span>`;
  if (data.used_smote) metaHtml += `<span class="results-meta__badge results-meta__badge--smote">SMOTE applied</span>`;
  resultsMeta.innerHTML = metaHtml;

  // Results per algorithm
  let bodyHtml = "";
  const plotTasks = []; // Array of functions to render Plotly charts after DOM update

  for (const [algoName, metrics] of Object.entries(data.results)) {
    const isBest = algoName === data.best_algorithm;
    const safeId = algoName.replace(/[^a-zA-Z0-9]/g, '');
    
    bodyHtml += `<div class="algo-block">`;
    bodyHtml += `<div class="algo-block__title">${escapeHtml(algoName)} ${isBest ? '<span class="best-tag">★ BEST</span>' : ''}</div>`;
    bodyHtml += `<div class="metrics-grid">`;

    if (data.task === "classification") {
      bodyHtml += metricCard(metrics.accuracy, "Accuracy");
      bodyHtml += metricCard(metrics.precision, "Precision");
      bodyHtml += metricCard(metrics.recall, "Recall");
      bodyHtml += metricCard(metrics.f1_score, "F1 Score");
    } else if (data.task === "regression") {
      bodyHtml += metricCard(metrics.mae, "MAE");
      bodyHtml += metricCard(metrics.mse, "MSE");
      bodyHtml += metricCard(metrics.r2_score, "R² Score");
    } else {
      bodyHtml += metricCard(metrics.silhouette_score, "Silhouette");
      bodyHtml += metricCard(metrics.n_clusters, "Clusters");
    }
    bodyHtml += `</div>`; // .metrics-grid

    // Chart Containers
    const chartsHtml = [];
    
    if (data.task === "classification" && metrics.confusion_matrix) {
      const cmId = `chart-${safeId}-cm`;
      chartsHtml.push(`<div id="${cmId}" class="chart-container"></div>`);
      plotTasks.push(() => renderConfusionMatrixPlot(cmId, metrics.confusion_matrix, data.label_names));
    }
    
    if (data.task === "clustering" && metrics.pca_x) {
      const pcaId = `chart-${safeId}-pca`;
      chartsHtml.push(`<div id="${pcaId}" class="chart-container"></div>`);
      plotTasks.push(() => renderPcaPlot(pcaId, metrics.pca_x, metrics.pca_y, metrics.pca_labels));
      
      const pieId = `chart-${safeId}-pie`;
      chartsHtml.push(`<div id="${pieId}" class="chart-container"></div>`);
      plotTasks.push(() => renderClusterPie(pieId, metrics.cluster_distribution));
    }

    if (data.task === "regression" && metrics.actual) {
      const scatterId = `chart-${safeId}-scatter`;
      chartsHtml.push(`<div id="${scatterId}" class="chart-container"></div>`);
      plotTasks.push(() => renderActualVsPred(scatterId, metrics.actual, metrics.predicted));
    }

    if (metrics.feature_importances && metrics.feature_importances.length > 0) {
      const fiId = `chart-${safeId}-fi`;
      chartsHtml.push(`<div id="${fiId}" class="chart-container"></div>`);
      plotTasks.push(() => renderFeatureImportance(fiId, metrics.feature_names, metrics.feature_importances));
    }
    
    if (chartsHtml.length > 0) {
      bodyHtml += `<div class="charts-grid">${chartsHtml.join("")}</div>`;
    }

    bodyHtml += `</div>`; // .algo-block
  }
  resultsBody.innerHTML = bodyHtml;
  
  // Render plots now that DOM has the containers
  plotTasks.forEach(task => task());
}

function metricCard(value, label) {
  return `<div class="metric-card"><div class="metric-card__value">${value}</div><div class="metric-card__label">${label}</div></div>`;
}

// ── Plotly Renderers ───────────────────────────────
const layoutBase = {
  autosize: true,
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#e2e8f0', family: 'Inter' },
  margin: { t: 80, l: 80, r: 80, b: 40 }
};

function renderConfusionMatrixPlot(id, cm, labelNames) {
  const z = cm;
  const yLabels = labelNames ? Object.values(labelNames).reverse() : cm.map((_, i) => `Class ${i}`).reverse();
  const xLabels = labelNames ? Object.values(labelNames) : cm.map((_, i) => `Class ${i}`);
  
  const data = [{
    z: z.slice().reverse(), 
    x: xLabels,
    y: yLabels,
    type: 'heatmap',
    colorscale: 'Blues',
    showscale: false
  }];
  
  const layout = {
    ...layoutBase,
    title: { text: 'Confusion Matrix', font: { size: 14 } },
    xaxis: { title: 'Predicted' },
    yaxis: { title: 'Actual' }
  };
  Plotly.newPlot(id, data, layout, { displayModeBar: false, responsive: true });
}

function renderFeatureImportance(id, names, importances) {
  const sorted = names.map((n, i) => ({ n, v: importances[i] })).sort((a, b) => a.v - b.v);
  const top10 = sorted.slice(-10);

  const data = [{
    type: 'bar',
    x: top10.map(item => item.v),
    y: top10.map(item => item.n),
    orientation: 'h',
    marker: { color: '#6c63ff' }
  }];
  
  const layout = {
    ...layoutBase,
    title: { text: 'Feature Importance (Top 10)', font: { size: 14 } },
    margin: { t: 80, l: 180, r: 80, b: 40 }
  };
  Plotly.newPlot(id, data, layout, { displayModeBar: false, responsive: true });
}

function renderActualVsPred(id, actual, predicted) {
  const minVal = Math.min(...actual, ...predicted);
  const maxVal = Math.max(...actual, ...predicted);
  
  const data = [
    {
      x: actual, y: predicted,
      mode: 'markers',
      type: 'scatter',
      name: 'Predictions',
      marker: { color: '#00d4aa', opacity: 0.6 }
    },
    {
      x: [minVal, maxVal], y: [minVal, maxVal],
      mode: 'lines',
      type: 'scatter',
      name: 'Ideal',
      line: { color: '#f56565', dash: 'dash' }
    }
  ];
  
  const layout = {
    ...layoutBase,
    title: { text: 'Actual vs Predicted', font: { size: 14 } },
    xaxis: { title: 'Actual' },
    yaxis: { title: 'Predicted' },
    showlegend: false
  };
  Plotly.newPlot(id, data, layout, { displayModeBar: false, responsive: true });
}

function renderPcaPlot(id, x, y, labels) {
  const data = [{
    x: x, y: y,
    mode: 'markers',
    type: 'scatter',
    marker: {
      color: labels,
      colorscale: 'Viridis',
      size: 8,
      opacity: 0.8
    }
  }];
  
  const layout = {
    ...layoutBase,
    title: { text: 'PCA Cluster Map', font: { size: 14 } },
    xaxis: { title: 'Component 1', zeroline: false },
    yaxis: { title: 'Component 2', zeroline: false }
  };
  Plotly.newPlot(id, data, layout, { displayModeBar: false, responsive: true });
}

function renderClusterPie(id, dist) {
  const labels = Object.keys(dist).map(k => `Cluster ${k}`);
  const values = Object.values(dist);
  
  const data = [{
    values: values,
    labels: labels,
    type: 'pie',
    hole: .4,
    marker: { colors: ['#6c63ff', '#00d4aa', '#f56565', '#fbbf24', '#3b82f6'] }
  }];
  
  const layout = {
    ...layoutBase,
    title: { text: 'Cluster Distribution', font: { size: 14 }, y: 0.95 },
    legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.2 },
    margin: { t: 60, l: 40, r: 40, b: 80 }
  };
  Plotly.newPlot(id, data, layout, { displayModeBar: false, responsive: true });
}

// ── Download ───────────────────────────────────────

downloadBtn.addEventListener("click", () => {
  if (!modelId) return;
  window.location.href = `${API}/api/download/${modelId}`;
});

// ── Helpers ────────────────────────────────────────

function resetResults() {
  resultsMeta.innerHTML = "";
  resultsBody.innerHTML = "";
  resultsSection.classList.add("hidden");
  downloadBtn.classList.add("hidden");
  modelId = null;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
