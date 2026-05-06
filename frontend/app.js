/* ═══════════════════════════════════════════════════════
   AutoML — Frontend Logic
   Handles file upload, task config, training, and download
   ═══════════════════════════════════════════════════════ */

const API = "http://localhost:5001";

// ── State ──────────────────────────────────────────
let sessionId = null;
let columns = [];
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
  });
});

function populateTargetSelect(cols) {
  targetSelect.innerHTML = cols.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
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
    alert("Training error: " + err.message);
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
  for (const [algoName, metrics] of Object.entries(data.results)) {
    const isBest = algoName === data.best_algorithm;
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

    // Confusion matrix
    if (data.task === "classification" && metrics.confusion_matrix) {
      bodyHtml += renderConfusionMatrix(metrics.confusion_matrix, data.label_names);
    }

    // Cluster distribution
    if (data.task === "clustering" && metrics.cluster_distribution) {
      bodyHtml += renderClusterDist(metrics.cluster_distribution);
    }

    bodyHtml += `</div>`; // .algo-block
  }
  resultsBody.innerHTML = bodyHtml;
}

function metricCard(value, label) {
  return `<div class="metric-card"><div class="metric-card__value">${value}</div><div class="metric-card__label">${label}</div></div>`;
}

function renderConfusionMatrix(cm, labelNames) {
  let html = `<div class="cm-wrap"><h4>Confusion Matrix</h4><table class="cm-table">`;
  cm.forEach((row, i) => {
    html += "<tr>";
    row.forEach(val => html += `<td>${val}</td>`);
    html += "</tr>";
  });
  html += "</table></div>";
  return html;
}

function renderClusterDist(dist) {
  let html = `<div class="cm-wrap"><h4>Cluster Distribution</h4><table class="cm-table"><tr>`;
  for (const k of Object.keys(dist)) html += `<th>Cluster ${k}</th>`;
  html += "</tr><tr>";
  for (const v of Object.values(dist)) html += `<td>${v}</td>`;
  html += "</tr></table></div>";
  return html;
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
