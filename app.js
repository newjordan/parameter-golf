const state = {
  data: null,
  records: [],
  filtered: [],
  selectedId: null,
};

const els = {};

const metricLabels = {
  val_bpb: "Validation BPB",
  cap_val_bpb: "Cap Validation BPB",
  diag_bpb: "Diagnostic BPB",
  sliding_bpb: "Sliding BPB",
  ngram9_bpb: "N-gram 9 BPB",
  delta: "Delta",
};

const metricOrder = ["val_bpb", "cap_val_bpb", "diag_bpb", "sliding_bpb", "ngram9_bpb", "delta"];

document.addEventListener("DOMContentLoaded", init);

async function init() {
  bindElements();
  bindEvents();
  await loadData();
}

function bindElements() {
  [
    "scopeLabel",
    "generatedAt",
    "statTotal",
    "statOk",
    "statWarn",
    "statError",
    "bestMetricValue",
    "bestMetricLabel",
    "searchInput",
    "categoryFilter",
    "statusFilter",
    "metricSelect",
    "sortDir",
    "resetBtn",
    "visibleCount",
    "visibleExperiments",
    "visibleErrors",
    "activeQuery",
    "recordsBody",
    "resultCount",
    "detailTitle",
    "detailStatusBadge",
    "detailPathCode",
    "detailMeta",
    "detailSnippet",
    "detailMetrics",
    "detailNotes",
    "copyPathBtn",
  ].forEach((id) => {
    els[id] = document.getElementById(id);
  });
}

function bindEvents() {
  els.searchInput.addEventListener("input", applyFilters);
  els.categoryFilter.addEventListener("change", applyFilters);
  els.statusFilter.addEventListener("change", applyFilters);
  els.metricSelect.addEventListener("change", applyFilters);
  els.sortDir.addEventListener("change", applyFilters);
  els.resetBtn.addEventListener("click", resetFilters);
  els.copyPathBtn.addEventListener("click", copySelectedPath);
}

async function loadData() {
  try {
    const response = await fetch("./hub_index.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load index: ${response.status}`);
    }
    state.data = await response.json();
    state.records = Array.isArray(state.data.records) ? state.data.records : [];
    hydrateControls();
    renderHeader();
    applyFilters();
  } catch (error) {
    renderLoadFailure(error);
  }
}

function renderLoadFailure(error) {
  els.resultCount.textContent = "Unable to load hub index.";
  els.recordsBody.innerHTML = `<tr><td colspan="5" class="empty-cell">Error: ${escapeHtml(error.message)}</td></tr>`;
  els.detailTitle.textContent = "Load failed";
  els.detailStatusBadge.textContent = "error";
  els.detailStatusBadge.className = "status-pill status-error";
  els.detailPathCode.textContent = "No path available.";
  els.detailSnippet.textContent = String(error.stack || error.message);
}

function hydrateControls() {
  const categories = new Set(["all"]);
  const metrics = new Set(metricOrder);

  state.records.forEach((record) => {
    if (record.category) categories.add(record.category);
    Object.keys(record.metrics || {}).forEach((key) => metrics.add(key));
  });

  fillSelect(els.categoryFilter, [...categories], "All categories");
  fillMetricSelect(els.metricSelect, [...metrics]);
  els.sortDir.value = "asc";
}

function fillSelect(select, values, allLabel) {
  const existing = select.value;
  select.innerHTML = "";
  if (allLabel) {
    const option = document.createElement("option");
    option.value = "all";
    option.textContent = allLabel;
    select.appendChild(option);
  }

  values
    .filter((value) => value !== "all")
    .sort((a, b) => a.localeCompare(b))
    .forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });

  if (existing && [...select.options].some((option) => option.value === existing)) {
    select.value = existing;
  }
}

function fillMetricSelect(select, values) {
  const existing = select.value;
  select.innerHTML = "";

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Sort by path / recency";
  select.appendChild(placeholder);

  [...values]
    .filter((value) => value && value !== "all")
    .sort((a, b) => {
      const ai = metricOrder.indexOf(a);
      const bi = metricOrder.indexOf(b);
      if (ai === -1 && bi === -1) return a.localeCompare(b);
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi;
    })
    .forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = metricLabels[value] || value;
      select.appendChild(option);
    });

  if (existing && [...select.options].some((option) => option.value === existing)) {
    select.value = existing;
    return;
  }

  select.value = metricOrder.find((key) => values.includes(key)) || "";
}

function renderHeader() {
  const counts = state.data?.counts || {};
  els.scopeLabel.textContent = (state.data?.source_roots || []).join(" | ") || "experiments";
  els.generatedAt.textContent = formatTimestamp(state.data?.generated_at);
  els.statTotal.textContent = counts.total_records ?? state.records.length;
  els.statOk.textContent = counts.by_status?.ok ?? 0;
  els.statWarn.textContent = counts.by_status?.warn ?? 0;
  els.statError.textContent = counts.by_status?.error ?? 0;
}

function applyFilters() {
  const query = els.searchInput.value.trim().toLowerCase();
  const category = els.categoryFilter.value;
  const status = els.statusFilter.value;
  const metric = els.metricSelect.value;
  const direction = els.sortDir.value;

  const filtered = state.records.filter((record) => {
    if (category !== "all" && record.category !== category) return false;
    if (status !== "all" && record.status !== status) return false;
    if (!query) return true;

    const haystack = [
      record.path,
      record.rel_path,
      record.category,
      record.experiment_group,
      record.run_tag,
      record.timestamp_hint,
      record.snippet,
      ...(record.notes || []),
      ...Object.entries(record.metrics || {}).map(([key, value]) => `${key} ${value}`),
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();

    return haystack.includes(query);
  });

  const sorted = [...filtered].sort((a, b) => compareRecords(a, b, metric, direction));
  state.filtered = sorted;

  updateVisibleStats(sorted, { query, category, status, metric, direction });
  els.resultCount.textContent = `${sorted.length} of ${state.records.length} records`;
  renderTable(sorted, metric);

  if (!sorted.length) {
    setDetail(null);
    updateBestMetricCard(metric, []);
    return;
  }

  const preferred = state.selectedId ? sorted.find((record) => record.id === state.selectedId) : null;
  setDetail(preferred || sorted[0]);
  updateBestMetricCard(metric, sorted);
}

function updateVisibleStats(records, filters) {
  const experiments = new Set(records.map((record) => record.experiment_group).filter(Boolean));
  const visibleErrors = records.filter((record) => record.status === "error").length;
  els.visibleCount.textContent = records.length;
  els.visibleExperiments.textContent = experiments.size;
  els.visibleErrors.textContent = visibleErrors;

  const activeBits = [];
  if (filters.query) activeBits.push(`query: ${filters.query}`);
  if (filters.category !== "all") activeBits.push(`category: ${filters.category}`);
  if (filters.status !== "all") activeBits.push(`status: ${filters.status}`);
  if (filters.metric) activeBits.push(`metric: ${filters.metric}`);
  activeBits.push(`order: ${filters.direction}`);
  els.activeQuery.textContent = activeBits.length ? activeBits.join(" | ") : "No active filter";
}

function compareRecords(a, b, metric, direction) {
  const factor = direction === "asc" ? 1 : -1;
  if (metric) {
    const av = numericMetric(a, metric);
    const bv = numericMetric(b, metric);
    if (av !== bv) {
      if (Number.isFinite(av) && Number.isFinite(bv)) return (av - bv) * factor;
      if (Number.isFinite(av)) return -1 * factor;
      if (Number.isFinite(bv)) return 1 * factor;
    }
  }

  const at = a.timestamp_hint || "";
  const bt = b.timestamp_hint || "";
  if (at !== bt) return at.localeCompare(bt) * -1;
  return (a.path || "").localeCompare(b.path || "");
}

function renderTable(records, metric) {
  els.recordsBody.innerHTML = "";
  if (!records.length) {
    els.recordsBody.innerHTML = '<tr><td colspan="5" class="empty-cell">No records match the current filter deck.</td></tr>';
    return;
  }

  const fragment = document.createDocumentFragment();
  records.forEach((record) => {
    const row = document.createElement("tr");
    row.dataset.id = record.id;
    row.className = `record-row status-${record.status || "unknown"}`;
    row.innerHTML = `
      <td><span class="status-pill status-${record.status || "unknown"}">${escapeHtml(record.status || "unknown")}</span></td>
      <td>
        <div class="record-main">
          <span class="record-title">${escapeHtml(record.run_tag || record.category || "record")}</span>
          <span class="record-sub">${escapeHtml(record.category || "unknown")}</span>
        </div>
      </td>
      <td>
        <div class="record-main">
          <span class="record-title">${escapeHtml(record.experiment_group || "unknown")}</span>
          <span class="record-sub">${escapeHtml(record.timestamp_hint || "no timestamp hint")}</span>
        </div>
      </td>
      <td>${formatMetricCell(record, metric)}</td>
      <td class="path-cell">
        <div class="record-main">
          <span class="record-path">${escapeHtml(record.rel_path || record.path || "")}</span>
        </div>
      </td>
    `;
    row.addEventListener("click", () => setDetail(record));
    fragment.appendChild(row);
  });
  els.recordsBody.appendChild(fragment);
  highlightSelected();
}

function formatMetricCell(record, metric) {
  if (metric && Number.isFinite(numericMetric(record, metric))) {
    return `<span class="metric-pill">${escapeHtml(metricLabels[metric] || metric)} ${escapeHtml(formatNumber(numericMetric(record, metric)))}</span>`;
  }

  const pairs = metricOrder
    .filter((key) => Number.isFinite(numericMetric(record, key)))
    .slice(0, 2)
    .map((key) => `${metricLabels[key] || key}: ${formatNumber(numericMetric(record, key))}`);

  return pairs.length ? escapeHtml(pairs.join(" | ")) : "-";
}

function setDetail(record) {
  state.selectedId = record ? record.id : null;
  highlightSelected();

  if (!record) {
    els.detailTitle.textContent = "No record selected";
    els.detailStatusBadge.textContent = "unknown";
    els.detailStatusBadge.className = "status-pill status-unknown";
    els.detailPathCode.textContent = "No path selected.";
    els.detailMeta.innerHTML = "";
    els.detailSnippet.textContent = "No record selected.";
    els.detailMetrics.innerHTML = '<div class="metric-item"><span>Metrics</span><strong>None extracted</strong></div>';
    els.detailNotes.innerHTML = '<li class="note-empty">No notes extracted.</li>';
    els.copyPathBtn.disabled = true;
    els.copyPathBtn.dataset.path = "";
    return;
  }

  els.detailTitle.textContent = record.run_tag || record.rel_path || record.path || record.id;
  els.detailStatusBadge.textContent = record.status || "unknown";
  els.detailStatusBadge.className = `status-pill status-${record.status || "unknown"}`;
  els.detailPathCode.textContent = record.rel_path || record.path || "";
  els.copyPathBtn.disabled = false;
  els.copyPathBtn.dataset.path = record.path || record.rel_path || "";

  const metaRows = [
    ["Category", record.category],
    ["Experiment group", record.experiment_group],
    ["Run tag", record.run_tag],
    ["Timestamp hint", record.timestamp_hint],
    ["Status", record.status],
    ["Absolute path", record.path],
  ];

  els.detailMeta.innerHTML = metaRows
    .map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value || "-")}</dd></div>`)
    .join("");

  els.detailSnippet.textContent = record.snippet || "No snippet extracted.";
  els.detailMetrics.innerHTML = renderMetricGrid(record.metrics || {});
  els.detailNotes.innerHTML = (record.notes || []).length
    ? record.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")
    : '<li class="note-empty">No notes extracted.</li>';
}

function renderMetricGrid(metrics) {
  const entries = Object.entries(metrics);
  if (!entries.length) {
    return '<div class="metric-item"><span>Metrics</span><strong>None extracted</strong></div>';
  }
  return entries
    .sort((a, b) => sortMetricKeys(a[0], b[0]))
    .map(([key, value]) => `<div class="metric-item"><span>${escapeHtml(metricLabels[key] || key)}</span><strong>${escapeHtml(formatNumber(value))}</strong></div>`)
    .join("");
}

function sortMetricKeys(a, b) {
  const ai = metricOrder.indexOf(a);
  const bi = metricOrder.indexOf(b);
  if (ai === -1 && bi === -1) return a.localeCompare(b);
  if (ai === -1) return 1;
  if (bi === -1) return -1;
  return ai - bi;
}

function updateBestMetricCard(metric, records) {
  if (!metric) {
    els.bestMetricValue.textContent = "-";
    els.bestMetricLabel.textContent = "Choose a metric to identify the strongest record in view.";
    return;
  }

  const ranked = records
    .map((record) => ({ record, value: numericMetric(record, metric) }))
    .filter((item) => Number.isFinite(item.value))
    .sort((a, b) => a.value - b.value);

  if (!ranked.length) {
    els.bestMetricValue.textContent = "-";
    els.bestMetricLabel.textContent = `No ${metric} values in the current slice.`;
    return;
  }

  const best = ranked[0];
  els.bestMetricValue.textContent = formatNumber(best.value);
  els.bestMetricLabel.textContent = `${metricLabels[metric] || metric} lead: ${best.record.run_tag || best.record.rel_path || best.record.path}`;
}

function highlightSelected() {
  document.querySelectorAll(".record-row").forEach((row) => {
    row.classList.toggle("selected", row.dataset.id === state.selectedId);
  });
}

function resetFilters() {
  els.searchInput.value = "";
  els.categoryFilter.value = "all";
  els.statusFilter.value = "all";
  els.sortDir.value = "asc";
  els.metricSelect.value = metricOrder.find((key) => els.metricSelect.querySelector(`option[value="${key}"]`))?.value || "";
  applyFilters();
}

async function copySelectedPath() {
  const path = els.copyPathBtn.dataset.path || "";
  if (!path) return;
  try {
    await navigator.clipboard.writeText(path);
    const original = els.copyPathBtn.textContent;
    els.copyPathBtn.textContent = "Copied";
    setTimeout(() => {
      els.copyPathBtn.textContent = original;
    }, 1200);
  } catch (error) {
    console.error(error);
  }
}

function numericMetric(record, metric) {
  const value = record?.metrics?.[metric];
  return typeof value === "number" ? value : Number.NaN;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 100) return value.toFixed(2);
  if (Math.abs(value) >= 10) return value.toFixed(3);
  return value.toFixed(4);
}

function formatTimestamp(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
