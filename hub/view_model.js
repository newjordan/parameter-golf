(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.HubView = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const STATUS_COLORS = {
    ok: "#90efb6",
    warn: "#ffb066",
    error: "#ff7f74",
    unknown: "#9ca8b5",
  };

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function escapeRegex(input) {
    return String(input).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function numericMetric(record, metric) {
    const value = record?.metrics?.[metric];
    return typeof value === "number" ? value : Number.NaN;
  }

  function isByteMetric(name) {
    const key = String(name || "").toLowerCase();
    return key.includes("byte") || key.includes("size") || key === "lowest_file_size";
  }

  function formatNumber(value) {
    if (!Number.isFinite(value)) return "-";
    if (Math.abs(value) >= 100) return value.toFixed(2);
    if (Math.abs(value) >= 10) return value.toFixed(3);
    return value.toFixed(4);
  }

  function formatBytes(value) {
    if (!Number.isFinite(value) || value < 0) return "n/a";
    const units = ["B", "KB", "MB", "GB"];
    let amount = value;
    let index = 0;
    while (amount >= 1000 && index < units.length - 1) {
      amount /= 1000;
      index += 1;
    }
    const digits = index === 0 ? 0 : 2;
    return `${amount.toFixed(digits)} ${units[index]}`;
  }

  function formatConsiderationValue(item) {
    if (!item || !Number.isFinite(item.value)) return "n/a";
    const metricUsed = item.metric_used || item.metric_key || "";
    if (isByteMetric(metricUsed) || item.category === "lowest_file_size") {
      return formatBytes(item.value);
    }
    return formatNumber(item.value);
  }

  function formatMetricValue(key, value) {
    if (isByteMetric(key)) return formatBytes(value);
    return formatNumber(value);
  }

  function compareRecords(a, b, metric, direction) {
    const factor = direction === "desc" ? -1 : 1;
    if (metric) {
      const av = numericMetric(a, metric);
      const bv = numericMetric(b, metric);
      const aOk = Number.isFinite(av);
      const bOk = Number.isFinite(bv);
      if (aOk !== bOk) return aOk ? -1 : 1;
      if (aOk && av !== bv) return (av - bv) * factor;
    }
    const at = a.timestamp_hint || "";
    const bt = b.timestamp_hint || "";
    if (at !== bt) return bt.localeCompare(at);
    return String(a.rel_path || a.path || "").localeCompare(String(b.rel_path || b.path || ""));
  }

  function recordHaystack(record) {
    return [
      record.path,
      record.rel_path,
      record.category,
      record.track,
      record.experiment_group,
      record.run_tag,
      record.author,
      record.timestamp_hint,
      record.snippet,
      ...(record.notes || []),
      ...(record.keywords || []),
      ...Object.entries(record.metrics || {}).map(([key, value]) => `${key} ${value}`),
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
  }

  function filterRecords(records, filters) {
    const query = String(filters.query || "").trim().toLowerCase();
    const category = filters.category || "all";
    const status = filters.status || "all";
    return records.filter((record) => {
      if (category !== "all" && record.category !== category) return false;
      if (status !== "all" && record.status !== status) return false;
      if (!query) return true;
      return recordHaystack(record).includes(query);
    });
  }

  function sortRecords(records, metric, direction) {
    return [...records].sort((a, b) => compareRecords(a, b, metric, direction));
  }

  function bestInView(records, metric) {
    let best = null;
    records.forEach((record) => {
      const value = numericMetric(record, metric);
      if (!Number.isFinite(value)) return;
      if (!best || value < best.value) best = { record, value };
    });
    return best;
  }

  function statusColor(name) {
    return STATUS_COLORS[name] || STATUS_COLORS.unknown;
  }

  return {
    ARTIFACT_CAP_BYTES: 16000000,
    STATUS_COLORS,
    escapeHtml,
    escapeRegex,
    numericMetric,
    isByteMetric,
    formatNumber,
    formatBytes,
    formatConsiderationValue,
    formatMetricValue,
    compareRecords,
    filterRecords,
    sortRecords,
    bestInView,
    statusColor,
  };
});
