const test = require("node:test");
const assert = require("node:assert/strict");
const hub = require("../hub/view_model.js");

test("escapeRegex quotes every metacharacter", () => {
  const literal = "shard.size+mismatch?";
  const expression = new RegExp(hub.escapeRegex(literal));
  assert.equal("shard.size+mismatch?".match(expression)[0], literal);
  assert.equal("shardXsizeXmismatchX".match(expression), null);
});

test("escapeHtml covers the characters used in record snippets", () => {
  assert.equal(hub.escapeHtml(`<img alt="x" onload='1'>&`), "&lt;img alt=&quot;x&quot; onload=&#39;1&#39;&gt;&amp;");
});

test("decimal byte formatting matches the 16000000 byte cap", () => {
  assert.equal(hub.formatBytes(16_000_000), "16.00 MB");
  assert.equal(hub.formatBytes(15990006), "15.99 MB");
  assert.equal(hub.formatBytes(512), "512 B");
  assert.equal(hub.ARTIFACT_CAP_BYTES, 16_000_000);
});

test("missing metrics stay last when the sort direction flips", () => {
  const rows = [
    { id: "missing", metrics: {}, timestamp_hint: "2026-03-02", rel_path: "b" },
    { id: "high", metrics: { val_bpb: 1.4 }, timestamp_hint: "2026-03-01", rel_path: "a" },
    { id: "low", metrics: { val_bpb: 1.1 }, timestamp_hint: "2026-03-03", rel_path: "c" },
  ];
  assert.deepEqual(
    hub.sortRecords(rows, "val_bpb", "asc").map((row) => row.id),
    ["low", "high", "missing"],
  );
  assert.deepEqual(
    hub.sortRecords(rows, "val_bpb", "desc").map((row) => row.id),
    ["high", "low", "missing"],
  );
});

test("filters search author and status, and the best value is the lowest", () => {
  const rows = [
    { category: "submission", status: "ok", author: "ada", run_tag: "Ada", metrics: { val_bpb: 1.2, model_size_bytes: 2000 } },
    { category: "run_log", status: "error", author: "grace", run_tag: "Grace", metrics: { val_bpb: 1.0 } },
  ];
  assert.equal(hub.filterRecords(rows, { query: "ada", category: "all", status: "all" }).length, 1);
  assert.equal(hub.filterRecords(rows, { query: "", category: "all", status: "error" })[0].run_tag, "Grace");
  assert.equal(hub.bestInView(rows, "val_bpb").value, 1.0);
  assert.equal(hub.formatConsiderationValue({ value: 16_000_000, metric_used: "model_size_bytes" }), "16.00 MB");
  assert.equal(hub.formatMetricValue("model_size_bytes", 16_000_000), "16.00 MB");
  assert.equal(hub.statusColor("ok"), "#90efb6");
  assert.equal(hub.statusColor("error"), "#ff7f74");
});
