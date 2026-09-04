"use strict";

const METRIC_ORDER = ["psnr", "ssim", "fsim", "rmse", "displacement_error", "csi"];
const HIGHER_IS_BETTER = new Set(["psnr", "ssim", "fsim", "csi", "pod"]);
const METHOD_COLOURS = { model: "#4fc3f7", linear: "#9aa4b2", farneback: "#ffb74d" };

const state = {
  runId: null,
  manifest: null,
  observed: [],   // manifest indices of observed frames
  cursor: 0,
  timer: null,
  charts: {},
};

const $ = (id) => document.getElementById(id);

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) return null;
  return response.json();
}

function frameUrl(runId, index) {
  return `/api/runs/${encodeURIComponent(runId)}/frames/${index}.png`;
}

function formatTime(iso) {
  return iso ? iso.replace("T", " ").replace("+00:00", "Z").slice(0, 19) : "—";
}

function renderSummary(manifest) {
  const stats = [
    ["Sensor", manifest.sensor],
    ["Input cadence", `${manifest.input_cadence_minutes} min`],
    ["Output cadence", `${manifest.output_cadence_minutes} min`],
    ["Factor", `${manifest.factor}x`],
    ["Observed frames", manifest.n_original],
    ["Synthesized frames", manifest.n_synthetic],
  ];
  $("summary").innerHTML = stats
    .map(([label, value]) =>
      `<div class="stat"><div class="value">${value}</div><div class="label">${label}</div></div>`)
    .join("");
}

function showFrame(cursor) {
  const frames = state.manifest.frames;
  state.cursor = Math.max(0, Math.min(cursor, frames.length - 1));
  const frame = frames[state.cursor];

  $("player-interpolated").src = frameUrl(state.runId, state.cursor);
  $("synthetic-badge").hidden = !frame.synthetic;

  // The original panel holds the most recent real observation at or before this time.
  let latest = state.observed.length ? state.observed[0] : 0;
  for (const index of state.observed) {
    if (index <= state.cursor) latest = index;
  }
  $("player-original").src = frameUrl(state.runId, latest);

  $("timeline").value = String(state.cursor);
  $("frame-label").textContent =
    `${formatTime(frame.timestamp)}  ·  ${state.cursor + 1}/${frames.length}` +
    `  ·  ${frame.synthetic ? "synthesized" : "observed"}`;
}

function stop() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  $("play-toggle").textContent = "Play";
}

function play() {
  stop();
  const delay = Number($("speed").value);
  state.timer = setInterval(() => {
    showFrame((state.cursor + 1) % state.manifest.frames.length);
  }, delay);
  $("play-toggle").textContent = "Pause";
}

function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
    state.charts[key] = null;
  }
}

function drawOverallChart(summary) {
  destroyChart("overall");
  const methods = Object.keys(summary.overall || {});
  if (!methods.length) return;
  const labels = METRIC_ORDER.filter((m) => m in (summary.overall[methods[0]] || {}));

  state.charts.overall = new Chart($("chart-overall"), {
    type: "bar",
    data: {
      labels,
      datasets: methods.map((method) => ({
        label: method,
        backgroundColor: METHOD_COLOURS[method] || "#7e57c2",
        data: labels.map((metric) => summary.overall[method][metric] ?? null),
      })),
    },
    options: {
      responsive: true,
      scales: {
        y: { type: "logarithmic", ticks: { color: "#9aa4b2" } },
        x: { ticks: { color: "#9aa4b2" } },
      },
      plugins: { legend: { labels: { color: "#e6e9ef" } } },
    },
  });
}

function drawCategoryChart(summary) {
  destroyChart("category");
  const categories = Object.keys(summary.by_category || {});
  if (!categories.length) return;
  const methods = summary.methods || [];

  state.charts.category = new Chart($("chart-category"), {
    type: "bar",
    data: {
      labels: categories,
      datasets: methods.map((method) => ({
        label: method,
        backgroundColor: METHOD_COLOURS[method] || "#7e57c2",
        data: categories.map((c) => (summary.by_category[c]?.[method] || {}).psnr ?? null),
      })),
    },
    options: {
      responsive: true,
      scales: {
        y: {
          title: { display: true, text: "PSNR (dB)", color: "#9aa4b2" },
          ticks: { color: "#9aa4b2" },
        },
        x: { ticks: { color: "#9aa4b2" } },
      },
      plugins: { legend: { labels: { color: "#e6e9ef" } } },
    },
  });
}

function drawTable(summary) {
  const methods = Object.keys(summary.overall || {});
  const table = $("metrics-table");
  if (!methods.length) {
    table.innerHTML = "<tbody><tr><td>No report generated for this run yet.</td></tr></tbody>";
    return;
  }
  const metrics = METRIC_ORDER.filter((m) => m in summary.overall[methods[0]]);
  const head = `<thead><tr><th>Metric</th>${methods.map((m) => `<th>${m}</th>`).join("")}` +
               `</tr></thead>`;
  const body = metrics.map((metric) => {
    const values = methods.map((m) => summary.overall[m][metric]);
    const finite = values.filter((v) => typeof v === "number");
    const best = finite.length
      ? (HIGHER_IS_BETTER.has(metric) ? Math.max(...finite) : Math.min(...finite))
      : null;
    const cells = values.map((v) => {
      const shown = typeof v === "number" ? v.toFixed(4) : "—";
      return `<td${v === best ? ' class="win"' : ""}>${shown}</td>`;
    }).join("");
    return `<tr><td>${metric}</td>${cells}</tr>`;
  }).join("");
  table.innerHTML = `${head}<tbody>${body}</tbody>`;
}

async function loadRun(runId) {
  stop();
  state.runId = runId;
  state.manifest = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
  if (!state.manifest) return;

  state.observed = state.manifest.frames
    .map((frame, index) => (frame.synthetic ? -1 : index))
    .filter((index) => index >= 0);

  renderSummary(state.manifest);
  $("timeline").max = String(state.manifest.frames.length - 1);
  $("download-nc").href = `/api/runs/${encodeURIComponent(runId)}/download`;
  $("label-original").textContent =
    `observed frames · ${state.manifest.input_cadence_minutes} min`;
  $("label-interpolated").textContent =
    `observed + synthesized · ${state.manifest.output_cadence_minutes} min`;
  showFrame(0);

  const report = await getJSON(`/api/runs/${encodeURIComponent(runId)}/report`);
  const summary = (report && report.summary) || { overall: {}, by_category: {}, methods: [] };
  drawOverallChart(summary);
  drawCategoryChart(summary);
  drawTable(summary);
}

async function init() {
  const runs = (await getJSON("/api/runs")) || [];
  const select = $("run-select");
  select.innerHTML = runs
    .map((r) => `<option value="${r.run_id}">${r.run_id} (${r.sensor})</option>`)
    .join("");

  select.addEventListener("change", () => loadRun(select.value));
  $("play-toggle").addEventListener("click", () => (state.timer ? stop() : play()));
  $("timeline").addEventListener("input", (event) => {
    stop();
    showFrame(Number(event.target.value));
  });
  $("speed").addEventListener("change", () => { if (state.timer) play(); });

  if (runs.length) {
    await loadRun(runs[0].run_id);
  } else {
    $("summary").innerHTML =
      '<div class="stat"><div class="value">No runs</div>' +
      '<div class="label">run sattsr infer first</div></div>';
  }
}

document.addEventListener("DOMContentLoaded", init);
