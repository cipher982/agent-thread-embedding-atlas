import Plotly from "plotly.js-gl3d-dist-min";
import "./style.css";

const CLUSTER_COLORS = [
  "#ffb347",
  "#6ae8cd",
  "#e46f6f",
  "#8cb5ff",
  "#ffd86b",
  "#9ef07e",
  "#ff9fb3",
  "#84f3ff",
  "#c8a4ff",
  "#ffe59a",
  "#64d5ff",
  "#f6b6ff"
];

const ROLE_COLORS = {
  user: "#ffcf70",
  assistant: "#78f1d2",
  tool: "#ff8f8f",
  system: "#9ea7ff",
  unknown: "#c0cad2"
};

const state = {
  dataset: null,
  colorBy: "cluster",
  pivotQuantile: 0.85,
  threadLimit: 18,
  showTrajectories: true,
  showOnlyPivots: false,
  pointById: new Map(),
  threadRank: [],
  plotReady: false
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * clamp(q, 0, 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function shortThreadId(id) {
  if (!id) return "unknown";
  return id.length > 24 ? `${id.slice(0, 12)}...${id.slice(-8)}` : id;
}

function palette(index) {
  return CLUSTER_COLORS[index % CLUSTER_COLORS.length];
}

function roleColor(role) {
  return ROLE_COLORS[role] || ROLE_COLORS.unknown;
}

function getColor(point) {
  if (state.colorBy === "role") return roleColor(point.role);

  if (state.colorBy === "thread") {
    const idx = state.threadRank.indexOf(point.threadId);
    return palette(idx >= 0 ? idx : 0);
  }

  if (state.colorBy === "pivot") {
    return point._effectivePivot ? "#ff7a7a" : "#58b39d";
  }

  return palette(Number.isFinite(point.cluster) ? point.cluster : 0);
}

function formatText(text, limit = 340) {
  if (!text) return "";
  const compact = text.replace(/\s+/g, " ").trim();
  return compact.length > limit ? `${compact.slice(0, limit)}...` : compact;
}

async function loadDataset() {
  const res = await fetch("/data/dataset.json");
  if (!res.ok) {
    throw new Error(`Missing dataset file: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

function initLayout() {
  const app = document.querySelector("#app");
  app.innerHTML = `
    <aside id="left" class="panel">
      <div>
        <h1>Agent Thread Embedding Atlas</h1>
        <p class="sub">3D map of semantic pivots across agent conversations</p>
      </div>

      <section class="metrics" id="metrics"></section>

      <section class="field">
        <label for="colorBy">Color By</label>
        <select id="colorBy">
          <option value="cluster">Cluster region</option>
          <option value="thread">Thread</option>
          <option value="role">Role</option>
          <option value="pivot">Pivot confidence</option>
        </select>
      </section>

      <section class="field">
        <label for="pivotQ">Pivot Sensitivity (<span id="pivotQValue"></span>)</label>
        <input id="pivotQ" type="range" min="0.5" max="0.99" step="0.01" value="0.85" />
      </section>

      <section class="field">
        <label for="threadLimit">Visible Threads (<span id="threadLimitValue"></span>)</label>
        <input id="threadLimit" type="range" min="3" max="80" step="1" value="18" />
      </section>

      <section class="field">
        <label><input id="showTraj" type="checkbox" checked /> Show trajectories</label>
        <label><input id="showPivotOnly" type="checkbox" /> Show pivots only</label>
      </section>

      <section>
        <p class="sub">Top threads in view</p>
        <div class="thread-list" id="threadList"></div>
      </section>
    </aside>

    <main id="vizWrap" class="panel">
      <div id="viz"></div>
    </main>

    <aside id="right" class="panel">
      <section>
        <p class="sub">Selected point</p>
        <div class="detail-box" id="pointDetails">Click a point to inspect thread text and metadata.</div>
      </section>
      <section>
        <p class="sub">Region labels</p>
        <div class="cluster-list" id="clusterList"></div>
      </section>
      <p class="note" id="datasetNote"></p>
    </aside>
  `;
}

function bindControls() {
  const colorBy = document.querySelector("#colorBy");
  const pivotQ = document.querySelector("#pivotQ");
  const threadLimit = document.querySelector("#threadLimit");
  const showTraj = document.querySelector("#showTraj");
  const showPivotOnly = document.querySelector("#showPivotOnly");

  colorBy.addEventListener("change", () => {
    state.colorBy = colorBy.value;
    renderPlot();
  });

  pivotQ.addEventListener("input", () => {
    state.pivotQuantile = Number(pivotQ.value);
    renderPlot();
  });

  threadLimit.addEventListener("input", () => {
    state.threadLimit = Number(threadLimit.value);
    renderPlot();
  });

  showTraj.addEventListener("change", () => {
    state.showTrajectories = showTraj.checked;
    renderPlot();
  });

  showPivotOnly.addEventListener("change", () => {
    state.showOnlyPivots = showPivotOnly.checked;
    renderPlot();
  });
}

function updateNote() {
  const note = document.querySelector("#datasetNote");
  const ds = state.dataset;
  const model = ds?.config?.embeddingModel || "unknown";
  const dims = ds?.config?.dimensions || "?";
  note.textContent = `Generated ${new Date(ds.generatedAt).toLocaleString()} | model ${model} (${dims}d)`;
}

function renderClusterList() {
  const clusterList = document.querySelector("#clusterList");
  const clusters = state.dataset.clusters || [];
  clusterList.innerHTML = "";

  for (const c of clusters) {
    const el = document.createElement("div");
    el.className = "cluster-item";
    const label = c.label || "mixed";
    el.innerHTML = `
      <span class="cluster-label">Region ${c.id}: ${label}</span><br />
      <span class="badge">${c.size} points</span>
    `;
    clusterList.appendChild(el);
  }
}

function renderThreadList(visibleThreads) {
  const threadList = document.querySelector("#threadList");
  threadList.innerHTML = "";

  for (const threadId of visibleThreads) {
    const info = state.dataset.threads.find((t) => t.id === threadId);
    const item = document.createElement("div");
    item.className = "thread-item";
    item.innerHTML = `
      <span class="thread-id" title="${threadId}">${shortThreadId(threadId)}</span>
      <span class="badge">${info?.pointCount || 0}</span>
    `;
    threadList.appendChild(item);
  }
}

function renderMetrics(allVisiblePoints, pivotCount, activeThreads) {
  const metrics = document.querySelector("#metrics");
  const ds = state.dataset;
  const cards = [
    ["Visible points", String(allVisiblePoints.length)],
    ["Visible threads", String(activeThreads.length)],
    ["Pivot points", String(pivotCount)],
    ["Total regions", String(ds.clusters?.length || 0)]
  ];

  metrics.innerHTML = cards
    .map(
      ([k, v]) => `
      <div class="metric">
        <span class="k">${k}</span>
        <span class="v">${v}</span>
      </div>
    `
    )
    .join("");

  document.querySelector("#pivotQValue").textContent = state.pivotQuantile.toFixed(2);
  document.querySelector("#threadLimitValue").textContent = String(state.threadLimit);
}

function buildPlotData(visiblePoints, pivotThreshold) {
  for (const p of visiblePoints) {
    p._effectivePivot = Number(p.pivotScore || 0) >= pivotThreshold && p.seq > 0;
  }

  const points = state.showOnlyPivots ? visiblePoints.filter((p) => p._effectivePivot) : visiblePoints;

  const markerTrace = {
    type: "scatter3d",
    mode: "markers",
    name: "Events",
    x: points.map((p) => p.projection[0]),
    y: points.map((p) => p.projection[1]),
    z: points.map((p) => p.projection[2]),
    text: points.map((p) => formatText(p.text, 180)),
    customdata: points.map((p) => [
      p.id,
      p.threadId,
      p.seq,
      p.role,
      p.cluster,
      Number(p.pivotScore || 0).toFixed(4),
      p.timestamp || "",
      p.sourceType || "",
      p.text
    ]),
    hovertemplate:
      "<b>%{customdata[1]}</b><br>" +
      "seq %{customdata[2]} | %{customdata[3]}<br>" +
      "cluster %{customdata[4]} | pivot %{customdata[5]}<br>" +
      "%{text}<extra></extra>",
    marker: {
      size: points.map((p) => (p._effectivePivot ? 7.5 : 4.2)),
      color: points.map(getColor),
      line: {
        width: points.map((p) => (p._effectivePivot ? 1.4 : 0.3)),
        color: points.map((p) => (p._effectivePivot ? "#ffe0a8" : "rgba(10, 30, 34, 0.5)"))
      },
      opacity: 0.9
    }
  };

  const traces = [markerTrace];

  if (state.showTrajectories) {
    for (const traj of state.dataset.trajectories) {
      const threadPoints = traj.pointIds
        .map((id) => state.pointById.get(id))
        .filter((p) => p && points.includes(p));

      if (threadPoints.length < 2) continue;

      traces.push({
        type: "scatter3d",
        mode: "lines",
        name: traj.threadId,
        showlegend: false,
        x: threadPoints.map((p) => p.projection[0]),
        y: threadPoints.map((p) => p.projection[1]),
        z: threadPoints.map((p) => p.projection[2]),
        line: {
          width: 2,
          color: "rgba(220, 245, 240, 0.22)"
        },
        hoverinfo: "skip"
      });
    }
  }

  return traces;
}

function renderPlot() {
  if (!state.dataset) return;

  const sortedThreads = [...state.threadRank];
  const visibleThreads = sortedThreads.slice(0, clamp(state.threadLimit, 1, sortedThreads.length));
  const visibleSet = new Set(visibleThreads);

  const visiblePoints = state.dataset.points.filter((p) => visibleSet.has(p.threadId));
  const pivotThreshold = quantile(
    visiblePoints.map((p) => Number(p.pivotScore || 0)).filter((v) => Number.isFinite(v)),
    state.pivotQuantile
  );

  const traces = buildPlotData(visiblePoints, pivotThreshold);

  const pivots = visiblePoints.filter((p) => p._effectivePivot);
  renderMetrics(visiblePoints, pivots.length, visibleThreads);
  renderThreadList(visibleThreads);

  const layout = {
    margin: { l: 0, r: 0, t: 10, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    scene: {
      bgcolor: "rgba(0,0,0,0)",
      xaxis: { title: "U1", showgrid: false, zeroline: false, showticklabels: false },
      yaxis: { title: "U2", showgrid: false, zeroline: false, showticklabels: false },
      zaxis: { title: "U3", showgrid: false, zeroline: false, showticklabels: false },
      camera: {
        eye: { x: 1.42, y: 1.22, z: 0.98 }
      }
    }
  };

  Plotly.react("viz", traces, layout, { responsive: true, displaylogo: false }).then(() => {
    if (state.plotReady) return;

    const viz = document.querySelector("#viz");
    viz.on("plotly_click", (evt) => {
      const hit = evt?.points?.[0];
      if (!hit?.customdata) return;

      const [id, threadId, seq, role, cluster, pivotScore, timestamp, sourceType, text] = hit.customdata;
      const details = document.querySelector("#pointDetails");
      details.textContent = [
        `point: ${id}`,
        `thread: ${threadId}`,
        `seq: ${seq}`,
        `role: ${role}`,
        `cluster: ${cluster}`,
        `pivot score: ${pivotScore}`,
        `source: ${sourceType}`,
        `timestamp: ${timestamp || "n/a"}`,
        "",
        formatText(text, 1200)
      ].join("\n");
    });

    state.plotReady = true;
  });
}

async function init() {
  initLayout();

  try {
    state.dataset = await loadDataset();
  } catch (error) {
    document.querySelector("#app").innerHTML = `
      <main class="panel" style="padding:20px; max-width:780px; margin:20px auto;">
        <h1>Dataset missing</h1>
        <p class="sub">Run <code>bun run build:dataset</code> to generate <code>data/dataset.json</code>.</p>
        <pre>${String(error.message || error)}</pre>
      </main>
    `;
    return;
  }

  for (const point of state.dataset.points) {
    state.pointById.set(point.id, point);
  }

  state.threadRank = [...state.dataset.threads]
    .sort((a, b) => b.pointCount - a.pointCount)
    .map((t) => t.id);

  const maxThreads = Math.max(3, state.threadRank.length);
  const threadInput = document.querySelector("#threadLimit");
  threadInput.max = String(maxThreads);
  threadInput.value = String(Math.min(state.threadLimit, maxThreads));
  state.threadLimit = Number(threadInput.value);

  bindControls();
  updateNote();
  renderClusterList();
  renderPlot();
}

init();
