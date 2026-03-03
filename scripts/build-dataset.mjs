#!/usr/bin/env node

import crypto from "node:crypto";
import path from "node:path";
import { promises as fs } from "node:fs";

import { Command } from "commander";
import fg from "fast-glob";
import { UMAP } from "umap-js";
import { kmeans } from "ml-kmeans";
import OpenAI from "openai";

const STOPWORDS = new Set([
  "the",
  "and",
  "that",
  "with",
  "from",
  "this",
  "have",
  "your",
  "into",
  "about",
  "there",
  "where",
  "when",
  "what",
  "which",
  "would",
  "could",
  "should",
  "here",
  "been",
  "only",
  "also",
  "them",
  "they",
  "their",
  "just",
  "then",
  "than",
  "will",
  "each",
  "more",
  "some",
  "over",
  "very",
  "using",
  "used",
  "user",
  "assistant",
  "tool",
  "json",
  "input",
  "output",
  "action",
  "final",
  "answer"
]);

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = (sorted.length - 1) * clamp(q, 0, 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const w = idx - lo;
  return sorted[lo] * (1 - w) + sorted[hi] * w;
}

function cosineDistance(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    const av = a[i] || 0;
    const bv = b[i] || 0;
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  if (na === 0 || nb === 0) return 1;
  const cosine = dot / (Math.sqrt(na) * Math.sqrt(nb));
  return 1 - cosine;
}

function normalizeVector(vec) {
  let norm = 0;
  for (const x of vec) norm += x * x;
  norm = Math.sqrt(norm) || 1;
  return vec.map((x) => x / norm);
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function random() {
    t += 0x6d2b79f5;
    let v = Math.imul(t ^ (t >>> 15), t | 1);
    v ^= v + Math.imul(v ^ (v >>> 7), v | 61);
    return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
  };
}

function tokenize(text) {
  return (text.toLowerCase().match(/[a-z][a-z0-9_-]{2,}/g) || []).filter((t) => !STOPWORDS.has(t));
}

function compactText(value, limit = 2400) {
  if (value === null || value === undefined) return "";
  const source =
    typeof value === "string"
      ? value
      : typeof value === "object"
        ? JSON.stringify(value)
        : String(value);
  const compact = source.replace(/\s+/g, " ").trim();
  if (!compact) return "";
  return compact.length > limit ? `${compact.slice(0, limit)}...` : compact;
}

function extractUserInput(promptText) {
  if (!promptText) return "";
  const marker = "USER'S INPUT";
  const idx = promptText.indexOf(marker);
  if (idx >= 0) {
    const tail = promptText.slice(idx + marker.length);
    return compactText(tail.replace(/^-+\s*/m, ""));
  }

  const humanIdx = promptText.lastIndexOf("Human:");
  if (humanIdx >= 0) {
    return compactText(promptText.slice(humanIdx + "Human:".length));
  }

  return compactText(promptText);
}

function extractJsonAction(outputText) {
  if (!outputText) return "";
  try {
    const parsed = JSON.parse(outputText);
    if (parsed?.action_input) return compactText(String(parsed.action_input));
  } catch {
    // pass-through
  }
  return compactText(outputText);
}

function parseWandbRunStamp(filePath) {
  const match = filePath.match(/run-(\d{8})_(\d{6})-[a-z0-9]+/i);
  if (!match) return new Date("2025-01-01T00:00:00.000Z");

  const [, ymd, hms] = match;
  const iso = `${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6, 8)}T${hms.slice(0, 2)}:${hms.slice(2, 4)}:${hms.slice(4, 6)}.000Z`;
  const dt = new Date(iso);
  if (Number.isNaN(dt.getTime())) return new Date("2025-01-01T00:00:00.000Z");
  return dt;
}

async function loadZetaSessionTables(rootPath, maxFiles) {
  const globPattern = path.join(rootPath, "OLD_omnisearch/wandb/run-*/files/media/table/session_analysis_*.table.json");
  const files = (await fg(globPattern, { dot: false })).sort().slice(0, maxFiles);

  const points = [];

  for (const filePath of files) {
    const raw = await fs.readFile(filePath, "utf8");
    const parsed = JSON.parse(raw);
    const cols = parsed.columns || [];
    const rows = parsed.data || [];

    const cPromptStep = cols.indexOf("prompt_step");
    const cPrompts = cols.indexOf("prompts");
    const cOutputStep = cols.indexOf("output_step");
    const cOutput = cols.indexOf("output");
    const cName = cols.indexOf("name");

    const stamp = parseWandbRunStamp(filePath);
    const fileTag = path.basename(filePath).replace(".table.json", "");
    const runTag = (filePath.match(/run-([^/]+)/) || [])[1] || "run";
    const threadId = `zeta-${runTag}-${fileTag}`;

    for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
      const row = rows[rowIndex];
      const promptText = extractUserInput(String(row[cPrompts] || ""));
      const answerText = extractJsonAction(String(row[cOutput] || ""));
      const modelName = cName >= 0 ? String(row[cName] || "") : "";

      const promptStep = Number(row[cPromptStep] ?? rowIndex * 2);
      const outputStep = Number(row[cOutputStep] ?? rowIndex * 2 + 1);

      if (promptText) {
        points.push({
          id: `${threadId}-u-${rowIndex}`,
          threadId,
          seq: promptStep,
          role: "user",
          text: promptText,
          timestamp: new Date(stamp.getTime() + rowIndex * 2000).toISOString(),
          sourceType: "zeta-session-analysis",
          sourcePath: filePath,
          modelName
        });
      }

      if (answerText) {
        points.push({
          id: `${threadId}-a-${rowIndex}`,
          threadId,
          seq: outputStep,
          role: "assistant",
          text: answerText,
          timestamp: new Date(stamp.getTime() + rowIndex * 2000 + 900).toISOString(),
          sourceType: "zeta-session-analysis",
          sourcePath: filePath,
          modelName
        });
      }
    }
  }

  return points;
}

function extractClaudeText(message) {
  if (!message) return "";

  if (typeof message.content === "string") return compactText(message.content);

  if (Array.isArray(message.content)) {
    const parts = [];
    for (const item of message.content) {
      if (!item || typeof item !== "object") continue;
      if (item.type === "text" && item.text) parts.push(item.text);
      if (item.type === "output_text" && item.text) parts.push(item.text);
      if (item.type === "input_text" && item.text) parts.push(item.text);
      if (item.type === "tool_result" && item.content) {
        const content = typeof item.content === "string" ? item.content : JSON.stringify(item.content);
        parts.push(`tool_result: ${content}`);
      }
    }
    return compactText(parts.join("\n"));
  }

  return "";
}

function parseClaudeLine(obj, lineNumber, filePath) {
  const type = String(obj.type || "");
  if (type !== "user" && type !== "assistant") return null;

  const msg = obj.message || {};
  const role = msg.role || type;
  const text = extractClaudeText(msg);
  if (!text) return null;

  const threadId = obj.sessionId || obj.session_id || path.basename(filePath).replace(/\.[^.]+$/, "");

  return {
    id: `${threadId}-${lineNumber}`,
    threadId,
    seq: lineNumber,
    role,
    text,
    timestamp: obj.timestamp || null,
    sourceType: "zerg-claude-fixture",
    sourcePath: filePath,
    modelName: msg?.model || "claude"
  };
}

function parseCodexLine(obj, lineNumber, filePath, threadHint) {
  if (obj.type === "session_meta") {
    const sessionId = obj?.payload?.id;
    return { metaOnly: true, threadId: sessionId || threadHint };
  }

  if (obj.type !== "response_item") return null;
  const payload = obj.payload || {};
  if (payload.type !== "message") return null;

  const role = payload.role || "assistant";
  const text = extractClaudeText(payload);
  if (!text) return null;

  const threadId = threadHint || path.basename(filePath).replace(/\.[^.]+$/, "");

  return {
    id: `${threadId}-${lineNumber}`,
    threadId,
    seq: lineNumber,
    role,
    text,
    timestamp: obj.timestamp || null,
    sourceType: "zerg-codex-fixture",
    sourcePath: filePath,
    modelName: "codex"
  };
}

async function loadZergFixtures(fixturesPath) {
  const jsonlFiles = (await fg(path.join(fixturesPath, "*.jsonl"))).sort();
  const jsonFiles = (await fg(path.join(fixturesPath, "*.json"))).sort();
  const points = [];

  for (const filePath of jsonlFiles) {
    const raw = await fs.readFile(filePath, "utf8");
    const lines = raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    let codexThread = "";

    for (let i = 0; i < lines.length; i += 1) {
      let obj;
      try {
        obj = JSON.parse(lines[i]);
      } catch {
        continue;
      }

      const maybeClaude = parseClaudeLine(obj, i, filePath);
      if (maybeClaude) {
        points.push(maybeClaude);
        continue;
      }

      const maybeCodex = parseCodexLine(obj, i, filePath, codexThread);
      if (!maybeCodex) continue;

      if (maybeCodex.metaOnly) {
        codexThread = maybeCodex.threadId || codexThread;
        continue;
      }

      points.push(maybeCodex);
    }
  }

  for (const filePath of jsonFiles) {
    const raw = await fs.readFile(filePath, "utf8");
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch {
      continue;
    }

    if (!parsed?.messages || !Array.isArray(parsed.messages)) continue;

    const threadId = parsed.sessionId || path.basename(filePath).replace(/\.[^.]+$/, "");

    for (let i = 0; i < parsed.messages.length; i += 1) {
      const msg = parsed.messages[i];
      const role = msg.type === "gemini" ? "assistant" : msg.type || "unknown";
      const text = compactText(msg.content || "");
      if (!text) continue;

      points.push({
        id: `${threadId}-${i}`,
        threadId,
        seq: i,
        role,
        text,
        timestamp: msg.timestamp || null,
        sourceType: "zerg-gemini-fixture",
        sourcePath: filePath,
        modelName: msg.model || "gemini"
      });
    }
  }

  return points;
}

function sortedByThread(points) {
  return [...points].sort((a, b) => {
    if (a.threadId !== b.threadId) return a.threadId.localeCompare(b.threadId);
    if ((a.timestamp || "") !== (b.timestamp || "")) return String(a.timestamp || "").localeCompare(String(b.timestamp || ""));
    return a.seq - b.seq;
  });
}

function fallbackEmbedding(text, dims) {
  const vec = new Array(dims).fill(0);
  const tokens = tokenize(text);

  if (!tokens.length) return vec;

  for (const token of tokens.slice(0, 400)) {
    const digest = crypto.createHash("sha256").update(token).digest();
    for (let i = 0; i < 4; i += 1) {
      const idx = ((digest.readUInt16BE(i * 2) + i * 131) >>> 0) % dims;
      const sign = digest[i + 8] & 1 ? 1 : -1;
      const mag = 1 + (digest[i + 12] % 17) / 17;
      vec[idx] += sign * mag;
    }
  }

  return normalizeVector(vec);
}

async function embedWithOpenAI(texts, { model, dimensions, batchSize, apiKey }) {
  const client = new OpenAI({ apiKey });
  const vectors = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const chunk = texts.slice(i, i + batchSize);
    const response = await client.embeddings.create({
      model,
      input: chunk,
      dimensions
    });

    for (const d of response.data) {
      vectors.push(normalizeVector(d.embedding.map(Number)));
    }

    process.stdout.write(`\rEmbedding ${Math.min(i + batchSize, texts.length)}/${texts.length}`);
  }

  process.stdout.write("\n");
  return vectors;
}

function project3D(vectors, seed = 1337) {
  const nNeighbors = clamp(Math.round(Math.sqrt(vectors.length) * 1.8), 8, 30);
  const reducer = new UMAP({
    nComponents: 3,
    nNeighbors,
    minDist: 0.18,
    random: mulberry32(seed)
  });

  const projected = reducer.fit(vectors);

  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];

  for (const p of projected) {
    for (let i = 0; i < 3; i += 1) {
      mins[i] = Math.min(mins[i], p[i]);
      maxs[i] = Math.max(maxs[i], p[i]);
    }
  }

  return projected.map((p) => {
    const out = [];
    for (let i = 0; i < 3; i += 1) {
      const span = maxs[i] - mins[i] || 1;
      const scaled = ((p[i] - mins[i]) / span) * 2 - 1;
      out.push(Number(scaled.toFixed(6)));
    }
    return out;
  });
}

function clusterProjected(points3D, seed) {
  if (points3D.length < 3) return new Array(points3D.length).fill(0);

  const k = clamp(Math.round(Math.sqrt(points3D.length / 2)), 3, 10);
  const result = kmeans(points3D, k, {
    initialization: "kmeans++",
    seed
  });

  return result.clusters;
}

function makeClusterLabels(points) {
  const byCluster = new Map();
  const global = new Map();

  for (const p of points) {
    const cluster = p.cluster;
    if (!byCluster.has(cluster)) byCluster.set(cluster, new Map());
    const cMap = byCluster.get(cluster);

    const seen = new Set();
    for (const token of tokenize(p.text)) {
      cMap.set(token, (cMap.get(token) || 0) + 1);
      if (!seen.has(token)) {
        global.set(token, (global.get(token) || 0) + 1);
        seen.add(token);
      }
    }
  }

  const labels = new Map();

  for (const [cluster, cMap] of byCluster.entries()) {
    const scored = [];
    for (const [token, count] of cMap.entries()) {
      const globalCount = global.get(token) || 1;
      const score = count / globalCount;
      scored.push([token, score]);
    }

    scored.sort((a, b) => b[1] - a[1]);
    labels.set(cluster, scored.slice(0, 4).map((x) => x[0]).join(" · ") || "mixed");
  }

  return labels;
}

function applyPivotScoring(points, vectors, pivotQuantile) {
  const byThread = new Map();
  for (const p of points) {
    if (!byThread.has(p.threadId)) byThread.set(p.threadId, []);
    byThread.get(p.threadId).push(p);
  }

  const distances = [];

  for (const threadPoints of byThread.values()) {
    threadPoints.sort((a, b) => a.seq - b.seq);
    for (let i = 0; i < threadPoints.length; i += 1) {
      const p = threadPoints[i];
      if (i === 0) {
        p.pivotScore = 0;
        continue;
      }

      const prev = threadPoints[i - 1];
      const dist = cosineDistance(vectors[p._vecIndex], vectors[prev._vecIndex]);
      p.pivotScore = Number(dist.toFixed(6));
      distances.push(dist);
    }
  }

  const threshold = quantile(distances, pivotQuantile);

  for (const threadPoints of byThread.values()) {
    threadPoints.sort((a, b) => a.seq - b.seq);
    for (let i = 0; i < threadPoints.length; i += 1) {
      const p = threadPoints[i];
      p.isPivot = i > 0 && p.pivotScore >= threshold;
      p.clusterTransition = i > 0 && p.cluster !== threadPoints[i - 1].cluster;
    }
  }

  return Number(threshold.toFixed(6));
}

function buildTrajectories(points) {
  const byThread = new Map();
  for (const p of points) {
    if (!byThread.has(p.threadId)) byThread.set(p.threadId, []);
    byThread.get(p.threadId).push(p);
  }

  const trajectories = [];
  for (const [threadId, pts] of byThread.entries()) {
    pts.sort((a, b) => a.seq - b.seq);
    trajectories.push({
      threadId,
      pointIds: pts.map((p) => p.id)
    });
  }

  trajectories.sort((a, b) => b.pointIds.length - a.pointIds.length);
  return trajectories;
}

function summarizeThreads(points) {
  const byThread = new Map();
  for (const p of points) {
    if (!byThread.has(p.threadId)) {
      byThread.set(p.threadId, {
        id: p.threadId,
        sourceType: p.sourceType,
        pointCount: 0,
        firstTimestamp: p.timestamp,
        lastTimestamp: p.timestamp,
        roles: new Set()
      });
    }

    const t = byThread.get(p.threadId);
    t.pointCount += 1;
    t.roles.add(p.role);

    if (p.timestamp && (!t.firstTimestamp || p.timestamp < t.firstTimestamp)) t.firstTimestamp = p.timestamp;
    if (p.timestamp && (!t.lastTimestamp || p.timestamp > t.lastTimestamp)) t.lastTimestamp = p.timestamp;
  }

  return [...byThread.values()]
    .map((t) => ({
      ...t,
      roles: [...t.roles]
    }))
    .sort((a, b) => b.pointCount - a.pointCount);
}

async function main() {
  const program = new Command();
  program
    .option("--zeta-root <path>", "Zeta root path", "/Users/davidrose/git/zeta")
    .option(
      "--zerg-fixtures <path>",
      "Zerg fixture path",
      "/Users/davidrose/git/zerg/apps/zerg/backend/tests/integration/fixtures"
    )
    .option("--max-zeta-files <n>", "Max session-analysis files", "22")
    .option("--max-points <n>", "Max total embedded points", "600")
    .option("--embedding-model <name>", "Embedding model", "text-embedding-3-small")
    .option("--dimensions <n>", "Embedding dimensions", "256")
    .option("--batch-size <n>", "Embedding batch size", "64")
    .option("--pivot-quantile <n>", "Pivot threshold quantile", "0.85")
    .option("--seed <n>", "Random seed", "1337")
    .option("--force-local", "Use local fallback embeddings only", false)
    .option("--output <path>", "Output JSON path", "data/dataset.json")
    .parse(process.argv);

  const opts = program.opts();
  const maxZetaFiles = Number(opts.maxZetaFiles);
  const maxPoints = Number(opts.maxPoints);
  const dimensions = Number(opts.dimensions);
  const batchSize = Number(opts.batchSize);
  const pivotQuantile = Number(opts.pivotQuantile);
  const seed = Number(opts.seed);

  const zetaPoints = await loadZetaSessionTables(opts.zetaRoot, maxZetaFiles);
  const zergPoints = await loadZergFixtures(opts.zergFixtures);

  let points = sortedByThread([...zetaPoints, ...zergPoints]).filter((p) => p.text && p.text.length >= 6);

  if (points.length > maxPoints) {
    const byThread = new Map();
    for (const p of points) {
      if (!byThread.has(p.threadId)) byThread.set(p.threadId, []);
      byThread.get(p.threadId).push(p);
    }

    const selected = [];
    const lists = [...byThread.values()].sort((a, b) => b.length - a.length);
    let cursor = 0;

    while (selected.length < maxPoints) {
      let picked = false;
      for (const threadPoints of lists) {
        if (cursor < threadPoints.length) {
          selected.push(threadPoints[cursor]);
          picked = true;
          if (selected.length >= maxPoints) break;
        }
      }
      if (!picked) break;
      cursor += 1;
    }

    points = sortedByThread(selected);
  }

  points = points.map((p, idx) => ({ ...p, _vecIndex: idx }));

  const texts = points.map((p) => p.text);
  const apiKey = process.env.OPENAI_API_KEY;
  const canUseOpenAI = !opts.forceLocal && Boolean(apiKey);

  let vectors;
  if (canUseOpenAI) {
    console.log(`Embedding ${texts.length} points with ${opts.embeddingModel} (${dimensions}d)`);
    try {
      vectors = await embedWithOpenAI(texts, {
        model: opts.embeddingModel,
        dimensions,
        batchSize,
        apiKey
      });
    } catch (error) {
      console.warn(`OpenAI embedding failed: ${String(error.message || error)}`);
      console.warn("Falling back to deterministic local embeddings.");
      vectors = texts.map((t) => fallbackEmbedding(t, dimensions));
    }
  } else {
    console.log("OPENAI_API_KEY missing or force-local enabled. Using deterministic local embeddings.");
    vectors = texts.map((t) => fallbackEmbedding(t, dimensions));
  }

  const projection = project3D(vectors, seed);
  const clusters = clusterProjected(projection, seed);

  for (let i = 0; i < points.length; i += 1) {
    points[i].projection = projection[i];
    points[i].cluster = clusters[i];
  }

  const pivotThreshold = applyPivotScoring(points, vectors, pivotQuantile);

  const clusterLabels = makeClusterLabels(points);
  const clusterSummary = [];

  const clusterBuckets = new Map();
  for (const p of points) {
    if (!clusterBuckets.has(p.cluster)) clusterBuckets.set(p.cluster, []);
    clusterBuckets.get(p.cluster).push(p);
  }

  for (const [clusterId, bucket] of [...clusterBuckets.entries()].sort((a, b) => a[0] - b[0])) {
    const centroid = [0, 0, 0];
    for (const p of bucket) {
      centroid[0] += p.projection[0];
      centroid[1] += p.projection[1];
      centroid[2] += p.projection[2];
    }
    centroid[0] /= bucket.length;
    centroid[1] /= bucket.length;
    centroid[2] /= bucket.length;

    clusterSummary.push({
      id: clusterId,
      size: bucket.length,
      label: clusterLabels.get(clusterId) || "mixed",
      centroid: centroid.map((v) => Number(v.toFixed(6)))
    });
  }

  const trajectories = buildTrajectories(points);
  const threads = summarizeThreads(points);

  const output = {
    generatedAt: new Date().toISOString(),
    config: {
      embeddingModel: canUseOpenAI ? opts.embeddingModel : "local-hash-fallback",
      dimensions,
      pivotQuantile,
      seed,
      sources: {
        zetaRoot: opts.zetaRoot,
        zergFixtures: opts.zergFixtures
      }
    },
    stats: {
      points: points.length,
      threads: threads.length,
      trajectories: trajectories.length,
      pivotThreshold,
      sourceCounts: points.reduce((acc, p) => {
        acc[p.sourceType] = (acc[p.sourceType] || 0) + 1;
        return acc;
      }, {})
    },
    clusters: clusterSummary,
    threads,
    trajectories,
    points: points.map((p) => ({
      id: p.id,
      threadId: p.threadId,
      seq: p.seq,
      role: p.role,
      text: p.text,
      timestamp: p.timestamp,
      sourceType: p.sourceType,
      sourcePath: p.sourcePath,
      modelName: p.modelName,
      projection: p.projection,
      cluster: p.cluster,
      pivotScore: p.pivotScore,
      isPivot: p.isPivot,
      clusterTransition: p.clusterTransition
    }))
  };

  const outputPath = path.resolve(opts.output);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");

  console.log(`Wrote dataset: ${outputPath}`);
  console.log(`points=${output.stats.points}, threads=${output.stats.threads}, pivotThreshold=${pivotThreshold}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
