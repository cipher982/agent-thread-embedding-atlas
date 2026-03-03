# Spec: Agent Thread Embedding Visualization (MVP)

## Goal

Build an interactive web app that maps agent-thread events into a 3D semantic space so topic pivots appear as visible transitions between regions.

## Scope (MVP)

- Input data:
  - `zeta` session-analysis tables (`session_analysis_*.table.json`)
  - `zerg` cross-provider fixture sessions (Claude/Codex/Gemini)
- Processing pipeline:
  - Normalize events into `{thread_id, seq, role, text, timestamp}`
  - Generate embeddings (`text-embedding-3-small`, configurable dimensions)
  - 3D projection via UMAP
  - Region clustering via k-means
  - Pivot scoring via consecutive cosine-distance change points
- UI:
  - 3D scatter plot
  - trajectory polylines per thread
  - pivot sensitivity slider (quantile threshold)
  - color modes (cluster/thread/role/pivot)
  - point detail inspector

## Non-goals (MVP)

- No server backend required
- No incremental streaming updates yet
- No full summarization/compression loop yet

## Data contracts

Output dataset (`data/dataset.json`) includes:

- `points[]`: normalized events + projection + cluster + pivot metadata
- `threads[]`: per-thread summary and counts
- `clusters[]`: region labels and centroids
- `trajectories[]`: ordered point IDs per thread
- `stats` and `config`

## Design choices

- Embeddings:
  - Default `text-embedding-3-small` for cost and speed
  - Deterministic local fallback if key/rate-limit issues
- Pivot detection:
  - High-quantile cosine jump is simple, explainable, and tunable
- Projection:
  - UMAP chosen for better global+local readability in conversation maps

## Extension path

1. Add explicit change-point algorithms (e.g., PELT/Binseg) instead of pure quantiles.
2. Add hierarchical regions (macro topic -> subtopic).
3. Add relevance-graph edges for conversation compression and retrieval.
4. Add live ingest from shipper/Longhouse event streams.
