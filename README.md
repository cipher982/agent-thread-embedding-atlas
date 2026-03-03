# Agent Thread Embedding Atlas

Interactive 3D visualization of agent conversation threads, focused on semantic pivots and region transitions.

## What it does

- Parses minimal real data from:
  - `~/git/zeta/OLD_omnisearch/.../session_analysis_*.table.json`
  - `~/git/zerg/apps/zerg/backend/tests/integration/fixtures/*`
- Embeds each event text (`text-embedding-3-small`, configurable) with local deterministic fallback.
- Projects vectors to 3D (UMAP), clusters regions (k-means), and flags pivots via embedding-distance change points.
- Renders an interactive web UI with:
  - 3D scatterplot + thread trajectories
  - pivot sensitivity controls
  - thread filtering and drill-down point details

## Quick start

```bash
bun install
bun run build:dataset
bun run dev
```

Open the printed local URL (default Vite behavior).

## Deploy (Coolify + Infisical)

1. Publish this folder to GitHub.
2. Create a Coolify app from that repo (Dockerfile build pack).
3. Keep all deployment secrets in Infisical and sync into Coolify env vars.
4. Deploy (not restart) after env changes.

Runtime for this app is static and does not require secrets by default, but deployment secrets should still be sourced from Infisical rather than `.env` files.

## Dataset build options

```bash
bun run build:dataset -- --help
```

Useful flags:

- `--max-zeta-files 22`
- `--max-points 600`
- `--embedding-model text-embedding-3-small`
- `--dimensions 256`
- `--pivot-quantile 0.85`
- `--force-local` (skip OpenAI API calls)

Output file defaults to `data/dataset.json`.

## Notes

- The builder uses `OPENAI_API_KEY` when available.
- If OpenAI embedding fails, it falls back to deterministic local hash embeddings so the app still runs.
- Prior-art notes and references are in `docs/prior-art.md`.
