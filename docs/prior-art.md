# Prior Art and Technique Notes

## Existing patterns worth reusing

1. Trajectory maps in embedding space
- Visualizing points plus line trajectories captures semantic drift better than static clusters.
- Implementation pattern: project to 2D/3D, color by region, overlay time-ordered paths.

2. UMAP for local+global structure tradeoff
- UMAP is usually easier than t-SNE for preserving both local neighborhoods and broader manifold shape.
- Key knobs: `n_neighbors` (local vs global structure) and `min_dist` (cluster tightness).

3. Cluster labeling from projected neighborhoods
- BERTopic-style workflow (UMAP + clustering + topic terms) gives practical region names.
- For MVP we can use simple token scoring rather than full transformer topic models.

4. Pivot detection as change-point signal
- A robust MVP heuristic: cosine distance between consecutive turn embeddings, with a high quantile cutoff.
- Future upgrade: explicit change-point algorithms (PELT/Binseg) from `ruptures`-style formulations.

5. Embedding model constraints
- OpenAI `text-embedding-3-small` is low-cost and supports reduced dimensions; suitable for first pass.
- Use one embedding model consistently for a dataset snapshot.

## Sources

- t-SNE paper (JMLR): https://www.jmlr.org/papers/v9/vandermaaten08a.html
- UMAP docs (parameter effects): https://umap-learn.readthedocs.io/en/latest/parameters.html
- BERTopic parameter tuning: https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html
- OpenAI embeddings model docs: https://platform.openai.com/docs/guides/embeddings
- TensorBoard Embedding Projector: https://projector.tensorflow.org/
- `ruptures` docs (change-point detection): https://centre-borelli.github.io/ruptures-docs/
