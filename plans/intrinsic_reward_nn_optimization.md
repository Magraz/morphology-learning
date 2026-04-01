# Intrinsic Reward — Nearest Neighbor Optimization

The current `_compute_intrinsic_reward` uses `torch.cdist` against the full episodic memory,
which scales O(n) per query as memory grows. Options for improving this:

## 1. Capped Ring Buffer (simplest)
- Fixed-size buffer, overwrite oldest entries.
- Keeps `cdist` cost constant.
- Trades off losing old states, but recent history matters most in practice.

## 2. `scipy.spatial.cKDTree`
- Classic spatial index with O(log n) lookups.
- Works well when `obs_dim < ~20`. Degrades toward brute-force in high dimensions.
- Rebuild the tree periodically as memory grows.

## 3. FAISS
- Facebook's library for fast nearest-neighbor search in high-dimensional spaces.
- `IndexFlatL2`: exact results, much faster than `cdist` at scale.
- `IndexIVFFlat`: approximate results, even faster.
- GPU-accelerated. Standard choice in NGU/Agent57-style episodic curiosity.

## 4. PyNNDescent / Annoy
- Approximate NN libraries. Good throughput.
- Less ecosystem support than FAISS, no GPU path.

## Recommendation
- If `obs_dim < 20`: capped ring buffer + `cKDTree`.
- If `obs_dim` is larger or GPU-native performance is needed: FAISS `IndexFlatL2`.
