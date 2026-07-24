# tests/

CPU-only tests that run anywhere (no GPU, no `torch`, no `tribev2`). Run with:

```bash
pytest            # from the repo root
```

## What's covered

- `test_cache.py` — `PredictionCache` round-trip, overwrite, and cross-session
  persistence (the durability that resume support depends on).
- `test_cache_keys.py` — cache-key determinism, mask-sensitivity, HDF5-safety.
- `test_claims.py` — the NeuroCheck DB loads, validates clean, no duplicate IDs.

## Deliberately not here yet (need a GPU / the model)

These are tracked in `.notes/plans/corticall/ROADMAP.md`, not stubbed as
failing tests:

- The ablation restore-in-`finally` path in `tribe_tools/model.py` (needs a
  loaded model or a mock of `TribeModel.data.features_to_use`).
- Output-shape invariants on `predict()` — `(n_kept_segments, n_vertices)`.
- The exact-Shapley efficiency identity (`Σ attributions = full − empty`), once
  the correlation heuristic in `brainlens/attribution.py` is replaced.
