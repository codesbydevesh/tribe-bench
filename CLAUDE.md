# CLAUDE.md — Standing Orders for Every Session

This file is read automatically at the start of every Claude Code session.
It is the law. No exceptions unless the human explicitly overrides.

---

## What This Project Is

tribe-bench is a research toolkit and benchmark for Meta FAIR's TRIBE v2 brain encoding
model. We are building: a shared Python library (tribe_tools/), four research builds
(BrainLens, NeuroGenre, ScaleLaw, NeuroCheck), Kaggle/Colab notebooks, and aiming for
2-3 bioRxiv publications.

The operator is a solo researcher in India. No GPU locally. No university affiliation.
No credit card. All compute is borrowed (Kaggle T4, Colab, ZeroGPU, Lightning AI).

## First Action Every Session

Before doing ANYTHING, read `/home/deveshb/workspace/AI/tribe-bench/ops/war-room.md`.
It contains: current status of every workstream, what's blocked, and what to work on next.
If the human doesn't specify a task, the war room decides.

## Build & Test Commands

```bash
# Install the package
cd /home/deveshb/workspace/AI/tribe-bench && pip install -e .

# Verify imports
python -c "from tribe_tools import model, inference, atlas, viz, cache"

# Run BrainLens CLI help
python -m brainlens.cli --help

# Validate claims YAML
python -c "import yaml; yaml.safe_load(open('neurocheck/claims_db/claims.yaml'))"

# Run tests (when they exist)
pytest tests/ -v
```

## Non-Negotiable Rules

1. NEVER write code against TRIBE v2 internals without checking
   `ops/source-of-truth.md` first. If the fact you need is marked UNVERIFIED or
   missing, STOP and verify it from the actual source code before proceeding.

2. NEVER change a function signature without updating `ops/interface-contracts.md`
   first. The contract file is the single source of truth for all APIs.

3. ALL code must run on CPU without errors. GPU paths must be wrapped in
   try/except or if-guards that print a clear message:
   "GPU not available. To run inference, use Kaggle/Colab with a T4 GPU."

4. Every NeuroCheck claim must follow `ops/claims-protocol.md`. No shortcuts.
   A sloppy claim database destroys the entire NeuroCheck paper.

5. NEVER use placeholder data, dummy values, or TODO stubs in shipped code.
   If a function can't be implemented yet, raise NotImplementedError with a
   specific explanation of what's needed.

6. Do not add type annotations, docstrings, or comments to code you didn't change.
   Do not refactor adjacent code. Touch only what the task requires.

7. End every session by updating `ops/war-room.md` and `ops/decision-log.md`
   if any decisions were made. Follow `ops/session-protocol.md` closing checklist.

## Architectural Boundaries

```
tribe_tools/     → shared library. NO build-specific logic here.
                   If it's specific to BrainLens, it goes in brainlens/.
brainlens/       → depends on tribe_tools/ only.
neurogenre/      → depends on tribe_tools/ only.
scalelaw/        → depends on tribe_tools/ only.
neurocheck/      → depends on tribe_tools/ only.
notebooks/       → thin wrappers that call the library. Minimal logic.
```

No circular dependencies. No build imports from another build.
tribe_tools/ is the foundation. Everything else sits on top.

## Data Shapes (Memorize These — VERIFIED from source code)

- Brain surface: fsaverage5, vertex count = FSAVERAGE_SIZES["fsaverage5"] per hemi
- Vertex ordering: left hemisphere first (0 to N-1), right hemisphere (N to 2N-1)
- Model output: `np.ndarray` shape `(n_kept_segments, n_vertices)`, dtype float32
  IMPORTANT: n_kept_segments ≠ n_total_segments — empty segments are dropped
- predict() returns: `(preds, all_segments)` tuple
- Atlas: HCP-MMP1 via MNE (NOT nilearn), 360 regions (180 per hemisphere)
- Encoders: V-JEPA 2 (vision), LLaMA 3.2-3B (text), Wav2Vec-BERT (audio)
  These are EXTERNAL extractors, NOT part of the brain model.
  They load sequentially, cache to disk, and free GPU automatically.

## Known Pitfalls (From Source Code Reading)

- Encoders are external extractors. Do NOT try to manually load/unload them.
  TRIBE v2's pipeline handles sequential loading via `_free_extractor_model()`.
- LLaMA 3.2-3B requires HuggingFace auth token. Tests must mock this.
- predict() returns (preds, segments). preds shape is (n_kept_segments, n_vertices),
  NOT (n_timesteps, 20484). Segments with no events are removed by default.
- Atlas uses MNE (`mne.datasets.fetch_hcp_mmp_parcellation()`), NOT nilearn.
  MNE atlas fetch needs internet on first run. Cache via lru_cache (already done
  in tribev2/utils.py).
- Modality ablation: `features_to_mask` does NOT work at inference time (only
  at model construction). Our code mutates `features_to_use` via
  `_find_features_to_use()` discovery. Exact attribute path on TribeModel is
  unverified (G018) — needs GPU test. If it fails, check `dir(model)` and
  `dir(model.data)` to find the correct path.
- Kaggle sessions timeout after 12 hours. HDF5 checkpointing is mandatory
  for any batch job. Save every 10 clips minimum.
- ZeroGPU gives 3.5 min/day. That's ~1 inference pass on a short clip.
  Pre-plan every second. See `ops/compute-playbook.md`.
- neuralset/neuraltrain pip availability is UNVERIFIED (G016). These are
  required dependencies — first GPU session must confirm they install.

## File Reference

- Full project plan: `PLAN.md`
- Operational system: `ops/` directory (10 files, read the README there)
- Research foundation: `/home/deveshb/workspace/AI/tribe-v2/` (13 docs, read-only)
- TRIBE v2 source: `/home/deveshb/workspace/AI/tribev2-source/` (Meta's code, read-only reference)

## When You're Unsure

Read the relevant ops/ file. If still unsure, ask the human. Do not guess.
A wrong guess costs more than a question.
