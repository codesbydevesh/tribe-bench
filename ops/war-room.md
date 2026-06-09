# War Room — Project Command Center

Last updated: 2026-06-08
Updated by: Post Three Musketeers review & fix session

---

## Overall Status: PHASE 1 — CODE COMPLETE (CPU), AWAITING GPU TEST

All code that can be written without a GPU is done. 37 project files exist.
tribe_tools/ (6 modules), BrainLens MVP (5 files), NeuroCheck claims (20/50),
Kaggle notebook, and full ops system. Three Musketeers review found 11 bugs —
all fixed and verified.

**Next milestone:** GPU smoke test on Kaggle (Session 3).

---

## Workstream Status

### tribe_tools/ (Shared Library)
**Status:** COMPLETE (CPU-verified). 6 modules written, all imports pass.
**Blocked by:** GPU for end-to-end test
**Next action:** Run on Kaggle T4

| Module | Status | Notes |
|--------|--------|-------|
| model.py | Done | Thin wrapper + `_find_features_to_use()` for ablation. Needs GPU test (G018). |
| inference.py | Done | Batch predict with HDF5 checkpointing, CUDA OOM handling, hash-based cache keys |
| atlas.py | Done | MNE-based HCP-MMP1 wrapper with lru_cache, 181 regions verified |
| viz.py | Done | Brain surface plots, RGB attribution, export_html (both hemispheres) |
| cache.py | Done | HDF5 cache with segment-loss documented |
| video_utils.py | Done | ffmpeg wrappers |

### BrainLens (Build 1)
**Status:** MVP COMPLETE (CPU). 5 files: inference.py, attribution.py, visualization.py, cli.py, __main__.py
**Blocked by:** GPU for first brain map
**Next action:** Run on Kaggle with a test video

### NeuroGenre (Build 2)
**Status:** NOT STARTED
**Blocked by:** tribe_tools/ GPU validation + clip manifest
**Next action:** After BrainLens demo on GPU

### ScaleLaw (Build 3)
**Status:** NOT STARTED
**Blocked by:** StudyForrest data download + GPU
**Next action:** After NeuroGenre

### NeuroCheck (Build 4 — Flagship)
**Status:** 20/50 claims written. Schema validated with atlas + source fields.
**Blocked by:** Nothing. Remaining 30 claims are zero-compute.
**Next action:** Write 30 more claims (Session 3)

### Compute Access
**Status:** NO APPLICATIONS SUBMITTED
**Next action:** Kaggle and Colab are already available (free tier). Test there first.

| Platform | Applied? | Status | Notes |
|----------|----------|--------|-------|
| Kaggle T4 | No need | Available | Free tier, 30 hrs/week |
| Google Colab | No need | Available | Free T4, 12hr sessions |
| HuggingFace ZeroGPU | No | — | 3.5 min/day, H200 |
| Lightning AI | No | — | 80 GPU hrs/month |

### Outreach
**Status:** NOT STARTED
**Blocked by:** Need a working demo (BrainLens) or at least a published repo
**Next action:** After BrainLens runs on GPU

### Papers
**Status:** NOT STARTED
**Next action:** NeuroCheck benchmark design paper can start after 50 claims curated

---

## What To Work On Next (Priority Order)

1. ~~Write project skeleton~~ DONE
2. ~~Write tribe_tools/ shared library~~ DONE
3. ~~Write BrainLens MVP~~ DONE
4. ~~Start NeuroCheck claims database (first 20 claims)~~ DONE
5. ~~Write Kaggle notebook skeleton~~ DONE
6. Complete 50 claims (30 more)
7. Run GPU smoke test on Kaggle
8. Fix any issues found on GPU (especially G018 — features_to_use path)

---

## Blockers

| Blocker | Impact | Resolution |
|---------|--------|------------|
| ~~Haven't read TRIBE v2 source code~~ | ~~Can't write model.py~~ | **RESOLVED 2026-06-08.** Source read, source-of-truth.md populated. |
| ~~features_to_mask API path unclear~~ | ~~Blocks BrainLens~~ | **PARTIALLY RESOLVED 2026-06-08.** Code uses `_find_features_to_use()` discovery. G017 closed (names are "video","audio","text"). G018 still open — needs GPU to confirm attribute path. |
| neuralset/neuraltrain pip availability unknown | Can't test on GPU | Try pip install on Kaggle (G016). First GPU session priority. |
| No GPU access tested yet | Can't validate inference | Run smoke test on Kaggle as first GPU task |
| No HuggingFace auth token | Can't load LLaMA 3.2 | Set up HF account, request access |

---

## Session History

| Date | What was done | Key decisions | Files changed |
|------|--------------|---------------|---------------|
| 2026-06-08 | Created PLAN.md and ops/ system (10 files) | Monorepo structure, 4 builds, HDF5 caching, CC BY-NC scope | PLAN.md, CLAUDE.md, ops/* |
| 2026-06-08 | Created Three Musketeers decision protocol | Use general-purpose agents, 6 task types, sequential flow | ops/three-musketeers/three-musketeers.md |
| 2026-06-08 | Cloned and read TRIBE v2 source code | Encoders are external (no custom loader needed), atlas uses MNE not nilearn, output shape is (n_kept_segments, n_vertices) | ops/source-of-truth.md |
| 2026-06-08 | Updated PLAN.md and all ops files with source code findings | Removed Phase 2 T4 hack, corrected all API contracts and shapes, closed 6 knowledge gaps, added 3 new gaps | PLAN.md, ops/interface-contracts.md, ops/knowledge-gaps.md, ops/pre-mortem.md, ops/compute-playbook.md, ops/war-room.md, ops/decision-log.md |
| 2026-06-08 | Built all CPU code: skeleton, tribe_tools/ (6), BrainLens (5), NeuroCheck claims (20), Kaggle notebook | pyproject.toml build-backend fix, fsaverage5=10242/hemi confirmed | 37 files created |
| 2026-06-08 | Three Musketeers review: found 11 bugs, fixed all | D009: features_to_use mutation for ablation, similarity-based attribution fallback, HDF5-safe cache keys | model.py, inference.py, viz.py, cache.py, atlas.py, attribution.py, claims.py, claims.yaml, interface-contracts.md, knowledge-gaps.md |
