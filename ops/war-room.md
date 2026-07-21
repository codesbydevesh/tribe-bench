# War Room — Project Command Center

Last updated: 2026-07-21 (Session 5)
Updated by: Session 5 — DOI audit + fixes, smoke-test notebook hardened, run environment set up

> **PICK UP HERE (next session): the run environment is fully set up — just RUN it.**
> Open the Quick-Saved Kaggle notebook → confirm GPU T4 x2 + Internet On + the `HF_TOKEN`
> secret is attached → **Run All** (~15-30 min) → paste back the final RECORD block +
> the step-8 ABLATION VERDICT. First confirm the LLaMA-3.2 access shows approved (not
> pending) on the HF model page. This single run closes G016/G005/G018.

---

## Overall Status: RESUMED AFTER ~6 WEEKS DORMANT — RESHAPED TO NeuroCheck-FIRST

All CPU-producible code and content is done (~1,413 LOC + the 50-claim DB), but
**nothing has ever run on a GPU** and three of four builds are still stubs. A full
strategic re-assessment on 2026-07-21 (see `ops/assessment-2026-07-21.md`) reshaped
scope. Headlines:

- **G016 (existential dep risk) is effectively DEAD** — `neuralset`/`neuraltrain`/`exca`
  are real public Meta PyPI wheels (0.0.2, `py3-none-any`), not private libs. Confirm at
  first `pip install`; `neuralset 0.0.2` needs Python >=3.12.
- **Top live risk is now G018** — modality ablation has no supported inference-time API
  upstream (`tribev2/model.py:190` zeros only absent-from-batch modalities; `:212` is
  training-only dropout). BrainLens depends on it. The <=1hr Kaggle smoke test kills or
  confirms this.
- **Scope reshaped: NeuroCheck-first.** Ship the DOI-verified benchmark dataset as a
  bioRxiv/HF resource. **NeuroGenre and ScaleLaw are CUT.** BrainLens kept only if the
  ablation test passes. Standout follow-on = an MCP "neural-engagement" tool (idea #1).
- The 50-claim DB commit (`ddb57cf`) is now **pushed** to `codesbydevesh/tribe-bench`.

**Next milestone:** the <=1hr GPU smoke test on Kaggle (install + one predict + the
ablation kill/confirm + per-extractor VRAM). See assessment §"Biggest live risk".

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
**Status:** CUT (2026-07-21, D012) — deprioritized indefinitely. 35% chance genres
don't cluster; not worth the risk vs the NeuroCheck-first focus.
**Next action:** none. Revisit only after a NeuroCheck preprint ships.

### ScaleLaw (Build 3)
**Status:** CUT (2026-07-21, D012) — deprioritized indefinitely. StudyForrest
fsaverage5 alignment + HRF timing is a ~30%-fail swamp.
**Next action:** none. Revisit only after a NeuroCheck preprint ships.

### NeuroCheck (Build 4 — Flagship)
**Status:** 50/50 claims written. **DOIs now machine-verified against CrossRef (2026-07-21,
D015): 17 claims had broken or mis-attributed citations — all fixed; 50/50 resolve + match,
zero duplicates.** Verification tooling in `scripts/` (verify/resolve/patch). Category
validation in claims.py.
**Blocked by:** GPU for the benchmark scoring run.
**Next action:** Write the scoring pipeline (only real code left for the flagship) after the
smoke test; ship the resource paper + HF Dataset (DOI report as supplement).

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
**Status:** NOT STARTED (but 50 claims are ready — design paper is now unblocked)
**Next action:** Draft NeuroCheck benchmark design paper outline

---

## What To Work On Next (Priority Order — reshaped 2026-07-21)

1. ~~Write project skeleton~~ DONE
2. ~~Write tribe_tools/ shared library~~ DONE
3. ~~Write BrainLens MVP~~ DONE
4. ~~Start NeuroCheck claims database (first 20 claims)~~ DONE
5. ~~Write Kaggle notebook skeleton~~ DONE
6. ~~Complete 50 claims (30 more)~~ DONE
7. ~~Three Musketeers review + fix 11 DOIs, 5 claims replaced~~ DONE
8. ~~Push the 50-claim DB commit to GitHub (backup)~~ DONE 2026-07-21
9. **Run the <=1hr GPU smoke test on Kaggle** — THE gate for everything. Notebook is
   hardened (Fable source-review; 4 run-killers fixed; per-extractor VRAM + no-WhisperX
   fallback added) and Quick-Saved on Kaggle. Environment READY: Kaggle GPU unlocked,
   HF account + LLaMA-3.2 access requested + `HF_TOKEN` secret attached, repo public so
   it clones directly. Only action left = open notebook → Run All → paste results.
10. **CrossRef-verify all 50 DOIs programmatically + de-dupe the 2 collisions** (CPU,
    no GPU — can do tonight). Makes the DB bulletproof and becomes a paper supplement.
11. Get HF gated approval for LLaMA 3.2-3B (G011 — one click).
12. Write the NeuroCheck scoring pipeline (only real code left for the flagship).
13. Ship the NeuroCheck resource paper (bioRxiv) + HF Dataset — before model scoring.
14. BrainLens on ZeroGPU drip demo — ONLY if step 9 confirms ablation works.

---

## Blockers

| Blocker | Impact | Resolution |
|---------|--------|------------|
| ~~Haven't read TRIBE v2 source code~~ | ~~Can't write model.py~~ | **RESOLVED 2026-06-08.** Source read, source-of-truth.md populated. |
| ~~features_to_mask API path unclear~~ | ~~Blocks BrainLens~~ | **PARTIALLY RESOLVED 2026-06-08.** Code uses `_find_features_to_use()` discovery. G017 closed. G018 still open — needs GPU to confirm the ablation path. NOW THE TOP LIVE RISK. |
| ~~neuralset/neuraltrain/exca pip availability unknown~~ | ~~Can't test on GPU~~ | **RESOLVED 2026-07-21 (G016 closed).** All three are public Meta PyPI wheels (0.0.2, `py3-none-any`). Confirm clean install on Kaggle; `neuralset` needs Python >=3.12. |
| No GPU access tested yet | Can't validate inference (G018, G005) | Run the <=1hr smoke test on Kaggle as the first GPU task. THE gate. |
| No HuggingFace auth token | Can't load LLaMA 3.2 (G011) | Set up HF account, request LLaMA 3.2-3B gated access (one click). |

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
| 2026-06-09 | Wrote 30 claims (NC021-NC050), completed 50/50 | 8 category targets hit | claims.yaml |
| 2026-06-09 | Three Musketeers review #2: found 11 wrong DOIs, 5 infeasible claims | D011: DOI verification required, category renamed multisensory→multimodal, 5 claims replaced with stimulus-driven alternatives | claims.yaml, claims.py, decision-log.md |
| 2026-07-21 | Resumed after ~6 weeks dormant. Deep-reasoning strategic re-assessment. Pushed the 50-claim DB commit to GitHub. | D012 (reshape NeuroCheck-first; cut NeuroGenre/ScaleLaw), D013 (smoke-test-first gate), D014 (MCP neural-engagement tool as standout direction). G016 closed (deps are public wheels). G018 elevated to top risk. TRIBE v2 dated to May 2026 (~2mo old), weakening the first-mover thesis. | assessment-2026-07-21.md, war-room.md, decision-log.md, knowledge-gaps.md, progress.md |
| 2026-07-21 (Session 5) | CrossRef DOI audit: 17 of 50 claims had broken/mis-attributed citations — all fixed, 50/50 now resolve+match (D015). Built + hardened the smoke-test notebook (Fable source-review found 4 run-killers; from_pretrained + features_to_use ablation confirmed correct vs source; per-extractor VRAM + no-WhisperX fallback added; zero wrapper fixes needed). Made repo public for Kaggle (D016). Set up the full run environment (Kaggle GPU, HF account + LLaMA access + HF_TOKEN secret, notebook Quick-Saved). | D015 (DOI audit), D016 (repo public). Next = Run All on Kaggle. | scripts/verify_dois.py, resolve_dois.py, patch_dois.py, doi_verification_report.md, claims.yaml, notebooks/01_setup_test.ipynb, decision-log.md, war-room.md, progress.md |
