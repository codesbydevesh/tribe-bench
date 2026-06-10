# Progress Tracker — Pit Stops

---

## Overall Completion: 45%

```
[█████████░░░░░░░░░░░] 45%
```

---

## The Full Map (weighted by importance to the end goal)

| # | Milestone | Weight | Status | Done? |
|---|-----------|--------|--------|-------|
| 1 | Ops & planning system | 5% | All 14 files written and updated | YES |
| 2 | Source code reading & verification | 5% | Cloned, read, source-of-truth populated | YES |
| 3 | Project skeleton (pyproject, gitignore, README) | 3% | pyproject.toml, .gitignore, README.md, all dirs | YES |
| 4 | tribe_tools/ shared library (6 modules) | 12% | All 6 modules written, imports verified | YES |
| 5 | BrainLens MVP (inference, attribution, viz, CLI) | 8% | 4 modules + CLI, all tested | YES |
| 6 | NeuroCheck claims database (50 claims) | 12% | 50/50 claims written and validated | YES |
| 7 | GPU smoke test (Kaggle T4) | 5% | — | NO |
| 8 | BrainLens demo on Kaggle (first brain map) | 5% | — | NO |
| 9 | BrainLens on HuggingFace ZeroGPU | 5% | — | NO |
| 10 | NeuroGenre batch inference + UMAP figure | 8% | — | NO |
| 11 | ScaleLaw replication (StudyForrest) | 10% | — | NO |
| 12 | NeuroCheck benchmark run (50 claims scored) | 10% | — | NO |
| 13 | Paper 1: NeuroCheck benchmark (bioRxiv) | 7% | — | NO |
| 14 | Paper 2: ScaleLaw / NeuroGenre (bioRxiv) | 5% | — | NO |

**Total: 100%**

---

## Pit Stops (Session Goals)

### Session 1 (2026-06-08) — COMPLETED
**Goal:** Build the entire ops system + read source code
**Delivered:** 14 ops files, Three Musketeers, source code analysis, all contracts defined
**Progress:** 0% → 10%

### Session 2 (2026-06-08 continued) — COMPLETED
**Goal:** Build everything that doesn't need a GPU
- [x] Project skeleton (#3): pyproject.toml, .gitignore, README.md, directory structure
- [x] tribe_tools/ (#4): all 6 modules (model.py, inference.py, atlas.py, viz.py, cache.py, video_utils.py)
- [x] BrainLens MVP (#5): inference.py, attribution.py, visualization.py, cli.py, __main__.py
- [x] NeuroCheck claims (#6): 20/50 claims written and validated
- [x] Kaggle notebook (#7 prep): 01_setup_test.ipynb
- [x] neurocheck/claims.py: claim loader + validator
- [x] All imports verified, CLI tested, cache roundtrip tested, atlas tested (181 regions, V1=523 vertices)
- [x] Confirmed fsaverage5 = 10,242 vertices/hemi, 20,484 total

**Result:** 10% → 33% (target was 38%, missed because claims are 20/50 not 50/50)

**37 files total in project**

**Bonus:** Three Musketeers review found 11 bugs (1 showstopper). All fixed:
- P0: predict_single() would crash (wrong attribute path) — rewrote with discovery
- P1-P4: device not passed, export_html half-broken, CUDA OOM masked, cache key collisions
- P5-P6: All 20 claims missing atlas+source fields, 4 wrong DOIs
- P7-P9: Dead import, missing lru_cache, wrong attribution fallback
- P10-P11: Ops docs synced

### Session 3 (2026-06-09) — IN PROGRESS
**Goal:** Finish claims + first GPU test
- [x] Complete 50 claims (#6): 30 new claims (NC021-NC050), all 8 categories hit targets
- [x] Three Musketeers review #2: found 11/30 wrong DOIs (37% error rate), 5 infeasible claims
- [x] Fixed all 11 DOIs (verified via CrossRef/web search)
- [x] Replaced 5 cognitive-state claims with stimulus-driven alternatives
- [x] Fixed 4 wrong journal names (NC026, NC034, NC036, NC037)
- [x] Fixed NC022 region (V3→V3A), NC038 first author (Rolls→Kringelbach), NC044 (review→empirical)
- [x] Fixed NC040 source (IAPS→OASIS), renamed multisensory→multimodal category
- [x] Added category validation to claims.py
- [ ] Run smoke test on Kaggle (#7)
- [ ] Fix any issues found on GPU

**Progress so far:** 33% → 45% (claims complete + reviewed, GPU test remaining)
**Target:** 45% → 50%

### Session 4
**Goal:** First real results
- [ ] BrainLens demo on Kaggle (#8) — first brain map ever generated
- [ ] Deploy to HuggingFace (#9)

**Target:** 50% → 60%

### Session 5-6
**Goal:** Batch science
- [ ] NeuroGenre 60-clip batch (#10)
- [ ] NeuroCheck run on GPU (#12)

**Target:** 60% → 78%

### Session 7-8
**Goal:** Papers
- [ ] ScaleLaw replication (#11)
- [ ] Write and submit papers (#13, #14)

**Target:** 78% → 100%

---

## Rules

- Update this file at the END of every session
- Check off completed items
- If a session doesn't hit its target, note WHY and adjust the next session
- Never move the goalposts — if something takes longer, it takes longer. Track honestly.
