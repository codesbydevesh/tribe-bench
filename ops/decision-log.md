# Decision Log — The Record of Why

Every significant decision is recorded here with its reasoning.
This prevents re-litigating settled decisions in future sessions.
If a decision needs to be revisited, add a new entry — don't edit the old one.

---

## How to Use

When a decision is made during a session, add an entry with:
- **Date**
- **Decision** — what was decided
- **Alternatives considered** — what else was on the table
- **Reasoning** — why this option was chosen
- **Implications** — what this means for other parts of the project
- **Revisit if** — conditions under which this should be reconsidered

---

## Decisions

### D001: Monorepo structure (2026-06-08)

**Decision:** Single monorepo `tribe-bench/` containing shared library + all 4 builds.

**Alternatives:**
- Separate repos per build (7 repos)
- Shared library as separate published package

**Reasoning:** Single citation target ("cite tribe-bench"), simpler dependency
management, one pip install gets everything, matches the strategist's recommendation
from musketeer-3-strategist.md. Solo researcher maintaining 7 repos is overhead.

**Implications:** All builds share the same version. Changes to tribe_tools/ affect
all builds simultaneously. pyproject.toml manages the whole thing.

**Revisit if:** A collaborator wants to use tribe_tools/ without the builds.
In that case, consider publishing tribe_tools/ as a separate PyPI package.

---

### D002: Glasser HCP-MMP1 atlas (2026-06-08)

**Decision:** Use Glasser atlas (360 regions) as the primary atlas for all ROI work.

**Alternatives:**
- Destrieux atlas (nilearn built-in, 150 regions, easier to access)
- Desikan-Killiany (FreeSurfer default, 68 regions)
- Yeo 7-network only (coarse but simple)

**Reasoning:** Glasser is the highest-resolution publicly available cortical atlas.
360 regions gives us precision for NeuroCheck claims. Neuroscience literature
increasingly uses Glasser. nilearn supports it.

**Implications:** Need to verify nilearn can fetch Glasser for fsaverage5 surface.
If not, may need to manually project from the HCP release.

**Revisit if:** nilearn doesn't support Glasser on fsaverage5 cleanly. Fallback
to Destrieux for initial implementation, Glasser for paper.

---

### D003: Sequential encoder loading as primary T4 strategy (2026-06-08)

**Decision:** Load one encoder at a time (sequential mode) as the primary strategy
for running on T4 GPUs.

**Alternatives:**
- Dual-T4 sharding (split across two GPUs)
- CPU offloading (keep model in RAM, move to GPU per layer)
- 4-bit quantization of all encoders

**Reasoning:** Sequential loading has ~50% success probability per musketeer-5-hacker.md
analysis. Peak VRAM ~10GB for V-JEPA 2 in FP16, fits T4 with margin. Works on ANY
single T4 (Colab, Lightning, Kaggle single GPU). Dual-T4 is backup.

**Implications:** Need to verify encoders can be isolated (see source-of-truth.md
verification queue). Inference is slower (3-4x due to load/unload overhead).
This is acceptable for research workloads.

**Revisit if:** Source code reading reveals encoders cannot be isolated. Or if
dual-T4 proves simpler than expected.

---

### D004: HDF5 for prediction caching (2026-06-08)

**Decision:** Use HDF5 (h5py) for caching predictions between sessions.

**Alternatives:**
- SQLite (simpler queries, worse for large arrays)
- NumPy .npz files (one file per prediction, many small files)
- Pickle (fast but fragile across Python versions)

**Reasoning:** HDF5 handles large numerical arrays efficiently. Supports partial
reads (read one prediction without loading all). Standard in neuroimaging (nilearn,
nibabel use it). Single file = easy to transfer between platforms.

**Implications:** h5py added as dependency. Need corruption handling (partial writes
if session dies mid-save).

**Revisit if:** HDF5 file corruption becomes a real problem. Fallback to one .npz
per prediction with a manifest JSON.

---

### D005: NeuroCheck as zero-compute first deliverable (2026-06-08)

**Decision:** Start curating the claims database immediately, in parallel with code.
The benchmark design paper is publishable without any model results.

**Alternatives:**
- Code first, claims later
- BrainLens demo first (needs GPU)

**Reasoning:** From musketeer-4-realist.md and musketeer-6-architect.md: 70% of project
value is zero-compute work. The claims database is the intellectual moat. Nobody else
will spend weeks curating 50 neuroscience findings. The GLUE benchmark paper was published
before any model was evaluated on it. Same model applies here.

**Implications:** Claims curation runs parallel to all coding work. Need to allocate
time explicitly for it (not just "when code is done").

**Revisit if:** Never. This is the right call regardless of compute situation.

---

### D006: CC BY-NC 4.0 scope (2026-06-08)

**Decision:** TRIBE v2's CC BY-NC 4.0 license applies to the model weights and
predictions. Our code (tribe-bench) will be licensed MIT. The claims database
will be CC BY 4.0 (allowing commercial use of the benchmark design).

**Alternatives:**
- License everything CC BY-NC to match
- License everything MIT

**Reasoning:** Our code is original work, not a derivative of Meta's model weights.
The claims database is curated from published literature, also not a derivative.
MIT for code maximizes adoption. CC BY for claims maximizes benchmark usage.
Only the model predictions themselves fall under Meta's license.

**Implications:** Document this clearly in README. Users who run the model must
comply with Meta's license. Users who use our benchmark design need not.

**Revisit if:** Legal review suggests the code is a derivative work (unlikely
given it's an independent wrapper).

---

### D007: Sequential loading not needed — Meta already handles it (2026-06-08)

**Decision:** Remove the custom sequential encoder loading strategy. TRIBE v2's
pipeline already loads each extractor one at a time, caches features to disk,
and frees GPU memory before loading the next one.

**Alternatives:**
- Proceed with custom sequential loading (redundant)
- Implement dual-T4 sharding (unnecessary now)

**Reasoning:** Reading the actual source code (main.py:59-79, main.py:214-217)
revealed that `_free_extractor_model()` deletes the encoder from GPU and calls
`gc.collect()` + `torch.cuda.empty_cache()` after each extractor runs. The
pipeline already does exactly what we planned to implement. Our model.py
becomes a thin wrapper around `TribeModel.from_pretrained()` instead of a
complex sequential loader.

**Implications:**
- Phase 2 "Engineering the T4 Hack" removed from PLAN.md
- model.py is dramatically simpler (thin wrapper, not custom loader)
- `load_model(mode="sequential")` API removed — no mode parameter needed
- T4 feasibility still depends on G005 (actual VRAM per extractor)
- All 4 research builds remain valid and needed
- Removed `notebooks/02_sequential_loading.ipynb`

**Revisit if:** Never. This is a fact about the source code, not an opinion.

---

### D008: Atlas implementation uses MNE, not nilearn (2026-06-08)

**Decision:** atlas.py wraps TRIBE v2's own MNE-based atlas functions
(`get_hcp_labels()`, `get_hcp_roi_indices()`, `summarize_by_roi()`) instead
of implementing our own via nilearn.

**Alternatives:**
- Use nilearn's atlas functions (wrong library — TRIBE v2 uses MNE)
- Implement from scratch (wasteful — TRIBE v2 already has these)

**Reasoning:** Source code (utils.py:214-318) shows TRIBE v2 already has
complete HCP-MMP1 atlas integration via MNE. Using their functions ensures
vertex indices match their output format exactly. No alignment bugs.

**Implications:** MNE is a required dependency (already required by tribev2).
The `GlasserAtlas` wrapper class from the old contract is replaced by
direct function calls wrapping TRIBE v2's utils.

**Revisit if:** Never. Using the same atlas code as the model ensures consistency.

---

### D009: Three Musketeers review — modality ablation approach (2026-06-08)

**Decision:** Use `features_to_use` mutation for runtime modality ablation,
discovered via `_find_features_to_use()` probing 4 candidate attribute paths.

**Alternatives:**
- `features_to_mask` config mutation (broken — only works at model construction, not inference)
- `config_update` parameter on `from_pretrained()` (correct but requires reloading the model 4x, too slow)
- Remove modality events from the DataFrame before `predict()` (clean but we don't know the DF structure)

**Reasoning:** Three Musketeers review (Athos/Porthos/Aramis) found that the
original code accessed `model.xp.cfg.data.features_to_mask` which doesn't exist
on TribeModel and would crash. `features_to_mask` only affects model construction
(projector creation), not inference. Mutating `features_to_use` controls which
extractors run; missing modalities get zeros substituted (model.py:189-192).
4 candidate paths are probed at runtime since the exact attribute path is unverified.

**Implications:** BrainLens ablation now has a working code path that needs GPU
testing (G018). If none of the 4 paths work on the actual model, the error
message directs to G018 for debugging.

**Revisit if:** GPU test reveals none of the probed paths work. Then investigate
the actual TribeModel attribute structure and update.

---

### D010: HDF5-safe cache keys via path hashing (2026-06-08)

**Decision:** Cache keys use `{video_stem}_{sha256_prefix}_{mask_key}` format
instead of raw absolute paths.

**Alternatives:**
- Use filename only (collision risk for same-named files in different dirs)
- Use full absolute path (HDF5 interprets `/` as group separators, creating nested groups)
- Use URL-encoded path (ugly but works)

**Reasoning:** HDF5 treats `/` as path separators, so cache key
`/home/user/video.mp4_full` silently creates nested groups instead of a flat key.
The hash approach gives unique, flat, human-readable keys.

**Revisit if:** Never. This is a correctness fix.

---

### D011: Claims batch 2 quality review and DOI verification (2026-06-09)

**Decision:** All claim DOIs must be verified against CrossRef/doi.org before
acceptance. LLM-generated DOIs have a ~37% hallucination rate (11/30 wrong in
batch 2). Category "multisensory" renamed to "multimodal" for consistency.
5 cognitive-state-dependent claims dropped and replaced with stimulus-driven ones.

**Alternatives:**
- Accept LLM-generated DOIs as-is (37% error rate found)
- Manually curate all claims without LLM assistance (too slow)

**Reasoning:** Three Musketeers review of batch 2 (NC021-NC050) found 11/30 DOIs
pointing to wrong papers. Root cause: LLMs generate plausible DOI formats but
can't reliably recall exact article identifiers. The DOI prefix (publisher code)
was usually correct but the article-specific suffix was fabricated.

Additionally, 5 claims (NC030, NC046, NC048, NC049, NC050) required active
cognitive tasks (verb generation, oddball detection, self-referential judgment,
rest-vs-task, n-back) that TRIBE v2 cannot test because it processes passive
video/audio/text stimuli, not task-evoked brain states.

**Implications:**
- Future claims must include DOI verification step
- Category validation added to claims.py
- All 50 claims now have verified DOIs and are testable with passive stimuli

**Revisit if:** CrossRef API integration is added to validation pipeline.

---

### D012: Reshape to NeuroCheck-first; cut NeuroGenre and ScaleLaw (2026-07-21)

**Decision:** Narrow the project from four builds to one primary deliverable — NeuroCheck
as a published benchmark dataset. Cut NeuroGenre and ScaleLaw indefinitely. Keep BrainLens
only if the GPU smoke test confirms modality ablation works (D013).

**Alternatives:**
- Keep all four builds (the original "standard toolkit" empire)
- Kill the project entirely (it went dormant ~6 weeks)

**Reasoning:** Strategic re-assessment 2026-07-21 (`ops/assessment-2026-07-21.md`). The
NeuroCheck 50-claim DB is the only real, ~90%-done, publishable-without-GPU asset (verified:
48 unique DOIs, correct landmark citations). NeuroGenre carries ~35% risk genres don't
cluster; ScaleLaw's StudyForrest fsaverage5/HRF alignment is a ~30%-fail swamp. The
"four-build standard toolkit" is scope creep for a solo, no-GPU operator. The realistic prize
is a bioRxiv preprint (~75%) and a portfolio/hiring story (~60%), both anchored on NeuroCheck.

**Implications:** Effort concentrates on the NeuroCheck scoring pipeline + resource paper +
DOI re-verification. NeuroGenre/ScaleLaw marked CUT in war-room. The niche is reframed from
"the standard toolkit for in-silico neuroscience" to "the first DOI-verified sanity-check
benchmark for brain-encoding models."

**Revisit if:** A NeuroCheck preprint ships and there's appetite/collaborator interest to
expand; or the smoke test reveals something that changes the calculus.

---

### D013: GPU smoke test is the gate before any more build code (2026-07-21)

**Decision:** Run a <=1hr Kaggle smoke test (install deps + one `model.predict()` + the
G018 ablation kill/confirm + per-extractor VRAM logging for G005) BEFORE writing any further
build code.

**Alternatives:**
- Keep building the scoring pipeline / BrainLens on faith
- Assume the ablation path works because the discovery code exists

**Reasoning:** Nothing has ever run on a GPU; every "COMPLETE" is CPU-import-only. The
upstream source has no supported inference-time modality-mask API (`tribev2/model.py:190`
zeros only absent-from-batch modalities; `:212` is training-only dropout). Our
`_find_features_to_use()` probes 4 attribute paths on hope. One hour converts the whole
project from "hope" to "de-risked" — or tells us exactly what to cut (BrainLens).

**Implications:** This test closes G018 and G005 in one session. If ablation outputs are
identical (full vs modality-removed), BrainLens is dropped and the project is NeuroCheck-only.

**Revisit if:** Never — this is basic due diligence before committing more effort.

---

### D014: MCP "neural-engagement" tool is the standout follow-on direction (2026-07-21)

**Decision:** Record — but do not start yet — an MCP tool that wraps TRIBE to return
predicted per-region cortical engagement (visual/auditory/language/emotion ROI scores) as
structured JSON callable by any agent, with a "neural-atypicality" headline score. Hold it
as the follow-on once the model is confirmed to run.

**Alternatives:** Pursue it now (premature — depends on the smoke test); ignore it.

**Reasoning:** It is the highest-strategic-ROI creative idea from the assessment: it fuses
the operator's day job (building MCP tool-servers) with this project, is a differentiated
portfolio artifact almost nobody else can credibly build, stays license-clean as a research
demo, and needs little compute (ZeroGPU drip or cached precompute).

**Implications:** Sequenced after D013 passes and after the NeuroCheck resource paper.
Pairs with the normative-deviation scorer (assessment idea #7) as its headline metric.

**Revisit if:** The smoke test fails (no working inference) — then this is not buildable as
described.

---

<!-- Add new decisions above this line -->
