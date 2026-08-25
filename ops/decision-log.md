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

### D015: Full CrossRef DOI audit — 17 of 50 claims had broken or mis-attributed citations (2026-07-21)

**Decision:** Machine-verify every claim DOI against CrossRef before the NeuroCheck resource
paper, and fix all failures. Ship the verification tooling + clean report as a paper
supplement.

**What the audit found (this is worse than "typo'd DOIs"):**
- 34/50 clean on first pass. **17 claims needed correction.**
- 10 DOIs did not resolve on CrossRef at all; 6 resolved to a *different paper* than the
  citation; 2 pairs shared a DOI.
- Several were not bad DOIs but **mis-attributed citations** — the wrong journal/year:
  NC006 (J Neurosci → Cerebral Cortex), NC016 (Science → Current Biology), NC020
  (Neuropsychologia → Nature Neuroscience), NC028 (J Neurophysiol → Cerebral Cortex),
  NC033 (J Neurophysiol → Neuron), NC004 (Tootell 1998 → the actual V1-contrast paper,
  Boynton 1996).
- De-duplicated shared sources: NC027 → Humphries 2010 (A1 tonotopy; Formisano is NC023's),
  NC024 → Vouloumanos 2001 (posterior-belt speech vs nonspeech; Scott 2000 kept for NC029).

**Reasoning:** D011 caught the 37% DOI-hallucination rate in claims batch 2 and hand-fixed
it, but batch 1 and others were never machine-verified end-to-end. A reviewer running the
same CrossRef check would have found ~16 bad references and discarded the benchmark. Every
correction here was confirmed by a direct CrossRef lookup — no LLM-guessed DOIs (that was
the original failure mode).

**Implications:**
- `scripts/verify_dois.py` (audit), `scripts/resolve_dois.py` (CrossRef candidate search),
  `scripts/patch_dois.py` (line-anchored corrections) are now in the repo. Re-run
  `verify_dois.py` before any claims release.
- Result: **50/50 DOIs resolve and match their citation, zero duplicates.**
- The clean `scripts/doi_verification_report.md` becomes a supplementary artifact — turns
  the 37% scar into a credibility signal.

**Revisit if:** New claims are added (re-run the verifier) or CrossRef metadata changes.

---

### D016: Make the tribe-bench GitHub repo public (2026-07-21)

**Decision:** Flip `codesbydevesh/tribe-bench` from private to public.

**Alternatives:**
- Keep private + upload the folder as a Kaggle Dataset to mount it
- Keep private + use a `GH_TOKEN` Kaggle secret for a tokenized clone

**Reasoning:** The smoke-test notebook runs on Kaggle and needs the tribe-bench package
installed there. A public repo clones with a plain `git clone` (no token, no dataset upload)
— the simplest path for a solo operator. The content (MIT code + CC-BY claims) is destined
to be public anyway, and per the assessment an open "research infrastructure" posture is a
trust signal, not a risk. The notebook's clone cell was updated to clone directly, with the
dataset-mount and `GH_TOKEN` paths kept as fallbacks in case it goes private again.

**Implications:** The code + 50-claim DB are now publicly visible. No secrets are in the repo
(verified). Can be flipped back to private anytime without breaking the notebook (fallbacks
remain).

**Revisit if:** We want the work private again before a formal release (flip back; the
notebook still works via dataset/token).

---

### D017: Pivot to the fused "Corticall" flagship; retire the four-build empire (2026-07-23)

**Decision:** Collapse the project from four builds (BrainLens, NeuroGenre, ScaleLaw, NeuroCheck)
into ONE fused flagship: a runnable NeuroCheck benchmark that reports where TRIBE generalizes vs
breaks, wrapped as a read-only agent-callable brain-response MCP instrument, fronted by a live HF
Space. Papers fall out as byproducts.

**Trigger:** the 2026-07-23 10-agent principal review (`ops/principal-review-2026-07-23.pdf`).

**Reasoning:** Three external facts broke the old premise — (1) CortexLab already shipped a solo
TRIBE toolkit, so "a toolkit on TRIBE" is taken; (2) TRIBE-derived "engagement scores" are already
published as a failure (arXiv 2607.01400); (3) the benchmark has incumbents (Neurosynth, Brain-Score)
to position against, not an empty field. The one uncontested lane is a read-only brain-response MCP —
which the operator's MCP day-job makes uniquely credible. Critically, the NeuroCheck scoring harness
is BOTH the paper's engine and the MCP backend, so "Great" (a citable finding) and "Legendary" (the
live instrument) share one spine — no either/or, no doubled effort.

**Implications:** All four review lenses agreed the load-bearing risk is construct validity (localizer
contrasts vs a movie-trained model), so the direction is reframed to "where does it generalize / break"
and gated on an in-distribution validation test (Phase 0) before any build. Roadmap:
`.notes/plans/corticall/ROADMAP.md`.

**Revisit if:** Gate 0 fails (TRIBE doesn't recover known maps in-distribution) → fall back to the
NeuroCheck static-resource paper + the ablation-mechanic write-up.

---

### D018: Rename the initiative to "Corticall" — docs-level only (2026-07-23)

**Decision:** Brand the initiative **Corticall** (cortical + call = the agent-callable cortex).
Apply it in `.notes/`, the roadmap, and README positioning ONLY.

**Reasoning:** "tribe-bench" reads as a generic harness and buries the instrument; a name that fuses
cortex + agent-callable matches the reframe and is uncontested (CortexLab/BrainExplore/NeuroProbe are
taken or collide). Full rationale: `.notes/plans/corticall/IDENTITY.md`.

**Implications:** Import paths (`tribe_tools`, `brainlens`, `neurocheck`), the `pyproject` project name,
and the GitHub repo are UNCHANGED so the working Kaggle pipeline doesn't break. The mechanical rename
(package + repo + notebook clone URL) is a separate, deliberate follow-up PR if the name sticks.

**Revisit if:** A better name surfaces (it's a find-and-replace across a few docs) or the mechanical
rename is greenlit.

---

### D019: Adopt a `.notes/` operating system; demote `ops/` to durable reference (2026-07-23)

**Decision:** Living state moves to `.notes/` (BRIEF_ME, LOOSE-ENDS, journal/, plans/), mirroring the
Bridge repo's system. `ops/` keeps only durable reference (source-of-truth, interface-contracts,
claims-protocol, compute-playbook, knowledge-gaps, decision-log); process-theater is moved to
`ops/archive/`. `CLAUDE.md` stays evergreen and points to `.notes/BRIEF_ME.md` as the first read.

**Reasoning:** The review found status + process ritual tangled into `CLAUDE.md` and a 15-file `ops/`
around ~1,869 LOC — process-as-progress. Separating evergreen rules / durable facts / living state
(the Bridge pattern the operator already uses) kills the clutter and makes "read this first" unambiguous.

**Implications:** Session ritual now: read `.notes/BRIEF_ME.md` → `LOOSE-ENDS.md` → last journal(s) →
`ROADMAP.md`. Update those at session end. Archived files remain in `ops/archive/` for history.

**Revisit if:** The split creates friction in practice (fold back), or a second initiative needs its
own `plans/` subtree.

---

### D020: Gate 0 — pre-registered in-distribution validation (GO/NO-GO for all of Corticall) (2026-07-23)

**Decision:** Before any Phase-1 build, run Gate 0 — test whether TRIBE v2's as-trained (full trimodal)
forward pass recovers **right-FFC (FFA) face-selectivity** on naturalistic movie clips. Pre-registered rule
(full detail in `.notes/plans/corticall/GATE-0.md`):
- Stimuli: 4 face + 4 scene ~20s clips, ALL from Tears of Steel (live-action, CC-BY), downscaled to 480p;
  face/scene-dominance confirmed via an in-notebook thumbnail montage before any GPU minute. No animated
  films in the face claim (Sintel/BBB faces are rendered → OOD).
- Primary ROI: right FFC (per NC001; left FFC is the VWFA so bilateral is avoided). Scene ROI
  (corroborating only): PHA1-3 ∪ VMV1-3. Specificity controls: V1, right LO2 (body/EBA), A1.
- Metric: per-clip spatial z-score; direction via Mann-Whitney U over 16 clip pairs, EXACT permutation null
  (C(8,4)=70; U≥15→p=0.029, U=16→p=0.014), implemented + unit-tested in `tribe_tools/roi_stats.py`.
- GO iff ALL: G1 U_FFCr≥15/16; G2 Δz_FFCr > 95th-pct of its own permutation null; G3 Δz_FFCr > Δz_V1 AND
  > Δz_LO2; G4 video-only Δz_FFCr>0 AND U≥12/16; G5 face/scene sets matched on flow/luminance/RMS.
- NO-GO if U_FFCr≤10/16, OR Δz_FFCr≤0, OR V1≥FFCr, OR video-only Δz_FFCr≤0. AMBIGUOUS otherwise → diagnose
  (within-clip segment contrast, cross-film live-action) before Phase 1. Never build on AMBIGUOUS.

**Trigger:** ROADMAP Phase 0; the principal review named construct-validity/OOD the load-bearing risk.
Designed via the Three Musketeers protocol (Athos design → Porthos teardown → Aramis ruling, on Fable).

**Reasoning:** Cost is asymmetric — a false GO wastes ~6 months on an unvalidated encoder; a false NO-GO
costs one experiment and falls back to D017's static-resource paper — so GO is deliberately hard.
Independently verified before committing: (a) the exact permutation null (Porthos's caught defect: 3v3
U≥7 = p=0.20, a coin flip; 4v4 U≥15 = p=0.029) — now pinned by `tests/test_roi_stats.py`; (b) the
precedent arXiv 2605.13904 is still-image feature-visualization, so FFC weights are face-tuned but a
clip-mean contrast magnitude is unmeasured → the rule leans on direction + specificity + video-only, not a
borrowed floor; (c) FFC spans FFA (right) and VWFA (left) per our own claims.yaml → primary ROI is right FFC;
(d) scene area = PHA1-3 ∪ VMV1-3; (e) ToS 720p is 372 MB and the 20 min/clip figure was 480p → clips are
downscaled and wall-clock budgeted honestly (~3.5 h expected, <12 h session).

**Implications:** New `notebooks/02_gate0_validation.ipynb` reuses the proven smoke-test setup cells verbatim
and adds prep+covariate+atlas-assert, a FULL loop, a video-only loop, and analysis+decision against the
verified APIs. No new VRAM risk (same 11.12 GB path). New tested module `tribe_tools/roi_stats.py`. Video-only
is a hard gate; the scene double dissociation is corroborating only (motion can fake a crossover).

**Revisit if:** Gate 0 returns NO-GO (→ D017 fallback) or AMBIGUOUS (→ run the deferred diagnostics before
any Phase-1 work); or the neuralset on-disk feature cache proves it reuses the video encode across mask
configs (then video-only is near-free).

---

### D021: Gate 0 v2 — frontal-face-present vs face-absent, single-source (resolves the AMBIGUOUS run) (2026-07-27)

**Decision:** Re-run Gate 0 as a pre-registered confirmatory test with a redesigned, single-source stimulus
set. Manifest frozen in `notebooks/gate0_v2_stimuli.json` before any GPU spend.

- **Source:** Charade (1963), public domain, color live-action Hollywood — in-distribution for a
  Friends/CNeuroMod-trained model. Tears of Steel abandoned (off-distribution stylized sci-fi; forced
  group/body framing). All conditions from ONE film → no cross-film low-level confound.
- **Conditions (mechanically defined, from sustained >=10s runs, >=45s apart):** FACE = 15 clips with one
  frontal face filling the frame; NONFACE = 15 clips where the frontal-face detector found NO dominant face
  across the whole window (people-with-backs/profiles, scenes, places). Both contain people, so the only
  systematic difference is a dominant frontal face — this isolates the face signal and controls for film,
  people/bodies, scene, and low-level stats at once, dissolving the body confound that broke Run 1.
- **ROIs (frozen):** primary = right FFC (FFA). Specificity controls = V1 (low-level) + EBA proxy (right
  LO2∪LO3∪V4t∪FST∪PH — replaces the broken 12-vertex LO2). Positive control = PPA (PHA1-3∪VMV1-3).
  Auditory control = A1.
- **PRIMARY GO (one rule):** right-FFC face>nonface, one-sided Monte-Carlo permutation on Mann-Whitney U
  (>=10,000 shuffles, seeded), p<=0.025, AND Δ(spatial-z) exceeds the 95th percentile of its own permutation
  null, AND FFC Δ > V1 Δ and FFC Δ > EBA Δ (specificity), AND the effect stays positive with p<=0.05 in the
  VIDEO-ONLY pass (rules out a speech/audio artifact).
- **NO-GO:** face<=nonface in FFC, or V1/EBA >= FFC, or video-only effect vanishes. **AMBIGUOUS:** positive
  but short of the bars → diagnose (finer stimulus curation / a second source), do not build Phase 1.
- **Confirmatory (reported, Holm-corrected, NOT gated):** PPA scenes>faces using clean McLintock landscapes
  (pipeline positive control, must reproduce the Run-1 place effect); EBA nonface>face crossover.

**Trigger:** Run 1 (D020) returned AMBIGUOUS — right-FFC face-selectivity was real and visual (video-only
strengthened it) but underpowered (U=14/16, p=0.057) and the body region out-responded FFC because the
"face" clips were groups full of bodies. A 5-agent research workflow (on Fable) diagnosed the cause
(off-distribution single source + body confound + broken 12-vtx control) and the fix.

**Reasoning / stimulus sourcing reality:** No single free public-domain naturalistic film yields clean
faces AND bodies AND objects AND scenes at n>=15 from one matched source. Charade is face-rich (167 clean
single-face candidates) but body/object/scene-poor as clean isolated categories; McLintock is body/scene-rich
but face-poor. Rather than mix films (reintroducing the low-level confound) or force a 4-way we can't source
cleanly, the within-Charade face-present-vs-absent contrast is confound-free, mechanically defined, and
directly tests the reframed question. The full 4-way (faces>objects AND faces>bodies with modern in-distribution
stock video) is the documented future upgrade — it needs a Pexels/Pixabay API key.

**Implications:** New notebook `03_gate0_v2_validation.ipynb` (downloads Charade + McLintock, cuts clips at
the frozen manifest, runs FFC/EBA/V1/PPA/A1 analysis via the unit-tested `roi_stats` Monte-Carlo permutation +
IUT). Stimuli + code frozen in git before the run. Run 1 is the declared pilot (validated pipeline, estimated
effect); no pilot-clip reuse; one primary rule; both GO and NO-GO pre-committed as reportable.

**Amendment (2026-07-27 PM, commits `5c7f554` + `f8b277d`):** after rendering the manifest and inspecting the
FACE/NONFACE montages, 4 clips were replaced — FACE 104:15 (dark night outlier) → 59:30, FACE 108:25 (raised
arm + busy board) → 96:30, NONFACE 112:35 (fragmented end-of-film frame) → 99:45, NONFACE 28:50 (struggle with
visible faces) → 8:00. All 15 FACE clips are now clean face-dominant. Three NONFACE clips (57:35 / 72:00 /
91:40) retain profile/partial faces by choice: a partial face in the face-ABSENT baseline is conservative — it
can only shrink face>nonface, never inflate it. Every amendment landed BEFORE any GPU spend and is recorded in
the manifest's `revision_note`, so the freeze-before-run commitment holds. No change to conditions, ROIs,
statistics, or the GO/NO-GO rules.

**Revisit if:** v2 returns AMBIGUOUS/NO-GO (→ finer curation or a stock-video key for the full 4-way), or a
Pexels/Pixabay key becomes available (→ upgrade to the modern-stimulus 4-way as the confirmatory-of-record).

---

### D022: Widen the ASR sentence-alignment guard for 10s clips; make a failed pass drop instead of abort (2026-07-28)

**Decision:** In `notebooks/03_gate0_v2_validation.ipynb`, force `AddSentenceToWords.max_unmatched_ratio`
to 0.99 at its construction site before the pass loops, and wrap each pass in a per-clip try/except that
records the failure and drops that clip from the contrast instead of killing the run.

**Reasoning:** the Gate 0 v2 run died 28 minutes in, at FACE_08 of 38 clips, on
`RuntimeError: Ratio of unmatched words is 0.1111 on 9 words while AddSentenceToWords.max_unmatched_ratio=0.05`
(G019). The 5% default assumes a full-film transcript; a 10s clip transcribes to 9–35 words, so any clip
under 20 words tolerates ZERO unalignable words. FACE_01–07 passed only because they happened to align
perfectly — this would have kept firing at random through all ~68 passes. The guard fires after every
annotation is already written and only protects the `sentence` field feeding the LLaMA text encoder, so
widening it costs nothing structural; 0.99 rather than 1.0 because `model_post_init` requires 0 ≤ r < 1,
which usefully leaves total alignment failure still raising. The field is in `_exclude_from_cls_uid()`, so
the change cannot invalidate cached features.

**Implications:** no change to conditions, ROIs, statistics, or the GO/NO-GO rules — this is pipeline
robustness, not design. Sentence context may be incomplete for a word or two on some clips, which can only
add noise to the text modality; the pre-registered G4 video-only pass (text masked entirely) is the clean
control that makes this irrelevant to the primary verdict. The analysis cell now intersects the surviving
clips across passes, prints every dropped pass, asserts n ≥ 12 per group, and writes `dropped` into
`gate0v2_results.json` — so any attrition is reported with the result, never silent. Cached passes make the
restart resume; the 8 completed FACE passes are not re-paid.

**Revisit if:** attrition ever exceeds 3 clips per group (then the stimulus set or the ASR step needs
fixing, not the tolerance), or the run reports a clip where alignment failed totally.

---

### D023: Corrections to D022 and a narrowed NO-GO rule, before any Gate 0 v2 data exists (2026-07-29)

**Context:** D022 was re-audited before spending the 7h Kaggle run (17-agent adversarial review, every
serious finding independently refuted). The audit falsified most of D022's stated rationale and, more
importantly, found that the largest false-verdict risk in notebook 03 is not the ASR guard at all.

**(a) NO-GO disposition narrowed — the single highest-value change.**
D021's NO-GO list included `Delta z_V1 >= Delta z_FFCr` and `Delta z_EBA >= Delta z_FFCr`, and cell 22
implemented them so that failing G3 ALONE printed `NO-GO -> stop; D017 fallback`. Those are bare
comparisons of two point estimates with no null, no CI and no margin, across ROIs of 58 (FFCr), 116 (EBA)
and 523 (V1) vertices — and the 58-vertex FFC mean is structurally the noisiest of the three, so the
comparison is biased against the hypothesis. Decisive evidence is historical, not simulated: in Run 1 the
body ROI out-responded FFC, and Run 1's NO-GO list contained only the V1 clause, so it routed to
AMBIGUOUS -> diagnose -> which is why D021 exists. D021 silently widened NO-GO to include the EBA
ordering, which would convert Run 1's own observed pattern into a project-ending NO-GO. Verified by
execution against mocks, not argument.
NO-GO is therefore restricted to the two SIGN conditions: `Delta z_FFCr <= 0` in the full pass, or
`Delta z_FFCr <= 0` in the video-only pass. **G3 remains a required conjunct of GO, unchanged.** A
non-significant ROI-ordering reversal now routes to AMBIGUOUS -> diagnose.
This LOOSENS a pre-registered stopping rule in the direction of continuing the project. It is recorded
before any v2 contrast numbers exist, and the FULL-vs-video-only and FFC-vs-V1/EBA orderings are reported
as headline numbers whatever the verdict.

**(b) ASR tolerance set to 0.15, not 0.99, and deletion is now recorded.**
D022's C1 ("only guards the `sentence` field ... loses nothing but the guard") and C2 ("can only add
noise") are FALSE. An unmatched word gets `sentence=""` (neuralset `text.py:186`), then
`AddContextToWords` gives it `context=""` (`text.py:271-274`), then `RemoveMissing` DELETES the row
(`basic.py:52-54`), and the affected TRs receive exact zeros from the text extractor
(`extractors/base.py:250-262, 302-305`). The ratio was the only cap on a destructive step. Worse, it
undercounts what it nominally bounds: `utils.py:298-311` back-fills `sentence` without `sentence_char` on
gap words, so such a word counts as matched at the check and is still deleted — deletion was never
bounded by the ratio at any setting.
Tolerance is therefore 0.15, the smallest value admitting the observed 1-of-9 that killed FACE_08 while
rejecting 2-of-9. Per-clip word count, unmatched count, unmatched ratio, deleted count and `n_kept` are
recorded in `gate0v2_results.json`. Conditions, ROIs, statistics and gate thresholds are unchanged.

**(c) Environment pinned.** `exca` is now pinned to 0.5.25 and the tribev2 clone SHA is recorded.
Verified: exca 0.5.20/0.5.25 import; 0.5.26 (2026-06-03) through 0.5.29 (2026-07-28) raise
`AttributeError: module 'exca.steps.base' has no attribute 'NoValue'` inside
`neuralset/events/study.py`. tribev2 pins `neuralset==0.0.2` but nothing pinned exca, and 0.5.29 shipped
the day of the crash. Run 1's tribev2 SHA is unrecoverable, so Run 1 stays a pilot, not a comparable prior.

**(d) Recording amendment, and no assert may destroy a finished run.** Per-clip z vectors, retained clip
names, per-clip `n_kept`, ROI vertex counts, seeds, `n_perm`, the tribev2 SHA, the exca version and the
tolerance are written to `gate0v2_results.json`, and the artifact is written BEFORE the plotting block.
D022's `assert min(len(FN), len(NN)) >= 12` is replaced by a printed `underpowered` warning plus a
recorded flag — an assert at hour 7 of an unrepeatable session destroys the result it was meant to
protect. Attrition BALANCE is printed too, since asymmetric drops bias the contrast rather than merely
shrinking n.

**(e) Corrections to D022's claims.** C1, C2, C4 and C6 are false as written. C3 is half false: the
video-only pass blocks a text-driven GO (G4 is a hard conjunct of GO) but cannot prevent a text-driven
NO-GO, because the NO-GO branch reads full-pass numbers — G4 is a filter, never a rescue. C5's conclusion
holds but its reason is inapplicable: the transforms run with `infra=None`
(`transforms/base.py:156-158`), so no exca uid is ever computed and `_exclude_from_cls_uid` is never
consulted; cache safety follows instead from the guard being the final statement of `_run` and from the
feature caches being content- or path-keyed. Also, D022's "total alignment failure still raises" is false
for the likeliest form of total failure: `text.py:174-176` returns without raising when there are no Word
events at all, silently dropping the whole text modality. G019 is reopened as *mitigated, not closed*.

**Settled NEGATIVE by the same audit — do not re-litigate:**
- Word deletion CANNOT change `n_kept_segments`. `demo_utils.py:370-371` keeps a TR if
  `len(s.ns_events) > 0`, and `ns_events` is OVERLAP-based with no type filter
  (`segments.py:250-262 -> 155-164 -> 89-104`, `mask = (starts < stop) & (stops > start)`), on top of the
  clip-long Video/Audio events and the per-timeline dummy `CategoricalEvent` (`main.py:186-195`).
- `ChunkEvents(min_duration=30)` is a verified no-op on a 10s event: `etypes.py:453` discards `t=0.0` on a
  strict inequality before `min_duration` is consulted.

**Alternatives rejected:** a permutation test on `Delta_FFC - Delta_V1` (more defensible, but its first
execution would be at hour 7 of an unrepeatable run); re-cutting to 20-30s clips (voids the >=10s
sustained-run curation D021 rests on); promoting video-only to co-primary (already a conjunct of GO);
`audio_only=True` as the primary path (`demo_utils.py:76` extracts audio outside the `if not audio_only`
block, so it is an AV pass, not video-only); a local CPU WhisperX dry run (`eventstransforms.py:107-108`
hardcodes `compute_type="float16"` with `--model large-v3`, and a CPU-viable model changes WHICH words go
unmatched, so it cannot predict the ratios that matter).

**Revisit if:** the run reports attrition imbalance > 2 clips, any `unmatched_ratio` at or above 0.15, a
clip with `words_before == 0`, or a non-empty `nonfinite_stats`.

**(f) Stimulus pre-flight MEASURED on CPU, 2026-07-30 — two findings, one of them unresolved.**
Both films verified byte-exact against archive.org's own md5 (Charade
`f2602d71c2279e834d48bdefe32b04a6`, McLintock `04671e70c46d1b3f3cb8d1df4217a666`); Charade is
6785.0s against the 6205s the last clip needs and decodes cleanly at every clip start; all 38 clips
cut valid (10.000-10.050s, height 480, video+audio present, clean decode). Full record:
`notebooks/gate0_v2_preflight.json`.
- **MOTION IS A REAL CONFOUND: FACE 0.914 vs NONFACE 3.039 mean frame-difference energy,
  two-sided permutation p = 0.0008.** Eleven of fifteen NONFACE clips exceed the FACE maximum, so
  the conditions are close to linearly separable on motion alone. This matters more than a generic
  covariate imbalance because `spatial_z` subtracts the whole-brain mean: anything that selectively
  raises non-FFC regions in NONFACE mechanically raises `Delta z_FFCr`. Motion does exactly that,
  and the chosen EBA proxy (LO2/LO3/V4t/FST/PH) is the motion-sensitive lateral occipitotemporal
  complex, so a motion asymmetry can inflate BOTH the primary statistic and the G3 specificity
  margin. **G4 does not protect against it** - motion is a visual property and survives video-only.
  GATE-0.md already anticipated this class of artifact for the PPA control ("a motion confound
  (low-motion faces vs high-motion wides) can fake a crossover"); it is now measured in the primary
  set. Disclosed, gating nothing; a GO must be reported as "face-selective or motion-asymmetry
  driven" until a motion-matched sub-analysis is run on the saved per-clip z vectors.
- Well matched, no action: luminance (p=0.91), audio RMS (p=0.50), voiced fraction (p=0.19),
  shot-change rate (p=0.17), early-cut fraction (p=0.14). **The voiced-fraction result refutes the
  working assumption that FACE clips are dialogue-dense and NONFACE quiet** - the measured means run
  the other way (FACE 0.571, NONFACE 0.674) and do not differ significantly. The speech-confound
  worry that motivated part of D023(b) is therefore NOT supported by measurement; D023(b) still
  stands on the deletion mechanism, which is independent of any condition asymmetry.
- **UNRESOLVED, needs a decision before the run: the clips are not sustained shots.** At a
  visually-validated 0.15 scene threshold, FACE averages 2.20 shot changes per 10s clip and NONFACE
  1.53; 13/15 FACE and 11/15 NONFACE contain at least one; 10/15 FACE and 5/15 NONFACE have one
  within the first 2.5s. Hand-inspected frame-0 and 0/2.5/5/7.5s filmstrips confirm the detector at
  5/5 checked clips, and show FACE_04, FACE_10 and FACE_13 opening on a non-face shot (a dark gate,
  a red case, a wide two-man room) and NONFACE_03 and NONFACE_14 opening on a prominent frontal
  face. D021's curation rule sampled the face detector every 5s, which is structurally blind to
  sub-5s cuts, so "sustained runs of >=2 consecutive 5s samples" does not imply a sustained shot.
  This compounds with frame-0 over-weighting (see D024(b)). Both directions of contamination are ATTENUATING, so this cannot manufacture a GO - it
  raises P(AMBIGUOUS) on a test whose power is already ~0.46 at AUC 0.70. Note the earlier 0.4
  threshold reported 0.13 cuts/clip and is unusable on these 480p re-encodes.
  **Options, all CPU-cheap and all pre-data:** (i) run as-is and disclose; (ii) re-cut with a
  mechanical, condition-blind shot-alignment rule (take the 10s from inside the longest cut-free
  span near each manifest start), which enforces the property the pre-registration already claims;
  (iii) trim the first 2.5s to 8s clips, which fixes frame-0 contamination only. **This is a
  stimulus-design decision and is NOT taken here.**


**(g) Stimulus set REBUILT to fix both pre-flight findings — Gate 0 v3 (2026-07-30, still pre-data).**
`gate0_v2_stimuli.json` is retained unchanged as history; `gate0_v3_stimuli.json` supersedes it and the
notebook now reads v3. Zero Gate 0 runs have completed, so no outcome informed any choice, and the whole
procedure is deterministic at seed 0. Pipeline + rationale committed at `scripts/gate0_v3_curation/`.

*Authority:* GATE-0.md's own G5 pre-registered "FACE and SCENE clip sets not differing in mean
optical-flow / luminance / RMS beyond tolerance **after reselection**". This executes a clause already
written; it does not invent one. The hypothesis, ROIs, statistics and gate thresholds are untouched.

*Rule (mechanical, outcome-blind):* shot boundary = any frame with ffmpeg `scene_score > 0.05` on a
160 px scan of all 169,625 frames — deliberately SENSITIVE, because over-detecting boundaries only costs
candidates while under-detecting is the defect being fixed (the 0.4 default reported 0.13 cuts/clip
against 2.20 actual). A window is 10.0 s lying strictly inside ONE shot with a 0.5 s cut-free guard both
sides. FACE = the two Haar frontal cascades' union finds a face of area >= 0.025 of frame in >= 80% of
the 21 samples, median >= 0.025, **at the first sample**, in >= 80% of the first 4 s, and the two
cascades AGREE in >= 30% of samples and at the first sample. NONFACE = the same high-precision detector
NEVER finds a face of area >= 0.020 anywhere in the window, and people are present (profile detections
in >= 30% of samples), preserving D021's "both conditions contain people" control. One clip per shot; no
two of the 30 clips within 45 s of each other, enforced ACROSS conditions; greedy best-first 1:1 pairing
on standardised [motion, luminance] distance. Acceptance: two-sided permutation p >= 0.20 on motion AND
|SMD| <= 0.25 AND |AUC-0.5| <= 0.10.

*Two rules earned by measurement, not taste.* (i) The frame-0 and first-4 s clauses exist because
frame 0 is over-weighted 17% (see D024(b)); a draft selection passed a
window-average criterion while frame 0 showed a wide establishing shot. (ii) Cascade AGREEMENT is
required because a persistent single-cascade false positive was observed directly: on a wide corridor
shot the union reported face area ~0.040 on every one of 20 samples while the two cascades agreed on
0.000 throughout. Requiring agreement excludes it; requiring full agreement everywhere would have cut
the pool to 9 shots, so >= 30% is the measured compromise (19 shots available, 15 needed).

*Result, verified on the 30 cut clips (`gate0_v3_verification.json`):*
- **Shot changes inside clips: 0** (v2: 2.20 FACE / 1.53 NONFACE per clip; 10/15 FACE cutting within 2.5 s).
- **Motion two-sided p = 0.246** (v2: 0.0008); SMD at selection -0.002, AUC 0.404.
- Luminance p = 0.393, audio RMS p = 0.589, voiced fraction p = 0.745 — all matched.
- All 30 clips 10.000-10.040 s, height 480, video+audio present, clean decode; min separation 51.0 s.
- **Visual audit passed** on frame-0 montages for both conditions: all 15 FACE open on a clear frontal
  face, all 15 NONFACE show people with no dominant frontal face. Frame 0 is audited specifically
  because it is the over-weighted frame. The notebook's own montage now renders at `-ss 0`, not `-ss 5`
  — sampling at 5 s is what let the v2 defects through.

*What this does NOT fix, and is disclosed rather than claimed away:* the face-absent baseline still
contains people and scenes rather than objects or scrambled controls, so "face-absent" means "something
else", as in any localiser; category labels rest on Haar cascades plus a visual audit, not hand-labelled
ground truth; and matching does not change power, which remains ~0.46 at AUC 0.70, so AMBIGUOUS is still
a live outcome meaning "underpowered".

**Revisit if:** the visual audit of a future re-cut disagrees with the detector labels, or the accepted
motion p falls below 0.20 on the cut clips.


---

### D024: Two corrections to D023 found by re-auditing the shipped v3 set (2026-07-30, still pre-data)

**(a) The v3 selection FAILED its own acceptance criterion; v3b replaces it.**
v3 matched on motion from a 160 px full-film scan, but the criterion is evaluated on the delivered 480 p
clips. On those files v3 gives motion AUC 0.631, i.e. |AUC-0.5| = 0.131 against its own cap of 0.10. The
p-value leg (0.246) passed, which is exactly how it went unnoticed: the gap between the two measurements
was recorded in the manifest, but only the p-value was re-checked on delivery, never the AUC.
Root cause: selection optimised a PROXY (pairwise distance on downscaled scans) rather than the quantity
the criterion measures (set-level rank balance on delivered files). Pairwise greedy on the delivered
measurements was tried and also failed (AUC 0.658, n = 14) - still the wrong objective.
Fix: all 42 shot representatives are cut at 480 p and measured with the SAME estimator used for
verification; the selection is then solved EXHAUSTIVELY over all **1536** feasible 15+15 sets (one clip
per shot, all 30 clips >= 45 s apart across conditions), minimising |AUC-0.5| of motion, tie-broken on
|SMD| of log motion - a global optimum over the feasible set, not a heuristic.
Delivered result: **motion p = 0.885, AUC = 0.520, |AUC-0.5| = 0.020, SMD(log) = +0.043**; luminance
p = 0.394, RMS p = 0.799, voiced p = 0.895; **0 shot changes inside any clip**; min separation 51.0 s;
all 30 clips 10.000-10.040 s at height 480, clean decode; visual audit passed both conditions. 10 of 30
clips changed.
**Balance is now certified in RANKS**, because G1 is a rank test. Raw SMD on a positive wide-range energy
measure is outlier-dominated and demonstrably uninformative here: rejected v3 had raw SMD -0.097
(passing) while AUC 0.631 failed. The criterion is STRICTER than before - it is what rejected v3.

**(b) "V-JEPA2 clamps to 4 s" is FALSE and is withdrawn.**
It slides a 4 s window: 20 stimulus bins at t = 0.5 .. 10.0 s x 64 frames = **1280 frames**, so the
encoder sees ALL TEN SECONDS in 20 overlapping spans. Frames taken from before the clip start clamp to
frame 0 and number exactly **217** (55+47+39+31+23+15+7 for bins t = 0.5 .. 3.5 s). Both figures
reproduce exactly under this reading and neither under a clamp reading - a true clamp would feed 64
frames, not 1280. Verified by direct arithmetic.
Frame 0 IS over-weighted, at 217/1280 = 17.0%, so the frame-0 selection clause and the frame-0 visual
audit are KEPT and remain justified. But every statement that the model "only sees the first 4 s" is
wrong and is corrected in place. No design decision depended on the false version: v3/v3b already
required whole-clip single-shot purity, correct under both readings.

**(c) The direction of the spatial_z motion nuisance was asserted without justification.**
D023(f) said a motion asymmetry "mechanically raises Delta z_FFCr". Not established: spatial_z subtracts
the whole-brain mean, so the bias depends on FFC's motion drive relative to the mean over 20,484 mostly
non-visual vertices - unmeasured. Since motion was HIGHER in NONFACE, it could equally have DEFLATED the
contrast. Corrected to "a nuisance term of unknown sign". The G3 sub-claim survives and is directional:
the EBA proxy contains most of the MT+ complex, so matching makes G3 harder, not easier.

**(d) Clause ordering fixed** - D023(g) had been filed above D023(f).

**Revisit if:** any future stimulus change is ever accepted on a measurement other than the delivered files.

---

### D025: Gate 0 v3b returned NO-GO. The D017 pivot is HELD pending six cached diagnostics (2026-08-01)

**The result (executed 2026-07-31 09:52-13:56 UTC, Kaggle T4, single ~4 h pass).**
`FFCr d=-0.244 U=79/225 p=0.9186`; `V1 d=-0.046`; `EBA d=-0.382`; `A1 d=+0.280 p=0.1448`; video-only
`FFCr d=-0.201 p=0.7687`; PPA place control `d=+2.529 U=120/120 p=0.0002`. G1-G4 all False.
**VERDICT: NO-GO.** Speech disclosure: face 23.3 words vs nonface 19.8, p=0.157. Full numbers, log audit
and provenance: journal 2026-08-01.

**The verdict is correctly computed and stands.** It fired on the two sign conditions, exactly as D023(a)
specifies. It would ALSO have fired under the un-amended rule (V1 -0.046 >= FFC -0.244), so the D023(a)
narrowing neither caused nor prevented it.

**The run is mechanically clean.** Zero attrition (15/15, 15/15, 8/8), zero failures, zero cache hits
(all 68 passes fresh), masking verified working (video-only passes load only the video extractor and run
in ~1.5 s), ASR deletion 1 word on 5 clips and near-symmetric across conditions, no shadowing, and a
positive control at perfect separation. This is a real measurement, not a plumbing failure.

**But the decision to pivot is HELD, for two reasons found by auditing the analysis code, not the log.**

**(a) The pre-registered statistic is zero-sum by construction.** Read from source:
`spatial_z = (g[verts].mean() - g.mean()) / g.std()` on the clip-mean map, so every clip's z-map has
mean exactly 0 and sd exactly 1. An ROI z is a SHARE of a fixed budget. The vertex-weighted sum of the
FACE-NONFACE delta over the brain is therefore exactly zero: if anything rises, something else must
fall. The observed pattern - the only positive ROI is the auditory one, all three visual ROIs negative
and ordered by distance from the brain average, in a condition carrying 18% more speech - is exactly
what a pure normalizer shift produces WITH ZERO CATEGORY INFORMATION PRESENT. The reported statistics
cannot distinguish that from "no face selectivity". Filed as G020.

**(b) There is no within-film positive control.** The PPA control is Charade vs McLintock, i.e.
cross-film. It proves the pipeline emits differentiated output between two movies; it does not prove
the pipeline resolves any within-film 15v15 10 s-clip category contrast. The one within-film contrast
that should have been near-guaranteed (more speech in FACE -> auditory cortex) came out A1 +0.280,
p=0.1448, not significant. So the run demonstrates no within-film sensitivity to anything. Filed as G021.

**Calibration, so "underpowered" is not used as a blanket excuse.** Observed AUC = 79/225 = 0.351;
null SE at 15v15 ~ 0.107; the observation sits ~3.3 null-SEs BELOW the hypothesised 0.70. Not a
coin-flip null. But under (a), that distance may be measuring the normalizer rather than cortex. Both
are true at once and both belong in any write-up.

**Correction recorded:** the notebook's printed `top-k FACE ROIs` is computed on the FACE grand-mean
map, NOT on the FACE-NONFACE contrast. It shows where TRIBE's output is largest on this material
overall (very likely identical for NONFACE). It is not evidence of a condition-specific auditory
response and must not be cited as one.

**Also recorded, in the other direction:** FFC (-0.244) beat EBA (-0.382). Run 1's body confound is
gone. G3 failed only on the V1 leg, and V1 (523 vertices, near the brain average) is close to being
the reference, so "FFC < V1" is nearly a restatement of "FFC fell relative to the global mean".

**THE DECISION.** Six diagnostics (CPU, on the cached predictions, minutes) run BEFORE any pivot:
(1) V1/EBA/A1 under video-only; (2) pooled auditory+STS within-film - the missing positive control;
(3) is `g.mean()`/`g.std()` condition-dependent - the direct zero-sum test; (4) FFC-EBA per clip,
permuted - compositional-immune specificity; (5) FFC re-normalized against a reference excluding
auditory+STS - does the sign survive; (6) all 360 parcels ranked by FACE-NONFACE delta, both passes,
with FFC's rank. Code parked in LOOSE-ENDS thread 1.

**None of these can convert NO-GO into GO** - that is the M001-class error this project keeps catching.
They are strictly diagnostic, and they separate two worlds:
- FFC stays negative under re-normalization AND the within-film control is significant -> the design
  had sensitivity; this is a clean publishable negative (strong place structure, no within-film face
  structure), which is the "where does it break" hook D017's pivot was built around.
- FFC flips, OR even the auditory control is null within-film -> Gate 0's STATISTIC failed, not TRIBE.
  D017's static-resource fallback is then the wrong response, and the correct move is a **Gate 0 v4**:
  non-compositional measure, a mandatory within-film positive control, and an object/scrambled baseline
  instead of profiles - pre-registered and frozen before it runs, and disclosed as motivated by these data.

**Revisit if:** the diagnostics run and the two branches resolve; or if the prediction cache is lost
before they run, in which case they cost another ~4 GPU-hours and this decision must be re-taken with
that price attached.

<!-- Add new decisions above this line -->

---

### D026: Gate 0's NO-GO does not trigger D017's fallback; Gate 0 is retired as a project-gate (2026-08-04)

**Decision:** D017's revisit clause ("Gate 0 fails → fall back to the NeuroCheck static-resource
paper") **does not fire**. `CHARTER.md` stands in full, including the live instrument. Gate 0 stops
being a pass/fail project-gate and becomes the instrument's sensitivity calibration. `CHARTER.md`
§6 moves from PENDING RULING to RULED, pointing here.

**Reasoning — five independent grounds, any one sufficient.**
1. **The antecedent is false.** TRIBE v2's own paper (arXiv 2605.04326, d'Ascoli et al., 2026-05-05)
   recovers FFA, PPA, EBA and VWFA in-silico, validated against IBC over the 360 HCP parcels. A
   third party (Bladon & Bent, arXiv 2605.13904) recovers FFA selectivity by feature visualisation.
   Place selectivity recovered in both our own runs. "Recovers some and not others" is verbatim the
   direction D017 set (I6).
2. **Meta also resolved a within-naturalistic-movie contrast** ("for Bang ... we simply contrast
   segments from the Algonauts dataset"), so the naturalistic regime is not inherently the hard one.
3. **Our verdict is not interpretable as a model result.** Simulation with ZERO face information,
   varying only auditory drive, reproduces the observed pattern (sim FFCr −0.239 vs observed −0.244;
   EBA −0.295 vs −0.382; V1 −0.124 vs −0.046; ordering exact). With a genuine 0.05 FFC effect
   injected, raw and reference statistics detect it at p=0.0005 while spatial_z reports −0.124,
   p=0.9985 → NO-GO. The statistic inverts real effects. See G020.

   > ⚠️ **The NUMBERS in this ground are superseded by D030 (2026-08-23); the CONCLUSION stands.**
   > That spatial_z is compositional and can invert a real effect is arithmetic and does not depend
   > on the simulation. But every figure quoted above came from ONE draw of a simulator with a
   > module-level mutable RNG, at a D_AUD selected by argmin over 25 single noisy draws on a grid
   > whose LAST point was the winner. Corrected: D_AUD **0.30** (interior, 2.43σ clear); FFCr
   > **−0.262 ± 0.035 (sd, 225 draws)**, gap to observed **0.018 (~0.5 sd)**, not "within 0.005";
   > **p < 5e-4** (that was the estimator floor 1/2001). **"Ordering exact" is withdrawn** — the
   > ordering is set by hand in `build_brain`. V1 and A1 are **not** reproduced (A1 +0.035 vs
   > +0.280). Detection rate over 40 seeds at a genuine 0.05 effect: raw/reference **40/40**,
   > spatial_z **0/40** — which is a stronger statement than the single p-value it replaces.
4. **The design was mis-specified in four independent ways:** n=15 against a field norm of 25-144+;
   all 30 clips from one film treated as a fixed effect (Westfall et al. 2017 measure 50-200%
   statistic inflation); the Glasser right-FFC parcel is a poor face fROI; 10 s clips fill ~11 of
   100 window rows. And no same-scale positive control existed (G021/M004) — the near-ceiling
   speech→auditory contrast came out p=0.1448.
5. **Operator instruction, 2026-08-04:** "STICK TO WHAT WE WANT TO BUILT DON'T CHANGE WHAT WE ARE
   BUILDING."

**What this gives up:** the option to treat Gate 0 as dispositive and stop cheaply. We accept more
work and the risk that calibration shows the contrast class sits below what free compute resolves.
Bounded by MASTER-PLAN §6 stop rules; a measured floor is itself reportable (I6).

**Drift check (CHARTER §5):** feeds both benchmark and MCP; one spine, fewer engines; read-only;
free compute; strictly more honest. Passes all five.

**Revisit if:** S2 fails to replicate the published in-silico localizer result with our pipeline.

---

### D027: Measurement doctrine — adopt the model authors' statistic; retire spatial_z as primary (2026-08-04)

**Decision:** the primary in-silico contrast statistic becomes the published one: predicted response
at t=+5 s post-onset minus the mean t=+5 s response across the other categories, on 1 s-flashed /
8 s-ISI stimuli. Add a GLM contrast z (effect/SE across time per vertex — what the paper's Fig 4
reports) and an ROI-minus-reference statistic. `spatial_z` is demoted to legacy, retained only as
the comparison in the write-up. Every verdict carries a detection floor. fROIs are defined
in-silico from an independent localizer half, not taken as whole atlas parcels.

**Reasoning:** spatial_z normalises across vertices within a stimulus, forcing each z-map to mean 0
and sd 1, so brain-wide condition deltas sum to exactly zero and an off-target shift inverts
on-target ROIs. This is not a new discovery — it is the global-signal-regression artifact (Murphy
et al. 2009) and, in the interpretability setting, arXiv 2512.18792 ("The Dead Salmons of AI
Interpretability", Méloux et al.). **Do not write it up as a novel methods contribution.** A GLM
z-score is not compositional; ours is. Also relevant: TRIBE's fMRI targets were z-scored per vertex
across session, which independently makes cross-vertex magnitudes incommensurable.

**Implications:** `tribe_tools/roi_stats.py` gains `event_locked_contrast`, `glm_contrast_z`,
`roi_minus_reference`, `define_froi`, `detection_floor`. The compositional simulation becomes a
regression test that asserts the artifact.

**Revisit if:** the floor estimator shows the new statistics are less sensitive than spatial_z on a
design where ground truth is known (it should be the reverse).

---

### D028: Compute doctrine — the prediction bank is the spine (2026-08-04)

**Decision:** GPU is used only to grow a durable, versioned, provenance-stamped bank of predictions
persisted **off-session** (HF Dataset; free tier gives 100 GB private) before anything is analysed.
All analysis is CPU-only against the bank. The MCP read path never requires a GPU. Every clip
filename carries a content hash. Batched runs use one unique `timeline` per clip,
`keep_in_ram: False` on all three extractors, and `chunk_size ≈ 25` with per-clip fallback.

**Reasoning:** `exca` keys features on absolute filepath + offset + duration and flushes each to
disk as computed, so a bank makes every future re-analysis free (warm re-run 1.5 s/clip vs ~378 s
cold) and makes a killed session survivable. Kaggle `/kaggle/working` is lost on interactive session
end unless Persistence was enabled beforehand — which is how ~4 GPU-hours went missing on 07-31.
Independently, HF now requires a paid plan to create any compute Space except up to 2 Gradio Spaces
on ZeroGPU, free ZeroGPU is 5 min/day (2 min for unauthenticated callers), and ZeroGPU is
Gradio-only — so serving precomputed predictions from CPU is the **only** architecture in which
charter I4 (a live free Space) and I5 (free compute only) are compatible.

**Implications:** the cross-session feature cache moves from a Phase-1 optimisation to the spine.
No GPU spend is authorised on a design whose value depends on one contrast landing (D-1).
"No precomputed public bank of model-predicted brain responses exists for any encoder" survived
three independent searches — this is the project's one surviving structural novelty.

**Revisit if:** HF free-tier storage or Space terms change again.

---

### D029: Positioning corrections — stop claiming four occupied contributions (2026-08-04)

**Decision:** remove these claims from all framing, and cite the incumbents instead.

| Dropped claim | Occupied by |
|---|---|
| The four-valued verdict / detection-floor-per-item idea | **Veritas**, arXiv 2604.12144 (Apache-2.0): Supported/Refuted/Underpowered/Invalid from a deterministic non-LLM operator, per-hypothesis power + SESOI, 64 hypotheses with 37 positive / 16 negative / 11 deliberately underpowered controls. Earlier still: Conditional Equivalence Testing, arXiv 1710.01771 |
| Optimal / adversarial stimulus search as a headline | **Bladon & Bent**, arXiv 2605.13904, gradient ascent on TRIBE v2 across 7 ROIs, code released |
| Building modality ablation | **CortexLab** (`cortexlab-toolkit` 0.2.0): `analysis/lesion.py`, `experiments/causal_modality_ablation.py`, row-permutation tests + BH-FDR, 283 tests |
| "Where does TRIBE break" / negative findings as the novelty | Algonauts 2025's own stated goal; TRIBE's published OOD evaluation; two 2026 preprints auditing TRIBE-predicted cortex |
| A live Space as the differentiator | Meta hosts `aidemos.atmeta.com/tribev2`; also **Cortex-Canvas** (CHI 2026 EA, doi:10.1145/3772363.3798548) and **BrainExplore** (arXiv 2512.08560) |
| "First agent-callable TRIBE instrument" | possibly `sissississi-013/cortex` (unverified). **Never claim first.** |

**What survives as ours:** the prediction bank (unclaimed); the design-requirements/sensitivity
result for in-silico contrasts on this model; breadth of DOI-verified claims scored with floors; an
MCP whose validity flag is mechanically derived from bank coverage and measured floors.

**Also decided:** no rename. The fear that product-like names plus a statistics-dense write-up
invite AI-content desk rejection is **refuted** — ICML 2026's official CFP permits LLM writing
assistance, and the mech-interp workshop's desk rejections were for incomprehensible abstracts or
egregious citation fabrications after human review. A DOI-verified bibliography defeats that screen.
But **do not lead with "MCP"** to developer audiences (four 2026 HN front-page threads, 295-460
points, arguing MCP is dead or worse than a CLI), and **do not lead with the instrument** to
bioRxiv, whose exclusion list names "announcements of tools or services without data".

**Revisit if:** any incumbent above turns out to be misidentified (see G024).

---

### D030: The compositional demo's published numbers are corrected; D_AUD was boundary-selected on single draws (2026-08-23)

**Decision:** correct every number sourced from `scripts/compositional_demo.py`. The simulation's
qualitative finding stands; its quantitative claims did not survive audit and are restated with
their real provenance and uncertainty.

**What was wrong** — four independent defects, all verified by execution, not inferred:

1. **Not reproducible.** The module held `RNG = np.random.default_rng(0)` at module scope, consumed
   by BOTH `build_brain()` and `run()`. Every call advanced shared state, so two `build_brain()`
   calls returned different brains (max |diff| **0.17**) and every published figure was one
   unrepeatable draw.
2. **The headline was a single draw.** `FFCr z_d = -0.239` was one realization. At a FIXED setting
   the per-draw sd is **0.035**; 12 independent draws span **-0.279 to -0.167**. One draw is not a
   result.
3. **D_AUD was selected on noise, at a grid boundary.** `main()` chose D_AUD by argmin over
   `np.arange(0.0, 0.25, 0.01)` evaluating ONE draw per point. The winner, **0.24, was the last
   point of that grid** — a boundary solution. The selected value was then reused to report
   performance, so results were conditioned on a parameter fitted to the same draws.
4. **`p=0.0005` was never measured.** It is the Monte-Carlo estimator floor `1/(n_perm+1) = 1/2001`
   at `n_perm=2000`. The only honest statement is `p < 5e-4`.

**What replaces it.** `scripts/sensitivity_surface.py` averages **225 seeded draws per grid point**
over a grid widened to **0.00–0.65**, using seeds **disjoint** from those used to report
(`0..224` select; `100000..100224` report). Result:

| | old | corrected |
|---|---|---|
| D_AUD | 0.24 (argmin of 25 single draws, on the grid boundary) | **0.30** (interior, but see below -- NOT a resolved optimum) |
| FFCr z_d | -0.239 (one draw) | **-0.262 +/- 0.035** (sd, 225 draws) |
| gap to observed -0.244 | claimed "within 0.005" | **0.018**, i.e. ~0.5 sd |
| p | "0.0005" | **< 5e-4 (n_perm=2000)** |
| SSE over 4 ROIs | 0.07985 (at 0.24) | **0.07155** (at 0.30) |

**What did NOT survive, and is now stated plainly:** the simulation reproduces the **sign and
ordering** of the 2026-07-31 pattern, and FFCr/EBA to within ~0.5-1.3 sd. It does **not** reproduce
V1 (-0.136 vs -0.046) or **A1 (+0.035 vs +0.280, ~8x short)**. The phrase **"ordering exact" is
withdrawn**: the relative baseline z of V1/FFCr/EBA is set by hand in `build_brain`, so the ordering
is a stipulated INPUT, not a measured agreement. "FFC within 0.005" is withdrawn.

**Also recorded:** `rho`, the within-parcel correlation of the prediction noise, is now a swept
parameter rather than an implicit `rho=0`. `rho=0` is not neutral — TRIBE's head is
`nn.Linear(hidden, low_rank_head=2048)` (`tribev2/model.py:139-141`, `grids/defaults.py:198`), so its
20,484-vertex output has **rank <= 2048** and spatially independent noise is structurally
impossible. `_parcel_noise` holds the marginal variance fixed (verified 0.0025 at every rho), so
floors across rho differ by noise STRUCTURE, not by total noise.

**Alternatives considered:** (a) leave the numbers and add a caveat — rejected: the docstring is the
public API surface and D026 ground 3 rests on these figures; (b) re-derive the old -0.239 under the
new generator — impossible, and pointless: clip seeding changed, so the old draw is not reproducible
even in principle, which is itself the finding.

**Implications:** D026's ground 3 keeps its conclusion (spatial_z is compositional and can invert a
real effect — that is arithmetic, not simulation-dependent) but its supporting numbers are
superseded by this entry. The commit-message claim *"spatial_z needs a 6.7x larger effect than the
simplest alternative"* (`29cb104`) is a value at ONE corner of a two-parameter space and is
superseded by the measured range in `data/floor_surface.md`:

| rho | spatial_z floor | best alternative | ratio |
|---|---|---|---|
| 0.0 | 0.1726 | 0.0210 | **8.22x** |
| 0.3 | 0.1858 | 0.0426 | **4.36x** |
| 0.6 | 0.1924 | 0.0500 | **3.85x** |
| 0.9 | 0.2074 | 0.0616 | **3.37x** |

**Corrected claim: 3.4x-6.0x against a FIXED comparator (`glm_contrast_z`), not 6.7x.** (Against the best-of-three selected per row it reads 3.4x-8.2x, but that per-row `min()` is itself a selection on the same data and inflates the rho=0 endpoint; against a fixed `roi_minus_reference` it reads 2.6x-8.2x. Lead with the fixed comparator.) The mechanism is that spatial
correlation costs the honest statistics dearly and barely touches the compositional one: across the
rho range `spatial_z`'s floor rises **+20%** while the best alternative's rises **2.9x**. Note the
two corrections push OPPOSITE ways -- fixing the boundary-selected D_AUD (0.24 -> 0.30) RAISES the
ratio at rho=0 from 6.7x to 8.2x, while realistic rho lowers it. 6.7x was not fabricated; it was a
real value at one unstated corner of a two-parameter space, reported as if it were the answer.

**The optimum is interior but NOT resolved, and this is stated rather than glossed.** The winner at
0.30 is separated from its nearest rival (0.35) by **2.43 sigma**. But the grid was searched 14
times, so 13 comparisons were implicitly made; the Bonferroni threshold for family-wise 0.05 is
**2.89 sigma**. 2.43 < 2.89, so **D_AUD in {0.30, 0.35} are statistically indistinguishable**.
0.30 is therefore the best available point estimate of a **stipulated nuisance level**, not a
fitted optimum, and must be reported as such. (An earlier version of the selection code used a bare
2-sigma rule, which would have called this "distinguishable" -- that criterion ignored the number of
looks and has been corrected in `scripts/sensitivity_surface.py`.)

Averaging did work as intended: 225 draws cut the objective's noise **15x**, from a single-draw sd
of 7.7% of the value to a sem of 0.5%. The problem is not insufficient averaging; it is that the
objective is genuinely flat between 0.30 and 0.35.

**A second finding, not anticipated:** the ranking of the three NON-compositional statistics
**reverses with rho**. `roi_minus_reference` has the lowest floor at rho=0 (0.0210) and the
**highest** at every rho >= 0.3 (0.0482 / 0.0668 / 0.0803), because subtracting an off-target
reference cancels a shared gain only when the noise is shared. `glm_contrast_z` is best at all
rho >= 0.3. So "the simplest statistic is the most sensitive" holds only at the one setting we have
shown is impossible for this model. Since **rho for real TRIBE predictions is unmeasured**, the
paper must either recommend `glm_contrast_z` or state the recommendation as conditional. Robust
across the entire range: `spatial_z` is the worst of the four at every rho.

**The rho endpoint was checked for the same boundary problem and is clean.** At rho=0.97 the ratio
is 3.62x, i.e. it RISES again after a minimum at rho=0.9 (3.37x), because spatial_z's floor keeps
degrading while glm_contrast_z's plateaus. The low end of the reported range is therefore bounded
by evidence, not by where the grid stopped.

**Revisit if:** the simulation is ever replaced by measurements from a real prediction cache, which
would make the whole synthetic-brain parameterisation moot. Until then, every figure it produces is
conditional on `build_brain`'s hand-set constants and must be reported as such.

---

### D031: Phase B — seven audited defects corrected; the Fig 4 attribution was false (2026-08-23)

**Decision:** apply the seven must-fix items from `data/audit_s1_SYNTHESIS.md` and
`data/audit_s1_CRITIC.md`, each with a regression test built from the finding's own failing input.

| id | defect | fix |
|---|---|---|
| **M1** | `define_froi(top_n=100)` on the 58-vertex right-FFC parcel returned the WHOLE parcel — `k = min(100, 58) = 58` — so the "fROI" was the unfixed anatomical ROI and S2 would have reported it while believing the ROI was fixed. **No lane filed this; the synthesis found it.** | raise when `top_n >= parcel size`. Selecting everything is not selection. |
| **M2** | pooled (equal-variance) SE in `glm_contrast_z` is not level-alpha at unequal n — the design S2 plans. Measured 10v40, 6:1 sd ratio: pooled z=10.74 vs Welch 5.59, **1.9x anticonservative** on the headline face claim. | Welch SE. Verified against `scipy.stats.ttest_ind` exactly, and identical to pooled at equal n — so the published 15v15 floor table is unchanged. |
| **M3** | non-finite values absorbed silently in five places. `np.argsort` sorts NaN LAST ascending and `[::-1]` promotes it to FIRST, so a dead vertex was ranked **maximally selective** and displaced a real one. | one shared `_require_finite` policy (NaN and both infinities) at every entry point — not five subtly different rules. Hardening: no NaN source is demonstrated in this pipeline. |
| **M4** | the docstring claimed the statistic is *"what Meta's Fig 4 actually reports"*. **False under both readings.** §5.9 describes the visual contrasts as the plain t=+5 s subtraction with **no GLM**; the Fig 4 caption separately describes a GLM on the predicted **time-series**. The shipped code is a two-sample contrast across **observations**. | state what it actually is — a recorded deviation, not a replication. `interface-contracts.md`'s ⚠ OPEN is **resolved**, and the stale restatement in MASTER-PLAN is annotated. |
| **S6** | `event_locked_contrast` accepted a 2-D target and averaged both axes, returning an attenuated contrast (0.0404 for a true 0.0941). `peri_event_timecourse`'s output is always in scope beside it. | reject non-1-D at the contract boundary; raise on an empty category rather than dropping it. |
| **C7** | the overlap guard was defeated by dtype: `np.intersect1d` compares a boolean mask's VALUES (0/1) against integer indices, so total overlap went undetected — the module's only defence against an undeclared normaliser. | normalise both selectors to integer indices before comparing. |
| **C5** | `peak_lag_trs` selected the peak from whatever course it was handed. Choosing the lag on the target category and testing at it is selection on the test statistic: measured type-I **0.0417** against a nominal 0.025. | the function now takes ALL categories' courses and pools them itself, requiring >= 2 — single-category selection is **not expressible** through the API. |

**Also closed (S1):** `glm_contrast_z` had zero value coverage — `return 0.0` *and* a sign flip both
passed the entire suite. Sign, antisymmetry and a magnitude floor are now pinned.

**Two API changes, both deliberate:**
- `define_froi` raises where it previously capped. `test_define_froi_caps_at_parcel_size` asserted
  the no-op was correct and was **rewritten in the same commit** — it had been blessing the bug.
- `peak_lag_trs(timecourse)` becomes `peak_lag_trs(category_timecourses)`. Enforcing the pooling in
  the signature rather than the docstring is the point: a docstring cannot prevent the error.

**Method:** every defect was reproduced against the pre-fix tree first, so each regression test is
built from the finding's own failing input rather than from a description of it.

**Revisit if:** a real NaN source is demonstrated in the prediction pipeline (M3 would move from
hardening to a live defect), or if the paper's §5.9 wording is ever read in full — two independent
fetches truncated before its body, so its exact GLM specification remains **unverified** and is
recorded as a gap, not a fact.

---

## D032 — Phase B closed under a stated stopping rule (2026-08-24)

**Decision.** Stop the adversarial review loop. Do not run a fifth independent reviewer to search
for further theoretical test gaps.

**Stopping rule, pre-committed and now met:**

> no demonstrated live wrong-number defect + independent numerical cross-checks pass + known
> residuals explicitly recorded → stop auditing, move to replication and paper work.

**The claim being made — deliberately narrow:**

> The latest independent audit found no remaining demonstrated wrong-number defect in the
> decision-critical paths; remaining issues are documented test/coverage or defensive-hardening
> gaps.

**Not** "the statistics module is proven correct". Four independent reviews each found something
the previous reviewer and the author had missed. A fifth would probably find something too. What
changed is the *kind* of finding, not the supply.

**Why stop here.** Marginal value of another reviewer is now below the marginal value of running
S2, checking whether the decision-critical analysis replicates, and writing the paper. Five days
remain. Round 4 found zero live wrong-number defects; its yield was two correct-but-unasserted
mechanisms, three operator-error guards, and one regression the author had introduced while fixing
round 3.

**Evidence.** `data/phase_b_closure.md` (live correctness vs residuals, the four C verdicts, the
independent numerical cross-checks), `data/phase_b_invariants.md` (I1-I5 + S6),
`data/phase_b_mutation_ledger.md` (69 mutations; eleven that proved nothing, recorded as such).
86 tests in `test_roi_stats.py`, 124 repo-wide, 69/69 mutations detected against a verified-green
baseline.

**Re-open criterion.** A concrete new correctness concern arising from S2 or from paper
generation — not another audit of this module.

---

## D033 — G024 resolved: Gate 0 loaded TRIBE v2 (2026-08-24)

`tribe_tools/model.py` has hardcoded `TribeModel.from_pretrained("facebook/tribev2")` in **every
commit since the initial commit f03833a (2026-06-09)**, seven weeks before the 2026-07-31 Gate 0
run. `notebooks/03_gate0_v2_validation.ipynb` contains **zero** `from_pretrained` calls and loads
the model only via `from tribe_tools.model import load_model`, so there is no independent load path
that could have reached a different checkpoint.

**Caveat, recorded rather than glossed:** this is a code-path argument over full git history, not a
runtime artifact. The 07-31 results JSON records the tribev2 commit
(`af58661791a351a448a489042a28f6c37e1c14b7`) but no tribe-bench SHA — the known provenance gap. The
conclusion is therefore strong but indirect. **The in-silico FFA/PPA/EBA/VWFA claim we are
replicating exists only for v2, and that claim is safe to rely on.**
