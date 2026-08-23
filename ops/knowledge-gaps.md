# Knowledge Gaps — What We Don't Know Yet

This file tracks things we need to learn, verify, or figure out.
Each gap has a priority and a plan for closing it.
When a gap is closed, move it to the "Closed" section with the answer.

---

## Standing methodological rules (earned the hard way)

### M001: Verify acceptance on the DELIVERED artifact, never on the proxy used to build it
**Rule:** whenever a selection or threshold is chosen using a cheap approximation (a downscaled scan, a
sampled subset, a smaller model), the acceptance test must be re-run on the exact artifact that will be
used, and on EVERY leg of the criterion — not the legs that are convenient.
**Why:** this failed twice on 2026-07-30. The Gate 0 v3 stimulus set was matched on motion from a 160 px
film scan while its acceptance criterion is evaluated on the delivered 480 p clips; on the real files it
gave motion AUC 0.631 against its own 0.10 cap and had to be thrown away. The discrepancy between the two
measurements had even been WRITTEN DOWN, and only the p-value leg was re-checked, not the AUC leg.
Earlier the same day the same shape of error appeared in the D022 review. See D024(a).
**How to apply:** state the criterion, then evaluate it once on the shipped artifact and paste the numbers
next to the artifact. If a proxy was used for search, say so and give both numbers.

### M002: Balance a rank test in ranks
**Rule:** if the decision statistic is a rank test (Mann-Whitney U, as G1 is), certify covariate balance
with a rank measure (AUC), not with a mean-based one.
**Why:** raw standardised mean difference on a positive, wide-range energy measure is outlier-dominated.
The rejected v3 set had raw SMD -0.097 (comfortably passing) while its rank separation AUC 0.631 failed.
Supplement with SMD on the LOG of the measure. See D024(a).

### M003: An automated stimulus label is not a label until it has been looked at
**Rule:** any detector-assigned stimulus category gets a mandatory visual audit of the frames that carry
the most weight, at the smallest scale that would reveal a mislabel.
**Why:** three real defects in the Gate 0 sets were invisible to every automated check and obvious on
sight — a clip passing a window-average face criterion while its first frame was a wide establishing
shot; a corridor shot admitted by a persistent single-cascade false positive (one cascade reported a face
on every sample, the two cascades agreed on none); and two "independent" clips that were adjacent shots
of one scene. See D023(g).

### M004: A contrast needs a positive control measured at the SAME contrast scale
**Rule:** an experimental contrast is only interpretable next to a positive control that shares its
scale — same film, same n, same clip length, same statistic. A control run at an easier scale proves
the pipeline emits output; it does not prove the pipeline can resolve the comparison you actually made.
**Why:** Gate 0 v3b's control was PPA scene>face across TWO films (d=+2.529, U=120/120, perfect
separation) while the experimental contrast was within one film. The run therefore cannot distinguish
"no face selectivity" from "no within-film sensitivity to anything" — and the one within-film contrast
that should have been near-guaranteed (FACE clips carry 18% more speech → auditory cortex) came out
A1 +0.280, p=0.1448, not significant. See D025 / G021.

### M005: Check whether your summary statistic is compositional before you interpret its sign
**Rule:** if a statistic normalises within a unit (subtracts a mean, divides by an sd, takes a share),
its components are constrained to sum to a constant. A negative on one component may be nothing but a
positive somewhere else. Verify by testing the normaliser itself for a condition difference.
**Why:** `spatial_z` makes every clip's z-map mean exactly 0 and sd exactly 1, so the brain-wide sum of
any condition delta is exactly zero. Gate 0 v3b's three negative visual ROIs and one positive auditory
ROI are exactly what a normaliser shift produces with zero category information present. See G020.

---

## Gate 0 methodology (HIGH — blocks interpreting the NO-GO)

### G020: Is Gate 0's NO-GO a face result or a zero-sum-statistic artifact?
**Question:** `spatial_z = (g[verts].mean() - g.mean()) / g.std()` forces every clip's z-map to mean 0
and sd 1, so ROI z is a share of a fixed budget and brain-wide condition deltas sum to exactly zero.
FACE clips carry 18% more speech (23.3 vs 19.8 words, p=0.157) and the model's output mass sits in
auditory/STS on this material. Did an off-target auditory difference mechanically drive FFC/V1/EBA
negative, with no face information involved either way?
**Priority:** HIGH — decides whether the NO-GO is a finding or a measurement failure, and therefore
whether D017's pivot or a Gate 0 v4 is the correct next move.
**How to close:** on the cached predictions (CPU, minutes) — (a) test `g.mean()` and `g.std()` FACE vs
NONFACE; (b) recompute FFC z against a reference mask excluding auditory+STS and check whether the sign
survives; (c) FFC−EBA per clip, permuted (compositional-immune); (d) rank all 360 parcels by
FACE−NONFACE delta, both passes, and locate FFC. Code in LOOSE-ENDS thread 1.
**Status:** OPEN — needs the HDF5 prediction cache from the 2026-07-31 Kaggle session.

### G021: Does the Gate 0 v3b design detect ANY within-film contrast?
**Question:** the only positive control in the run is cross-film (PPA, Charade vs McLintock). No
within-film control was specified. Even the near-guaranteed within-film speech→auditory contrast was
non-significant (A1 +0.280, p=0.1448). So we have no evidence the design has within-film sensitivity
to anything, and a null on faces cannot be attributed to faces specifically.
**Priority:** HIGH — same decision as G020; also a mandatory design element for any Gate 0 v4 (M004).
**How to close:** pooled auditory+STS parcels FACE vs NONFACE within Charade, full and video-only, on
the cached predictions. Significant → the design has sensitivity and the face null is real.
Null → the design is insensitive and the face null is uninformative.
**Status:** OPEN — same cache dependency as G020.

---

## TRIBE v2 Internals

### G005: Actual VRAM usage per extractor
**Question:** What is the real peak VRAM when each extractor runs on GPU?
Our estimates (V-JEPA 2 ~10GB FP16, LLaMA ~5GB, Wav2Vec ~1GB) are guesses.
**Priority:** HIGH — determines T4 feasibility
**How to close:** Run smoke test on Kaggle with `torch.cuda.max_memory_allocated()`
after each extractor runs and before `_free_extractor_model()` clears it.
**Status:** ✅ CLOSED 2026-07-22 (full Kaggle run). **Fits a T4 with headroom.** Overall peak
**11.12 GB** on a 15.6 GB T4 (~4.5 GB free). Per-extractor peaks (brain model ~0.77 GB resident
throughout, so these are stage totals): **text 7.17, audio 10.19, video 11.12 GB**; brain-model
forward pass alone 0.77 GB. The V-JEPA2 video encoder is the high-water mark. Our pre-mortem
guesses (video ~10, LLaMA ~5, Wav2Vec ~1) were in the right ballpark but the peak is higher
(~11 GB) because the brain model stays resident during each extractor. Output `(31, 20484)`,
load 13.7s, inference 1176.7s (~20 min, dominated by the T4 video encode + one-time weight
downloads). Measured via the `_free_extractor_model` hook. Got here after fixing the import
shadow + a numpy load-order issue (see source-of-truth "SECOND KAGGLE RUN").

### G016: neuralset/neuraltrain pip availability
**Question:** Are `neuralset`, `neuraltrain`, and `exca` pip-installable from
PyPI? Or are they Meta-internal libraries bundled in the tribev2 repo? If not
pip-installable, what's the install path?
**Priority:** CRITICAL — can't install tribev2 without these
**How to close:** Try `pip install neuralset neuraltrain exca` on Kaggle.
Check if they're in the tribev2 repo's setup.py/pyproject.toml dependencies.
**Status:** CLOSED 2026-07-21 — all three are real, public, pure-Python (`py3-none-any`)
PyPI wheels from Meta FAIR (`facebookresearch/neuroai`), version 0.0.2. NOT private/internal.
The existential "DOA on free hardware" risk is resolved false. Remaining nuance (confirm at
first install, not a blocker): `neuralset 0.0.2` requires Python >=3.12 — if Kaggle is on
3.11, install via uv/conda. "Exists on PyPI" is not yet "installs cleanly with all transitive
deps on a T4" — the smoke test confirms that.
**✅ CONFIRMED ON HARDWARE 2026-07-22 (first Kaggle run):** Kaggle is **Python 3.12.13** (so the
>=3.12 caveat is moot), `git clone` + `pip install -e tribev2` returned **rc=0**, neuralset/
neuraltrain/exca resolved, torch 2.6.0+cu124, 2× Tesla T4 15.6GB, ffmpeg + uvx present, numpy
2.0.2, HF login OK. The install genuinely works on Kaggle's free image. **G016 fully closed —
this was the ~40%-fatal pre-mortem risk, now empirically dead.**

### G018: features_to_use mutation path on TribeModel
**Question:** What is the exact attribute path to `features_to_use` on a loaded
TribeModel instance? Our code probes multiple candidate paths (`model.data.features_to_use`,
`model.xp.cfg.data.features_to_use`, etc.) but we need GPU verification.
**Priority:** HIGH — blocks BrainLens modality ablation. **NOW THE TOP LIVE RISK
(2026-07-21)** now that G016 is closed. Upstream has NO supported inference-time mask API:
`tribev2/model.py:190` zeros a modality only when it is absent from the batch; `:212` is
training-only `modality_dropout`. If none of the 4 probed paths work, BrainLens produces
identical maps for every modality and the demo is fake → pivot to NeuroCheck-only (D012/D013).
**How to close:** Load TribeModel on GPU and inspect `dir(model)`, `dir(model.data)`,
etc. to find where `features_to_use` lives. Then verify that mutating it before
`model.predict()` controls which extractors run — assert full vs modality-removed outputs DIFFER.
**Status:** ✅ CLOSED 2026-07-22 (full Kaggle run) — **ABLATION WORKS. BrainLens's core mechanic
is real.** `_find_features_to_use()` resolved to `Data.features_to_use = ['text','audio','video']`
(path #1, exactly as predicted from source). Masking audio+text (video-only pass) vs the full
pass gave **`max abs diff 0.65430`** — clearly non-zero → mutating `features_to_use` before
`predict()` genuinely controls which extractors run and changes the output. `features_to_mask`
config confirmed inference-inert (`[]`). **BrainLens is NOT cut — the D012/D013 "pivot to
NeuroCheck-only if ablation fails" contingency does NOT trigger.** Note: the video-only pass
reused cached video features (fast, ~1s dataloader), so re-running ablations is cheap once the
full pass is cached.

---

## Atlas & Neuroimaging (MEDIUM)

### G007: Vertex-to-region mapping precision
**Question:** When we map vertex 7342 to "FFA", how precise is this? Is there
ambiguity at region boundaries? Do different atlas versions disagree?
**Priority:** MEDIUM — affects claim accuracy
**How to close:** Compare Glasser region boundaries with published FFA coordinates
**Status:** OPEN

---

## Neuroscience (MEDIUM — needed for claims curation)

### G009: Modern consensus on FFA selectivity
**Question:** Is the "FFA is face-selective" claim still the consensus? Recent
work suggests it may respond to expertise in general, not just faces.
**Priority:** MEDIUM — affects claim NC001
**How to close:** Read Kanwisher's latest review papers (2020+)
**Status:** OPEN

### G010: Glasser region labels for classic areas
**Question:** What Glasser region labels correspond to: FFA, PPA, EBA, STS,
Broca's area, Wernicke's area, primary visual cortex, primary auditory cortex?
**Priority:** HIGH — needed for writing claims
**How to close:** Cross-reference Glasser 2016 paper Table S1 with classic anatomy
**Status:** OPEN

---

## Engineering (MEDIUM — needed before GPU sessions)

### G011: HuggingFace auth for LLaMA 3.2
**Question:** What is the exact process for getting access to LLaMA 3.2-3B
through HuggingFace? Is there a waitlist? How long does approval take?
**Priority:** MEDIUM — blocks any model loading
**How to close:** Go to huggingface.co/meta-llama, check access request process
**Status:** OPEN

### G012: Kaggle persistent storage
**Question:** Where exactly should we save HDF5 caches on Kaggle so they
persist between sessions? `/kaggle/working/` is session-scoped. Is there
a way to save to a Kaggle dataset?
**Priority:** MEDIUM — needed for batch jobs
**How to close:** Read Kaggle docs on persistent storage, test with a small file
**Status:** OPEN

### G013: ffmpeg availability on Kaggle/Colab
**Question:** Is ffmpeg pre-installed on Kaggle and Colab? If not, can we
install it? video_utils.py depends on it.
**Priority:** LOW — easy to work around
**How to close:** Run `which ffmpeg` on both platforms
**Status:** OPEN

---

## Publishing (LOW — needed later)

### G014: bioRxiv submission without affiliation
**Question:** Can we submit to bioRxiv as "Independent Researcher"? Do they
require institutional email? Is there a verification process?
**Priority:** LOW — not needed until paper is ready
**How to close:** Read bioRxiv submission guidelines
**Status:** OPEN

### G015: arXiv endorsement
**Question:** arXiv requires an endorsement from an existing author for first
submissions. How do we get one? Can any of the 5 groups endorse us?
**Priority:** LOW — bioRxiv is the primary target anyway
**How to close:** Check arXiv endorsement policy, consider asking collaborators
**Status:** OPEN

---

## Closed Gaps

| ID | Question | Answer | Closed Date |
|----|----------|--------|-------------|
| G001 | Can each encoder be loaded independently? | YES — encoders are EXTERNAL feature extractors via `neuralset`, not part of the brain model. They run sequentially, cache to disk, and free GPU via `_free_extractor_model()` (main.py:59-79). | 2026-06-08 |
| G002 | What are the exact encoder attribute names? | They're NOT attributes on the model. They're configured as `data.text_feature`, `data.audio_feature`, `data.video_feature` in the Data config (main.py:98-100). The brain model (`FmriEncoderModel`) only has `self.projectors` keyed by modality name. | 2026-06-08 |
| G003 | What is the tensor shape at the fusion boundary? | Each modality is projected to `hidden // len(feature_dims)` dims (model.py:119-123), then concatenated to `hidden` (256 by default). Input to each projector: `(B, num_layers * feature_dim, T)` with layer_aggregation="cat". | 2026-06-08 |
| G004 | How is modality dropout implemented? | Per-sample random mask: `mask = torch.rand(data.shape[0]) < self.config.modality_dropout` (model.py:212). Zeros out the entire projected modality tensor for selected batch elements. Confirms zeroing is a valid ablation. | 2026-06-08 |
| G006 | Does nilearn provide Glasser on fsaverage5? | WRONG QUESTION — TRIBE v2 uses MNE, not nilearn. `get_hcp_labels()` in utils.py calls `mne.datasets.fetch_hcp_mmp_parcellation()` and maps to fsaverage5. Our atlas.py wraps these MNE-based functions directly. | 2026-06-08 |
| G008 | What is TRIBE v2's vertex ordering? | Left hemisphere first (indices 0 to expected_size-1), right hemisphere second (indices expected_size to 2*expected_size-1). Confirmed in utils.py:242 `index_offset = expected_size if hemi == "right" else 0`. | 2026-06-08 |
| G017 | Exact feature names for features_to_mask? | Feature names are "video", "audio", "text" — confirmed from main.py:98-100 `data.video_feature`, `data.audio_feature`, `data.text_feature`. The `features_to_mask` config uses these names (main.py:486-504). However, `features_to_mask` only works at model construction time, not at inference. For runtime ablation, mutate `features_to_use` instead. | 2026-06-08 |
| G016 | neuralset/neuraltrain/exca pip-installable? | YES — all three are real, public, pure-Python (`py3-none-any`) PyPI wheels from Meta FAIR (`facebookresearch/neuroai`), v0.0.2. NOT private. Existential "DOA on free hardware" risk resolved false. `neuralset 0.0.2` needs Python >=3.12; confirm clean install (transitive deps) at first Kaggle run. | 2026-07-21 |
| G005 | Real peak VRAM per extractor on GPU? | FITS A T4. Overall peak 11.12 GB / 15.6 GB (per-extractor: text 7.17, audio 10.19, video 11.12; brain fwd 0.77). Video encoder = high-water mark. | 2026-07-22 |
| G018 | features_to_use mutation path + does ablation change output? | WORKS. Path = `Data.features_to_use` (#1). Video-only vs full max abs diff 0.654 (non-zero) → masking controls extraction. BrainLens mechanic confirmed; NOT cut. | 2026-07-22 |
| G011 | HF auth for LLaMA 3.2 | Resolved in practice — `HF_TOKEN` Kaggle secret + granted LLaMA-3.2 gated access; text extractor loaded LLaMA-3.2 fine in the 2026-07-22 run. | 2026-07-22 |
| G013 | ffmpeg on Kaggle? | YES — `ffmpeg` (and `uvx`) present on PATH on the Kaggle T4 image (2026-07-22 run). | 2026-07-22 |
| G019 | Why does a pass die with "Ratio of unmatched words is X while AddSentenceToWords.max_unmatched_ratio=0.05"? | **REOPENED 2026-07-29 — MITIGATED, NOT CLOSED.** Mechanism confirmed: `tribev2/demo_utils.py:85` builds `AddSentenceToWords(max_unmatched_ratio=0.05)` at call time; `neuralset` 0.0.2 `text.py:215-224` raises when the fraction of Word rows with an empty `sentence` field exceeds it; a 10s clip yields 9-35 words, so under 20 words one unalignable word = 0.11 > 0.05. **But the 2026-07-28 fix rationale was wrong.** The guard is NOT cosmetic: an unmatched word gets `sentence=""` (`text.py:186`), `AddContextToWords` gives it `context=""` (`text.py:271-274`), and `RemoveMissing` DELETES the row (`basic.py:52-54`), whose TRs then receive exact zeros (`extractors/base.py:250-262, 302-305`). So widening the ratio trades a loud abort for silent, condition-correlated word deletion. Tolerance is now **0.15** (admits the observed 1/9, rejects 2/9) and deletion is instrumented per clip into `gate0v2_results.json` — see D023(b). **Residual, not fixed:** (1) deletion is not actually bounded by the ratio, because `utils.py:298-311` back-fills `sentence` without `sentence_char` on gap words so they count as matched and are still deleted; (2) a clip with ZERO ASR words returns at `text.py:174-176` without raising at ANY tolerance and silently loses the whole text modality (pre-existing, present in Run 1 too). NOTE: `predict_single()` builds events before applying `features_to_mask`, so ASR runs even for video-only passes — the patch must precede every pass loop. | 2026-07-29 |

### M006: Read the model's own paper before designing a test of the model
**Rule:** before building any experiment that probes a model's behaviour, read that model's own
paper and repo for the experiment you are about to run. Record what they did, their stimulus design,
and their statistic, in `source-of-truth.md`, before designing anything.
**Why:** Gate 0 was designed and rebuilt four times (D020, D021, D023, D024) to test whether TRIBE
recovers face selectivity. TRIBE v2's own paper — titled "A foundation model of vision, audition and
language for **in-silico neuroscience**" — already reports recovering FFA, PPA, EBA and VWFA, using
1 s-flashed images at 8 s ISI and a raw t=+5 s across-category contrast. We used 10 s naturalistic
clips and a compositional spatial z-score, differing on every axis, and spent ~4 GPU-hours plus ten
days reaching an uninterpretable null. None of this was in our record until 2026-08-04. See G022.

> **M007 is intentionally absent from this file.** It is an operational rule about the shared
> development machine and it is kept local-only in `.notes/BRIEF_ME.md`, because this repo is public
> (D016) and the rule carries the employer's infrastructure details. The numbering gap is deliberate.

### M008: Fix the CLASS, not the reported instance — then check the neighbours
**Rule:** when an audit reports a defect, fix every place that shares its mechanism, not the one
example that was filed. Before declaring a fix done, ask: (a) does the same pattern exist in a
sibling function? (b) can the bad behaviour be reached by a different representation of the same
input? (c) does the guard sit on every argument, or only the one that was reported? Then write the
test against the CLASS, not the example.
**Why:** on 2026-08-23 all seven Phase B fixes passed their own regression tests and were confirmed
by two independent mutant batteries — and **four of them were still incomplete**, each in exactly
this way:
- `_as_vertex_indices` was written to fix the boolean/int overlap bug and then **not applied to
  `define_froi`**, the one selector entry point that most needed it (a bool-mask parcel returns ten
  copies of vertex 0). It also handles only `dtype == bool`, so an `int8` mask now returns a WRONG
  ROI value — worse than the bug it replaced, which only defeated a guard.
- The 2-D rejection in `event_locked_contrast` was placed on `target_responses` and not on
  `other_responses`, where the same array shape does the same damage.
- `peak_lag_trs` was given a `>= 2 categories` rule to prevent selecting on the target's own
  timecourse. `[tc, tc]` satisfies it. Type-I 0.2032 against a nominal 0.025. The restriction was
  syntactic; the statistical dependence survived untouched.
- `_require_finite` was added at five entry points under a docstring claiming "ONE policy, used at
  every entry point". Seven other public paths still absorb non-finite input, and `u_statistic`
  returns a FINITE WRONG answer for a NaN.
**How to apply:** after fixing, grep the module for the same construct (`argsort`, a bare `>`/`<=`
on possibly-NaN data, `np.asarray(..., dtype=int)`, an unguarded argument) and list every hit before
closing the finding. If a docstring says "everywhere", a test must prove "everywhere" or the
docstring must be narrowed.

### M009: A green suite is not a safety net until you have tried to break it
**Rule:** for any decision-critical function, run mutation testing before trusting its tests.
Replace the body with a constant, flip a sign, change a `ddof`, swap an axis, delete a guard — and
see what survives.
**Why:** on 2026-08-23, **55 of 66 one-line mutations survived a 79-test suite**. Three functions
had direction-only coverage with zero value coverage, so `raw_roi_mean`'s `g[verts].mean()` could be
replaced by `g.mean()` — **ignoring the ROI entirely** — and every test still passed. This is the
same gap that was already recorded for `glm_contrast_z` (`return 0.0` and a sign flip both passed);
it had simply never been checked for the neighbouring functions.
**How to apply:** the cheap version is a handful of hand-written mutants per critical function. The
standard: if reverting the implementation while keeping the test does not fail the test, the test
proves nothing.

## Positioning / prior art (HIGH — decides what may be claimed)

### G022: What has already been published on TRIBE v2, and by whom?
**Question:** the record contained nothing about TRIBE v2's own in-silico results, its training/test
split, or the three 2026 groups working the same lane. Standing gap: keep this current.
**Priority:** HIGH — it decides what can be claimed as a contribution.
**How to close:** maintain the occupancy map in `MASTER-PLAN.md` §3.12; re-scan before any submission.
**Status:** PARTLY CLOSED 2026-08-04. Known now: TRIBE v2 = arXiv 2605.04326, weights 2026-03-24,
CC-BY-NC-4.0, ungated; training 25 subjects / 451.6 h (CNeuroMod 4/268.7, BoldMoments 10/61.9,
Lebel2023 8/85.8, Wen2017 3/35.2); testing 695 subjects / 666.1 h (NNDb, LPP, Narratives, HCP 7T).
Our "700-subject model" line describes the TEST set. TRIBE v1 (arXiv 2507.22229,
facebookresearch/algonauts-2025) is a DIFFERENT, earlier model — see G024.

### G023: What does an in-silico category contrast on this model need to have sensitivity?
**Question:** n per condition, stimulus duration, flashed vs naturalistic, fROI definition,
statistic, number of source films. Our null varied none of these deliberately.
**Priority:** HIGH — this is now the project's primary empirical question (MASTER-PLAN S2/S3).
**How to close:** S2 (replicate the published protocol) then S3 (cross duration × n × ROI-definition
× statistic on a multi-film corpus, with negative and deliberately-underpowered controls).
**Status:** OPEN.

### G024: Did Gate 0 run TRIBE v2 or TRIBE v1?
**Question:** the in-silico localizer result exists only for v2 (`facebook/tribev2`). If any run
loaded v1 weights from `facebookresearch/algonauts-2025`, its interpretation changes entirely. The
07-31 results JSON records the tribev2 commit `af58661791a351a448a489042a28f6c37e1c14b7` but the
tribe-bench clone FAILED that session and no tribe-bench SHA was recorded.
**Priority:** HIGH — cheap to close, and it gates the interpretation of every result so far.
**How to close:** check the checkpoint identifier and `from_pretrained` argument in the notebook and
in `tribe_tools/model.py:load_model`; confirm against the HF model card.
**Status:** OPEN.
