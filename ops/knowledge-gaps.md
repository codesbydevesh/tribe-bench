# Knowledge Gaps — What We Don't Know Yet

This file tracks things we need to learn, verify, or figure out.
Each gap has a priority and a plan for closing it.
When a gap is closed, move it to the "Closed" section with the answer.

---

## TRIBE v2 Internals

### G005: Actual VRAM usage per extractor
**Question:** What is the real peak VRAM when each extractor runs on GPU?
Our estimates (V-JEPA 2 ~10GB FP16, LLaMA ~5GB, Wav2Vec ~1GB) are guesses.
**Priority:** HIGH — determines T4 feasibility
**How to close:** Run smoke test on Kaggle with `torch.cuda.max_memory_allocated()`
after each extractor runs and before `_free_extractor_model()` clears it.
**Status:** OPEN — first Kaggle run (2026-07-22) reached the model-load phase but was blocked
by an import-shadow bug (the clone dir `/kaggle/working/tribev2` shadowed the package as a
namespace package → `ModuleNotFoundError: tribev2.demo_utils`). Fixed 2026-07-22 (clone to
`tribev2_src` + defensive guards). Notebook already instruments per-extractor VRAM via a
`_free_extractor_model` hook — rerun will close this.

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
**Status:** OPEN — code has discovery logic in `_find_features_to_use()`, needs the <=1hr GPU
smoke test (D013). First run (2026-07-22) never reached this — blocked by the tribev2 import
shadow (see G005), now fixed. Rerun pending.

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
