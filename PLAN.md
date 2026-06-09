# TRIBE v2 Research Toolkit — Master Plan

## What We Have

TRIBE v2 (TRImodal Brain Encoder v2) is Meta FAIR's open-source foundation model that
predicts whole-brain fMRI responses from video, audio, and text. It won Algonauts 2025
(1st out of 263 teams), explains ~54% of brain variance, and reaches 0.77-0.85
correlation in key regions.

The architecture: three pretrained encoders — V-JEPA 2 (vision), LLaMA 3.2-3B
(language), and Wav2Vec-BERT (audio) — run as external feature extractors that cache
their outputs to disk. A small fusion model (projectors + 8-layer transformer +
predictor) reads the cached features and outputs ~20,484 vertices on the fsaverage5
cortical surface.

Key finding from source code analysis (2026-06-08): the encoders are NOT part of the
brain model. They run sequentially via the `neuralset` library, each one loading into
GPU, extracting features, caching to disk, and being freed from memory before the next
one loads. Meta already solved the sequential loading problem. See
`ops/source-of-truth.md` for full verified details.

Code and weights are public: github.com/facebookresearch/tribev2 (CC BY-NC 4.0).
API: `TribeModel.from_pretrained("facebook/tribev2")`, `model.predict(events=df)`.

After 1 year public, only 5 papers cite or use TRIBE v2. The ecosystem is empty:
- No public benchmark exists for v2
- No interpretation layer (raw voxels only, no "engagement score" or ROI summaries)
- No aggregate metrics, no hosted API, no tutorials, no community
- No validation against real fMRI for downstream tasks
- No one has built standard tooling around it

We have completed deep research across 13 documents in /tribe-v2/ covering architecture,
gaps (15 identified), build-on-top ideas (11 across 3 tiers), ecosystem analysis, and
multi-perspective execution plans. We have also cloned and read the TRIBE v2 source code,
verifying the actual architecture and internal APIs.

## What We Want to Achieve

We are building **tribe-bench**: the standard research toolkit and benchmark suite that
the neuroscience-AI community will use to work with TRIBE v2. Because the field is so
nascent (5 papers in 1 year), whoever ships the first quality toolkit owns the space.

Our four concrete builds:
1. **BrainLens** — Modality ablation explorer. Feed a video, get RGB brain maps showing
   which regions respond to visual (red), auditory (blue), or language (green) content.
   Works by running TRIBE v2 with `features_to_mask` to isolate each modality.
2. **NeuroGenre** — Genre clustering analysis. Process 60 clips across 6 genres, show
   that TRIBE v2 predictions cluster by genre in UMAP space. Publication-quality figure.
3. **ScaleLaw** — Scaling law replication. Compare TRIBE v2 predictions against real
   fMRI from the StudyForrest dataset (Forrest Gump movie). Plot the scaling curve.
4. **NeuroCheck** — The flagship. A curated database of 50 neuroscience claims
   ("faces activate FFA more than houses") structured as testable contrasts. Run TRIBE v2
   on each, score pass/fail. This is a publishable benchmark WITHOUT running the model.

The shared library (`tribe_tools/`) wraps TRIBE v2's TribeModel with a clean API,
adds brain atlas integration (via MNE's HCP-MMP1), visualization, caching, and
batch inference with checkpointing — reusable across all builds.

## Constraints

- India-based researcher, no credit card, no GPU, no active university affiliation
- Laptop + wifi only for local development
- Viable compute: Kaggle dual T4 (30 hrs/week), Google Colab free T4,
  HuggingFace ZeroGPU (3.5 min/day), Lightning AI free (80 GPU hrs/month)
- T4 feasibility: TRIBE v2's pipeline already loads encoders one at a time.
  Each individual extractor needs to fit in 16GB. VRAM per extractor is UNVERIFIED
  (see ops/knowledge-gaps.md G005) but estimated at ~10GB peak for V-JEPA 2 in FP16.
- All code must run locally without GPU (graceful fallback with "no GPU" messages)
- Non-commercial use only (CC BY-NC 4.0 license)

## Why This Matters

TRIBE v2 is the most capable brain encoding model ever released, but it has zero
infrastructure around it. Nobody has systematically tested if its predictions match
known neuroscience. Nobody has built tools to interpret its output. Nobody has made
it accessible on free-tier hardware. We fill all three gaps. The NeuroCheck benchmark
alone is publishable as a standalone paper. Early mover advantage in a field with
5 papers is massive.

---

## Execution Plan

### Phase 1: Project Setup & Foundation (TODAY)

#### 1.1 Create monorepo structure

```
/home/deveshb/workspace/AI/tribe-bench/
├── PLAN.md                      (this file)
├── CLAUDE.md                    (standing orders for Claude sessions)
├── README.md
├── pyproject.toml
├── .gitignore
├── ops/                         (operational system — 10 files + three-musketeers)
├── tribe_tools/                 (shared library)
│   ├── __init__.py
│   ├── model.py                 (TribeModel wrapper, modality masking)
│   ├── inference.py             (batch inference, progress, checkpointing)
│   ├── atlas.py                 (HCP-MMP1 atlas via MNE, ROI extraction)
│   ├── viz.py                   (nilearn brain surface plots)
│   ├── cache.py                 (HDF5 prediction cache)
│   └── video_utils.py           (ffmpeg wrappers, image-to-video)
├── brainlens/                   (Build 1: modality ablation explorer)
│   ├── __init__.py
│   ├── inference.py             (4-pass ablation via features_to_mask)
│   ├── attribution.py           (per-vertex modality contribution)
│   ├── visualization.py         (RGB brain maps)
│   ├── app.py                   (Gradio UI)
│   └── cli.py                   (CLI entry point)
├── neurogenre/                  (Build 2: genre clustering)
│   ├── __init__.py
│   ├── corpus.py                (clip manifest, yt-dlp download)
│   ├── batch_inference.py
│   ├── analysis.py              (UMAP, clustering, stats)
│   └── visualization.py
├── scalelaw/                    (Build 3: scaling law replication)
│   ├── __init__.py
│   ├── download.py              (StudyForrest download)
│   ├── align.py                 (fMRI-to-fsaverage5 alignment)
│   ├── inference.py
│   ├── correlate.py             (vertex-wise correlation)
│   ├── fit.py                   (curve fitting, R-squared)
│   └── plot.py
├── neurocheck/                  (Build 4: neuroscience benchmark)
│   ├── __init__.py
│   ├── claims.py                (YAML loader, claim dataclass)
│   ├── stimuli.py               (stimulus preparation per claim)
│   ├── contrast.py              (TRIBE v2 inference + stats)
│   ├── atlas.py                 (ROI extraction, vertex masking)
│   ├── scorecard.py             (pass/fail, tables, brain maps)
│   └── claims_db/
│       ├── claims.yaml          (50 structured claims)
│       └── README.md            (database documentation)
├── notebooks/                   (Kaggle/Colab notebooks)
│   ├── 01_setup_test.ipynb
│   ├── 02_brainlens_demo.ipynb
│   └── 03_neurogenre_batch.ipynb
└── scripts/
    ├── apply_compute.md         (checklist for compute applications)
    └── outreach_emails.md       (templates for the 5 research groups)
```

#### 1.2 Write shared library (`tribe_tools/`)

**`tribe_tools/model.py`** — Thin wrapper around TRIBE v2's TribeModel:
- `load_model(device, cache_folder)` — calls `TribeModel.from_pretrained("facebook/tribev2")`
- `predict_single(model, video_path, features_to_mask)` — wraps `model.predict()`
  with modality masking support. Uses `features_to_mask` config to zero out modalities.
- No custom sequential loading needed — TRIBE v2's pipeline handles it internally.
  Each extractor loads, runs, caches to disk, and frees GPU before the next one.

**`tribe_tools/inference.py`** — Batch inference wrapper with:
- Progress bars (tqdm)
- HDF5 checkpointing (resume if session dies)
- Modality masking per-run (full, video-only, audio-only, text-only)

**`tribe_tools/atlas.py`** — HCP-MMP1 atlas integration via MNE:
- Wraps TRIBE v2's own `get_hcp_labels()` from `tribev2/utils.py`
- `get_vertices(region_name)` -> vertex indices (uses `get_hcp_roi_indices()`)
- `summarize_by_roi(data)` -> mean activation per ROI (uses TRIBE v2's function)
- `get_topk_rois(data, k)` -> top-k activated regions
- Supports wildcard matching: "V1*" matches all V1 subregions

**`tribe_tools/viz.py`** — Brain surface visualization:
- `plot_activation(data, title, cmap)` using nilearn
- `plot_contrast(a, b, title)` for A-vs-B maps
- `plot_rgb_attribution(visual, auditory, language)` for BrainLens maps
- `export_html(fig)` for interactive plotly export

**`tribe_tools/cache.py`** — HDF5-based prediction cache:
- Key = hash of input file + modality mask config
- Saves predictions + metadata
- Essential for Kaggle (persist across sessions)

**`tribe_tools/video_utils.py`** — Media preprocessing:
- `extract_audio(video_path)` via ffmpeg
- `image_to_video(image_path, duration=3)` for static images
- `segment_video(video_path, segment_duration=30)`

#### 1.3 Write BrainLens MVP

The first visible output. CLI script that:
1. Takes a video path
2. Runs 4 TRIBE v2 passes using `features_to_mask`:
   - full (no mask), video-only (mask audio+text), audio-only (mask video+text),
     text-only (mask video+audio)
3. Computes per-vertex modality contribution
4. Saves a PNG brain map (red=visual, blue=auditory, green=language)

#### 1.4 Start NeuroCheck claims database

The zero-compute moat. Create `neurocheck/claims_db/claims.yaml` with the first
20 structured claims. Each claim has:
- ID, plain-English claim, citation
- ROI (HCP-MMP1 region label), expected contrast direction
- Stimulus A and B specifications
- Effect size threshold

This is publishable as a benchmark paper WITHOUT running the model.

---

### Phase 2: Validate on GPU (Week 1-2)

#### 2.1 Smoke test on Kaggle T4

`notebooks/01_setup_test.ipynb`:
- Install tribev2 and tribe-bench
- Run `TribeModel.from_pretrained("facebook/tribev2")` — verify it loads
- Run prediction on a single short clip
- Measure VRAM per extractor with `torch.cuda.max_memory_allocated()`
- Verify output shape: `(n_kept_segments, n_vertices)`
- Record all measurements in `ops/source-of-truth.md`

#### 2.2 Check dependency availability

Verify that `neuralset`, `neuraltrain`, and `exca` (Meta's internal libraries)
are pip-installable. If not, determine the installation path — they may be bundled
in the tribev2 repo or available through Meta's PyPI.

#### 2.3 Test modality masking

On Kaggle, run TRIBE v2 with different `features_to_mask` configs:
- `features_to_mask=[]` (full)
- `features_to_mask=["audio", "text"]` (video-only)
- Verify outputs differ (confirms masking works)

---

### Phase 3: First Results (Week 2-4)

#### 3.1 BrainLens demo on Kaggle

Run the 4-pass ablation on a single video clip. Generate the first brain map:
- A demo screenshot for emails and applications
- Validation that the toolkit works end-to-end

#### 3.2 Deploy BrainLens on HuggingFace ZeroGPU

Create a Gradio app in a HuggingFace Space:
- Uses `@spaces.GPU` decorator
- 3.5 min/day = enough for 1-2 clips (each pass needs its own extractor run)
- Live demo link for all outreach

#### 3.3 NeuroGenre batch inference

On Kaggle (30 hrs/week), process 60 clips:
- 10 per genre x 6 genres
- Generate UMAP scatter plot
- If genres cluster, that is a publication-quality figure

#### 3.4 Complete NeuroCheck claims database (50 claims)

Finish curating all 50 claims. Release on HuggingFace Datasets.
Write the benchmark design paper (publishable WITHOUT GPU results).

#### 3.5 Send outreach emails

Contact the 5 TRIBE v2 research groups + Meta FAIR:
1. Benchetrit et al. (synthetic fMRI)
2. Bladon & Bent (feature visualization)
3. Meta FAIR TRIBE v2 authors
4. Algonauts organizers
5. StudyForrest maintainers

---

### Phase 4: Papers & Scale (Week 5-8)

#### 4.1 ScaleLaw replication

Download StudyForrest data (~20GB). Run TRIBE v2 on Forrest Gump.
Compare against real fMRI. Plot scaling curve. Submit to bioRxiv.

#### 4.2 NeuroCheck with results

Run the 50 claims through TRIBE v2. Score pass/fail.
Write up the flagship paper. Submit to bioRxiv.

#### 4.3 SynBrain

Generate synthetic fMRI for 1000 COCO images. Train a Ridge regression decoder.
Show improvement over real-only baseline.

---

## What We Build in the First Session

Priority order:

1. **Project skeleton** — pyproject.toml, .gitignore, README, directory structure
2. **`tribe_tools/` shared library** — model.py (TribeModel wrapper), inference.py,
   atlas.py, viz.py, cache.py, video_utils.py
3. **`brainlens/` MVP** — inference.py, attribution.py, visualization.py, cli.py
4. **`neurocheck/claims_db/claims.yaml`** — first 20 structured claims
5. **Kaggle notebook skeleton** — 01_setup_test.ipynb

---

## Verification Criteria

- `pip install -e .` succeeds in the monorepo
- `python -c "from tribe_tools import model, inference, atlas, viz, cache"` imports clean
- `python -m brainlens.cli --help` shows usage
- Claims YAML validates with a simple Python script
- All code runs locally without GPU (graceful "no GPU" messages)

---

## Key Technical Reference (Verified from Source Code)

- TRIBE v2 API: `TribeModel.from_pretrained("facebook/tribev2")`,
  `model.predict(events=df)` returns `(preds, segments)`,
  `model.get_events_dataframe(video_path=...)`
- Source: `/home/deveshb/workspace/AI/tribev2-source/`
- Brain model: `FmriEncoderModel` in `tribev2/model.py` — projectors + transformer + predictor
- Encoders: external extractors via `neuralset`, loaded/freed sequentially in `main.py`
- Modality ablation: `data.features_to_mask` config zeros out specified modalities
- Missing modalities: model substitutes zeros (model.py:190-192)
- Atlas: HCP-MMP1 via MNE in `tribev2/utils.py` — `get_hcp_labels()`,
  `get_hcp_roi_indices()`, `summarize_by_roi()`
- Output shape: `(n_kept_segments, n_vertices)` — segments with no events are dropped
- Surface mesh: fsaverage5, vertex ordering: left hemi first (0 to N-1), right hemi (N to 2N-1)
