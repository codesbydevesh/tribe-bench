# Source of Truth — Verified TRIBE v2 Internals

Every fact we use to write code is listed here. Each fact is marked:
- **VERIFIED** — confirmed by reading the actual source code in tribev2/
- **FROM-DOCS** — from Meta's paper, blog, or README (not source-verified)
- **ASSUMED** — inferred from documentation, not directly confirmed

Source code location: `/home/deveshb/workspace/AI/tribev2-source/`

NEVER write code against an ASSUMED fact without attempting verification first.
When you verify something, update this file immediately.

---

## CRITICAL DISCOVERY (2026-06-08)

**The encoders (V-JEPA 2, LLaMA, Wav2Vec) are NOT part of the brain model.**

They run as external feature extractors via the `neuralset` library. The pipeline:
1. Each extractor loads its encoder, runs it on the input, caches features to disk
2. `_free_extractor_model()` (main.py:59-79) deletes the encoder from GPU and calls `gc.collect()` + `torch.cuda.empty_cache()`
3. The next extractor loads, runs, caches, frees
4. The brain model (`FmriEncoderModel`) then reads cached features and runs the fusion

**This means sequential loading is already built into TRIBE v2's pipeline.**
We do NOT need to implement our own sequential loading hack.
The T4 feasibility depends on whether each individual extractor fits in 16GB, not the total.

---

## Model Architecture

| Fact | Status | Source |
|------|--------|--------|
| Three encoders: vision, language, audio | VERIFIED | main.py:98-100 `data.text_feature`, `data.audio_feature`, `data.video_feature` |
| Encoders are EXTERNAL feature extractors, not part of the brain model | VERIFIED | main.py:171-173, main.py:214-217 |
| Encoders cache features to disk via `extractor.prepare(events)` | VERIFIED | main.py:216 |
| Encoders are freed from GPU after caching: `_free_extractor_model()` | VERIFIED | main.py:59-79, main.py:217 |
| Brain model class: `FmriEncoderModel` | VERIFIED | model.py:89 |
| Brain model config class: `FmriEncoder` | VERIFIED | model.py:49 |
| Brain model has: projectors (per modality), combiner, transformer encoder, predictor | VERIFIED | model.py:103-157 |
| Extractor aggregation: "cat" (concatenate modality features) | VERIFIED | model.py:62, default value |
| Layer aggregation: "cat" (concatenate layer features) | VERIFIED | model.py:63, default value |
| Modality dropout: zeros out entire modality for a batch element | VERIFIED | model.py:211-213 |
| Modality dropout is per-sample random mask, not per-modality toggle | VERIFIED | model.py:212 `mask = torch.rand(data.shape[0]) < self.config.modality_dropout` |
| Missing modalities are handled by substituting zeros | VERIFIED | model.py:190-192 |
| Output shape from brain model: `(B, n_outputs, T)` | VERIFIED | model.py:175-178 |
| Temporal resolution: 2 Hz (from neuro extractor frequency) | FROM-DOCS | architecture.md, confirmed by Data.TR property in main.py:146 |

## Forward Pass Chain (VERIFIED)

```
FmriEncoderModel.forward(batch):
  1. aggregate_features(batch)          → (B, T, H)     [model.py:164]
     - For each modality in feature_dims:
       - If modality in batch.data: project features via self.projectors[modality]
       - If modality NOT in batch.data: substitute zeros
       - Apply modality_dropout if training
     - Concatenate all modality tensors on dim=-1 (cat) or dim=1 (stack) or sum
  2. temporal_smoothing (if configured)  → (B, T, H)     [model.py:166-167]
  3. transformer_forward(x, subject_id)  → (B, T, H)     [model.py:169]
     - combiner MLP
     - add time positional embedding
     - add subject embedding (if configured)
     - transformer encoder
  4. low_rank_head (if configured)       → (B, T, bottleneck) [model.py:171-172]
  5. predictor (SubjectLayers)           → (B, O, T)     [model.py:173]
  6. pooler (AdaptiveAvgPool1d)          → (B, O, T')    [model.py:175]
```

## TribeModel Wrapper (VERIFIED)

| Fact | Status | Source |
|------|--------|--------|
| TribeModel extends TribeExperiment | VERIFIED | demo_utils.py:133 |
| `from_pretrained()` downloads config.yaml + best.ckpt from HuggingFace | VERIFIED | demo_utils.py:150-241 |
| `from_pretrained()` loads checkpoint, builds model, loads state_dict | VERIFIED | demo_utils.py:228-241 |
| Model is stored in `xp._model` | VERIFIED | demo_utils.py:240 |
| `predict()` returns `(preds, all_segments)` | VERIFIED | demo_utils.py:322-392 |
| `preds` shape: `(n_kept_segments, n_vertices)` — note: NOT `(T, 20484)` | VERIFIED | demo_utils.py:378-381 |
| Segments with no events are removed by default (`remove_empty_segments=True`) | VERIFIED | demo_utils.py:148, 370-376 |
| `get_events_dataframe()` accepts exactly one of: text_path, audio_path, video_path | VERIFIED | demo_utils.py:243-320 |
| For video input: extracts audio, chunks long clips, transcribes words | VERIFIED | demo_utils.py:66-95 |

## Output Shape Correction

**IMPORTANT: Output is `(n_kept_segments, n_vertices)` not `(n_timesteps, 20484)`.**

The difference: segments with no events are dropped. So if a video has silent/empty
sections, those timesteps won't appear in the output. The `all_segments` list tells
you which timestamps each row corresponds to.

## Modality Ablation (How It Actually Works)

**IMPORTANT UPDATE (2026-06-08):** `features_to_mask` does NOT work at inference
time. It only affects model construction (which projectors are created). For
runtime ablation, mutate `features_to_use` instead.

**Option A (BROKEN for inference): `features_to_mask` config**
- `data.features_to_mask = ["video"]` would mask video features
- main.py:101-103, main.py:486-504 — masked features get `feature_dims[modality] = None`
- model.py:107-111 — if feature_dims is None, no projector is created
- **This only works during model construction, NOT after the model is loaded**

**Option B (OUR APPROACH): Mutate `features_to_use` at runtime**
- `features_to_use` controls which extractors run during `predict()`
- Removing a feature name → that extractor doesn't run → no features produced
- The brain model substitutes zeros for missing modalities (model.py:189-192)
- Our code: `_find_features_to_use(model)` probes 4 candidate attribute paths
- **UNVERIFIED on GPU (G018)** — exact path needs confirmation

**Option C: Remove modality events from DataFrame**
- Don't provide video events in the DataFrame → no video features extracted
- The model handles missing modalities by substituting zeros (model.py:190-192)
- Requires knowing the DataFrame structure (not yet explored)

## Atlas / HCP Labels (VERIFIED)

| Fact | Status | Source |
|------|--------|--------|
| HCP-MMP1 atlas is loaded via MNE, NOT nilearn | VERIFIED | utils.py:214-257 `get_hcp_labels()` |
| Uses `mne.datasets.fetch_hcp_mmp_parcellation()` | VERIFIED | utils.py:220-221 |
| Maps to fsaverage5 by default | VERIFIED | utils.py:214 `mesh="fsaverage5"` |
| Vertex indices are offset: left=0, right=+expected_size | VERIFIED | utils.py:242 |
| `get_hcp_roi_indices(rois, hemi, mesh)` returns vertex indices | VERIFIED | utils.py:268-284 |
| Supports wildcard ROI matching: "V1*" matches all V1 subregions | VERIFIED | utils.py:275-279 |
| `summarize_by_roi(data)` computes mean activation per ROI | VERIFIED | utils.py:287-306 |
| `get_topk_rois(data, k=10)` returns top-k activated ROIs | VERIFIED | utils.py:309-318 |
| FSAVERAGE_SIZES imported from neuralset | VERIFIED | utils.py:19 |
| fsaverage5 vertex count per hemisphere: check FSAVERAGE_SIZES dict | VERIFIED | utils.py:241, 261 |

## Memory Requirements

| Fact | Status | Source |
|------|--------|--------|
| V-JEPA 2 FP32: ~14GB | ASSUMED | musketeer-5-hacker.md estimate |
| V-JEPA 2 FP16: ~7GB weights + 2-3GB activations | ASSUMED | musketeer-5-hacker.md estimate |
| LLaMA 3.2-3B FP16: ~5GB | ASSUMED | Standard 3B model sizing |
| Wav2Vec-BERT: ~1GB | ASSUMED | musketeer-5-hacker.md estimate |
| Brain model (FmriEncoderModel): small — projectors + transformer + linear | VERIFIED | model.py:89-157, likely < 1GB |
| Sequential loading is handled by the pipeline itself | VERIFIED | main.py:214-217, _free_extractor_model |
| Each extractor runs alone, caches to disk, then freed | VERIFIED | main.py:214-217 |

## API Details

| Fact | Status | Source |
|------|--------|--------|
| HuggingFace model ID: "facebook/tribev2" | FROM-DOCS | code.md |
| Requires HF auth for LLaMA 3.2 | FROM-DOCS | code.md |
| Output shape: (n_kept_segments, n_vertices) | VERIFIED | demo_utils.py:378-381 |
| Returns tuple: (preds_array, segments_list) | VERIFIED | demo_utils.py:392 |
| Dependencies include: neuralset, neuraltrain, exca, mne, einops | VERIFIED | imports across all files |

## Key Dependencies (from source imports)

| Package | Used For | Source |
|---------|----------|--------|
| neuralset | Dataset loading, feature extraction, segments | model.py:12, main.py:21 |
| neuraltrain | Model configs, optimizers, losses, metrics | model.py:13-15, main.py:32-35 |
| exca | Config management, task infrastructure | main.py:27 |
| mne | HCP atlas loading, parcellation | utils.py:13 |
| einops | Tensor rearrangement | model.py:10 |
| lightning | Training module | pl_module.py:13 |
| pydantic | Config validation | demo_utils.py:15 |

---

## Verification Queue (Updated 2026-06-08)

1. [x] Clone tribev2 repo
2. [x] Read tribev2/model.py — FmriEncoderModel class, forward chain
3. [x] Read tribev2/demo_utils.py — TribeModel wrapper
4. [x] Identify how to isolate each encoder's forward pass → ANSWER: already isolated as external extractors
5. [x] Identify the exact tensor shapes at each stage boundary
6. [x] Check if modality dropout = zeroing out one encoder's output → ANSWER: yes, per-sample random mask
7. [ ] Check actual memory usage by loading model on GPU (needs GPU)
8. [x] Read tribev2/utils.py — HCP atlas functions (get_hcp_labels, get_hcp_roi_indices, summarize_by_roi)

Remaining:
- [x] Check FSAVERAGE_SIZES dict for exact vertex count → ANSWER: 10,242 per hemisphere, 20,484 total
- [ ] Verify neuralset/neuraltrain are pip-installable or if they're Meta-internal
- [ ] Test `TribeModel.from_pretrained("facebook/tribev2")` on Kaggle
- [ ] Measure actual VRAM per extractor on GPU
