# Pre-Mortem — What Kills Each Build

For each build, we imagine it's 8 weeks from now and the build has failed.
What went wrong? Then we address those risks NOW, before starting.

---

## How to Use

Before starting any build, read its pre-mortem section.
Address the "Prevention" column for each risk.
After the build ships, update this file with what actually happened.

---

## tribe_tools/ (Shared Library)

| Risk | Probability | Impact | Prevention |
|------|------------|--------|------------|
| ~~Encoders can't be isolated~~ | ~~30%~~ | ~~FATAL~~ | **RESOLVED 2026-06-08.** Encoders are external extractors, already load/run/free sequentially. |
| neuralset/neuraltrain not pip-installable | 40% | FATAL — can't install tribev2 at all | Try `pip install` on Kaggle immediately. If they're Meta-internal, check if bundled in tribev2 repo. See G016. |
| V-JEPA 2 doesn't fit on T4 even with sequential loading | 20% | HIGH — need fallback to 4-bit quant | Measure actual VRAM in smoke test (G005). Each extractor runs alone, so only one needs to fit at a time. |
| ~~nilearn doesn't support Glasser on fsaverage5~~ | ~~40%~~ | ~~MEDIUM~~ | **RESOLVED 2026-06-08.** Atlas uses MNE, not nilearn. TRIBE v2 already has HCP-MMP1 via `mne.datasets.fetch_hcp_mmp_parcellation()`. |
| ~~TRIBE v2 output vertex ordering unknown~~ | ~~25%~~ | ~~HIGH~~ | **RESOLVED 2026-06-08.** Left hemi first, right hemi second, confirmed in utils.py:242. |
| ~~features_to_mask config path unclear~~ | ~~35%~~ | ~~HIGH~~ | **PARTIALLY RESOLVED 2026-06-08.** G017 closed: feature names are "video","audio","text". `features_to_mask` doesn't work at inference time. Code now uses `_find_features_to_use()` discovery to mutate `features_to_use`. G018 still open: exact attribute path needs GPU verification. |
| HuggingFace denies LLaMA 3.2 access | 10% | HIGH — can't load model at all | Apply for access immediately. Check if TRIBE v2 bundles its own copy. |
| Cache corruption on Kaggle session death | 30% | MEDIUM — lose hours of compute | Implement atomic writes: write to temp file, rename on completion. Verify cache after each write. |

**Critical path:** neuralset/neuraltrain availability (G016) is the new existential risk. Cannot write or test any model code until confirmed.

---

## BrainLens (Build 1)

| Risk | Probability | Impact | Prevention |
|------|------------|--------|------------|
| ~~Modality ablation doesn't work~~ | ~~25%~~ | ~~HIGH~~ | **PARTIALLY RESOLVED.** Modality dropout zeros entire modality (model.py:211-213), confirming ablation is valid. Code uses `_find_features_to_use()` discovery. Still need GPU test to confirm the attribute path (G018). |
| All modalities produce nearly identical brain maps | 15% | HIGH — no interesting result to show | Expected from architecture: vision should dominate V1, audio should dominate A1. If not, the model may not differentiate well. Run on a clip with clear visual vs. audio content. |
| RGB brain map is visually unreadable | 20% | MEDIUM — need better visualization | Have fallback: three separate maps (visual, auditory, language) side by side instead of RGB overlay. |
| Processing 4 passes takes too long for ZeroGPU | 60% | MEDIUM — can't deploy live demo | 4 passes x ~1 min each = 4 min. ZeroGPU limit = 3.5 min. May only fit 3 passes. Workaround: skip text-only pass (least interesting). |

**Critical path:** Risk 1. If ablation doesn't work, BrainLens needs a fundamentally different approach.

---

## NeuroGenre (Build 2)

| Risk | Probability | Impact | Prevention |
|------|------------|--------|------------|
| Genres don't cluster in UMAP space | 35% | FATAL — no publication-quality result | This is a genuine research risk. If genres don't cluster, the result is "TRIBE v2 doesn't differentiate genres" — which is still publishable as a negative result, but less impressive. |
| Copyright issues with video clips | 20% | MEDIUM — can't share data | Use Creative Commons clips or academic datasets (Kinetics-700, MovieNet). Document licenses. |
| 60 clips is too few for statistical significance | 25% | MEDIUM — weak statistics | Run power analysis beforehand. If 10/genre is too few, increase to 15-20 (need more compute time). |
| Batch inference crashes at clip 30 of 60 | 40% | LOW — checkpointing handles this | Checkpoint every 5 clips. Resume from cache. |

**Critical path:** Risk 1 is a genuine unknown. We won't know until we run it. Mitigate by choosing clips with very distinct genre characteristics.

---

## ScaleLaw (Build 3)

| Risk | Probability | Impact | Prevention |
|------|------------|--------|------------|
| StudyForrest fMRI data format is incompatible with fsaverage5 | 30% | HIGH — need alignment pipeline | StudyForrest data is in MNI space or native space. Need FreeSurfer alignment to fsaverage5. This is a known neuroimaging preprocessing step. Budget time for it. |
| StudyForrest is too large to download on Kaggle | 20% | MEDIUM — need alternative storage | Full dataset is ~300GB. We only need the movie-watching runs (~20GB). Download only what's needed. |
| Correlation between predicted and real fMRI is low | 25% | MEDIUM — scaling law doesn't replicate | This is a research risk. Meta reports 0.54 normalized correlation on their data. StudyForrest is different data. Some degradation expected. Report honestly. |
| Temporal alignment between video and fMRI is wrong | 30% | HIGH — garbage correlations | fMRI has ~5s hemodynamic delay. Need to account for it. TRIBE v2 may already handle this (check source). |

**Critical path:** Risk 1 (data alignment) and Risk 4 (temporal alignment). Both need significant neuroimaging expertise. Consider partnering with someone who has FreeSurfer experience.

---

## NeuroCheck (Build 4 — Flagship)

| Risk | Probability | Impact | Prevention |
|------|------------|--------|------------|
| Claims database has errors (wrong ROIs, wrong directions) | 20% | FATAL for paper credibility | Follow claims-protocol.md rigorously. Red-team every 10 claims. Have a neuroscientist review if possible. |
| TRIBE v2 fails most claims (score < 20/50) | 30% | MEDIUM — but still publishable | A low score is a finding: "current brain encoding models fail to reproduce X% of established neuroscience." Frame as a gap analysis, not a success story. |
| Can't find public stimuli for some claims | 25% | LOW — reduce claim count | Pre-check stimulus availability before committing to a claim (Step 5 in claims-protocol.md). |
| Reviewer says benchmark is too simple (just a t-test per claim) | 20% | MEDIUM — need more sophisticated analysis | Add effect size reporting, confidence intervals, brain maps per claim, and meta-analysis across claim categories. |
| Someone publishes a similar benchmark before us | 10% | HIGH — lose first-mover advantage | Unlikely given the 5-paper ecosystem. But move fast. Get the design paper on bioRxiv ASAP. |

**Critical path:** Risk 1 (claim quality). The claims are the product. Everything else is infrastructure.

---

## Overall Project Risks

| Risk | Probability | Impact | Prevention |
|------|------------|--------|------------|
| Never get GPU access | 5% | FATAL for demos/results (but design paper still publishable) | Apply to all 6+ platforms simultaneously. TRIBE v2's built-in sequential loading means any single T4 works. |
| TRIBE v2 model is taken offline | 2% | FATAL | Download and cache weights immediately when GPU access is obtained. |
| Meta releases TRIBE v3, making v2 obsolete | 15% | MEDIUM — benchmark framework transfers | Our toolkit design is model-agnostic. Claims database works with any brain encoding model. Update model loading code. |
| Burnout / motivation loss | 30% | FATAL | Ship something visible every week. First brain map → first 20 claims → first paper draft. Visible progress sustains motivation. |
| Scope creep into all 4 builds simultaneously | 40% | HIGH — nothing ships | Strict sequencing: BrainLens first, NeuroCheck claims in parallel, everything else waits. War room enforces this. |

---

## What Actually Happened (Fill In Later)

| Build | Predicted Risk | What Actually Happened | Date |
|-------|---------------|----------------------|------|
| — | — | — | — |
