# Compute Playbook — GPU Session Battle Plans

Every GPU minute is borrowed. Zero improvisation. Every session is scripted
in advance. You open the notebook, paste the script, hit run, and collect results.

---

## The Rules

1. NEVER start a GPU session without knowing exactly what you will run.
2. ALWAYS save results to persistent storage before the session ends.
3. ALWAYS run the smallest possible test first (1 clip, 1 pass) before batch jobs.
4. ALWAYS set checkpoints so a crashed session can resume.
5. Time your first run. Extrapolate. Decide if the batch fits in the session window.

---

## Platform Cheat Sheet

| Platform | GPU | VRAM | Session Limit | Storage | Weekly Budget |
|----------|-----|------|--------------|---------|---------------|
| Kaggle | 2x T4 | 2x 16GB | 12 hrs | 20GB persistent | 30 hrs |
| Colab Free | 1x T4 | 16GB | ~12 hrs (flaky) | Google Drive | Unlimited* |
| HF ZeroGPU | H200 | 70GB | 3.5 min/day | HF Space storage | 24.5 min |
| Lightning AI | T4 | 16GB | No idle timeout | 10GB persistent | 80 hrs/month |

*Colab may disconnect randomly. Checkpoint aggressively.

---

## Session Type 1: Smoke Test (Any Platform, 5 minutes)

**Goal:** Verify the model loads and produces output of the expected shape.

```python
# 1. Install tribev2 and dependencies
!pip install tribev2  # or install from source if not on PyPI
# Check if neuralset/neuraltrain are available (G016)
!pip install neuralset neuraltrain exca

# 2. Load model (thin wrapper — no mode param needed)
from tribe_tools.model import load_model
model = load_model(device="cuda")

# 3. Single prediction
from pathlib import Path
from tribe_tools.model import predict_single
preds, segments = predict_single(model, Path("test_video.mp4"))

# 4. Verify shape
print(f"Shape: {preds.shape}")  # Expected: (n_kept_segments, n_vertices)
print(f"Dtype: {preds.dtype}")  # Expected: float32
print(f"Range: [{preds.min():.4f}, {preds.max():.4f}]")
print(f"Mean: {preds.mean():.4f}, Std: {preds.std():.4f}")
print(f"Segments kept: {len(segments)}")

# 5. Check VRAM (after all extractors have run and been freed)
import torch
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# SAVE THE OUTPUT
import numpy as np
np.save("smoke_test_prediction.npy", preds)
print("Smoke test PASSED")
```

**What to record:** Shape, n_vertices, dtype, value range, VRAM peak, wall clock time.
Update `ops/source-of-truth.md` with verified facts.

---

## Session Type 2: BrainLens Single Clip (Kaggle/Colab, ~20 minutes)

**Goal:** Run 4-pass ablation on one clip. Generate first brain map.

```python
from brainlens.inference import run_ablation
from brainlens.attribution import compute_attribution
from brainlens.visualization import create_brain_map
from tribe_tools.model import load_model
from pathlib import Path

# Load model (TRIBE v2 handles sequential encoder loading internally)
model = load_model(device="cuda")

# Run 4 passes (full, video-only, audio-only, text-only)
results = run_ablation(model, Path("demo_clip.mp4"), cache_dir=Path("cache/"))

# Compute attribution (pass just the prediction arrays, not segments)
preds_dict = {mod: r[0] for mod, r in results.items()}
visual, auditory, language = compute_attribution(preds_dict)

# Generate brain map
fig = create_brain_map(visual, auditory, language,
                       title="BrainLens: Demo Clip",
                       output_path=Path("brainlens_demo.png"))

# Save raw data
import numpy as np
np.savez("brainlens_results.npz",
         visual=visual, auditory=auditory, language=language,
         full=preds_dict["full"], video_only=preds_dict["video_only"],
         audio_only=preds_dict["audio_only"], text_only=preds_dict["text_only"])
print("BrainLens demo COMPLETE")
```

**What to record:** Total time for 4 passes, VRAM peak per pass, output image.
This image goes in the README and outreach emails.

---

## Session Type 3: NeuroGenre Batch (Kaggle, ~4 hours)

**Goal:** Process 60 clips (10 per genre x 6 genres).

```python
from tribe_tools.model import load_model
from tribe_tools.inference import batch_predict
from pathlib import Path

model = load_model(device="cuda")

# Load manifest (pre-prepared list of video paths + genre labels)
import json
manifest = json.load(open("neurogenre/corpus_manifest.json"))
video_paths = [Path(v["path"]) for v in manifest]

# Batch inference with checkpointing every 5 clips
results = batch_predict(
    model, video_paths,
    cache_dir=Path("cache/neurogenre/"),
    checkpoint_every=5,
)

# Save everything
import pickle
with open("neurogenre_results.pkl", "wb") as f:
    pickle.dump(results, f)
print(f"Processed {len(results)} clips")
```

**Timing estimate:** ~4 min/clip x 60 clips = ~4 hours.
Fits in one Kaggle session (12 hr limit). Checkpoint every 5 clips (20 min).
If session dies after 30 clips, re-run: cached clips are skipped automatically.

---

## Session Type 4: NeuroCheck Claims Testing (Kaggle, ~2 hours)

**Goal:** Test N claims by running TRIBE v2 on stimulus pairs.

```python
from neurocheck.claims import load_claims
from neurocheck.stimuli import prepare_stimuli
from neurocheck.contrast import run_contrast
from neurocheck.scorecard import generate_scorecard
from tribe_tools.model import load_model
from pathlib import Path

model = load_model(device="cuda")
claims = load_claims(Path("neurocheck/claims_db/claims.yaml"))

results = []
for claim in claims:
    # Prepare stimulus videos
    stim_a, stim_b = prepare_stimuli(claim)

    # Run contrast
    result = run_contrast(model, claim, stim_a, stim_b,
                          cache_dir=Path("cache/neurocheck/"))
    results.append(result)
    print(f"{claim.id}: {'PASS' if result.passed else 'FAIL'} "
          f"(d={result.effect_size:.3f}, p={result.p_value:.4f})")

# Generate scorecard
scorecard = generate_scorecard(results)
scorecard.save(Path("neurocheck_scorecard.html"))
print(f"Score: {scorecard.passed}/{scorecard.total}")
```

**Timing estimate:** ~2 min/claim x 50 claims = ~100 min.

---

## Session Type 5: ZeroGPU Quick Test (HuggingFace, 3.5 minutes)

**Goal:** One inference pass on one short clip. Verify model works on ZeroGPU.

This platform has 70GB VRAM (H200) so all extractors can run without VRAM pressure.
But you only get 210 seconds. Use them wisely.

```python
import spaces

@spaces.GPU
def run_test():
    import time
    start = time.time()

    from tribe_tools.model import load_model, predict_single
    from pathlib import Path

    model = load_model(device="cuda")
    preds, segments = predict_single(model, Path("short_clip_5s.mp4"))

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s. Shape: {preds.shape}")
    return preds

result = run_test()
```

**Critical:** Use a clip under 10 seconds. Loading + inference must finish in 210s.
If it works, this becomes the BrainLens demo on HuggingFace Spaces.

---

## Session Type 6: ScaleLaw Full Run (Kaggle, ~6 hours)

**Goal:** Run TRIBE v2 on StudyForrest movie segments, compare to real fMRI.

```python
from scalelaw.inference import run_forrest_predictions
from scalelaw.correlate import compute_vertex_correlation
from scalelaw.fit import fit_scaling_curve
from scalelaw.plot import plot_scaling_law
from tribe_tools.model import load_model
from pathlib import Path

model = load_model(device="cuda")

# Run predictions on all movie segments
predictions = run_forrest_predictions(
    model, data_dir=Path("data/studyforrest/"),
    cache_dir=Path("cache/scalelaw/"),
    checkpoint_every=3,
)

# Compare to real fMRI
correlations = compute_vertex_correlation(predictions, real_fmri_dir=Path("data/studyforrest/fmri/"))

# Fit scaling curve
curve = fit_scaling_curve(correlations)
fig = plot_scaling_law(curve, output_path=Path("scalelaw_figure.png"))
print(f"R-squared: {curve.r_squared:.4f}")
```

---

## Emergency Procedures

**Session dying mid-inference:**
- All batch jobs use HDF5 caching. Re-run the same script. Cached clips skip.
- Check cache integrity: `h5py.File("cache/file.h5", "r").keys()` to see what's saved.

**Out of memory:**
- TRIBE v2 already handles sequential encoder loading and GPU cleanup internally
- Reduce batch size to 1
- Check if any tensors are being held in a list (accumulating memory)
- Verify `_free_extractor_model()` is being called (it should be automatic)

**Model download failing:**
- Pre-download weights in a separate cell before the timer-critical code
- Cache HuggingFace models to persistent storage: `HF_HOME=/path/to/persistent`
- For Kaggle: download to /kaggle/working/ which persists within a session

**Wrong output shape:**
- Update source-of-truth.md with the actual shape
- Update interface-contracts.md
- File a bug/note: the documentation may be wrong

---

## Pre-Session Checklist

Before starting ANY GPU session:

- [ ] Script written and tested for syntax errors (python -m py_compile script.py)
- [ ] Test video uploaded to the platform
- [ ] Cache directory path set to persistent storage
- [ ] HuggingFace auth token configured
- [ ] Expected runtime estimated (will it fit in the session window?)
- [ ] Know exactly what files to download before session ends
- [ ] Checkpoint interval set (every 5-10 items for batch jobs)
