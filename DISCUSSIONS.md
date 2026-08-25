# Discussions

Operational notes and decisions that came out of working sessions. Runbooks live here so
they are version-controlled and can be followed without re-deriving them.

---

## Uploading the S2 inputs to Kaggle (2026-08-24)

### Two corrections up front

**fLoc is 15 MB, not 15 GB.** 125 images totalling 15,041,551 bytes, plus a 2.7 MB video —
**17.7 MB** in total. This is a trivial upload, not a bulk-data problem, and the strategy
below is chosen for structural safety rather than for size.

**The path is neither of the two candidates.** `scripts/s2_run.py` originally hardcoded
repo-relative `data/floc` and `data/s2_stimulus.mp4`, so *both* proposed
`/kaggle/input/...` layouts would have failed inside the GPU session — the exact discovery
we were trying to avoid. It now takes one knob.

### What goes where

| file | ships in git? | action |
|---|---|---|
| `data/s2_manifest.json` | **yes** (force-added) | arrives with the repo clone |
| `data/s2_stimulus_probe.json` | **yes** | arrives with the repo clone |
| `data/floc/` (125 jpgs) | no — fLoc has **no licence** | **upload** |
| `data/s2_stimulus.mp4` | no — derived from fLoc | **upload** |

After upload, `s2_run.py` reads:

```
<stimulus-root>/floc/
<stimulus-root>/s2_stimulus.mp4
```

where `<stimulus-root>` comes from `--stimulus-root` (or `S2_STIMULUS_ROOT`, default
`data`). On Kaggle that is `/kaggle/input/<dataset-slug>`.

### ZIP, not a folder

Upload **a ZIP**, and not because of size. Kaggle's web uploader can flatten or partially
select nested folders depending on how they are dragged in. A ZIP is one atomic object that
Kaggle auto-extracts with the directory structure intact. The failure it prevents is the
silent one: a missing image, or a flattened tree that only surfaces mid-run.

```bash
cd /home/deveshb/workspace/AI/tribe-bench
zip -r ~/s2_inputs.zip data/floc data/s2_stimulus.mp4 -x '*.DS_Store'

cd ~ && python3 -c "
import zipfile; z=zipfile.ZipFile('s2_inputs.zip')
n=z.namelist(); print(len([x for x in n if x.endswith('.jpg')]),'jpgs')
print('video present:', any(x.endswith('s2_stimulus.mp4') for x in n))"
```

Expect **125 jpgs** and `video present: True`.

> ⚠️ The zip contains a top-level `data/` folder, so the mount becomes
> `/kaggle/input/<slug>/data/floc/`. Pass `--stimulus-root /kaggle/input/<slug>/data`.
> Either layout works — the verifier reports which one you actually got, before anything
> else runs.

### Upload and attach

1. kaggle.com → **Datasets** → **New Dataset**
2. Drag `s2_inputs.zip` in; wait for extraction to finish
3. Title `corticall-s2-inputs`; note the **slug** shown under the title
4. Visibility: **Private**
5. **Create**

In the notebook editor: right sidebar → **Input** → **+ Add Input** → **Datasets** → **Your
Datasets** → select it. Kaggle mounts it **read-only** at `/kaggle/input/<dataset-slug>/`.
Confirm with `!ls /kaggle/input/`.

### The order of operations, which is not negotiable

```bash
# 1. verify the upload — CPU only, read-only, writes nothing
python3 scripts/s2_verify_inputs.py --stimulus-root /kaggle/input/<slug>
```

Expect `8/8 checks PASS`. It checks: the video exists at the path the run will actually
consume; its sha256 is exactly
`5564c0104e2bff552714cdc02a1e47377ccbf5ec8365333eaa1bf77301ff25ba`; all 125 images are
present; every image hash matches the manifest; nothing is a placeholder; and it echoes the
resolved paths. It resolves those paths through the same logic `s2_run.py` uses, so it
cannot verify one location while the run reads another.

If anything fails, **stop and re-upload.** Tested against four real failure modes, all
caught: wrong nesting (`data/` left inside the dataset), a truncated video, three images
silently omitted, and one image corrupted while keeping the correct filename.

```bash
# 2. only after 8/8 — the one check that needs the GPU environment
python3 scripts/s2_check_frame_sampling.py
```

### The decision tree

**Timestamp-based** → proceed:

```bash
python3 scripts/s2_go_no_go.py --review-clean --neuralset-timestamp   # expect GPU GO
python3 scripts/s2_run.py --infer --stimulus-root /kaggle/input/<slug>
python3 scripts/s2_check_compliance.py data/s2_report.json
```

**Frame-index-based** → **STOP. Do not run S2.**

Return to a machine you control, re-render at 16 fps, verify the new artifact, re-upload.
`--prepare` now *refuses* to run when the stimulus root is not writable, so a re-render
cannot happen by accident on the read-only Kaggle mount.

> At 16 fps the video roughly doubles to ~5.5 MB **and the design fingerprint changes**,
> which invalidates the current manifest. That makes it a full re-prepare locally, not a
> tweak in the session.

### Why the frame-sampling check gates everything

`vjepa2-vitg-fpc64-256` takes 64 frames per 4 s clip — 16 fps of intended coverage. Our
stimulus is 8 fps.

- **Timestamp-based selection** → each clip still spans 4 s and frames resolve to duplicates
  of their neighbours. Harmless for a *static* image, and the paper's stimulus was static
  too. 8 fps is fine.
- **Index-based selection** → 64 consecutive frames at 8 fps span **8 seconds, not 4**. That
  halves each 1 s presentation's weight inside its tubelet and smears it across neighbouring
  events at an 8 s SOA. Blocking.

Ten seconds to check, versus a wasted 3-hour run.

### Do not, while doing any of this

- modify the experiment
- re-render anything
- start S2 merely because the upload succeeded

The upload working says nothing about whether the stimulus is usable. That is what the
frame-sampling check is for.
