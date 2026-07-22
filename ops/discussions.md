# Discussions — running log

Live working log of active debugging threads. Newest at top. Move settled items into
`war-room.md` / `decision-log.md` / `source-of-truth.md` once resolved.

---

## Kaggle GPU smoke test — getting `import tribev2` to load (2026-07-22) — ✅ RESOLVED / GO

**OUTCOME (2026-07-22): smoke test PASSED end to end. G016/G005/G018 all closed, BrainLens
mechanic confirmed alive.** Numbers recorded in source-of-truth "SECOND KAGGLE RUN" +
knowledge-gaps. G005 peak 11.12/15.6 GB, G018 ablation WORKS (0.654 diff). Debug trail below
kept for reference (three failures + fixes). Reusable run recipe: **Run All → Restart & Clear →
Run All** (numpy needs a fresh interpreter after the 2.2.6 upgrade). Full detail retained below.


Goal: get `notebooks/01_setup_test.ipynb` to reach Phase 7/8 on Kaggle (T4 x2) so we finally
measure VRAM (G005) and the modality-ablation verdict (G018). Both still UNMEASURED — the run
keeps dying before the model loads. Sequence of failures + fixes:

1. **Run 1 — import shadow (clone dir `tribev2`).** `import tribev2` bound to a bare
   `/kaggle/working/tribev2` dir (cwd on sys.path) with no `demo_utils`. Fixed by cloning to
   `tribev2_src`. (commit before 58a7a94)

2. **Run 2 — shadow persisted.** Phase 4 wrapper import passed, Phase 6 died with
   `ModuleNotFoundError: No module named 'tribev2.demo_utils'`. Cause: editable-install `.pth`
   doesn't activate mid-session, so `import tribev2` still missed `tribev2_src`. Fix (commit
   **58a7a94**): cell-2 puts `tribev2_src` on `sys.path` explicitly, pops cached `tribev2*`,
   and the fail-fast asserts `tribev2.__file__` is under `tribev2_src` and RAISES.

3. **Run 3 — shadow DEAD (win), new error: numpy.** `import tribev2` now correctly resolves
   into `/kaggle/working/tribev2_src/tribev2/__init__.py` and finds `demo_utils`. But the
   dep chain `neuralset -> sklearn -> scipy -> numpy` blew up with:
   `ImportError: cannot import name '_center' from 'numpy._core.umath'` — a mixed/broken numpy
   (newer `strings.py` vs older compiled `umath`). Added a numpy force-reinstall to cell-2
   (commit **f263748**): `pip install --force-reinstall --no-deps numpy==2.2.6` after the
   tribev2 install, before the first numpy import.

4. **Run 4 — numpy fix DIDN'T take.** cell-2 printed `numpy: 2.0.2` (NOT the 2.2.6 requested),
   same `_center` ImportError. So the version request is being overridden. **Hypothesis:
   Kaggle's `PIP_CONSTRAINT` env var pins numpy to the base image (2.0.2)**, silently clamping
   both tribev2's `numpy==2.2.6` dep and our force-reinstall.

5. **Diagnostic result — PIP_CONSTRAINT theory WRONG; real cause found.** The diagnostic
   cell printed `PIP_CONSTRAINT = None`, and pip reported the **existing on-disk numpy was
   already 2.2.6** (rc=0, reinstalled 2.2.6 over 2.2.6). So disk = correct 2.2.6, but the
   kernel imported 2.0.2 and broke. **Real cause: Kaggle pre-imports numpy 2.0.2 at kernel
   boot.** cell-2 then upgrades numpy on disk to 2.2.6 mid-session; the cached 2.0.2 stays in
   memory (`numpy.__version__` -> 2.0.2), but a lazily-loaded submodule (`numpy.strings`) reads
   the NEW 2.2.6 file off disk and asks the already-loaded 2.0.2 `umath` for `_center` (a
   symbol not in 2.0.2) -> ImportError. Classic "package upgraded after it was already
   imported." Not our code, not a version-choice problem — a load-order problem.

**FIX (in progress): kernel restart.** Disk is already 2.2.6, so: **Restart the kernel**
(Restart & Clear — NOT Factory reset, which wipes disk back to base 2.0.2 and reproduces the
bug), then Run All. Fresh kernel boots against the 2.2.6 already on disk; cell-2's reinstalls
become no-ops (no churn); `import numpy` comes up clean at 2.2.6.

**If restart alone doesn't hold** (e.g. Kaggle restart also wipes pip installs, or Factory
reset is the only option): durable fix = make cell-2 install numpy 2.2.6 and then self-restart
the kernel BEFORE the first numpy import, or detect `numpy.__version__ != installed` and raise
a clear "restart now" message. Decide after seeing the restart result.

6. **RESTART FIXED IT — run now reaches live inference (2026-07-22).** After Restart & Clear +
   Run All: cell-2 prints `numpy: 2.2.6` + `tribev2 resolves to: .../tribev2_src/...` + `OK:
   tribev2.demo_utils imports`. Then clean sail: Phase 2 torch 2.6.0+cu124, 2× Tesla T4 15.6 GB,
   ffmpeg+uvx OK; Phase 3 HF login OK; Phase 4 wrapper OK; Phase 5 Sintel clip made; **Phase 6
   MODEL LOADED in 13.7s** (the step that failed 3×). Phase 7 in progress: WhisperX ASR ran
   (~2m15s), spacy en_core_web_lg auto-installed, all 3 encoders downloaded+loaded (LLaMA-3.2
   6.43G text, Wav2Vec-BERT 2.32G audio, V-JEPA2 4.14G video, brain 709M) with no OOM so far.
   Currently in the V-JEPA2 **video encode** (~15s/segment, 60 segments ≈ 8–15 min) — the
   expensive step, NOT stuck. So **G016 = DEAD (confirmed live), install + load + fit all
   working.** Awaiting Phase 7 `=== FULL PREDICTION ===` (G005 VRAM) + Phase 8 `ABLATION
   VERDICT` (G018) + the Phase 9 record block.

**DONE — durable fix for the numpy/restart issue:** cell-2 installs numpy 2.2.6 on disk; because
Kaggle pre-imports numpy at boot, the FIRST Run-All after a fresh/factory session still imports
the stale 2.0.2 and errors — the reliable recipe is **Run All once (puts 2.2.6 on disk) →
Restart & Clear (NOT factory reset) → Run All again**. Consider baking a "loaded numpy !=
installed → raise 'restart now'" guard into cell-2 later so this is self-explaining.
