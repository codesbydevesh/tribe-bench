# Gate 0 v3 stimulus curation (D023(g))

Rebuilds the FACE / NONFACE stimulus set from Charade (1963) so that it is **sustained-single-shot**
and **motion-matched**. Run order, all CPU-only:

1. **Full-film scans** (three cheap ffmpeg passes at 160 px over all 169,625 frames) producing
   per-frame `scene_score`, frame-difference motion energy, and luminance. Everything downstream is
   aggregation over these.
2. `facescan.py START DUR OUT.npy` — 2 fps face timeline. Run one process per core; it sets
   `cv2.setNumThreads(1)` so workers do not oversubscribe. Records both a **high-precision** and a
   **high-recall** cascade pass, because the two errors are not symmetric: a false positive puts a
   non-face clip into FACE, a false negative puts a face into the face-**absent** baseline.
3. `candidates.py` — shot segmentation and candidate-window enumeration.
4. `select.py` — labelling, one-per-shot reduction, 45 s independence, matched selection, acceptance
   test. Writes `selection.json`.
5. `preflight.py` — integrity (md5 vs archive.org), per-clip validity, covariates, montages.

Outputs committed to `notebooks/`: `gate0_v3_stimuli.json` (manifest + full provenance) and
`gate0_v3_verification.json` (per-clip measurements on the cut clips).

**Why the thresholds are what they are** is documented inline. Two are load-bearing and were set from
measurement, not taste: the scene threshold is deliberately *sensitive* (0.05) because over-detecting
boundaries only costs candidates while under-detecting is the defect being fixed; and FACE requires the
two Haar cascades to **agree**, because a persistent single-cascade false positive was observed
(union area ~0.040 on every sample of a wide corridor shot, agreement 0.000 throughout).

Nothing here consults brain data — none exists. The whole procedure is deterministic at seed 0.
