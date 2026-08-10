# Scope: the lookahead / causal-ceiling run

**Status: PROPOSED, awaiting go. Not built.** Written 2026-08-10.

## The one-sentence question

> How much of TRIBE v2's reported brain-prediction accuracy is bought with stimulus the subject
> had not yet seen — and what is the accuracy ceiling for a real-time (causal) encoder?

## Why it is real (not another premise that dies on contact)

Verified from the released model, not the literature:

- The encoder is **maskless / bidirectional** (`config.yaml: causal:false` →
  `neuraltrain/models/transformer.py:69-72` returns `Encoder`, not `Decoder`).
- Therefore the prediction for within-window TR *p* attends over the whole ~100-TR window,
  including the `(99−p)` TRs **after** *p*.
- fMRI BOLD is HRF-smoothed and temporally autocorrelated over ~10–20 s, so future stimulus within
  a few TRs of *p* is genuinely informative about the BOLD at *p*. So the effect is
  mechanistically **expected to be non-zero** — the run measures its *size*, which is the one
  honest unknown.

This does **not** use `time_pos_embed` (proven dead, see `position_embed_dead.md`). It manipulates
the input window directly, so it is robust to how position is encoded.

## Design — window placement (preferred), not causal masking

Two ways to remove future access. The second is cleaner and is the one to build.

**A. Causal mask at inference (rejected as primary).** Add a causal mask so TR *p* attends only to
0..*p*. Confound: the model was *trained* bidirectional, so a causal mask is off-distribution, and
the accuracy drop conflates "lost future info" with "off-distribution inference." Keep only as a
secondary cross-check.

**B. Window placement (primary).** To predict the fMRI at true stimulus moment *t*, feed the
full-length in-distribution window `[t−k .. t+(99−k)]` and read output position *k*. Every forward
pass is a normal full 100-TR window — **no off-distribution confound**. Sweep *k*:

- `k = 99` → window `[t−99 .. t]` → **0 future TRs** → the causal operating point / ceiling.
- `k = 0`  → window `[t .. t+99]` → **99 future TRs** → maximal lookahead.

Define *future context* = `99 − k`. The curve is **accuracy(future_context)**, where accuracy is
the Pearson r between the predicted BOLD at *t* and the true BOLD at *t*, aggregated over many *t*
and parcels. Three numbers fall out of one curve:

1. **Causal ceiling** = accuracy at future_context = 0.
2. **Bidirectional accuracy** = accuracy at the placement the benchmark actually scores.
3. **Lookahead value** = the gap. If large, the leaderboard numbers are substantially future-fed.

## The move that makes this cheap and clean: do it LOCALLY first

The main curve needs **ground-truth fMRI**, and the public **Friends s1–6** responses are in hand
(the challenge's own `.h5`, no withheld data, no submission, no ethics surface). So:

- **Result 1 (local, no leaderboard):** the full accuracy(future_context) curve on a few Friends
  episodes. This is the paper's core figure and it touches nothing withheld.
- **Result 2 (one own-model submission):** submit the *causal-placement* OOD predictions to
  Codabench 9483 and read the score, to state the causal ceiling on the held-out OOD films where
  we lack ground truth. One legitimate submission of our own model. No probing, no third-party
  data — none of the flagged exploit behaviour.

## Cost

Features are extracted **once** per stimulus (the expensive step). Re-windowing the feature tensor
and re-running only the small encoder+head per placement is cheap and warm.

- Feature extraction: a few Friends episodes ≈ handful of GPU-hours (dominated by ASR + video
  decode, per the measured cost model).
- The *k*-sweep: many encoder passes, all warm, minutes.
- Estimate: **~4–6 GPU-hours** for a solid local curve; +~3–8 for the OOD causal submission if we
  do Result 2. Well under the 14 h budget.

## Pre-registered prediction (freeze before running)

Monotone non-increasing accuracy as future_context → 0, with a **non-trivial** drop (point
estimate: ≥15% relative at the causal end). If the drop is <5% relative, the finding is weak and
we say so plainly rather than dress it up — that is the small-effect failure mode and it is a real
possibility given text is TRIBE's weakest modality.

## Two things to do BEFORE building (both cheap, both gate the build)

1. **Occupancy re-check under THIS framing.** The judges checked "lookahead rig" occupancy, but
   "bidirectional encoders see the future" is a known idea in time-series/BCI. Targeted search:
   has anyone quantified the causal ceiling of a *naturalistic-fMRI encoding model on a public
   benchmark*, in prediction units? If yes with this exact framing, re-scope. ~30 min, no GPU.
2. **Plumbing spike (CPU, ~1 day).** Confirm we can call `model.transformer_forward` on a
   hand-assembled feature window and read a chosen output position, bypassing the event pipeline.
   The judges flagged this as "a real bypass of Meta's data pipeline, not a config change." If the
   forward path resists hand-assembled windows, the whole design needs rework — so this is the
   true feasibility gate, and it is CPU-testable with random tensors before any GPU.

## Honest status line

Existence of the effect: architectural, verified. Size of the effect: unknown, and the run exists
to measure it. Novelty of the framing: strong pending the 30-minute occupancy re-check. This is
the first candidate in the project whose premise survived its own kill test — the test killed the
*mechanism* (the position embedding) and the question stood up on the architecture instead.
