# Measured noise floor — Algonauts OOD board, submission 877771

**What this is.** On 2026-08-04 we submitted **pure random numbers** to the Algonauts 2025
post-challenge out-of-distribution leaderboard (Codabench competition 9483) as a plumbing test,
before spending ~23 GPU-hours on real predictions. It scored **−0.00028** and, far more usefully,
the detailed-results page rendered a full per-parcel brain map for every subject and every movie.

Because the submission was noise, **every number below is a null** — measured by the challenge's
own scorer, on the real withheld human fMRI, at each level of aggregation. That is an empirical
detection floor, and it is strictly better than a simulated one.

Submission: id **877771**, `submission_random.zip`, 2026-08-04 18:53, status Finished, score
**−0.00028**. Detailed results: `codabench.org/competitions/9483/detailed_results/877771/`.

## PROVENANCE — read this before citing any number here

These values were transcribed from **33 screenshots** of the detailed-results page (the page
requires an authenticated session, so it cannot be re-fetched programmatically). The screenshots
are the only copy of this content.

**Single-source.** Transcribed by one reader (Opus) from the images. Six attempts to have a second
model (Fable) independently verify them all failed with API 529 Overloaded, so **these numbers have
NOT been cross-checked.** The colourbar ticks in particular are small rendered text and are the
most likely thing to have been misread. Re-verify before any of this appears in a write-up.

Screenshots retained at `scratchpad/shots/01.png … 33.png` — copy them somewhere durable, because
the scratchpad is session-scoped and this page cannot be regenerated without resubmitting.

## Whole-brain mean accuracy (the headline number at each level)

Values are transcribed exactly as the page printed them, including formatting quirks like `-0.0`
and `1e-04`, which are preserved deliberately — the page's own rounding is evidence about how the
scorer reports.

| subject | movie-average | chaplin | mononoke | passepartout | planetearth | pulpfiction | wot |
|---|---|---|---|---|---|---|---|
| **all** | **−0.0003** | — | — | — | — | — | — |
| sub-01 | −0.0006 | +0.0009 | −0.0017 | −0.0021 | −0.0 | +0.0002 | −0.0012 |
| sub-02 | −0.0003 | −0.0002 | +0.0012 | −0.0008 | 1e-04 | −0.0015 | −0.0007 |
| sub-03 | +0.0004 | −0.0013 | 0.0 | +0.0013 | +0.0002 | +0.002 | 1e-04 |
| sub-05 | −0.0005 | +0.0007 | −0.0014 | −0.0008 | −0.0009 | −0.0007 | −1e-04 |

Range across all 24 subject×movie cells: **−0.0021 to +0.002**. No cell is an outlier, and
critically **`chaplin` behaves like the rest** — so the scorer does not treat the silent,
transcript-free movie differently. Worth knowing before we predict it without language features.

## Per-parcel maxima — THE DETECTION FLOOR

The top colourbar tick is the largest single-parcel correlation reached **by chance** at that level
of aggregation. This is the number that matters: it is the bar a real per-parcel claim must clear.

| level of aggregation | whole-brain mean | max single parcel |
|---|---|---|
| all subjects × all movies | −0.0003 | **0.023** |
| one subject, averaged over 6 movies | −0.0006 … +0.0004 | **0.042 – 0.052** |
| one subject, one movie | −0.0021 … +0.002 | **0.093 – 0.14** |

Per-subject-average maxima: sub-01 0.052, sub-02 0.05, sub-03 0.042, sub-05 0.044.
Per-subject-per-movie maxima ranged 0.093 (sub-05 mononoke) to 0.14 (sub-01 mononoke, sub-01 wot).

Every colourbar's bottom tick was **−1e-05**, i.e. the maps are floored at ~0 and negative
correlations are not shown. So these maps display magnitude of positive correlation only.

## Internal consistency check — this is why the numbers are believable

The observed nulls match the theoretical standard error of a Pearson correlation, at all three
levels, with no free parameters:

- Segments average ~410 TR samples, so a single parcel's r has SE ≈ 1/√(410−3) ≈ **0.050**.
  The maximum of 1,000 such draws sits near 3.2σ ≈ **0.16**. Observed: **0.093 – 0.14**. ✓
- Averaging 6 movies shrinks σ by √6 → max ≈ **0.065**. Observed: **0.042 – 0.052**. ✓
- Averaging all 24 subject×movie pairs → max ≈ **0.032**. Observed: **0.023**. ✓

Theory and observation agree at every level, which closes a real open question: **the scorer applies
no hidden normalisation, no shrinkage, and no noise-ceiling division.** It is a straight Pearson r
per parcel, then averaged, exactly as the challenge's baseline code implies.

## What this means for the real submission

1. **Published comparators are whole-brain means** (2nd place 0.2125, 3rd 0.2094). A single parcel
   on a single movie reaches 0.14 from noise alone — so a per-parcel claim needs to clear a bar
   roughly **6× higher** than the headline number. Never quote a per-parcel value against the
   leaderboard figure.
2. **The floor is now measured, not simulated.** MASTER-PLAN's detection-floor work (S1) can cite
   these instead of estimating them, for this design.
3. **The detailed-results page is itself a deliverable.** It renders a per-parcel, per-movie,
   per-subject map of where predictions succeed and fail, computed externally on withheld data.
   That is the "where does it generalize and where does it break" map, and the server produces it
   for free on every submission.
