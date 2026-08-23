# Phase B — three read-only verification passes against the frozen tree (e0a424f)

Implementation frozen at `e0a424f` before any agent ran. All three passes confirmed the repo was
untouched; all scratch work was done in /tmp. No agent modified anything.

## Verdict on the seven fixes: ALL SEVEN ORIGINAL FAILURE MODES ARE GONE

Confirmed independently by two separate mutant batteries (Pass 2: 14 mutants + a no-op control;
Pass 3: 66 mutants). Reverting any of the seven is killed by the suite, as are
`glm_contrast_z -> return 0.0` and a sign flip.

**M2 (Welch) is verified to the strongest standard available:** bit-identical (diff exactly 0.0)
against an independently derived formula AND `scipy.stats.ttest_ind(equal_var=False)`, across equal n,
unequal n in BOTH variance directions, n_a=2, zero-variance vertices, and a proper ROI subset.
ddof=1 confirmed by hand. The equal-n invariant is exact to ~9e-16 (2 ulp) over 2000 random draws,
and `detection_floor_table.py` is strictly 15v15 — so **the published floor table is genuinely
unchanged**. That was an argument at commit time; it is now a measurement.

## But: the safety net is far weaker than "79 tests pass" implies

**55 of 66 one-line mutations survived the full suite.**

## Findings all three passes agree on (highest confidence)

- **Zero-SE guard** (Pass1 / F9 / A7) — found INDEPENDENTLY BY ALL THREE. `se == 0 -> z = 0` means a
  perfectly separated vertex (true statistic +/-inf, the strongest possible evidence) contributes
  0.0, the value meaning "no evidence". ROI mean shrinks by the dead fraction (-2.5% at 1 vertex of
  40, -20% at 8); a fully degenerate ROI returns exactly 0.0. Directly contradicts the policy
  written six lines above it: "Raise rather than renormalise: a silently shrinking ROI is itself a
  result the operator must see (M3)".
- **NaN onsets / row times resolve silently** (F1, F2 / A1) — `err > tol` and `diff <= 0` are both
  False for NaN. `event_locked_response(..., [nan], ...)` returns the LAST row, no error. This
  resurrects the exact row-drift failure `row_times_from_segments` exists to prevent.
- **Empty-ROI guard runs before normalisation** (F8 / A2) — an all-False boolean mask has
  `len == n_vertices`, so it passes and then selects nothing: `glm_contrast_z -> nan`.
  `raw_roi_mean`/`roi_minus_reference` normalise first and correctly raise. Policy inconsistent
  across four functions.
- **Negative indices defeat the overlap guard** (F5 / A8) — `roi_minus_reference(g, [-1], [19])`
  returns 0.0 with no error although ROI[-1] IS vertex 19.

## Four ways MY OWN Phase B fixes were incomplete

1. **A0 — `define_froi` is the ONLY vertex-selector entry point that does NOT call
   `_as_vertex_indices`.** I added the helper for C7 and did not apply it to the function M1 was
   about. A boolean-mask parcel becomes a 0/1 array: `define_froi(a, b, bool_mask, top_n=10)`
   returns ten copies of vertex 0, and the `top_n >= pv.size` guard mis-fires because pv.size is the
   mask length, not the parcel size. Silent wrong ROI.
2. **F3 — S6's 2-D guard is one-sided.** `ELC(tgt, [tc])` raises; `ELC(tgt, tc)` does not — a bare
   2-D array iterates to rows and each row becomes a "category". Returns 6.5 silently; on a design
   where the truth is 1.0000 it gives 1.4583.
3. **F6/A5 — C5's ">= 2 categories" is syntactic, not statistical.** `[tc, tc]`, `[tc, tc.copy()]`,
   `[tc[:2], tc[2:]]`, `[tc, zeros_like(tc)]` all restore target-only selection: measured type-I
   0.2032 vs a nominal 0.025. And the deeper point: pooled is the unweighted mean of category means,
   so a target that responds more strongly than the others — the normal case in a selectivity study —
   makes the "pooled" peak BE the target's peak. The API restriction does not remove the dependence.
4. **F4 — `_as_vertex_indices` special-cases only `dtype == bool`.** A 0/1 int8/uint8/float mask
   (i.e. any mask that has been through arithmetic or serialisation) is read as indices {0,1}:
   `raw_roi_mean(g, mask.astype(np.int8))` returns 0.15 instead of 11.0. Arguably worse than the
   original C7, which only defeated a guard; this returns a wrong value.

## The glm_contrast_z precedent has REPEATED on three more functions

`raw_roi_mean`, `roi_minus_reference` and `detection_floor` all have direction-only coverage and
**zero value coverage**. Survived:
- `raw_roi_mean`: `g[verts].mean()` -> `g.mean()` (ignores the ROI ENTIRELY), -> `.sum()`, -> `+1.0`
- `roi_minus_reference`: `-> +1.0`
- `detection_floor`: `return hi` -> `hi*2.0`, -> `lo`; all three input guards deleted; defaults
  `alpha 0.025 -> 0.5` and `power 0.80 -> 0.20`
- `preds.mean(axis=0)` -> `preds[0]` survived in `spatial_z`, `raw_roi_mean`, `roi_minus_reference`
  (every test feeds a (1,N) clip, so the documented row-aggregation contract has NO coverage)
- `peak_lag_trs`: `np.mean([c.mean(0) for c in courses])` -> `np.vstack(courses).mean(0)` survived —
  the docstring's explicit "a category with more events does not dominate" guarantee is untested,
  and the mutant changes the answer under the real 1:4 design

## My own test that claims the most and proves the least

`test_non_finite_policy_is_consistent_across_entry_points` — docstring says "One policy, not five
subtly different ones — NaN and both infinities are rejected everywhere." It checks **3 of ~10 call
sites**, and its `perm_p` call never reaches `mc_perm_p` at all (C(6,3)=20 dispatches to
`exact_perm_p`). Seven `_require_finite` calls can be deleted with this test still green.
Pass 3's verdict: "a test whose docstring claims universal coverage while covering 30% of the call
sites is worse than no test, because it retires the question."

## A Phase A gap that Phase A missed

The corrected figures in `spatial_z`'s docstring (FFCr -0.262 +/- 0.035, EBA -0.325 +/- 0.042) and in
`test_spatial_z_inverts_a_real_effect`'s docstring are **checked nowhere**.
`test_forty_seed_aggregation_is_reproducible` proves `run_many` is reproducible, never that it still
produces those values. The generator can drift and every published figure goes stale with a green
suite. Same for `test_reading_at_lag_5_attenuates_a_real_effect`: it asserts `< 0.35` where the
measured value is 0.2186 and `ops/source-of-truth.md:51` cites this very test as verification for
"21.9%". The attenuation could regress to 34% and the test still passes.

## Same vulnerable pattern still elsewhere

`spatial_z` (returns nan), `u_statistic` (returns a FINITE WRONG U — NaN compares False so it scores
as a loss with no tie credit), `perm_null_deltas` (all-nan null silently fails G2), `iut_pass`
(nan p-value becomes a quiet False). `define_froi` and `spatial_z` still unnormalised for selectors.
Negative and duplicate indices slip through every guard.

## Genuinely well protected (clean findings)

- `glm_contrast_z`'s core estimator — the scipy oracle at 1e-9 kills ddof changes, the pooled/Welch
  swap, an na/nb swap and any rescaling; the sign tests kill `return 0.0` and the flip. The S1
  precedent is closed **for this function's value**.
- The exact permutation machinery (1/70, 2/70, 1/20, 4/20 all pinned; strict-`>` killed;
  `u_statistic` ties pinned; `spatial_z`'s `sd == 0 -> 0.0` killed).
- `event_locked_contrast`'s category-mean semantics; `peri_event_timecourse`'s time axis;
  `define_froi`'s M1 boundary on all three sides.
- `peak_lag_trs`'s signature change is fully propagated — contracts, decision log, and no caller
  anywhere passes a bare array.

## Triage context

`define_froi`, `event_locked_contrast` and `peak_lag_trs` have **no production callers yet** (tests
only), so F3, F6, F7, A0, F11, F12 are latent S2 traps rather than wrong numbers already published.
A1, A2, F4, F5, A7 sit in functions S2 will call directly. A3/A4/A6 affect the inputs to the
published floor table.
