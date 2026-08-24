# Phase B — mutation ledger (mechanism level)

Run against `3e515f8`+ with 91 tests. Source restored and checksum-verified after every mutation (`ff25a7c7e63bd61e`).

Each row: a deliberate revert of one mechanism-level fix, the finding it came from, the test
family expected to catch it, and which test actually failed.

| id | behaviour deliberately broken | finding | expected catcher | result | test that fired |
|---|---|---|---|---|---|
| M01 | define_froi bypasses the selector validator | A0 | selector-CLASS | **DETECTED** | `test_bool_mask_and_equivalent_indices_agree_at_every_entry_point, test_empty_selection_raises_at_every_entry_point_not_just_two` |
| M02 | validator special-cases only dtype==bool (0/1 int mask read as indices) | F4 | selector-CLASS | **DETECTED** | `test_every_selector_entry_point_rejects_the_whole_bad_selector_CLASS` |
| M03 | negative indices accepted | A8 | selector-CLASS | **DETECTED** | `test_every_selector_entry_point_rejects_the_whole_bad_selector_CLASS` |
| M04 | duplicate indices accepted | dedup | selector-CLASS | **DETECTED** | `test_every_selector_entry_point_rejects_the_whole_bad_selector_CLASS` |
| M05 | out-of-range indices accepted | range | selector-CLASS | **DETECTED** | `test_every_selector_entry_point_rejects_the_whole_bad_selector_CLASS` |
| M06 | no guard on other_responses (bare 2-D iterates to rows) | F3 | shape-BOTH-args | **DETECTED** | `test_event_locked_contrast_rejects_bad_shapes_on_BOTH_arguments` |
| M07 | 0-d input falls through to TypeError | 0-d | shape-BOTH-args | **DETECTED** | `test_event_locked_contrast_rejects_bad_shapes_on_BOTH_arguments` |
| M08 | identical category courses accepted | F6 | peak-lag-degenerate | **DETECTED** | `test_peak_lag_rejects_every_degenerate_configuration_that_satisfies_the_count` |
| M09 | flat/constant category course accepted | F6b | peak-lag-degenerate | **DETECTED** | `test_peak_lag_rejects_every_degenerate_configuration_that_satisfies_the_count` |
| M10 | empty (0-event) category accepted | F7 | peak-lag-degenerate | **DETECTED** | `test_peak_lag_rejects_every_degenerate_configuration_that_satisfies_the_count` |
| M11 | u_statistic unguarded (NaN scores as a loss -> finite WRONG U) | M3-D | nonfinite-EVERY | **DETECTED** | `test_inf_is_treated_exactly_like_nan_everywhere, test_non_finite_is_rejected_at_EVERY_entry_point_the_docstring_claims` |
| M12 | perm_null_deltas unguarded (all-NaN null silently fails G2) | M3-D | nonfinite-EVERY | **DETECTED** | `test_non_finite_is_rejected_at_EVERY_entry_point_the_docstring_claims` |
| M13 | _resolve_rows unguarded (NaN > tol is False -> silent wrong row) | A1 | resolve-rows-direct | **DETECTED** | `test_resolve_rows_rejects_non_finite_directly_not_only_via_its_callers` |
| M14 | row_times_from_segments unguarded (NaN defeats diff<=0) | A1b | row-times-direct | **DETECTED** | `test_non_finite_is_rejected_at_EVERY_entry_point_the_docstring_claims, test_row_times_from_segments_rejects_non_finite_directly` |
| M15 | empty-selection check placed BEFORE normalisation | A2 | empty-EVERY | **DETECTED** | `test_empty_selection_raises_at_every_entry_point_not_just_two` |
| M16 | spatial_z bypasses the selector validator | A0b | selector-CLASS | **DETECTED** | `test_empty_selection_raises_at_every_entry_point_not_just_two, test_every_selector_entry_point_rejects_the_whole_bad_selector_CLASS` |

**16/16 detected.**

## Two weak tests found BY this process, not by running the suite

Ordinary execution passed in both cases. Only deliberate breakage exposed them.

1. **A generic `pytest.raises(ValueError)` let M02 survive.** With the ambiguity check removed, a
   0/1 int8 mask becomes `[0,0,...,1,1,1,...]`, which trips the *duplicate* check instead — so
   something still raised and the assertion passed while the fix was gone. **Strengthened:** every
   bad-selector case now pins its specific reason (`match="ambiguous"`, `match="negative"`, ...).
2. **Testing `_resolve_rows` only through its caller let M13 survive.** `peri_event_timecourse`
   guards onsets before `_resolve_rows` ever sees them, so the outer guard masked the inner one.
   **Strengthened:** `_resolve_rows` and `row_times_from_segments` now have direct tests.

A third gap was found the same way while wiring the discovery contract: `roi_minus_reference` takes
TWO selectors and the harness only ever exercised the first — the same 'guarded one argument of two'
pattern that produced F3, this time in the test suite. Both positions are now exercised.

---

## Second mechanism pass — mutation battery (39 mutations, all detected)

Run against a **verified-green baseline**: the script now refuses to run unless the unmutated
suite passes first. An earlier run of this battery reported 39/39 while one baseline test was
failing — under `pytest -x` a red baseline makes every mutant report DETECTED, so that result
was void. The guard exists because that happened.

Groups: **ORIGINAL** = the 16 author-designed mutations from the first pass; **REVIEWER** = the 5
that survived the independent review, now mandatory regressions; **CLASS** = new mutations
probing each stated invariant (I1-I5), including cases neither the author nor the reviewer
demonstrated.

| id | group | detected | invariant / defect reintroduced |
|----|-------|----------|----------------------------------|
| O01 | ORIGINAL | yes | negative indices accepted; they defeat the overlap guard |
| O02 | ORIGINAL | yes | duplicate indices double-weight a vertex in every ROI mean |
| O03 | ORIGINAL | yes | 0/1 ambiguity resolved by guessing, the original silent wrong ROI |
| O04 | ORIGINAL | yes | out-of-range indices accepted |
| O05 | ORIGINAL | yes | empty category fabricates a lag via all-NaN argmax |
| O06 | ORIGINAL | yes | flat filler category dilutes the pool back to target-only |
| O07 | ORIGINAL | yes | pooling reduced to the target's own course (C5) |
| O08 | ORIGINAL | yes | pooled SE instead of Welch; not level-alpha at unequal n |
| O09 | ORIGINAL | yes | biased variance in the contrast SE |
| O10 | ORIGINAL | yes | top_n >= parcel size, a silent no-op selection |
| O11 | ORIGINAL | yes | empty parcel accepted |
| O12 | ORIGINAL | yes | empty ROI returns nan instead of raising |
| O13 | ORIGINAL | yes | fROI returned unsorted |
| O14 | ORIGINAL | yes | non-integer float indices silently truncated |
| O15 | ORIGINAL | yes | permutation p can be exactly zero; invalid estimator |
| O16 | ORIGINAL | yes | contrast with no comparison category returns a bare mean |
| R01 | REVIEWER | yes | F1: 2-D boolean mask of size n read in flat C order -> wrong vertex set |
| R02 | REVIEWER | yes | F1: 1-D rule exempts booleans, the exact shape of the original bypass |
| R03 | REVIEWER | yes | F4: spatial_z guards only the ROI while dividing by whole-map statistics |
| R04 | REVIEWER | yes | F5: syntactic duplicate check; scaled/offset/row-duplicated copies pass |
| R05 | REVIEWER | yes | F6: u_statistic recomputes from exhausted iterators -> finite wrong U=0.0 |
| C01 | CLASS | yes | I1: integer selectors keep caller order; representations diverge |
| C02 | CLASS | yes | I1: bare scalar selector accepted |
| C03 | CLASS | yes | I1: boolean mask of the wrong length accepted |
| C04 | CLASS | yes | I1: object/string dtype reaches np.isfinite and raises TypeError |
| C05 | CLASS | yes | I1/F7: unstable tie-break; two encodings give different fROIs |
| C06 | CLASS | yes | I2: spatial_z accepts non-finite anywhere and returns nan |
| C07 | CLASS | yes | I2: only one of two conditions guarded |
| C08 | CLASS | yes | I2: only the ROI guarded, not the reference |
| C09 | CLASS | yes | I3: exact_perm_p re-reads consumed arguments |
| C10 | CLASS | yes | I3: mc_perm_p measures an argument it has already consumed |
| C11 | CLASS | yes | I3: perm_null_deltas calls len() on a consumed argument |
| C12 | CLASS | yes | I4: degeneracy judged from the first event only, not the mean course |
| C13 | CLASS | yes | I4: degeneracy check disabled entirely |
| C14 | CLASS | yes | I4: no mean-centring, so a constant-offset duplicate slips through |
| C15 | CLASS | yes | I5/F3: coverage keyed on function name again, so one argument stands for all |
| C16 | CLASS | yes | I5: an array argument silently loses its non-finite coverage |
| C17 | CLASS | yes | I5/F2: selector discovery returns nothing, so every selector rule is vacuous |
| C18 | CLASS | yes | I5/F2: enumeration filters on __module__ again, hiding partials/wrappers |

Reproduce: `python3 scripts/mutate_roi_stats.py --jobs 3` (each mutation runs in a fresh copy
under /tmp; the repository is never modified).

### Mutations that were redesigned because they proved nothing

Recorded rather than quietly replaced — a mutation that cannot fail is not evidence.

* **R02** originally read the mask with `ravel(order='F')`. For a 1-D array Fortran order *is* C
  order, and 2-D masks are now rejected upstream, so the mutant was **equivalent** — it survived
  because it changed nothing. Replaced with `arr.ndim != 1 and arr.dtype != bool`, which
  reintroduces the F1 bypass exactly.
* **R05** deleted the reassignment from the validated array. The names had already been bound to
  materialised lists above, so this too was **equivalent**. Replaced with iteration over the
  original `face_vals`/`scene_vals`, which is the actual F6 defect.
* **C09** never applied: the search string carried the wrong error label. A `NOT_APPLIED`
  mutation is reported as STALE rather than counted as a pass.
* **C17** originally injected an early `return` into a test. Mutating a test into a no-op can
  never be detected by that same suite, so it was an **invalid** mutation by construction.
  Replaced with a mutation of the selector-discovery helper, which the planted-violation
  self-tests do detect.

### What the battery found that the author did not anticipate

Seven survivors in the first valid run, each a real gap rather than a bad mutation:

| survivor | what had no test |
|----------|------------------|
| O13 | the fROI was never asserted to be ascending, and the fixture's contrast rank happened to coincide with vertex order |
| O15 | nothing pinned the `(ge + 1) / (n_perm + 1)` estimator floor, so `p = 0` was reportable |
| O16 | a contrast with no comparison category could return a bare mean under the name of a contrast |
| C08 | `roi_minus_reference` was poisoned only inside the ROI, so dropping the reference-side guard changed nothing |
| C12 | every fixture had identical events, so a guard reading only the first event was indistinguishable from one reading the mean |
| C14 | no constant-offset duplicate was tested, so mean-centring could be removed |
| R04 | the degenerate-configuration test contained no rescaled, offset, or row-duplicated course |

Two further defects were found by the new tests rather than by mutation:

* `perm_null_deltas` calls `len()` on an already-consumed argument — a **sixth** instance of the
  I3 class, found by the representation-invariance harness on its first run. Neither the author
  nor the independent reviewer had identified it.
* `_selector_entry_points` still filtered on `__module__` after `_public_functions` had been
  rewritten not to. The planted-partial self-test caught it. The F2 fix had been applied to the
  enumeration helper but not to the selector discovery that actually uses it.

---

## Third battery — after the second review (45 mutations, all detected)

107 tests. Baseline verified green before the run.

| id | group | detected | invariant / defect reintroduced |
|----|-------|----------|----------------------------------|
| O01 | ORIGINAL | yes | negative indices accepted; they defeat the overlap guard |
| O02 | ORIGINAL | yes | duplicate indices double-weight a vertex in every ROI mean |
| O03 | ORIGINAL | yes | 0/1 ambiguity resolved by guessing, the original silent wrong ROI |
| O04 | ORIGINAL | yes | out-of-range indices accepted |
| O05 | ORIGINAL | yes | empty category fabricates a lag via all-NaN argmax |
| O07 | ORIGINAL | yes | pooling reduced to the target's own course (C5) |
| O08 | ORIGINAL | yes | pooled SE instead of Welch; not level-alpha at unequal n |
| O09 | ORIGINAL | yes | biased variance in ONE arm of the contrast SE (asymmetric estimator) |
| O10 | ORIGINAL | yes | top_n >= parcel size, a silent no-op selection |
| O11 | ORIGINAL | yes | empty parcel accepted |
| O12 | ORIGINAL | yes | empty ROI returns nan instead of raising (first occurrence: spatial_z) |
| O13 | ORIGINAL | yes | fROI returned unsorted |
| O14 | ORIGINAL | yes | non-integer float indices silently truncated |
| O15 | ORIGINAL | yes | permutation p can be exactly zero; invalid estimator |
| O16 | ORIGINAL | yes | contrast with no comparison category returns a bare mean |
| R01 | REVIEWER | yes | F1: 2-D boolean mask of size n read in flat C order -> wrong vertex set |
| R02 | REVIEWER | yes | F1: 1-D rule exempts booleans, the exact shape of the original bypass |
| R03 | REVIEWER | yes | F4: spatial_z guards only the ROI while dividing by whole-map statistics |
| R05 | REVIEWER | yes | F6: u_statistic recomputes from exhausted iterators -> finite wrong U=0.0 |
| C01 | CLASS | yes | I1: integer selectors keep caller order; representations diverge |
| C02 | CLASS | yes | I1: bare scalar selector accepted |
| C03 | CLASS | yes | I1: boolean mask of the wrong length accepted |
| C04 | CLASS | yes | I1: object/string dtype reaches np.isfinite and raises TypeError |
| C05 | CLASS | yes | I1/F7: unstable tie-break; two encodings give different fROIs |
| C06 | CLASS | yes | I2: spatial_z accepts non-finite anywhere and returns nan |
| C07 | CLASS | yes | I2: only one of two conditions guarded |
| C08 | CLASS | yes | I2: only the ROI guarded, not the reference |
| C09 | CLASS | yes | I3: exact_perm_p re-reads consumed arguments |
| C10 | CLASS | yes | I3: mc_perm_p measures an argument it has already consumed |
| C11 | CLASS | yes | I3: perm_null_deltas calls len() on a consumed argument |
| O06 | CLASS | yes | a category whose MEAN course is flat dilutes the pool back to target-only |
| R04 | REVIEWER | yes | F5: pooled-collinearity rule removed; scaled/offset/row-duplicated copies pass |
| C12 | CLASS | yes | I4: degeneracy judged from the first event only, not the mean course |
| C13 | CLASS | yes | I4: the <3-lag hole reopened, where every course pair is collinear |
| C14 | CLASS | yes | I4: no mean-centring, so a constant-offset duplicate slips through |
| C19 | CLASS | yes | I4: total cancellation accepted; argmax of a flat pool fabricates lag 0 |
| C20 | CLASS | yes | I4 over-rejection: a weak but real mean response treated as flat |
| C21 | CLASS | yes | I1: 3-D preds accepted; n_vertices then describes an axis verts never indexes |
| C22 | CLASS | yes | I1: masked selector silently unmasked, two encodings of one set disagree |
| C23 | CLASS | yes | I3: row_times_from_segments rejects a single-pass iterable |
| C24 | CLASS | yes | I3: row_times_s generator becomes a 0-d object array |
| C15 | CLASS | yes | I5/F3: coverage keyed on function name on BOTH sides, the faithful F3 revert |
| C16 | CLASS | yes | I5: an array argument silently loses its non-finite coverage |
| C17 | CLASS | yes | I5/F2: selector discovery returns nothing, so every selector rule is vacuous |
| C18 | CLASS | yes | I5/F2: enumeration filters on __module__ again, hiding partials/wrappers |

### A mutation that could not fail, and what it cost

The second review found that **R03** — the *mandatory* F4 regression, the one mutation whose whole
job was to prove `spatial_z`'s guard extent was asserted — was invalid:

```python
'g = _require_finite(g[verts], "spatial_z map") and g or g'
```

`ndarray and ndarray` raises `ValueError: truth value ... is ambiguous` on **every** call,
including clean data. It was therefore detected by ordinary `spatial_z` tests and by not one
non-finite test. The previous ledger row claiming R03 established F4 coverage was an overclaim,
and it is withdrawn. Narrowing the guard back to the ROI by hand left the suite **green at 62
passed** with F4 fully restored.

R03 now perturbs behaviour only on the non-finite path. The real correction is not the mutation
but `test_each_guard_covers_exactly_the_data_its_statistic_reads`, which declares each function's
read extent and asserts it in **both** directions.

### Other battery defects the second review found

* **C15** did not reintroduce F3. It only *added* bare names to `covered`, which cannot make
  anything missing, and it was detected by an unrelated `stale` assertion. A faithful F3 revert —
  keying both sides on the function name — passed the suite. Fixed by factoring the coverage
  computation into a helper shared by the completeness test and a new planted-violation self-test,
  then retargeting C15 at that helper.
* **O09** and **O12** use `replace(..., 1)` and so hit only the first occurrence. They test
  something narrower than their descriptions claimed; the descriptions now say so.
* **O06** and **R04** were left duplicated across two sections after the I4 rewrite and reported
  STALE. Deduplicated.

Running total of mutations that proved nothing and were redesigned rather than quietly replaced:
**seven** (R02, R05, C09, C17 from the first pass; R03, C15, plus the O06/R04 duplicates here).
That count is kept deliberately: it is the honest measure of how much of a mutation score is
evidence and how much is bookkeeping.

---

## Fourth battery — after the third review (55 mutations, all detected)

`tests/test_roi_stats.py`: **76 tests** (the suite the battery runs). Whole repo: 114.
Both numbers are stated because the earlier ledger said "107 tests" while the battery only ever
ran one file — the mutation score is backed by the 76, not the 114.

| id | group | detected | invariant / defect reintroduced |
|----|-------|----------|----------------------------------|
| O01 | ORIGINAL | yes | negative indices accepted; they defeat the overlap guard |
| O02 | ORIGINAL | yes | duplicate indices double-weight a vertex in every ROI mean |
| O03 | ORIGINAL | yes | 0/1 ambiguity resolved by guessing, the original silent wrong ROI |
| O04 | ORIGINAL | yes | out-of-range indices accepted |
| O05 | ORIGINAL | yes | empty category fabricates a lag via all-NaN argmax |
| O07 | ORIGINAL | yes | pooling reduced to the target's own course (C5) |
| O08 | ORIGINAL | yes | pooled SE instead of Welch; not level-alpha at unequal n |
| O09 | ORIGINAL | yes | biased variance in ONE arm of the contrast SE (asymmetric estimator) |
| O10 | ORIGINAL | yes | top_n >= parcel size, a silent no-op selection |
| O11 | ORIGINAL | yes | empty parcel accepted |
| O12 | ORIGINAL | yes | empty ROI returns nan instead of raising (first occurrence: spatial_z) |
| O13 | ORIGINAL | yes | fROI returned unsorted |
| O14 | ORIGINAL | yes | non-integer float indices silently truncated |
| O15 | ORIGINAL | yes | permutation p can be exactly zero; invalid estimator |
| O16 | ORIGINAL | yes | contrast with no comparison category returns a bare mean |
| R01 | REVIEWER | yes | F1: 2-D boolean mask of size n read in flat C order -> wrong vertex set |
| R02 | REVIEWER | yes | F1: 1-D rule exempts booleans, the exact shape of the original bypass |
| R03 | REVIEWER | yes | F4: spatial_z guards only the ROI while dividing by whole-map statistics |
| R05 | REVIEWER | yes | F6: u_statistic recomputes from exhausted iterators -> finite wrong U=0.0 |
| C01 | CLASS | yes | I1: integer selectors keep caller order; representations diverge |
| C02 | CLASS | yes | I1: bare scalar selector accepted |
| C03 | CLASS | yes | I1: boolean mask of the wrong length accepted |
| C04 | CLASS | yes | I1: object/string dtype reaches np.isfinite and raises TypeError |
| C05 | CLASS | yes | I1/F7: unstable tie-break; two encodings give different fROIs |
| C06 | CLASS | yes | I2: spatial_z accepts non-finite anywhere and returns nan |
| C07 | CLASS | yes | I2: only one of two conditions guarded |
| C08 | CLASS | yes | I2: only the ROI guarded, not the reference |
| C09 | CLASS | yes | I3: exact_perm_p re-reads consumed arguments |
| C10 | CLASS | yes | I3: mc_perm_p measures an argument it has already consumed |
| C11 | CLASS | yes | I3: perm_null_deltas calls len() on a consumed argument |
| O06 | CLASS | yes | a category whose MEAN course is flat dilutes the pool back to target-only |
| R04 | REVIEWER | yes | F5: pooled-collinearity rule removed; scaled/offset/row-duplicated copies pass |
| C12 | CLASS | yes | I4: degeneracy judged from the first event only, not the mean course |
| C13 | CLASS | yes | I4: the <3-lag hole reopened, where every course pair is collinear |
| C14 | CLASS | yes | I4: no mean-centring, so a constant-offset duplicate slips through |
| C19 | CLASS | yes | I4: total cancellation accepted; argmax of a flat pool fabricates lag 0 |
| C20 | CLASS | yes | I4 over-rejection: a weak but real mean response treated as flat |
| C21 | CLASS | yes | I1: 3-D preds accepted; n_vertices then describes an axis verts never indexes |
| C22 | CLASS | yes | I1: masked selector silently unmasked, two encodings of one set disagree |
| C23 | CLASS | yes | I3: row_times_from_segments rejects a single-pass iterable |
| C24 | CLASS | yes | I3: row_times_s generator becomes a 0-d object array |
| C25 | CLASS | yes | S6: u_statistic accepts a (n_events, n_lags) time course |
| C26 | CLASS | yes | S6: mc_perm_p broadcasts a 2-D input to a finite wrong p-value (0.567 -> 0.0005) |
| C27 | CLASS | yes | S6: the G2 magnitude null means over both axes |
| C28 | CLASS | yes | I1: rank collapse takes row 0 instead of the row mean |
| C29 | CLASS | yes | I1: rank collapse averages the wrong axis |
| C30 | CLASS | yes | I1: n_vertices from a raw .shape again, skipping the rank precondition |
| C31 | CLASS | yes | I2/I5: read-extent requirement no longer derived, so a new function is invisible |
| C32 | CLASS | yes | I2/I5: read-extent derivation returns nothing |
| C33 | CLASS | yes | pre_trs outside the lag grid reports a lag that does not exist |
| C34 | CLASS | yes | row index treated as TR index; the confusion the module was built to prevent |
| C15 | CLASS | yes | I5/F3: coverage keyed on function name on BOTH sides, the faithful F3 revert |
| C16 | CLASS | yes | I5: an array argument silently loses its non-finite coverage |
| C17 | CLASS | yes | I5/F2: selector discovery returns nothing, so every selector rule is vacuous |
| C18 | CLASS | yes | I5/F2: enumeration filters on __module__ again, hiding partials/wrappers |

### Two more mutations that could not fail

The third review found **C10** and **C11** deleted a binding along with the line they targeted and
raised `NameError` on every call. C11's damage was invisible in the battery output because
`_require_finite` raises before the mutated line is reached, so it looked like a clean targeted
detection. **The honest score at the time was 43/45, not 45/45.** Both repaired to mutate only the
measured expression; both are now detected by `test_iterable_representations_of_one_sample_agree`,
so the I3 invariant was genuinely asserted and nothing was hiding behind them.

Running total of mutations that proved nothing and were redesigned rather than quietly replaced:
**nine** — R02, R05, C09, C17 (first pass), R03, C15 (second), C10, C11 and the O06/R04 duplicates
(third). The count is kept because it is the honest measure of how much of a mutation score is
evidence and how much is bookkeeping.

### Ambiguous search patterns anchored

R01, R02 and C02 each matched **two** places in the module — `_as_vertex_indices` and
`_as_event_vector` — and hit the intended one only because it appears earlier in the file.
Reordering the module would have silently retargeted three mandatory regressions without any
STALE warning. All three now anchor on a unique surrounding string. O09 and O12 remain
deliberately narrow and their descriptions now say so.

### Two survivors in the first run of this battery, both in the evidence layer

* **C25** — the S6 rank guard on `u_statistic` could be removed with a green suite. The Finding-1
  fix had been verified interactively at all five entry points and reported as closed, and no test
  was ever written for it. This is the *verified but not asserted* pattern the third review named,
  committed one turn after reading that review. Closed by `_event_vector_entry_points()`, which
  derives all ten (function, argument) pairs from the signatures.
* **C31** — the read-extent derivation could be removed from the completeness test without failing
  anything, because the planted-violation self-test computed the derivation *separately*. Two
  tests, one invariant, no shared code, and the weakening slipped between them. Closed by a single
  `_read_extent_undeclared()` helper that both call.

### Server hygiene

`ignore_patterns` excluded `data/` and `notebooks/` but not `results/algonauts/`, so every battery
run copied a 73 MB zip 56 times — about **4 GB through /tmp per run** on a shared box with 21 GB
free. Now excludes `results`, `*.zip`, `*.h5`, `*.pdf`.
