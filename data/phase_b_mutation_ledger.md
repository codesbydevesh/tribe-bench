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
