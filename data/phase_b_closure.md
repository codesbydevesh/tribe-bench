# Phase B — closure record

**Closed 2026-08-24** under a stated stopping rule, not under a claim of proof.

> **Stopping rule:** no demonstrated live wrong-number defect + independent numerical
> cross-checks pass + known residuals explicitly recorded → stop auditing, move to
> replication and paper work.

## The claim being made

> The latest independent audit found no remaining demonstrated wrong-number defect in the
> decision-critical paths; remaining issues are documented test/coverage or defensive-hardening
> gaps.

That is the whole claim. It is deliberately narrower than "the statistics module is correct",
which is **not** claimed and is not supported by this evidence. Four independent reviews each found
something the previous one and the author had missed; a fifth would very likely find something
too. What changed is the *kind* of thing being found, not the supply of things to find.

---

## 1. Live correctness

**The fourth independent review found zero demonstrated live wrong-number defects in the shipped
decision-critical paths.** Every finding in that round was either a guard against operator error,
a correct computation that no test asserted, or a regression with no live caller.

Independent numerical cross-checks performed by that reviewer, against implementations it wrote
itself or against SciPy:

| check | oracle | result |
|---|---|---|
| `u_statistic` | `scipy.stats.mannwhitneyu` | 400/400, heavy ties, unequal n |
| `_u_fast` | `u_statistic` | 400/400 |
| `exact_perm_p` | independent brute-force enumeration | 60/60 at unequal n with ties |
| `exact_perm_p` validity under H₀ | 3v5 simulation | 0.0345 / 0.0740 / 0.1965 at α = 0.05 / 0.10 / 0.20 |
| `perm_null_deltas` | independent enumeration | exact, null centred at 0 |
| Welch SE in `glm_contrast_z` | `scipy.stats.ttest_ind(equal_var=False)` | exact to 1e-9 at 7v19 |
| `detection_floor` bisection | monotonicity of `achieved_power` under fixed seed | sound; bracket valid |
| all 55 prior mutations | re-run with per-test attribution | 55/55 detected, 0 stale, 0 equivalent |

## 2. Residual verification and coverage risks

**Not** live defects. Recorded so they are not mistaken for either proof or for bugs.

| id | residual | status |
|---|---|---|
| RB-1 | `exact_perm_p` / `mc_perm_p` / `perm_null_deltas` unequal-n arm split | **correct**; was insufficiently asserted — every fixture was 4v4/3v3/8v8/15v15, so the split index was invisible. Tests added this round against brute-force enumeration. |
| RB-2 | `_u_fast` tie term | **correct**; was unasserted, though it is used by every Monte-Carlo p-value and by `detection_floor`. Dropping it moves a real Gate 0 curation p-value from 0.2589 to 0.1699, across the 0.20 gate. Now pinned against `u_statistic` over tied integer samples. |
| RB-3 | `event_locked_contrast` lost single-pass-iterable support | a **regression introduced by this diff** (the one event-vector boundary of six that skipped `_materialise`). **No live caller passes a generator**, so no result was affected. Fixed and asserted. |
| RB-4 | defensive guards on scalars | `pre_trs`, `post_trs`, `lag_trs`, `n_perm` now share one rule. Scalars remain the parameter class with the weakest coverage machinery: they have declared kinds but no derived harness. |
| RB-5 | `glm_contrast_z` scores a zero-SE vertex as zero evidence | **M010**, open. Pre-existing, out of Phase B scope. Can only lose a true effect, never manufacture a false one. Needs a decision, not a patch. Revisit in Phase C against real predictions. |
| RB-6 | `spatial_z` population-vs-sample sd convention | unpinned; negligible at 20,484 vertices. |
| RB-7 | `_IMPORTED_CALLABLES` | exempts by name from all four derived harnesses. Now checked that every member is genuinely not defined by this module. |
| RB-8 | `peak_lag_trs` inherent limits | `[tc[:k], tc[k:]]` is undetectable, and a dominant target makes the pooled peak be the target's peak. Both documented in the docstring; no input check can fix either. The sound protection is re-selecting the lag inside every permutation. |

## 3. History — preserved deliberately

Not compressed into "Phase B passed after several iterations". The sequence is the evidence.

| stage | what was found |
|---|---|
| original 7 findings | fixed at the **example** level |
| **review 1 → C** | 5 of 16 mutations survived. Four fixes had not closed their class: a `(30,2)` boolean mask read in flat C order selected a different ROI and let the overlap guard pass on an ROI compared against **itself**; `spatial_z` returned a silent `nan`; the degeneracy guard was syntactic; `u_statistic` returned a finite wrong `U = 0.0` |
| **review 2 → C** | `spatial_z`'s guard was *correct but asserted by nothing*, and the mandatory regression mutant was **invalid** — it raised on every call, so F4 was reintroducible with a green suite. The degeneracy guard still missed flat-*mean* categories and mutually cancelling triples |
| **review 3 → C** | `_as_event_vector` guarded **1 of 5** entry points: `perm_p` returned **0.0005 where the correct call returns 0.567**. `_as_vertex_map`'s collapse arithmetic was asserted by nothing because every 2-D fixture had identical rows. Read-extent completeness compared two hand-written lists |
| **review 4 → C** | **zero live wrong-number defects.** Six findings: one regression the author introduced while fixing review 3, two correct-but-unasserted mechanisms, three defensive-guard gaps |

Three facts worth keeping visible:

1. **Some defects were introduced by the fixes.** RB-3 was caused by the edit that fixed the
   `perm_p` defect. A `pre_trs` guard was written for one function while a source comment claimed
   its neighbour was covered — it was not. A mutation was broken by *improving an error message*:
   the formula `(ge + 1) / (n_perm + 1)` was written into the message text, so the first-occurrence
   match began hitting the string instead of the code.
2. **The weakest point migrated out of the code and into the evidence about the code** — bad
   guards → assertions written against one fixture placement → mutations that could not fail →
   completeness checks that were not derived. Each round the reviewers were told this explicitly,
   and each round they found the next instance of it.
3. **Eleven of the mutations proved nothing** and were redesigned rather than quietly replaced —
   two equivalent, two stale, one invalid by construction, two raising `NameError` on every call,
   and one retargeted onto an error string. The count is kept because it is the honest measure of
   how much of a mutation score is evidence and how much is bookkeeping. It also means two
   previously reported scores were wrong: "39/39" was reported over a mutant that could not fail,
   and "45/45" was really 43/45.

## 4. Final state

```
tests/test_roi_stats.py    86 tests
whole repository           124 tests
mutations                  69/69 detected, verified-green baseline
invariants                 I1-I5 + S6, stated in data/phase_b_invariants.md
independent reviews        4 completed (all C, all addressed), 1 stalled and relaunched
```

Phase B is closed under the stopping rule at the top of this file — **not** under a claim that the
module is proven correct. The next concrete correctness concern should come from S2 or from
paper generation, not from another audit of this module.
