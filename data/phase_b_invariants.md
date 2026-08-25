# Phase B — the invariants behind the findings

Written **before** the second implementation pass, per the standing instruction: fix the
mechanism, not the reproduction. The independent review returned **C (blocking)** with five
surviving mutants (F1, F4, F5, F6, F7) plus two mechanism findings about the coverage machinery
itself (F2, F3). They are not seven bugs. They are five invariants that were never stated, and
therefore never tested.

The failure mode common to all of them, in one sentence:

> The guards were written against **one representation of one argument of one function**, and the
> coverage machinery verified **function names** rather than **(function, argument, representation)**
> triples. So every rule was true of the example that motivated it and false of its neighbours.

---

## I1 — Selector canonicalisation

**A vertex selector denotes a SET of vertices. Every accepted representation canonicalises to
exactly one array: 1-D, integer, strictly ascending, unique, within `[0, n_vertices)`. Two
representations that denote the same set must produce byte-identical canonical output. An input
with no unambiguous 1-D reading is rejected, never guessed.**

- The shape contract is a precondition on the **raw input**, established *before* any
  representation-specific conversion. F1 existed because the boolean branch returned before the
  dimensionality rule, so a `(30, 2)` mask over 60 vertices was read in flat C order.
- Order is **not** semantically meaningful in any selector argument — every consumer takes a mean
  over the selected vertices or uses the selector as an index set. Canonical order is therefore
  ascending, and `define_froi` is the only function that returns a subset, so it is the only place
  a tie-break rule is observable.
- Rejection is a valid answer. Guessing is what produced a silently wrong ROI.

Consequence for testing: assert **representation equivalence**, not per-example behaviour.

## I2 — Guard exactly the data the computation reads

**A validation guard must cover precisely the data its statistic consumes — no less, and the
guarded extent is a per-function declaration rather than a default.**

`raw_roi_mean`, `roi_minus_reference`, `glm_contrast_z` and `peri_event_timecourse` read only
their selected region, and guarding the region is correct for them. `spatial_z` divides by the
**brain-wide** mean and sd, so guarding only `g[verts]` left a non-finite value at any other
vertex to produce a silent `nan` (F4). The rule was right for four functions and wrong for the
fifth because it was copied rather than derived.

## I3 — Consume once, compute on the validated copy

**Any iterable accepted at a public boundary is materialised exactly once, and every subsequent
validation and computation reads that materialised array — never the original argument.**

F6: `u_statistic` materialised a list for the guard, discarded it, then re-read the now-exhausted
iterators and returned `U = 0.0` — finite, wrong, and maximally anti-selective. The same shape is
present in `exact_perm_p` (which consumes its arguments three times) and in the `len(face_vals)`
calls in `mc_perm_p` and `perm_p`.

Consequence: for identical logical data, `list` / `tuple` / generator / `ndarray` must give an
identical result. "It did not crash" is not the assertion.

## I4 — Guards target semantic degeneracy, not syntactic form

**A guard that rejects a degenerate configuration must be expressed in terms of the quantity the
computation actually derives, so that every representation of that degeneracy is rejected
together.**

F5: `peak_lag_trs` compared raw event matrices under a `shape ==` precondition, while pooling
consumes only `c.mean(axis=0)`. So `[tc, tc]` was rejected and `[tc, vstack([tc, tc])]`,
`[tc, tc*2]`, `[tc, tc+1]` were accepted — all of them argmax-identical to the target alone, all
returning the C5 answer with measured type-I 0.2032.

The real invalidity is: **the pooled course carries no information the target's own course did not
already carry.** That is a statement about the mean courses, up to positive scale and offset — not
about row counts.

## I5 — Coverage is keyed on (function, argument, representation)

**Function coverage and representation coverage are separate guarantees, and the first does not
imply the second.**

- F2: discovery filtered on `__module__ == roi_stats`, so a module-level `functools.partial` with
  a parameter named `verts` evaded all three safeguards with a green suite. The two claimed
  safeguards were not independent — `_call_with_selector` can only fire for a function discovery
  already found.
- F3: the non-finite cross-check keyed on `k.split(" ")[0]` — the bare function name — so one
  poisoned argument satisfied completeness for a function with several. This is the
  `roi_minus_reference`-two-selectors precedent recurring *inside the mechanism written to close
  it*.

Consequence: the coverage key is a `(function, parameter)` pair derived from the signature, an
unrecognised public callable is a failure rather than a silent miss, and a **new** parameter name
fails by default instead of being missed.

---

## What these invariants say about the previous pass

The 16/16 mutation result was true and is not withdrawn: the reviewer's own mutants against the
ambiguity, duplicate, range, empty-category, flat-course, identical-course and `n_vertices`
plumbing rules were all caught. The per-argument selector harness was a genuine advance.

The gap is describable in one line, and it is a **testing-strategy** gap rather than a collection
of coding slips:

> Every new rule was written and tested against 1-D, C-contiguous, correctly-shaped, already-
> materialised inputs, and the completeness mechanisms verified names rather than coverage.

Five of the reviewer's sixteen mutants survived, all inside that one band.

---

# Second review (2026-08-24) — verdict C again, and what it changed

The second independent reviewer, told explicitly *not* to re-confirm the first seven findings but
to hunt alternative manifestations of the same classes, returned **C (blocking)** with two
findings. Both were verified independently before any edit. Neither was a repeat.

## What it proved about the difference between a fix and an assertion

**Finding A did not find a bug in the source.** `spatial_z`'s guard was correct. What the reviewer
found is that **nothing asserted it**, and that the mandatory F4 regression mutant was invalid:

```python
'g = _require_finite(g[verts], ...) and g or g'   # ndarray and ndarray -> ValueError
```

That raises on *every* call including clean data, so it was caught by ordinary `spatial_z` tests
and by no non-finite test at all. Narrowing the guard back to the ROI by hand left the suite
**green at 62 passed** with F4 fully restored. A 39/39 mutation score had been reported over a
mutant that could not fail.

Every non-finite fixture in the suite placed its poison **inside** the selected region — the one
placement where a too-narrow guard still looks correct. The author had reasoned explicitly about
read extent for `define_froi` and `roi_minus_reference` and then copied the inside-the-region
convention to the single function where the outside case is the whole point.

The correction is not a test for `spatial_z`. It is `_WHOLE_MAP_CONSUMERS` /
`_REGION_LOCAL_CONSUMERS`: the read extent is now **declared per function and asserted in both
directions** — a whole-map consumer must reject poison anywhere, a region-local consumer must
still return a finite answer, because a guard that is too wide is also a defect. A second test
checks the declaration against the implementation, so a function that starts dividing by a
whole-map statistic cannot keep a region-local declaration.

## Finding B: I4 was still syntactic, one level up

The pooled-degeneracy guard compared **pairs of category means** and skipped a category whose mean
course had zero norm (`if ni == 0 or nj == 0: continue`), while the flat check keyed on the raw
event matrix. So a category whose *events* are structured but whose *mean course* is flat — events
disagreeing in sign, ordinary in real data — passed both guards, and so did three categories where
no pair is collinear but two cancel:

```
[tc, vstack([r, -r])]  -> 4     mean course identically zero
[tc, a, b]             -> 4     a and b cancel; no pair collinear
```

4 is the target-only answer: C5 restored, type-I 0.2032.

The rule is now expressed on the **pooled course itself** — the only quantity the function derives.
Reject a category whose mean-centred mean course is zero; reject a flat pool; reject a pool that is
argmax-identical to any single category. One rule covers duplicates, rescaling, offsets,
row-duplication, flat-mean contributors and mutual cancellation. Inputs with fewer than three lags
are refused outright instead of being given a weaker guard.

**Deliberately not rejected:** a category whose mean response is weak but non-zero. "Flat" means
zero to floating point at the data's own scale, not statistically small. A condition that genuinely
shows no time-locked response is a real result, not malformed input, and refusing to compute would
be a worse error than the one being prevented. That case remains the documented dominance limit,
which no input check can fix.

## The pattern across both reviews

First review: guards written against one representation of one argument.
Second review: **assertions** written against one placement of one fixture, and a mutation that
could not fail. The defect moved from the code to the evidence about the code — which is why the
second review had to be independent of the mutation work as well as of the implementation.

---

# Third review (2026-08-24) — verdict C, and a sixth invariant

Three blocking findings. Two were in the code, one in the evidence. All verified before any edit.

## S6 — one rank contract, enforced at every argument that takes an event vector

**The worst defect found in any of the three reviews**, and it was hiding in plain sight behind a
docstring that claimed the opposite.

`_as_event_vector` was written for exactly this failure: `peri_event_timecourse` returns
`(n_events, n_lags)` and is always in scope beside `event_locked_response`'s `(n_events,)`, so the
wrong one can be passed to any argument that takes a per-event vector. Its docstring said *"Used
for EVERY such argument (F3)."* It was used in **one function out of five**.

```
perm_p(A[:,0], B[:,0])   correct 1-D  ->  0.567216
perm_p(A,      B)        2-D          ->  0.000500
```

`_u_fast` broadcasts `(na,1,L) > (1,nb,L)` and `.sum()` collapses it to a scalar, so a time course
produces a *number* rather than an error. `perm_null_deltas` — the G2 magnitude null — averages
over both axes. `u_statistic` and `exact_perm_p` did raise, but only numpy's "truth value is
ambiguous", not a contract error, and both accepted `(n,1)` silently.

This is a finite wrong p-value in the module whose opening docstring says *"a wrong permutation
null turned Gate 0's original '7/9 pairs' rule into a p=0.20 coin flip."* A null result reads as a
headline result. Live callers route through it: `scripts/detection_floor_table.py`, the Gate 0
curation scripts, `detection_floor` itself.

All five entry points now validate, materialising once so I3 is preserved. Coverage is derived from
the signatures over ten `(function, argument)` pairs, not hand-listed.

## What the review proved about the shape of the remaining risk

**Finding 2 — the new helper's only arithmetic was asserted by nothing.** `_as_vertex_map` was
introduced by this very diff and four public statistics depend on it. Its rank *precondition* was
tested; its *computation* was not, because every 2-D `preds` fixture in the suite has identical
rows (`np.tile(...)`, `np.ones(...)`, `g[None,:]`). So `mean over rows` and `row 0` were
indistinguishable in every test, and `return arr[0]` passed all 69.

**Finding 3 — F4 was still reintroducible in a NEW function with a green suite.** The read-extent
machinery built in response to review 2 compared two hand-written lists, so a function in neither
was invisible. It was the only completeness check in the suite not derived from `inspect.signature`
— F3's own class ("keyed on names rather than derived from the signature") recurring inside the
mechanism written to close F4, which had itself recurred inside the mechanism written to close F3.

## The trajectory across four passes

| pass | where the defect was |
|------|----------------------|
| original | the code |
| review 1 | guards written against one representation of one argument |
| review 2 | assertions written against one fixture placement; a mutation that could not fail |
| review 3 | a helper introduced *by the fix* with untested arithmetic; the completeness check that was not derived |

Each round the code got more correct and the weakest point moved further into the evidence about
the code. That is why every reviewer after the first was told to audit the mutation battery as well
as the source, and why the count of mutations that proved nothing is tracked openly: **nine** so
far, out of fifty-five.

Two of this round's own repairs were themselves caught by the battery rather than by me — C25
showed the S6 fix had been verified interactively and never asserted, one turn after reading the
review that named that pattern; C31 showed a weakening could slip between two tests that computed
the same derivation separately instead of sharing it.
