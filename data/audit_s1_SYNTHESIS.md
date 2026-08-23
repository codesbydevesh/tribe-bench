# S1 audit — final synthesis

Repo untouched at 29cb104. Here is the synthesis.

---

# S1 STATISTICS MODULE — FINAL AUDIT SYNTHESIS

**Audited revision: 29cb104 (clean tree). 32 tests pass in 3.8 s. Deadline is now 6 days out (2026-08-29 AoE), not 10.**

**Critical process note first:** the brief's file map is stale. `roi_stats.py` is **613 lines, not 462**. Commits `57bd8eb`, `29e91bf`, `29cb104` all landed *after* the audit snapshot. Everything below is judged against the code as it exists now, re-verified by execution, not against the lane reports.

---

## 1. ONE-LINE VERDICT

**Sound in its core arithmetic — every statistic computes what it claims, the compositional bug is genuinely gone, and the two worst findings (HRF double-apply, arithmetic row indexing) are already fixed — but it ships five silent-failure paths and one silent no-op that will bite S2 specifically, plus a false attribution sentence headed for the paper.**

---

## 2. MUST FIX BEFORE THE PAPER

Ranked strictly by: *would this produce a wrong scientific conclusion in a paper due in six days.*

### M1. `define_froi` is a silent no-op at its default on the project's own parcel — **never filed by any lane**

`top_n: int = 100`, and `k = min(int(top_n), pv.size)`. The right-FFC parcel is **58 vertices**. Verified:

```
parcel size=58, default top_n=100 -> fROI size=58
fROI == whole parcel (no selection happened): True
```

S2's checklist item *"Define the face fROI from half the exemplars, test on the other half (§3.4)"* calls this function. With the documented parcel and the documented default, **no selection occurs**: the "fROI" is bit-identical to the Gate 0 anatomical parcel, with no warning. The paper then claims it fixed the ROI (§3.4's whole point — the FFC parcel is a poor FFA) while reporting the unfixed one. If S2 finds no face selectivity, the pre-committed stop rule fires and the project concludes "our pipeline is broken" when the ROI step never ran.

**Exact change** in `define_froi`, after `k = min(int(top_n), pv.size)`:
```python
if k == pv.size:
    raise ValueError(
        f"top_n={top_n} >= parcel size {pv.size}: this returns the whole parcel and "
        "performs no functional selection. Pass a larger search region or a smaller "
        "top_n (the right-FFC parcel is 58 vertices; the default 100 is a no-op on it)."
    )
```
And decide the search region explicitly before S2 runs: either widen the candidate region beyond FFC, or set `top_n` well below 58 (e.g. 30). Record the choice in `ops/decision-log.md`.

*Two skeptics noticed this in passing and both explicitly routed it out of scope ("SEPARATE ISSUE… belongs to the correctness lane and is materially more serious"). It was never filed, so it was never adjudicated.*

### M2. `glm_contrast_z` uses a pooled SE — not level-alpha at the unequal n S2 plans

`roi_stats.py:491-492` is Student's pooled two-sample SE. My independent run (V=58, ρ=0.6, true null, nominal one-sided α=0.025, 300 reps × 250 perms):

| design | sd_a/sd_b | POOLED (shipped) | WELCH (fix) |
|---|---|---|---|
| 50v200 | 1.00 | 0.0400 | 0.0300 |
| 50v200 | 1.50 | **0.0733** | 0.0233 |
| 50v200 | 0.67 | 0.0067 | 0.0367 |
| 50v200 | 0.33 | 0.0033 | 0.0433 |
| 15v15 | 0.33 | 0.0133 | 0.0133 |

MASTER-PLAN S2 is faces (≥50) vs the other categories; pooling them is the natural reading of §3.1's *"the other categories"*, giving 1:4 unequal n with the pooled arm heterogeneous by construction. The anticonservative branch (2.9× nominal) is a **false-positive risk on the paper's headline face-selectivity claim**, caught by nothing.

**Exact change** — delete the pooling line, replace the SE:
```python
se = np.sqrt(av.var(axis=0, ddof=1) / na + bv.var(axis=0, ddof=1) / nb)
```
**This is provably risk-free.** I verified pooled ≡ Welch at equal n *even at a 3:1 sd ratio* (diff 0.0 at 5v5, 1.1e-16 at 15v15). The only caller, `detection_floor_table.py`, is 15v15 — the published floor table is bit-identical after the fix. Both existing glm tests are equal-n or raise earlier.

Do **not** repeat the finding's "can never reject at any effect size" framing — three skeptics disproved it. The deflated direction is conservative (inflated floor, ~45% MDE inflation), not blind.

### M3. Non-finite values are absorbed silently in four places, and the chain ends in a confident NO-GO

All verified at HEAD in one run, under `warnings.simplefilter('error')` — nothing raises, nothing warns:

```
define_froi: NaN vertex 123 selected into fROI: True
             5 NaN vs 10 real, top_n=10 -> [13,14,15,16,17, 50,51,52,53,54]   # NaN displaced 5 real
raw_roi_mean with one NaN vertex: nan
u_statistic([nan,nan],[0,0]): 0.0
perm_p all-NaN vs all-NaN: 1.0
exact_perm_p with 1 NaN in 4v4:        0.028571
exact_perm_p dropping that clip (3v4): 0.028571   # identical — the observation is silently deleted, n still reports full
glm_contrast_z clean(58v)=2.53575  with 12 NaN verts=2.10176  honest 46-vert=2.65004
             dirty/honest = 0.793103 == 46/58 exactly
             entire ROI NaN -> 0.0   # a clean-looking "no effect"
```

`np.argsort` sorts NaN **last** ascending; `[::-1]` promotes it to **first**. So a dead vertex is ranked *maximally selective*. `glm_contrast_z` is the only statistic here that converts total corruption into a plausible finite number.

**Exact changes** (~8 lines total):
- `define_froi`, before ranking: `bad = pv[~np.isfinite(sel)]` → raise naming `bad.size` and the first few indices.
- `glm_contrast_z`, after slicing: `if not (np.isfinite(av).all() and np.isfinite(bv).all()): raise ValueError(...)` naming the offending vertex count. Raise rather than renormalise — a silently shrinking ROI is itself a result the operator must see.
- `raw_roi_mean` / `roi_minus_reference`: assert `np.isfinite(g[verts]).all()`.
- `peri_event_timecourse`: assert `np.all(np.isfinite(out))` after the loop.
- `exact_perm_p` / `mc_perm_p` (public entry points **only** — not `_u_fast`, which is the inner loop; a guard there measured +62%): reject non-finite input up front.

Honest caveat: **no NaN source is demonstrated in this pipeline.** Predictions are dense fp32 over all 20,484 fsaverage5 vertices; there is no medial-wall masking in `tribe_tools` and no NaN path in `tribev2-source`. This is hardening, not a live corruption. It earns MUST-FIX status because the guard is trivial, the failure is undetectable, and CLAUDE.md mandates HDF5 checkpointing across 12-hour Kaggle sessions that time out — a realistic partial-write vector.

### M4. Delete the false Fig 4 attribution from `glm_contrast_z`

`roi_stats.py:461`: *"What Meta's Fig 4 actually reports, and non-compositional."* This is false under **both** readings of the paper — the Fig 4 caption describes a GLM fit on the predicted **time-series**, and §5.9 describes the visual contrasts as the plain t=5 subtraction with **no GLM at all**. The shipped code is a two-sample z across **observations**. The project's own `ops/interface-contracts.md:445` already records this as *"⚠ OPEN … Under audit"* — the shipped docstring overclaims relative to the project's own record.

This is the paper's central methodological sentence ("we retracted a compositional statistic and adopted the model authors' own"). A reviewer at a venue titled *Failure Modes of AI in Biology* checks exactly this.

**Exact change:** replace line 461 with:
> *A per-vertex two-sample contrast z averaged over the parcel — structurally analogous to Fig 4E's parcel-averaging of z-scores, but **not** the paper's estimator. Chosen by us because it is non-compositional. An explicit, recorded deviation (see ops/interface-contracts.md, D027).*

Resolve the `⚠ OPEN` in the same commit. **Do not** record the nilearn/AR(1)/glover detail as verified — two independent fetches truncated before §5.9's body. Recording an unread protocol detail as fact is the exact error class this audit exists to prevent.

### M5. Pin S2's ROIs to the paper's §5.9 parcels for the replication-of-record

`ops/source-of-truth.md` has **no ROI/parcel table at all**. MASTER-PLAN S2 names ROIs functionally only ("PPA (places), EBA (bodies)") and defines no parcels, so an implementer reaches for D021's Gate-0 unions — where our single **EBA proxy contains both V4t (the paper's EBA) and PH (a parcel the paper assigns to a different category)**. The paper's category boundary falls *inside* one of our ROIs. S2's branch rule is a hard stop ("Not recovered → Stop. Do not run S3"), so this can burn the remaining runway chasing an ROI-definition mismatch.

**Exact change:** add the §5.9 mapping to `ops/source-of-truth.md` as a FROM-DOCS fact, recording the **9-names-to-8-labels misalignment explicitly** rather than guessing (FFA=FFC is the only secure high-level anchor; the A5→A5 / 45→45 anchors establish only that exactly one of EBA/PPA/VWFA has no stated label — do **not** assert PPA=PH). In S2, report Meta's single parcels as the replication-of-record and our Gate-0 unions as a labelled secondary. Restrict the "Not recovered → Stop" branch so it can only fire on the replication-of-record parcels.

Keep our PPA (PHA1-3∪VMV1-3) — that is the standard literature definition and Gate 0 v3b already got d=+2.529, p=0.0002 with it. The live dilution risk is EBA only.

---

## 3. SHOULD FIX

Ordered by value per minute.

| # | Item | Change |
|---|---|---|
| S1 | **`glm_contrast_z` has zero value coverage** — `return 0.0` **and** a sign flip both pass all 32 tests (I re-ran both in a sandbox: `32 passed` each). The sign flip is *not* monotone, so it does not cancel in the permutation test and would publish `spatial_z`-beats-the-replacement — the paper's central comparison, inverted, suite green. | Add: (a) direction guard — positive when A>B, negative when swapped; (b) scipy oracle at **unequal n** (7v19; at equal n the ddof mutant is a pure constant and escapes): `abs(glm_contrast_z(a,b,roi) - np.mean([ttest_ind(a[:,v],b[:,v],equal_var=True).statistic for v in roi])) < 1e-9`. Word it as pinning the *implemented* semantics, with a pointer to the open across-time question. |
| S2 | **`detection_floor`'s `alpha` and `power` are dead to the suite**; halving the injected effect doubles the MDE with 32 passing. | `assert detection_floor(n,sd,alpha=0.05,...) < detection_floor(n,sd,alpha=0.005,...)` and the same for `power=0.5 < 0.9`. These are the assertions that actually kill the mutants — the analytic band does **not** (mutant-b lands at ratio 0.930, inside it). Pin the value at n=15 or n=24 only; at n=6 the ratio reaches 1.382 and a 1.30 ceiling would flake. |
| S3 | **`detection_floor`'s absolute `tol=1e-3`** skips the bisection entirely below that scale. Verified: sd=1.0→ratio 1.118, sd=0.029→1.125, **sd=0.001→2.000** (raw doubling bracket returned, 79% high). | `tol: float | None = None` → `tol = 1e-3 * noise_sd`. At the project's real scales (0.016–0.029) the current bias is only ~1.1% and conservative, so this is prophylaxis for when S2 produces real-unit statistics. |
| S4 | **`detection_floor` misdiagnoses an `n_perm` limit as an underpowered design** when `alpha < 1/(n_perm+1)`; burns ~21 doublings then blames the design. | Validate up front: `if alpha < 1.0/(n_perm+1): raise ValueError(f"alpha={alpha} is below the MC p-value floor {1/(n_perm+1):.4g}; increase n_perm")`. |
| S5 | **`define_froi` independence is warn-only** — verified **20/20 false positives at p=0.001** (the MC floor) on pure null data when the same arrays are used for both roles; the honest split gives 0/20. | Add `define_froi_split(preds_a, preds_b, parcel, n_folds=2) -> (froi, held_out_idx)` that partitions at the API boundary. **Do not** add a `localizer_is_independent: bool` flag — `define_froi` never sees the test data, so it is an unverifiable self-attestation that supplies false assurance. Note `interface-contracts.md:447` records warn-only as a deliberate decision, so changing it needs a decision-log entry. |
| S6 | **`event_locked_contrast` accepts a 2-D target** and averages both axes: `event_locked_contrast(np.ones((2,3)), [np.array([1.,2.])])` → `-0.5`, no error. The documented workflow guarantees the hazard — `peri_event_timecourse` (n_events, n_lags) is always in scope beside `event_locked_response` (n_events,). Measured cost: true 0.0941 vs 0.0404 — right sign, 2.3× attenuated. It also silently drops empty other-categories, changing k. | `if tgt.ndim != 1: raise ValueError(...)` naming the shape; same per element of `others`; raise (not drop) on an empty category. |
| S7 | **`D_AUD = 0.24` is the last point of `np.arange(0.0, 0.25, 0.01)`**, documented in three places as *"best match"*. Confirmed at `compositional_demo.py:186`. Averaged over ≥25 draws the objective has a real interior minimum near 0.32; the reported point is ~3.4 sem worse. The `spatial_z` floor moves 0.1416 → ~0.18 (+30%). | Widen the sweep to ≥0.60, average ≥25 runs per grid point before any argmin (single-draw sd is 0.012 — the "flat plateau" the finding reported is noise), and either report a sensitivity band or relabel `D_AUD` an openly stipulated level. Correct "best matches" everywhere. |
| S8 | **SOA is 9 s in the plan; the paper says 8 s.** §2.5 verbatim: *"flashed for 1 second every 8 seconds."* MASTER-PLAN encodes it in the arithmetic (`5 × 50 × 9 s = 2250 s`). Worse, it has propagated into `ops/source-of-truth.md:51` as **VERIFIED**: *"measured 21.9% at a 9 s SOA"*, backed by `test_roi_stats.py:323` (`soa=9`). | Set SOA to 8 s. Recompute: 5×50 = 2000 s (6.4 h), 3×50 = 1200 s (3.8 h), 5×25 = 1000 s (3.2 h). Update the test and restate the source-of-truth row. Note only **two** mentions are verified (§2.5 unambiguous; the Fig-4 caption ambiguous) — a third §5.9 wording could not be confirmed, do not cite it. Drop the confound argument: 1 s in a 7–8 s recovery window is not a credible alternative explanation. |
| S9 | Remaining test gaps: `raw_roi_mean` (whole-brain mean, sum, single-vertex and median **all** pass), `define_froi` (both tests use all-zero `loc_b`, so contrast ≡ raw amplitude), `_resolve_rows` `tol` (unpinned in **both** directions — 0.0 and 90.0 both pass). | Three ~4-line tests. For `raw_roi_mean`: `g=np.arange(10.); v=np.array([0,1,9]); assert raw_roi_mean(g[None,:],v) == approx(10/3)` kills all five mutants at once. For `define_froi`: give `loc_b` structure (high-amplitude/zero-selectivity block vs low-amplitude/selective block). For `tol`: accept at 3.4 s, raise at 0.6 s on a thinned grid, and one case that fails at `tol=0.0`. |
| S10 | `"ordering exact"` (`roi_stats.py:35`, `decision-log.md:909`) restates a stipulated input. Swapping the two hand-set constants inverts the simulated ordering. | Strike the phrase; state in `build_brain`'s docstring that the relative baseline z of V1/FFCr/EBA is an **assumed input**. Do **not** adopt the proposed "measure it from the prediction cache" fix — that cache does not exist and cannot be produced before the deadline. Also drop the ⚠-unverified "Bladon & Bent use ~104 vertices" parenthetical or restore its flag; it sits at `roi_stats.py:510`, outside the scope of §9 row 7's verification sweep. |

---

## 4. RECORD AS A KNOWN LIMITATION (state it; do not fix it)

1. **`glm_contrast_z` is not on the z scale.** It is a mean of per-vertex *t* statistics; its null SD is ≈ `sqrt((ρ̄ + (1-ρ̄)/V) · df/(df-2))`, measured 0.14 at ρ̄=0 to 1.04 at ρ̄=1. On this project's own synthetic design ρ̄=0.202 and the null SD is 0.474. **Never threshold at 1.96; never compare numerically to Meta's per-vertex z-map; always permute.** (No caller does today — `detection_floor_table.py` correctly permutes.) Add the formula to the docstring and pin it in a test.
2. **The floor table's *ranking* is conditional on one nuisance channel.** The auditory confound is disjoint from FFCr/V1/EBA/REST, so the three replacements' floors are *bit-identical* at `d_aud` ∈ {0.00, 0.12, 0.24}. `spatial_z` is the **only** statistic here robust to a condition-correlated *global gain*; under that nuisance the ranking reverses (measured FP at 3% multiplicative gain, zero true effect: spatial_z 0.00, raw 0.52, ref 0.45, glm 0.53). State: *these floors rank sensitivity, not general robustness.* One caption paragraph.
3. **The floor is measured at 15v15; S2 would run ~50v200.** Whatever SE is used, a floor computed at a different n is not the floor of the test being run. Say so beside the table.
4. **`roi_minus_reference` cancels an additive gain but leaks a multiplicative one**, with residual exactly `c·(ROI_baseline − ref_baseline)` — so the mandated *low-drive* off-target reference is the choice that **maximises** the leak, biasing positive for an above-average ROI. The docstring's unqualified "cancels" reads broader than it is.
5. **Floors are in synthetic units.** The table already says this; keep it.
6. **§5.9's exact GLM specification is UNVERIFIED** (fetches truncated). Record as a gap, not a fact.
7. **The demo's `−0.124 / p=0.9985` is one seeded realization** of a construction whose z_d ranges ≈ −0.16 to +0.004 across drive realizations. And **`p=0.0005` is the estimator floor** `1/(n_perm+1)` = 1/2001, not a measurement — report `p < 5e-4 (n_perm=2000)` wherever it appears (`roi_stats.py:37`, `decision-log.md:910`, the test docstring), or raise `n_perm` to 20000.

---

## 5. THE S2 DECISION — does an event-locked read at onset+5 s land in real data?

**This is the highest-stakes output of the audit. I re-verified every link myself from primary source rather than accepting the lane report.**

### The answer, in two parts

**(a) It does NOT land in zero padding — but that is the trap, not the reassurance.**

The per-timeline dummy `CategoricalEvent` spans `timeline.start.min()` → `timeline.stop.max()` (`tribev2/main.py:186-195`, read directly), and the keep rule is `len(s.ns_events) > 0` (`demo_utils.py:371`). So **every TR inside the timeline span is returned**, regardless of whether any extractor produced features there. The read always lands inside "real output." That means: **if the ISI is not rendered as actual video frames, the ISI rows are still KEPT while their video features are exact zeros, and `event_locked_contrast` returns ≈0 for every category with no error raised.** That is the paper-sinking silent failure, and it is a property of stimulus construction, not of the statistics module.

**(b) A read at onset+5 s lands ~5 s PAST the model's own response peak — the lag would be double-applied.**

TRIBE v2's predictions are already hemodynamically aligned. Verified four ways, three of them by me directly from source:

- `tribev2-source/README.md:32` — *"They are offset by 5 seconds in the past, in order to compensate for the hemodynamic lag."*
- `tribev2/grids/defaults.py:64-68` — `neuro_extractor = {"name": "FmriExtractor", "offset": 5, "frequency": 1, ...}` (so TR = 1.0 s and 5 TRs = 5 s).
- Meta's demo pairs `preds[i]` with the stimulus frame at second `i`, unshifted.
- neuralset `FmriExtractor` semantics (*not* locally installable — uncorroborated in this environment, and not needed).

My own from-scratch simulation under the verified convention `preds[r] = BOLD(r+5)`, run through the repo's actual functions:

```
SOA=9s  true contrast=0.4250   MEASURED peak lag = 0 TRs
    lag 0: +0.0753 (100.0%)   lag 3: +0.0376 (50.0%)   lag 5: +0.0124 (16.5%)  -> 6.1x loss
SOA=8s  true contrast=0.4250   MEASURED peak lag = 0 TRs
    lag 0: +0.0767 (100.0%)   lag 3: +0.0360 (46.9%)   lag 5: +0.0134 (17.5%)  -> 5.7x loss
```

**Status: ALREADY FIXED at `57bd8eb`.** `event_locked_response` now defaults to `lag_trs=0`, `peri_event_timecourse` is the primary readout, `peak_lag_trs` measures the peak, and two regression tests pin both. `ops/source-of-truth.md:50-51` records it as VERIFIED. **No code change is required — verify the fix holds, do not re-fix.** Record this finding as *confirmed and remediated*, never as *invalid*; two skeptics already made that error by reading the repaired file.

### How S2 must build its stimulus — six concrete rules

1. **ONE continuous silent video file.** Render the ISI as real mid-grey frames, never as a gap in the event list. ≥5 s blank lead-in.
2. **Build the events DataFrame BY HAND — a single `Video` row. Do NOT call `get_events_dataframe`.** I read the chain: `get_events_dataframe(video_path=…)` → `get_audio_and_text_events(...)`, which **unconditionally** runs `ExtractAudioFromVideo()`, `ChunkEvents(Audio, max_duration=60, min_duration=30)`, `ChunkEvents(Video, max_duration=60, min_duration=30)`, `ExtractWordsFromAudio()` (whisper over silence), `AddText`, `AddSentenceToWords`, `AddContextToWords`, `RemoveMissing`. Hand-building a single Video row bypasses all of it, so `main.py:200-212` genuinely drops the audio and text extractors — which is what MASTER-PLAN's line *"audio and text extractors are dropped when no matching events exist"* assumes but does not guarantee on the `get_events_dataframe` path. Hand-building also means `ChunkEvents` never runs, which moots the 60 s-boundary and trailing-chunk concerns entirely.
3. **SOA = 8 s (1 s on / 7 s blank)**, per §2.5. Not 9 s. Recompute the cost table (5×50 = 2000 s ≈ 6.4 h; 3×50 = 1200 s ≈ 3.8 h; 5×25 = 1000 s ≈ 3.2 h).
4. **Randomize category order across the whole sequence. Never block.** The encoder is maskless and bidirectional over the full 100 s window, so blocking confounds window identity with category.
5. **Never index `preds` arithmetically.** Use `row_times_from_segments(all_segments)` → `peri_event_timecourse`. Assert `preds.shape[0] == len(all_segments)`. Pass segments for **one timeline at a time** — the helper raises on non-monotonic start times, which is correct.
6. **Do not hardcode a lag.** Extract the peri-event time course and report the **measured** peak from `peak_lag_trs`. Expected answer is 0. This costs zero GPU, settles the one remaining ambiguity by measurement, and is the honest thing to publish in a paper about measurement failure modes.

### Confidence, explicitly

- **VERY HIGH** — the 5 s offset is already applied; `lag_trs=0` is correct; lag 5 costs ~5–6×. Three primary-source confirmations plus my own independent simulation, all agreeing.
- **HIGH** — every TR inside the timeline span is kept (read `main.py:186-195` and `demo_utils.py:370-378` directly).
- **HIGH** — `get_events_dataframe` runs the audio/whisper/chunking chain unconditionally (read the source; the transform list is not conditional on the file having audio content).
- **UNVERIFIED, cheap to close (~5 min, no GPU):** whether `ExtractAudioFromVideo()` errors, or emits an empty Audio event, for a video with **no audio stream at all** versus one with a silent stream. Rule 2 makes this moot — but check it if anyone insists on the convenience path.
- **UNVERIFIED:** arXiv 2605.04326 §5.9's exact GLM description (two fetches truncated). Read the PDF by hand (~10 min) or leave it recorded as a gap.
- **UNVERIFIED:** `ChunkEvents` behaviour in the pinned `neuralset==0.0.2`. **Moot under rule 2** — hand-built events never reach it.
- **The countervailing evidence is handled correctly:** Fig 4A says activity *"peaks 5 seconds after stimulus onset"*, consistent with plotting the same array on the BOLD clock. Rule 6 (measure the peak) answers this empirically at zero cost rather than adjudicating it.

---

## 6. IS S1 ACTUALLY COMPLETE?

The checklist, verbatim from `.notes/plans/corticall/MASTER-PLAN.md`:

> ### S1 — The measure · owner: me · CPU · **STATUS: not started**
> - [ ] `tribe_tools/roi_stats.py`: add `event_locked_contrast` (§3.1…), `roi_minus_reference`, `glm_contrast_z` (**effect/SE across time per vertex** — non-compositional, and what Meta's Fig 4 actually reports), and `detection_floor` …
> - [ ] Add `define_froi` — top-N most face-selective vertices inside a parcel from an **independent** localizer run (§3.4). This fixes the ROI, not just the statistic.
> - [ ] Demote `spatial_z` to legacy with a docstring citing G020, Murphy et al. 2009 and arXiv 2512.18792. Do not delete it — it is the comparison.
> - [ ] Promote the §3.3 simulation to `scripts/compositional_demo.py` + a test that **asserts** the artifact. The bug becomes a regression test.
> - [ ] Record the detection floor for the v3b design under each statistic.
>
> **done-when:** `pytest tests/ -v` green, floor table recorded.

**The stated done-when is met.** 32 tests pass; `data/floor_table_v3b.md` exists (written today, 16:03) with all four floors. Items 3 and 4 are fully and correctly done — the `spatial_z` demotion cites all three sources and the body is byte-identical (CLAUDE.md rule 6 is **not** violated), and two regression tests assert the artifact. `ops/interface-contracts.md:381+` has a complete `roi_stats.py` section, so **CLAUDE.md rule 2 is satisfied** — the two doctrine findings claiming otherwise are refuted at HEAD.

**What remains — three substantive deviations plus bookkeeping:**

1. **Item 1 was not built to spec.** The checklist says `glm_contrast_z` = *"effect/SE across **time** per vertex"*; the shipped function is across **observations**. `interface-contracts.md:445` already flags this ⚠ OPEN. **Decide it now:** either implement the temporal form, or amend D027 and MASTER-PLAN S1 to record what was built and why, and strike the Fig 4 attribution (M4). Do not ship the contradiction.
2. **Item 2 is built but degenerate at its default** — see M1. `define_froi` as parameterised does not "fix the ROI" on the project's own parcel.
3. **Item 5 is recorded but with two caveats that belong beside it** — the boundary-picked `D_AUD` (S7) and the 15v15-vs-50v200 mismatch (limitation 3).
4. **Bookkeeping:** all five boxes are unchecked and `STATUS:` still reads *"not started"* for work that is substantially complete. Update it, or the plan re-arms deliberation it exists to prevent.

**Honest summary: S1 is complete apart from those three items.** None requires new science; M1 and M4 are ~15 minutes each.

---

## 7. DISMISSED TOO READILY

**1. `define_froi`'s default `top_n=100` is a no-op on the 58-vertex parcel — never filed at all.** Two skeptics hit it while checking *other* findings and both explicitly routed it out of scope ("SEPARATE ISSUE… not part of this finding", "belongs to the correctness lane and is materially more serious than the docstring wording"). Because neither filed it, it never entered the refutation pipeline and no skeptic ever adjudicated it. I verified it in one line. **This is now my M1** — it is the only defect in the set that is *certain* to fire on S2's documented call with documented defaults.

**2. "The floor is measured at 15v15 while S2 runs 50v200."** Raised by one skeptic as *"A separate, larger issue this exposes but the finding does not name"* and never filed. It is real and it affects a published number: whatever SE is used, a floor computed at a different n is not that test's floor. Now limitation 3.

**3. The `glm_contrast_z` sign-flip mutation was nearly lost inside a refuted finding.** The finding led with a *scale* error (2.449×) and built its whole impact case on it — but scale is provably invariant under the permutation test, so all three skeptics correctly demolished it. The genuinely dangerous mutation (sign flip, which is **not** monotone and drives measured power to 0.00) was found only *during* refutation and appears in the corrections, not the claim. Had the finding been dropped on its refuted headline, the real risk would have gone with it. I confirmed both mutations survive (`32 passed` each). Now S1.

**4. "The audio and text extractors are dropped for a silent video" — marked DID NOT SURVIVE; I partially disagree.** The skeptic concluded *"the claim breaks in the middle."* Reading the source myself, the chain holds for the `get_events_dataframe` path: that function calls `get_audio_and_text_events`, whose transform list runs `ExtractAudioFromVideo()`, both `ChunkEvents`, and `ExtractWordsFromAudio()` **unconditionally** — none of it gated on the video actually containing audio. The refutation may be right about the *last* link (whether an empty Audio event survives `RemoveMissing`), which I could not verify. Either way the operational conclusion is unchanged and cheap: **hand-build the events DataFrame** (S2 rule 2), which makes the whole question moot. I flag it because the dismissal, read quickly, could license the convenience path.

**5. "The fit never fits A1 at any `d_aud`."** Surfaced in a skeptic's correction as *"worth flagging separately, outside this finding's scope"* — A1 contributes 0.065 of the 0.080 minimum error (sim +0.02 vs obs +0.280, an order of magnitude short). A later skeptic showed one extra parameter (core-vs-belt drive ratio k=1.25) fits all four ROIs at sum-sq 0.0117 with A1 +0.288 vs +0.280. That is a *stronger* result than what is currently written up and it closes the obvious reviewer objection. Nobody filed it as an action. It belongs in S10's rewrite — and note the finding's proposed wording (*"state that the simulation does not reproduce A1"*) would put an untrue self-indictment in the paper.

**6. Process, not a defect: the audit judged a moving baseline.** Commit `57bd8eb` landed *during* the refutation stage and was itself the commit that introduced the audited block. Findings 6 and 9 were marked "did not survive" by lanes reading the repaired file, with reasoning like *"FICTIONAL SIGNATURE — there is no `onsets_tr` parameter."* Both were real and both are now fixed. Two further commits (`29e91bf`, `29cb104`) landed after that. **Any resumed audit must pin the reviewed SHA** — a git worktree at the frozen revision, with the SHA recorded in each lane result. Fixing mid-audit does not merely waste the audit; it converts real bugs into recorded non-bugs.

---

**Net assessment:** the replacement statistics are mathematically sound and genuinely non-compositional — I found no repeat of G020 and the regression tests for it are real. The module is sound apart from the five items in §2. Three of those five (M1, M4, M5) are under 20 minutes each; M2 is one line and provably neutral to everything already computed; M3 is eight lines. All of it is comfortably deliverable inside six days.