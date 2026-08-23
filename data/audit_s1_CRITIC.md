# S1 audit — completeness critic

What the eight lanes and ~200 skeptics failed to look at.
Run 2026-08-23 against 29cb104.

---

All runs completed cleanly (exit 0, full output read); every grep was scoped to named subdirectories with `--include` filters; no search was killed or truncated.

---

## What the audit missed, ranked by expected value

### 1. No lane audited the generative model. Every published number is a functional of it.

The eight lanes were: correctness, protocol, geometry, doctrine, silent-failure, tests, hostile review, blind re-implementation. All eight audit **the code**. The blind re-implementation diffs code against code. Not one lane asks whether the simulated world the code is validated in is a fair model of TRIBE.

That matters because both published artifacts — `/home/deveshb/workspace/AI/tribe-bench/data/floor_table_v3b.md` and the demo cited in `roi_stats.py:35` — are functionals of the 20 lines of `build_brain`/`clip_map` in `/home/deveshb/workspace/AI/tribe-bench/scripts/compositional_demo.py`. Two free parameters in there control the headline. I swept both (my harness reproduces the shipped table at shipped settings — spatial_z 0.1420 vs published 0.1416, raw 0.0295 vs 0.0294, glm 0.0283 vs 0.0286 — so the sweeps are calibrated):

**The stipulated auditory confound `D_AUD`:**

| d_aud | spatial_z | raw | ref | glm | ratio |
|---|---|---|---|---|---|
| 0.00 | 0.0450 | 0.0295 | 0.0223 | 0.0283 | **2.02×** |
| 0.12 | 0.0939 | 0.0295 | 0.0223 | 0.0283 | 4.21× |
| 0.24 | 0.1420 | 0.0295 | 0.0223 | 0.0283 | **6.37×** |

**Within-parcel correlation of the prediction noise (marginal variance held fixed, d_aud=0.24):**

| rho | spatial_z | raw | ref | glm | ratio |
|---|---|---|---|---|---|
| 0.0 | 0.1420 | 0.0295 | 0.0223 | 0.0283 | **6.37×** |
| 0.3 | 0.1476 | 0.0439 | 0.0392 | 0.0421 | 3.77× |
| 0.6 | 0.1592 | 0.0545 | 0.0497 | 0.0492 | 3.23× |
| 0.9 | 0.1711 | 0.0694 | 0.0675 | 0.0672 | **2.55×** |

Commit `29cb104`'s message is "spatial_z needs a 6.7x larger effect than the simplest alternative." That 6.7× is the value at one corner of a two-parameter space. About two-thirds of it is a knob the project set itself — `D_AUD=0.24`, chosen by an argmin over 25 single-draw realizations on a grid whose boundary it sits on (the audit's S7). Remove the confound and the intrinsic penalty is 2.0×. Add realistic spatial correlation and it falls to 2.5×, because spatial_z's floor barely moves (+20% across the whole rho range) while every replacement's floor rises 2.4–3.0×.

The i.i.d. assumption is not merely unverified — it is structurally impossible for TRIBE. `grids/defaults.py:198-199` sets `low_rank_head: 2048`, `hidden: 1152`, and `model.py:145-146` builds `SubjectLayers(in_channels=bottleneck, out_channels=n_outputs)`. The whole 20,484-vertex map is a linear image of a 2048-dim latent, i.e. rank ≤ 2048. `clip_map` generates rank-20,484 i.i.d. noise, the least-correlated structure possible, and that is precisely what lets averaging 58 ROI vertices and 2000 reference vertices be nearly free.

**The direction of the conclusion survives at every setting I tested — the replacements always beat spatial_z. The magnitude does not.** The honest claim is "2–8× depending on two parameters we stipulated," not "6.7×."

### 2. A major finding was filed, never adjudicated, and never reached the synthesis — and it is the one that decides whether S2 can work at all.

Lane 2 filed, at severity **major**: *"event_locked_contrast attributes the t=+5 read to one event, but TRIBE's encoder is verified maskless and bidirectional over a 100 s window."* Its concrete failure: each 100 s window mixes ~11 exemplars from all categories, so the contrast is measured on a diluted quantity, and "if S2 under-recovers FFA relative to Meta the authors have no way to distinguish leakage from a pipeline bug — which is exactly the branch point MASTER-PLAN S2 says determines whether to spend the remaining GPU budget."

I traced it: **absent from `all_verdicts` entirely** (no skeptic ever saw it), absent from the 20 survivors, and the synthesis contains zero occurrences of "occupancy" or "leakage." Worse, the synthesis uses the *same premise* in S2 rule 4 — "the encoder is maskless and bidirectional over the full 100 s window, so blocking confounds window identity with category" — to mandate randomization, which is the choice that maximizes within-window mixing. It took the fact and drew the opposite operational conclusion, without ever having seen the finding.

And Lane 2 is the lane whose `search_failures` field is the **empty string** while all nine others wrote explicit clean-run attestations. The audit's broadest reading lane is the one that cannot certify it completed.

I do not think the finding's "order of magnitude" is established — `position_embed_dead.md` shows RoPE is live and relative, so locality is learnable, and a contrast between two categories sharing the same windows may largely survive. The point is that **nobody adjudicated it**, and its proposed control costs nothing on top of a GPU run already planned: read the ISI baseline as a fourth "category" (flat if attribution holds) and record window packing beside every contrast. That converts an unfalsifiable null into a diagnosable one.

### 3. The audit hardened a function with no production caller and never audited the one that produces the numbers.

A finding titled *"Two divergent MDE estimators: the shipped floor table never calls the tested detection_floor"* was marked **invalid**. The refutation is right about the remedy — the two estimators measure different quantities, verified head-to-head — but it explicitly confirms the premise: *"Confirmed: the shipped floors come from an untested path."* Killing the remedy killed the premise with it.

I verified independently: `detection_floor` is referenced only at `roi_stats.py:544`, `tests/test_roi_stats.py:125/412-418`, and two ops docs. **It has no production caller.** Every number in `floor_table_v3b.md` comes from `interpolate_floor` at `/home/deveshb/workspace/AI/tribe-bench/scripts/detection_floor_table.py:96-105` — a two-point linear interpolation across a sigmoid, over bins of ~50% relative width, reported to four significant figures, with zero tests and no lane coverage. The synthesis then spent S2, S3 and S4 — three of ten SHOULD-FIX slots — on `detection_floor`.

### 4. Reproducibility was not a lane, and the demo's headline number is not reproducible.

`compositional_demo.py` uses a module-level mutable `RNG` consumed by *both* `build_brain()` and `run()`. Measured:

- `build_brain()` returns a different brain every call (max |Δ| = 0.170).
- `run(0.24)` twice in one process: **−0.2612**, then **−0.2135**.
- Across 40 realizations of the published setting: FFCr z_d mean **−0.205**, sd 0.0267. The value cited in `roi_stats.py:35` and in the regression test's docstring — **−0.239**, "vs obs −0.244" — is a ~1.3 sd favourable draw. The expected value sits 0.039 from the observation, not 0.005.
- A1 is sim **+0.017** vs obs **+0.280** in every realization — 16× short. (The audit's §7 item 5, confirmed, still unactioned.)
- The ordering *does* hold: 40/40 realizations reproduce EBA < FFCr < V1 < A1. So "ordering exact" is robust to noise, though still a stipulated input.

The published value is reachable only by executing `main()` top-to-bottom in its current order; any added RNG draw upstream silently changes it, and `detection_floor_table.py` imports from this module.

### 5. The module warns about double-dipping in space and has no equivalent in time — and its own recommended workflow creates it.

`define_froi` carries a prominent independence warning; the audit adds S5 to harden it. Meanwhile synthesis S2 rule 6 says *"Do not hardcode a lag. Extract the peri-event time course and report the measured peak from `peak_lag_trs`."* That selects a read-out lag from 12 candidates and then tests at it. Type-I error, true effect zero, nominal one-sided α=0.025, 50v50 events, 600 reps (SE ≈ 0.008):

- fixed lag 0: **0.0050**
- peak from the **pooled** timecourse: **0.0100**
- peak from the **target category's** timecourse: **0.0417**

`peak_lag_trs` takes whatever array you hand it and nothing in its docstring says which. One sentence fixes it: select from the pooled/grand-average time course, or re-select inside every permutation. Separately, the fixed-lag arm came out conservative (0.005 vs 0.025) because my events alternated deterministically — which means the synthesis's rule 4 is load-bearing for the permutation test's *validity*, not only for confound avoidance. Worth stating explicitly.

### 6. The audit's own M1 remedy turns the suite red.

Synthesis M1 prescribes raising in `define_froi` when `k == pv.size`. `tests/test_roi_stats.py::test_define_froi_caps_at_parcel_size` asserts exactly the opposite: `define_froi(np.zeros(10), np.zeros(10), np.arange(0,5), top_n=100).size == 5`. The shipped test *blesses* the no-op — probably why it survived review — and the synthesis reports "32 tests pass" without noting that its own top-ranked fix breaks one.

### 7. Minor, unfiled: the overlap guard is defeated by a dtype.

`roi_minus_reference(g, bool_mask_selecting_{10,11,12}, np.array([10,11]))` returns **0.5 with no error** — `np.intersect1d` compares boolean values against integers, so total overlap goes undetected. This is the module's only defence against an undeclared normaliser. Relatedly, `tribe_tools/atlas.py:151` returns `np.concatenate([...])` with no `np.unique`, so an overlapping pattern list silently double-weights vertices (`raw_roi_mean` 11.0 → 11.4 on my probe). The audit filed "No function validates `verts`" and a skeptic refuted the boolean case on "NO BOOLEAN-MASK PRODUCER EXISTS" — sound as far as it goes, but it did not consider the mixed-type overlap bypass or the concatenate-without-unique path.

---

## Direct answers

**1. What class no lane was pointed at:** *simulation-validity and inference-from-simulation.* Every lane checks whether the code computes what it claims; none checks whether the world it is validated in resembles TRIBE. Two subclasses also uncovered: reproducibility/provenance of published artifacts (§4), and selection-on-the-same-data along the time axis (§5).

**2. What every lane assumed:** that `build_brain`/`clip_map` is a fair noise model — specifically that prediction noise is spatially independent across 20,484 vertices (§1); that `detection_floor` is the thing producing the floors (§3); that `compositional_demo` is seeded and reproducible (§4); and that a per-event read is attributable to that event (§2). The pre-existing statistics above line 179 I did check — `exact_perm_p`, `mc_perm_p`, `perm_p`'s 20000 threshold against `_MAX_LABELINGS`, and the `(ge+1)/(n_perm+1)` estimator are all correct.

**3. Most expensive if false, with six days left:** the spatial-independence assumption. It is the only one I could show is *provably* load-bearing — ρ=0.3 alone nearly halves the headline ratio — it sits under a number already committed to a file and a commit message, and it needs no GPU to address. Second is per-event attribution (§2), which is more dangerous but unquantifiable before the run; its mitigation is a control, not a fix.

**4. Was anything dropped:** yes, two, and both matter. The window-attribution finding was **never adjudicated at all** (§2) — filed major, absent from `all_verdicts`, absent from the synthesis, by the one lane with a blank `search_failures` attestation. The divergent-MDE finding was marked invalid on a wrong remedy, taking a true and still-unaddressed premise with it (§3). Neither was dropped for budget: the budget statements in the JSON concern GPU hours, not audit effort.

**5. Cheapest CPU check today:** re-run `detection_floor_table.py` as a **sensitivity surface over `(D_AUD, rho)`** instead of a point estimate, and publish the floor ratio as a range. Roughly one CPU hour, no GPU, no new science, no dependency on anything not already in the repo — I ran a reduced version of exactly this in about fifteen minutes. It is the only check that can change which statistic the paper recommends, and it pre-empts the precise objection a *Failure Modes of AI in Biology* reviewer will raise: that the retraction was replaced by a number tuned on a simulation built to reproduce the thing being retracted. Bundle two five-minute fixes with it — pass an explicit `default_rng(seed)` through `build_brain`/`run` so the demo's numbers are reproducible, and report the demo's FFCr delta as **−0.205 ± 0.027 over 40 seeds** rather than the single draw **−0.239**.