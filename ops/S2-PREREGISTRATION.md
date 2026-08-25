# S2 — pre-registration

**Written before any GPU time and before any S2 result exists.**
Design fingerprint `8e743096ac3f2583` · frozen 2026-08-24.

The executable version of this document is `neurocheck/s2_design.py`; the machine-readable
version is `data/s2_manifest.json`. Where prose and code disagree, **the code is the design** —
this file exists so a human can read the commitments without running Python.

Phase B's lesson was *passing the example is not enough; verify the mechanism*. The
experimental form of that rule is: **a runnable S2 is not enough — freeze the design before
seeing the answer.**

---

## 1. What S2 is

Reproduce the in-silico visual functional localizer of TRIBE v2 (arXiv 2605.04326 §2.5, §5.9)
with our own pipeline: flash category images at the model as a silent video and test whether
category-selective responses appear in the parcels the paper names.

It is simultaneously the project's **positive control**. Our July Gate 0 run produced a
confident NO-GO that its own author retracted (D026): the statistic was compositional and
inverted real effects. If a published, in-silico-verified result does not reproduce here, no
downstream negative result from this pipeline means anything.

## 2. Stimulus — frozen

| parameter | value | source |
|---|---|---|
| SOA (cycle) | **8.0 s** | §5.9 verbatim: *"for one second every eight seconds"* |
| presentation | 1.0 s | same |
| ISI | 7.0 s = SOA − presentation | derived |
| lead-in / tail-out | 25 s / 25 s | Phase C brief; ≥5 s in MASTER-PLAN, 25 s is stricter |
| categories | faces, bodies, places, objects, characters | §2.5 |
| exemplars | 25 per category (125 events) | cost-bounded; see §7 |
| order | randomised, seed `20260824` | §5.9: *"presented … in a randomized order"* |
| form | ONE continuous silent video, 8 fps, 256×256 | §5.9: *"transforming them to static videos"* |
| total | 1050 s rendered | derived |

**SOA is a cycle, not an ISI.** An earlier plan line read "1 s on / 8 s ISI" — a 9 s cycle.
That was an 11% cost error and a protocol deviation from the paper being replicated.

**Every ISI is physically rendered as grey frames.** The ISI is not an absence of stimulus; it
is what the model sees between presentations. Rendering only the presentations would produce a
video ~8× too short and misplace every onset after the first.

**No whisper, no `get_events_dataframe`.** Over a silent video WhisperX transcribes nothing
while still costing ~65 s/clip, and any timing it derived would come from the audio track
rather than from the frozen schedule. The events table is built from the same schedule that
generated the pixels, and `check_three_way_consistency` proves the video, the manifest and the
analysis events agree.

## 3. ROIs — what may and may not decide anything

Methods §5.9 states the mapping, and it is **misaligned**: nine functional names against eight
parcel labels.

> FFA, EBA, PPA, VWFA, A5, 45, STS, TPJ, MTG respectively correspond to the following ROI
> labels: FFC, V4t, PH, A5, 45, STSv, PGi, TE1a.

`A5` and `45` appear on both sides and must correspond to themselves; A5 is left-position 5 but
right-position 4, so the off-by-one is already present by position 5 and **the omission lies
among the four visual regions**.

| ROI | parcel | role | may stop the study |
|---|---|---|---|
| FFA | FFC (right) | replication-of-record | **yes** |
| EBA | V4t | replication-of-record | **yes** |
| PPA | PH — *contested* | reported | no |
| VWFA | PH — *contested* | reported | no |
| PPA_literature | PHA1-3 ∪ VMV1-3 | secondary | no |
| EBA_gate0_union | V4t ∪ PH | secondary | no |
| V1_control | V1 | secondary control | no |

`PH` is claimed by PPA (if VWFA is the omission) or by VWFA (if PPA is). **Unresolvable from
the paper**, so neither may gate. §2.5 also orders the regions FFA, PPA, EBA, VWFA while §5.9
orders them FFA, EBA, PPA, VWFA — zipping against the results text instead of the methods text
swaps PPA and EBA, which is how our Gate-0 EBA proxy came to pool V4t with PH.

## 4. The lag conflict — resolved by measurement, not by choosing

| source | says |
|---|---|
| §5.9 | read the contrast at **t = 5**, *"which is the peak of the response"* |
| `source-of-truth.md:51` (VERIFIED) | output is already offset 5 s, so a read at 5 lands on **BOLD(onset+10)**, ~18% of peak |

Both cannot be true, and reading the wrong one costs ~5.5× in amplitude — a failed replication
for a reason unrelated to the model.

**One peri-event timecourse yields every lag from a single forward pass**, so both reads come
free from the same run. Primary is the paper's `t=5`; `t=0` is pre-specified; lags −2…9 are all
reported; the measured peak adjudicates and is reported in every case.

**If X, then Y:**

| outcome | reported as |
|---|---|
| recovers at t=5 | replication succeeds on the paper's own protocol; t=0 shown as secondary |
| fails at t=5, recovers at t=0, measured peak ≈ 0 | *"not replicated at the published lag; recovered at the lag implied by the model card"* — **evidence for the double-lag, NOT a plain replication** |
| fails at both | Not recovered for that parcel; stop rule applies |

`peak_lag_trs` is a **diagnostic only**. It is never used to choose the lag the test is then run
at: selecting a lag on the same data inflates type-I to a measured 0.2032.

## 5. Contrast and statistics — frozen

- **Primary:** `event_locked_contrast` — category response minus the mean of the other
  categories, per §5.9. **No GLM, no z-scoring.** §5.9 assigns the GLM to the *language*
  experiments; Figure 4's caption says otherwise and contradicts its own Methods. The Methods
  text is the more specific statement and is what is implemented.
- **Secondary, always reported, never decisive:** `glm_contrast_z`, `roi_minus_reference`,
  `spatial_z` (the retracted statistic, shown for comparison).
- **α = 0.025**, one-sided. Permutations: 10,000, seed 0.
- **Recovery requires a detection floor** (doctrine D-3): significant *and* above the minimum
  detectable effect. A significant effect below its own floor is **not** a recovery.

## 6. Decision rules — pre-registered

- **Blocking gate.** Speech → auditory cortex must be strongly positive. Both TRIBE papers put
  auditory/language near ceiling and our v3b run got p = 0.1448. If this fails, the pipeline is
  broken and nothing downstream is interpretable.
- **Recovered** iff p < α **and** effect > that parcel's **own** detection floor — computed from
  that parcel's noise and n, not a shared constant. Every parcel is scored at **both**
  pre-specified lags.
- **Not recovered → Stop** fires **only** if every stop-eligible parcel produced a usable result
  and failed **at both lags**. Currently FFA and EBA. Every other ROI is structurally incapable
  of firing it (`Parcel.stop_eligible=False`, asserted in `_validate_parcels`). A parcel that
  did not run, or returned a non-finite p / effect / floor, makes the evidence **incomplete** —
  reported as such and blocking the stop rule, because ending GPU spend requires having measured
  the thing.
- **C2 / the 100 s window.** A 100 s window packs ~11.4 exemplars of mixed categories, which may
  dilute the contrast. The ISI-baseline read is a **pre-specified secondary** because it answers
  a *different question* — response vs rest, not vs the other categories. It is secondary by
  construction, not by outcome, and may not be promoted after the results are seen.
- **Nothing secondary may become primary retrospectively.** Not the ISI baseline, not a
  secondary statistic, not the Gate-0 unions, not the literature PPA.

## 7. Compute

Derived from the **actual rendered timeline** — lead-in, every ISI and the tail-out are real
frames the model consumes. Costing only the 125 seconds of presentations would understate this
by ~8×.

| design | events | rendered | est. GPU |
|---|---|---|---|
| 5 × 25 (**chosen**) | 125 | 1050 s | **3.35 h** |
| 3 × 50 | 150 | 1250 s | 3.99 h |
| 5 × 50 | 250 | 2050 s | 6.55 h |

One model, one run, no subject dimension, no repetitions — stated rather than assumed. Basis:
§3.6's 11.5 s compute per 1 s of stimulus, measured at N=1 cold. **This anchor was measured with
the audio and text extractors running; a silent video should be cheaper, so treat 3.35 h as an
upper bound pending the first timing observation.**

## 8. Stop rule for GPU spend

Pre-committed and unchanged from the MASTER-PLAN: **if S2 does not replicate, no further GPU
spend** until it does. Debug against Meta's protocol. The paper remains writable in that branch
— *"our pipeline could not reproduce the published in-silico result; here is what we ruled
out"* — but weaker.

## 9. What this design cannot claim

- That responses appear in the parcels **Meta** used, for PPA and VWFA. Their parcel is
  contested and we say so.
- Anything from a secondary analysis.
- A replication, if recovery occurs only at the non-published lag.
- That the statistics module is proven correct — see `data/phase_b_closure.md`. The claim there
  is narrower: no *demonstrated* wrong-number defect in the decision-critical paths.

## 9b. Pinned inputs

Both resolved and verified 2026-08-24, wired into `S2Config` and recorded in
`data/s2_manifest.json`.

| input | pin |
|---|---|
| model | `facebook/tribev2` @ `f894e783020944dcd96e5568550afe2aa9743f9f` |
| checkpoint | `best.ckpt`, 708,856,138 bytes, sha256 `9c79ffff…f321` |
| stimuli | `VPNL/fLoc` @ `de6a26cc269a2c7075461a4c839bfd628f225c95` |
| images | 125 selected, each sha256'd into the manifest |

`TribeModel.from_pretrained` has **no** `revision` parameter — it downloads from the
floating branch. The pin is therefore applied by `tribe_tools.model.fetch_pinned_checkpoint`,
which resolves the exact SHA, verifies the checkpoint hash (mismatch is fatal), and hands
`from_pretrained` a local directory. fLoc states **no licence**, so the images are gitignored
and their identity travels as hashes.

## 10. Gate

`python3 scripts/s2_go_no_go.py` re-derives every checklist item from the frozen design and
from artefacts on disk, so "GO" is computed rather than asserted. It currently reports **NO-GO**
pending the independent design review.
