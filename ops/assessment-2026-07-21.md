# Strategic Assessment — 2026-07-21

Project resumed after ~6 weeks dormant (last activity ~2026-06-10). This is a full
strategic re-assessment produced with a deep-reasoning pass, plus the corrected facts,
the decision to reshape scope, and the direction from here.

---

## TL;DR — Reshape, then resurrect. Do not kill.

The scariest existential risk is dead. What remains is not "will it work" but "has anyone
run it" — and the answer is still no. The move: one decisive GPU smoke test, then collapse
four builds down to the one asset that is already ~90% done and needs no GPU to publish
(**NeuroCheck as a dataset**). Cut NeuroGenre and ScaleLaw.

---

## Two facts we had wrong (both in our favor)

1. **G016 is not open — it's effectively dead.** `neuralset`, `neuraltrain`, `exca` are
   real, public, pure-Python (`py3-none-any`) wheels on PyPI from Meta FAIR
   (`facebookresearch/neuroai`), version 0.0.2. They are NOT private internal libraries.
   The ~40% "dead on arrival on free hardware" risk is resolved false.
   - Caveat: "exists on PyPI" is not the same as "installs cleanly with all transitive
     deps on Kaggle." One real install detail surfaced: `neuralset 0.0.2` requires Python
     **>=3.12**. If Kaggle is on 3.11, install via uv/conda. Confirm at first `pip install`.
2. **The repo is not at zero commits.** `tribe-bench/` has 2 real commits
   (`f03833a` initial, `ddb57cf` 50-claim DB, dated 2026-06-09/06-10) and a private remote
   `codesbydevesh/tribe-bench`. As of 2026-07-21 the 50-claim commit is **pushed** — the DB
   is now backed up on GitHub.

## One fact that weakens the thesis

**TRIBE v2 is ~2 months old, not a year.** It is arXiv 2605.04326 = **May 2026**. Low
citation count is *expected for a 2-month-old model*, not proof of a tooling gap we can
own. First-mover advantage is real (nobody has had time yet), but "proven latent demand"
is NOT established. We may be building tools for a model people have not adopted yet, for
reasons unrelated to tooling. The "~5 papers in a year" figure was unsourced.

---

## How great it is / how it is not

**Genuine strengths**
- **The NeuroCheck claims DB is a real asset, not a skeleton.** 50 claims, 48 unique DOIs,
  correct landmark citations (NC001 FFA → Kanwisher 1997 `10.1523/JNEUROSCI.17-11-04302.1997`;
  NC002 PPA → Epstein & Kanwisher 1998 `10.1038/33402`). ~48KB of curated, verifiable domain
  content that survives independent of whether the model ever runs.
- **The ops system is unusually disciplined for a solo project** (source-of-truth vs
  knowledge-gaps vs decision-log separation, calibrated pre-mortem, interface contracts).
- **The core technical read of TRIBE is correct and load-bearing** — encoders are external
  extractors that load/cache/free sequentially, so per-extractor peak VRAM (not the ~32GB
  sum) is what a 16GB T4 must fit.

**Genuine weaknesses**
- **Nothing has ever executed on a GPU.** Every "COMPLETE" is CPU-import-only. ~1,413 LOC;
  three of four builds (NeuroGenre, ScaleLaw, NeuroCheck scoring) are 1-line `__init__.py`
  stubs. It is a well-documented skeleton with one real organ (the claims DB).
- **The first-mover thesis is partly wishful** (see the date-error note above).
- **The 37% DOI hallucination history (11/30) is a process scar.** Caught and hand-fixed;
  the shipped DB spot-checks clean; but it has never been programmatically re-verified
  end-to-end against CrossRef, and 2 of 50 DOIs are duplicated.

**Does it stand out?** Not yet — it is a plan and a dataset. Potentially yes, but narrower
than the pitch: not "the standard toolkit for in-silico neuroscience," but **"the first
DOI-verified sanity-check benchmark for brain-encoding models."** The four-build empire is
scope creep.

---

## Biggest live risk + the <=1-hour test that settles it

With G016 dead, the top live risk is **G018: does modality ablation work at inference?**
BrainLens (the MVP / demo / outreach screenshot) depends entirely on it, and the upstream
source has **no supported inference-time mask API** — the model only zeroes a modality when
a feature is absent from the batch (`tribev2/model.py:190`) or via training-only
`modality_dropout` (`:212`). Our wrapper's `_find_features_to_use()` blindly probes 4
attribute paths hoping one exists. If none do, BrainLens produces identical maps for every
"modality" and the flagship demo is fake.

**The decisive Kaggle test (dual-T4 notebook, ~1hr):**
1. Check Python version first — `neuralset 0.0.2` needs >=3.12; if Kaggle is on 3.11,
   install via uv/conda.
2. `pip install exca neuralset==0.0.2 neuraltrain==0.0.2` then `pip install -e .` on the
   tribev2 source. Expect success (public wheels). If it fails, stop — everything is wrong.
3. Get HF gated approval for LLaMA 3.2-3B first (G011 — one click, do today).
4. `model.predict()` on one 10s clip; log `torch.cuda.max_memory_allocated()` per extractor
   → closes G005 with a real number.
5. KILL/CONFIRM: `print(dir(model)); print(dir(getattr(model,'data',None)))` to find where
   `features_to_use` actually lives. Run predict twice — full vs audio-removed — and assert
   the outputs DIFFER. If identical, BrainLens is dead as designed → pivot to NeuroCheck-only
   (which needs no ablation).

Do this before writing another line of build code.

---

## Completion plan (~4 weeks part-time). Cut NeuroGenre + ScaleLaw.

- **Week 0 (now):** the smoke test above. HF/LLaMA approval. Push the DB commit (DONE
  2026-07-21). Programmatically CrossRef-verify all 50 DOIs + de-dupe the 2 collisions.
  Milestone: "it runs, and the DB is bulletproof."
- **Week 1:** Ship the **NeuroCheck resource paper** (dataset + protocol) to bioRxiv + HF
  Datasets, before any model scoring — the legitimate GLUE move. Milestone: a citable
  artifact exists.
- **Week 2:** Run the 50 claims through TRIBE on Kaggle (HDF5-checkpoint every claim — 12hr
  timeouts are real). Milestone: first real results.
- **Week 3:** BrainLens on HF ZeroGPU as a pre-baked drip demo (3.5 min/day is enough for
  one pre-baked clip). Milestone: a live link for outreach.
- **Week 4:** NeuroCheck-with-results v2 + outreach to the ~2 real builder groups + Meta
  authors.

**Where it realistically lands (honest probabilities)**
| Outcome | Probability |
|---|---|
| bioRxiv NeuroCheck preprint | ~75% (safe outcome) |
| Portfolio piece that helps get noticed/hired in ML | ~60% (the real prize) |
| HF Space demo that turns heads | ~50% (contingent on the ablation test passing) |
| A benchmark others actually adopt | ~15% (needs the model itself to have adopters) |
| A career in neuro-AI specifically | <10% (do not optimize for this) |

The real prize is the portfolio/hiring story: "solo, no GPU, first person to build tooling
+ a benchmark on Meta's newest brain model, shipped a live demo."

---

## Does NeuroCheck survive scrutiny?

The GLUE analogy is half-right. As a **resource paper** (a curated, well-designed eval
suite), it is publishable pre-results — legitimate. What is NOT legitimate is implying it is
a validated *measurement instrument* before showing it separates good models from bad
(GLUE's authors ran baselines; we will need to too, eventually).

Skeptical-reviewer objections and the fix for each:
- **n=1 per contrast, no error bars** (predictions are deterministic). Fix: bootstrap over
  stimulus *sets* (multiple exemplars per condition), report a distribution. Design stimuli
  as sets now, not single clips.
- **50 comparisons, no correction.** Fix: report FDR-corrected pass rates. Trivial to add,
  fatal to omit.
- **ROI ambiguity** (is "FFA" = which Glasser subregion? — our own G010/G007). Fix: document
  the exact HCP-MMP1 vertex set per claim with a citation for each mapping; flag contested
  mappings (e.g. modern FFA-selectivity debate, G009).
- **The 37% DOI history.** Fix: publish the CrossRef verification script + its clean output
  as a supplementary artifact. Turn the scar into a credibility signal.

Honest bottom line: as specified (single exemplar), it is **under-powered** — a qualitative
direction-check, fine for a v1 resource paper *if we say so*. Requiring stimulus sets makes
it bulletproof.

---

## Creative idea slate (ranked by impact x feasibility at $0/no-GPU)

`[DE-RISKED]` = path is visible. `[HOPE]` = depends on unverified things.

1. **MCP "neural-engagement" tool — the day-job bridge. [DE-RISKED-ish, highest strategic
   ROI]** Wrap TRIBE as an MCP tool: content in → predicted per-region cortical engagement
   (visual/auditory/language/emotion ROI scores) as structured JSON, callable by any agent.
   Fuses both halves of the operator's work (he builds MCP servers by day). Almost nobody
   else can credibly build this. Backend = ZeroGPU drip or cached precompute. License-clean
   (research/demo). THIS is the surprise: the day job is the differentiator.
2. **Agentic in-silico neuroscience experimenter. [HOPE, highest ceiling]** An agent that
   takes a natural-language hypothesis ("do faces drive FFA more than scenes?"), auto-designs
   the contrast, gets stimuli, runs TRIBE, does the stats, writes a result card. NeuroCheck
   automated — a strong blog/HN piece. Depends on the ablation test + orchestration to build.
3. **NeuroCheck as a model-agnostic leaderboard. [DE-RISKED]** Score any brain encoder, not
   just TRIBE. We own the benchmark; others bring models + compute. The actual "own the
   niche" move; de-risks single-model dependency.
4. **Encoder-agnostic toolkit pivot. [DE-RISKED]** Refactor `tribe_tools/` to wrap any
   fsaverage5-output encoder. High-floor insurance — survives even if TRIBE adoption stalls.
5. **Format-bridge fsaverage5 <-> NSD/native as standalone infra. [DE-RISKED, sleeper]** The
   tedious resampling utility everyone needs and nobody writes. Infra utilities get quietly
   adopted/cited more than flashy demos.
6. **Clean-room license escape. [HOPE, long game]** The recipe (frozen public encoders →
   small fusion head → cortex) is not encumbered; only Meta's weights are CC-BY-NC. Retrain
   a fusion head on CC0 data (Natural Scenes Dataset) → own weights, commercial-viable. Needs
   free training compute (NSF ACCESS / NAIRR, no-PI, ~1-2 day approval).
7. **Normative-deviation content scorer. [DE-RISKED-ish]** Reframe TRIBE as a normative
   model: score any stimulus by how atypically it drives cortex. No IRB, no real fMRI, no
   ground truth. Pairs as the headline metric for idea #1.
8. **PerceptLoop encode→decode consistency. [HOPE, lowest priority]** Chain TRIBE with an
   existing brain→image decoder; measure round-trip fidelity as a benchmark.
9. **Non-commercial license as the angle, not the blocker.** Brand everything "open research
   infrastructure, non-commercial by design." In a field nervous about commercialized
   neurotech, NC is a trust signal. Free positioning win.

**Highest-leverage weekend artifact:** the CrossRef-verified NeuroCheck dataset on HF
Datasets + a 2-page bioRxiv-ready resource note. No GPU, ~90% done, citable the moment it is up.

---

## Where we are moving now — the one thing this month

**Run the smoke test — this week, before anything else.** Everything is in the "hope"
column for one reason: nothing has run. The scariest unknown is already dead (deps are real
wheels), so the test is likely to succeed — and one good hour flips the project from
"elaborate plan" to "the only working tooling on Meta's newest brain model." That is when
the portfolio story becomes true. Do the DOI re-verification and keep the repo pushed as
insurance the same day.

Direction of record: **NeuroCheck-first.** Ship the DOI-verified benchmark dataset as a
bioRxiv/HF resource; keep BrainLens only if the ablation test passes; hold the MCP
neural-engagement tool (idea #1) as the standout follow-on once the model is confirmed to
run. NeuroGenre and ScaleLaw are cut.

Facts still unverified without a GPU: G018 (ablation path) and G005 (per-extractor VRAM) —
exactly what the smoke test closes.
