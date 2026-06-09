# Red Team — Adversarial Review Protocol

Before shipping any artifact — code, claims, papers, data — run the appropriate
red team checklist below. The goal is to find problems BEFORE a reviewer, user,
or GPU session finds them for us.

---

## How to Use This

For each artifact type, there are two sections:
1. **Attack vectors** — specific ways the artifact can fail
2. **Checklist** — items to verify before shipping

Run the checklist yourself. For papers and claims, also ask Claude to take the
role of an adversarial reviewer: "Act as a hostile peer reviewer. Find every
weakness in this claim/section/argument."

---

## Red Team: NeuroCheck Claims

### Attack Vectors

| Attack | What Goes Wrong | How to Check |
|--------|----------------|--------------|
| Citation fabrication | DOI doesn't resolve, paper doesn't exist | HTTP HEAD on doi.org/DOI |
| Citation misrepresentation | Paper says X, we claim it says Y | Re-read the abstract and results section |
| Unreplicated finding | Only one group found this result | Search for replication attempts |
| Wrong ROI mapping | FFA is not at the Glasser region we listed | Cross-reference coordinates from paper with Glasser atlas |
| Outdated finding | Result was overturned by later work | Check citations of the original paper for contradictions |
| Wrong direction | Paper says A > B, we coded B > A | Re-read the specific results paragraph |
| Unavailable stimuli | Dataset taken offline or requires registration | Try to access the download URL |
| Effect size inflation | Original d=0.8, meta-analysis says d=0.3 | Prefer meta-analysis numbers |
| TRIBE v2 can't test this | Claim requires temporal dynamics, connectivity, or subcortical | Check against "What Is NOT a Valid Claim" in claims-protocol.md |

### Checklist

For each claim:
- [ ] DOI resolves to the correct paper
- [ ] Read the paper's actual results (not just abstract)
- [ ] Found at least one independent replication
- [ ] Glasser region label exists in nilearn's atlas
- [ ] Contrast direction matches what paper reports
- [ ] Stimulus source dataset is publicly accessible right now
- [ ] No overlap with another claim in the database
- [ ] Effect size comes from meta-analysis (preferred) or original paper
- [ ] Claim is testable with TRIBE v2 (cortical, activation-based, stimulus-driven)

---

## Red Team: Code (tribe_tools/ and builds)

### Attack Vectors

| Attack | What Goes Wrong | How to Check |
|--------|----------------|--------------|
| Shape mismatch | Function expects (T, 20484), gets (20484,) | Check all array operations handle both shapes |
| Untested GPU path | Code works on CPU, crashes on CUDA | Flag all device-dependent code paths |
| Hard-coded paths | Works on your machine, fails on Kaggle | Grep for absolute paths |
| Memory leak | Tensors accumulate across loop iterations | Check for del/detach/no_grad in loops |
| Silent wrong results | No error, but output is garbage | Add shape assertions, range checks |
| Missing dependency | Import fails because package not in pyproject.toml | Run fresh install in clean venv |
| Cache corruption | HDF5 file partially written, unreadable on resume | Test: kill process mid-write, try to resume |
| Atlas region not found | Typo in region name, no helpful error | Check KeyError has suggestion for closest match |

### Checklist

Before merging any code:
- [ ] All functions match interface-contracts.md signatures
- [ ] No hard-coded absolute paths (use Path objects, relative or configurable)
- [ ] GPU code wrapped in availability checks
- [ ] Array shapes asserted at function boundaries
- [ ] Imports work without GPU: `python -c "from tribe_tools import X"`
- [ ] No print() for progress — use logging or tqdm
- [ ] Edge case: empty input (0-length video, empty list)
- [ ] Edge case: single timestep (T=1, shape (1, 20484) vs (20484,))
- [ ] pyproject.toml lists all dependencies

---

## Red Team: Papers (bioRxiv Submissions)

### Attack Vectors (What Reviewer 2 Will Say)

| Attack | Reviewer's Version | Our Defense |
|--------|-------------------|-------------|
| "No real fMRI comparison" | "You only show model predictions, not validation against real data" | ScaleLaw paper handles this. For NeuroCheck: the benchmark design is the contribution, not the model's score. GLUE was published before any model was evaluated on it. |
| "Sample size" | "50 claims is arbitrary. Why not 100?" | 50 covers 8 categories across visual, auditory, language, multimodal, motor, emotion, attention, cognition. We document inclusion/exclusion criteria. Community can extend. |
| "Cherry-picked claims" | "You only picked claims the model would pass" | We include claims across difficulty levels. We expect some failures. A model that passes 100% would be suspicious. |
| "No statistical correction" | "50 tests, no multiple comparison correction" | Apply Bonferroni or FDR correction. Report both corrected and uncorrected. |
| "Effect sizes are from different paradigms" | "TRIBE v2 sees video, original studies used static images" | Acknowledged as limitation. We convert images to short videos. Compare effect sizes with and without video conversion. |
| "Not novel" | "You just ran an existing model on known findings" | The contribution is the benchmark framework, not the model. Nobody has systematically validated brain encoding models against established neuroscience. |
| "Affiliation" | "Independent researcher, no institutional backing" | Irrelevant to the quality of the work. Peer review is blind. |

### Pre-Submission Checklist

- [ ] Every claim in the paper is traceable to a specific DOI
- [ ] Statistical methods section describes exact tests used
- [ ] Multiple comparison correction applied and reported
- [ ] Limitations section is honest and thorough
- [ ] Code and data availability statement included
- [ ] Figures have colorblind-accessible colormaps
- [ ] Supplementary materials include full claims database
- [ ] LaTeX compiles without warnings
- [ ] All co-authors (if any) have approved the manuscript

---

## Red Team: GPU Session Results

### After Every GPU Session, Verify:

- [ ] Output arrays have expected shape (T, 20484)
- [ ] Values are in a reasonable range (not all zeros, not NaN, not constant)
- [ ] Predictions differ across modalities (video-only != audio-only)
- [ ] Predictions differ across stimuli (face video != house video)
- [ ] Cache files are readable: `h5py.File("cache.h5", "r").keys()`
- [ ] Results are saved to persistent storage (not just RAM)
- [ ] VRAM peak is logged (for source-of-truth.md)
- [ ] Wall clock time is logged (for compute-playbook.md timing estimates)

If any check fails, DO NOT proceed to analysis. Debug first.

---

## Scheduling Red Team Reviews

| Artifact | When to Review | Who Reviews |
|----------|---------------|-------------|
| Claims batch (every 10) | After adding 10 new claims | Claude as adversarial reviewer |
| Code module | Before marking as "done" in war-room | Code checklist above |
| GPU results | Immediately after session ends | Results checklist above |
| Paper draft | Before bioRxiv submission | Full paper checklist + Claude as Reviewer 2 |
| Full database (50 claims) | Before HuggingFace release | Full claims checklist on all 50 |
