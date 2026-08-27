# S2 Guard Coverage Matrix

**Audited at** `436a4a9` ("S2 integration lockdown"), 2026-08-27.
**Subject: the WIRING, not the modules.** Every module in `tribe_tools/` was built and
mutation-tested in isolation and passes. The independent review returned NO-GO with one
sentence:

> "Every module was built correctly, mutation-tested in isolation, and then either
> never called, or called with the argument that switches its guard off."

Seven blockers (B1..B7) were then wired into `scripts/s2_run.py`. This document audits
that wiring by **reading and executing code**. `S2-STATUS.md` and
`ops/S2-PRE-GPU-REPORT.md` were read only to learn what the blockers were called and
what the report claims; **no verdict below rests on either document**. Where a verdict
contradicts them, the code is cited and a probe is quoted.

## Grading rules

A row is **RED** if any of these hold:

1. the guard has **no real caller** on the execution path;
2. the caller can **disable** it — an optional argument whose omission skips the check;
3. the guard is called but **cannot execute** on the real path (wrong argument, wrong
   type, unreachable object);
4. the only test is a **presence** test (`hasattr`, "the name exists", a source-string
   match), or there is **no test in `tests/`** at all.

Otherwise **GREEN**. A GREEN row means *this guard is wired and provable*; it does not
mean the stage works, because several GREEN guards sit downstream of a RED and can never
be reached on the real path. Those are marked `(unreachable)`.

## Evidence classes

* `DEMONSTRATED` — a probe was executed and its output is quoted below. Probe sources
  are in the session scratchpad at `gm/p*.py`; each is reproduced inline in §4 so the
  finding survives the scratchpad.
* `SOURCE-TRACED` — traced through version-correct third-party source
  (neuralset 0.0.2, exca 0.5.20, tribev2) with file:line.
* `UNPROVEN` — cannot be settled on this box (`tribev2`, `neuralset`, `torch` absent).
  Listed in §5 rather than graded.

---

## 1. Totals

| | count |
|---|---|
| **GREEN** | 20 |
| **RED** | 25 |
| rows audited | 45 |

Of the 20 GREEN, **eleven** (A2, A3, B1, B3, C1, E2, F1, I3, L1, M1, M2) sit downstream
of a run-stopper and cannot execute on the real path today — the Stage-1 ones because
Stage 1 aborts at `:553` before it loads the model, the Stage-2 ones because Stage 2
aborts on the first line of `stage2_infer`. A twelfth (D1) executes and is then defeated
by H2. Counting only guards that are wired, reachable **and** not defeated:
**9 GREEN, 25 RED**.

The 25 REDs sort into five kinds:

* **five unconditional run-stoppers** — the stage aborts on a *perfect* input:
  **I2** (Stage 1 dies with `WeightFileMissing` before it encodes anything),
  **E1** (Stage 2 dies with `TypeError` on the first line of `stage2_infer`),
  **A1** (if E1 were fixed, every payload read raises and a good artifact is condemned
  as poisoned), **H2** (artifact resolution rejects every candidate), **F2**
  (persistence raises `FileNotFoundError` immediately after the GPU work);
* **five guards disabled by an omitted or wrong-shaped argument** — N1
  (`assert_atlas_ready` without `parcels=`), D3 (`require_artifact_location` without
  `sidecar_probe=`), E3 (a zero-argument `sidecar_probe` where a one-argument one is
  called), E4 (a probe that ignores the directory it is given), O5 (`arm()` without
  `sitehook=` in the only stage that runs V-JEPA);
* **five rows for four functions wired nowhere at all** — `cuda_guard.preflight`
  (O3, and L2/L3 for the two distinct invariants it alone would enforce),
  `stage_stimulus` (G3), `ledger.resume_state` (P1); plus `probe_video`, which
  `scripts/s2_run.py:54` imports and never calls (folded into G3's narrative,
  not separately numbered);
* **three false claims of coverage** — B2 (`--infer --stub`'s docstring names four
  things it exercises and exercises none of them), I4 (the integration test written to
  catch B1 goes green over I2), K1 (the "absent"-version guard has no test in `tests/`);
* **seven residues** — D2, C2, G2, G4, K2, O4, O6.

Three REDs produce a remedy string that **prescribes another 4h45m encode over an
artifact that is fine**: A1 (`ArtifactCorrupt` → "You MUST `rm -rf` … and re-run
`--extract-features`"), H2 and D2 (`ArtifactNotFound` → "run stage 1 explicitly"). That
is the exact failure shape B1, B2, B3 and B7 exist to eliminate.

---

## 2. The matrix

Line numbers are `scripts/s2_run.py` unless another file is named.

### A — digest verification

| # | Invariant | Guard (file:line) | Real caller (file:line) | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| A1 | every payload byte is re-hashed against a write-time sha256 before Stage 2 consumes it | `feature_artifact.py:173-191` | `s2_pipeline.py:186` ← `s2_run.py:784` | 2 | `test_s2_pipeline.py` (corrupt payload); `test_s2_integration.py:1109` | **RED** — cannot execute. `read_item` is `extractor.infra.cache_dict[uid]` (`:781-782`) on a model loaded with `cache_folder=None` (`:770`). `demo_utils.py:206-207` writes that `None` into `data.<mod>_feature.infra.folder` unconditionally, `base.py:338-339` then returns `uid_folder() is None`, and `map.py:200-204` constructs `CacheDict(folder=None, keep_in_ram=False)`, which raises (`cachedict/core.py:126-127`). Every one of the 2100 items lands in `unreadable` → `ArtifactCorrupt`, whose remedy is `rm -rf` + re-encode. DEMONSTRATED (P6), SOURCE-TRACED |
| A2 | Stage 1 materialises and re-reads its own output before writing `COMPLETE` | `s2_pipeline.py:157` + `:144` | `s2_run.py:568-575` (`list(extractor._get_data(...))`, then `_read`) | 1 | `test_s2_integration.py:821` | **GREEN** — Stage 1 passes a real `cache_folder` (`:542,:562`), so `infra.folder` is set and the read-back works. This is the hole at `neuralset/extractors/base.py:201` closed |
| A3 | an artifact under construction cannot carry the previous session's `COMPLETE` | `feature_artifact.py:73-80` | `s2_pipeline.py:137` (first act of `stage1_extract`) | 1 | `test_feature_artifact.py`; `test_s2_pipeline.py` | **GREEN** |

### B — modality contract

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| B1 | video must be present and never exactly zero across a timestep; audio/text must stay absent | `s2_pipeline.py:87-123` | `s2_pipeline.py:205`, with `REQUIRED_MODALITIES` / `EXPECTED_ABSENT_MODALITIES` (`:606-607`) passed at `:797-798` | 2 | `test_s2_pipeline.py:304-340` (through the real `stage2_infer`, not the helper); `test_s2_integration.py:916,932` | **GREEN** (unreachable — behind E1/A1) |
| B2 | `--infer --stub` exercises the modality contract, the persistence step and the report writer (docstring, `:650-655`) | — | — | 2 (CPU) | — | **RED** — DEMONSTRATED. Instrumenting `stage2_infer`, `assert_modality_contract`, `verify_artifact`, `require_artifact_location`, `EncodeCounter` and `_persist_predictions`, then running `infer(S2, stub=True)`: `rc=0`, `called: NONE OF THEM`, `_persist_predictions calls: 0`. `_infer_stub` (`:650-674`) calls `analyse()` directly and never enters `stage2_infer`. The one CPU wiring check the script advertises reports success while every RED here stands |
| B3 | the contract is evaluated on a real batch, not on config or object presence | `s2_run.py:715-736` (`_probe_modalities`), a **mandatory** `Stage2Deps` field (`s2_pipeline.py:75`) | `s2_pipeline.py:205` | 2 | `test_s2_integration.py:942` (a batch with no `.data` refuses) | **GREEN** (unreachable). `tribev2/main.py:485-506` confirms `batch.data` is a real mapping and `main.py:160-274` confirms `get_loaders(..., split_to_build="all")` returns a `dict`. See §5 for the residual `KeyError` risk |

### C — frozen parcels

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| C1 | `analyse()` scores only vertices verified before the GPU; no live mne, no implicit lookup | `s2_run.py:224-228` (refuses a parcel absent from `parcels`) | `:795` (real) and `:673` (stub), fed by `load_frozen_parcels` at `:758` / `:659` | 2 | `test_s2_integration.py:986,1001,1024` | **GREEN**. B5 closed: `tribe_tools.atlas.get_vertices` no longer appears anywhere in `scripts/s2_run.py` |
| C2 | a refusal inside `analyse()` is recorded in the ledger | `s2_pipeline.py:235-237` | — | 2 | none | **RED** — `die()` raises `SystemExit`, which derives from `BaseException`; `except Exception` at `s2_pipeline.py:235` does not catch it. A parcel refusal, and every `die()` inside `_probe_modalities` (`:722`), escapes without an `ABORTED` record. Telemetry only, but the ledger exists precisely so a dead session can be diagnosed |

### D — artifact resolution

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| D1 | Stage 2 searches every durable location instead of reconstructing a local path | `durable_store.py:330-361` | `s2_run.py:776-778` | 2 | `test_s2_integration.py:872,892` | **GREEN** — B2 wired. `_search_paths()` (`:806-813`) covers `ARTIFACT_ROOT`, `S2_DURABLE_ROOT`, `S2_ARTIFACT_SEARCH` |
| D2 | the resolved location is what Stage 2 actually reads from | — | — | 2 | `test_s2_integration.py:1156` | **RED** — nothing carries `artifact_dir` back into `data.<mod>_feature.infra.folder`. `artifact_dir` steers only the manifest lookup in `verify_artifact`; both payload seams (`read_item` at `:781-782`, `sidecar_probe` at `:795`) hang off an extractor pointed nowhere. A Stage 1 artifact found on a `/kaggle/input` mount is located, printed, and then not read |
| D3 | the resolution search also checks exca's provenance sidecars | `durable_store.py:181` | `s2_run.py:776-778` — **`sidecar_probe=` omitted** | 2 | none | **RED** — rule 2. With `sidecar_probe=None`, `verify_location` passes `sidecars=None` and `feature_artifact.py:158` skips the entire block. D2 demonstrated that deleting `uid.yaml`/`config.yaml`/`full-uid.yaml` leaves the cache serving, and that a read-only stage re-creates them from its own config. During resolution that check is off |

### E — sidecar probe

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| E1 | Stage 2 compares exca's sidecar digests against the ones recorded at write time | `feature_artifact.py:157-171` | `s2_pipeline.py:187` ← `s2_run.py:795` | 2 | `test_s2_integration.py:1079,1089` | **RED** — DEMONSTRATED. The probe is `sidecar_digests(extractor.infra.uid_folder())`; with `cache_folder=None` that argument is `None` and `feature_artifact.py:68` does `Path(None)`: `TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'NoneType'`. It is evaluated as an *argument* to `verify_artifact`, so it fires on the first executable line of `stage2_infer`, before any verification at all |
| E2 | Stage 1 records those digests in the artifact manifest | `feature_artifact.py:60-70`, `:97` | `s2_run.py:583-585` → `s2_pipeline.py:158` | 1 | `test_s2_integration.py:1071` | **GREEN** |
| E3 | `publish` checks the sidecars of what it stores | `durable_store.py:181` (`sidecar_probe(root)` — **one** positional argument) | `s2_run.py:596` — `lambda: sidecar_digests(...)`, **zero** arguments | 1 | `test_s2_integration.py:1188` | **RED** — DEMONSTRATED: `TypeError: <lambda>() takes 0 positional arguments but 1 was given`. Fires whenever `S2_DURABLE_ROOT` is set, i.e. on Kaggle, *after* the local artifact is finalized: the encode survives the process and nothing survives the session. A one-argument probe publishes cleanly (control run) |
| E4 | the sidecars checked are the **copy's**, not the source's | `durable_store.py:390-391` states the requirement verbatim | `s2_run.py:596` closes over the source `extractor.infra.uid_folder()` and ignores `root` | 1 | none | **RED** — the same defect B3 fixed for `reader_factory` was left in place for `sidecar_probe` one line below it. Latent behind E3 |

### F — prediction persistence

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| F1 | the only copy of ~86 MB of predictions reaches disk before `analyse()` is entered | `s2_pipeline.py:227`; `persist` is a mandatory `Stage2Deps` field (`:77`) | `s2_run.py:793-794` | 2 | `test_s2_integration.py:956,974` | **GREEN** (unreachable) — B6 wired, and ordered before `analyse` |
| F2 | that write is atomic | `s2_run.py:704-712` | as F1 | 2 | `test_s2_integration.py:1210` | **RED** — DEMONSTRATED. `np.savez` appends `.npz` to any name that does not already end in it, so `np.savez(dest.with_suffix(".npz.tmp"), ...)` writes `preds.npz.tmp.npz` and the next line renames `preds.npz.tmp`: `FileNotFoundError: [Errno 2] ... 'preds.npz.tmp' -> 'preds.npz'`. Files actually on disk after the call: `['preds.npz.tmp.npz']`. This fires immediately after the GPU work and before `analyse`, so the step that exists to protect the predictions is the step that loses them |

### G — stimulus root

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| G1 | the stimulus path is absolute, so exca item keys do not depend on the working directory | `s2_run.py:66` (`.resolve()` at import) | module import | 1+2 | `test_s2_integration.py` runs both stages in fresh subprocesses | **GREEN** for the default and the `S2_STIMULUS_ROOT` route |
| G2 | …including when the documented CLI knob is used | — | `s2_run.py:838-839` | 1+2 | none | **RED** — `STIMULUS_ROOT = Path(args.stimulus_root)`, no `.resolve()`. `--stimulus-root data` yields relative item keys. SOURCE-TRACED: `neuralset/extractors/video.py:246-247` keys on `event.study_relative_path()`, and `neuralset/events/etypes.py:372-385` returns `Path(self.filepath)` verbatim when `extra["study"]` is unset — which it is, because `demo_utils.py:218` sets `data.study.path = "."`. Stage 1 from the repo root and Stage 2 from a notebook then disagree, and the resulting error prescribes a re-encode to fix a `cd`. The Kaggle invocation in the docstring passes an absolute path, so this is armed only for the local/relative case |
| G3 | the stimulus is staged content-addressed so the key is mount-independent | `durable_store.py:693-754` (`stage_stimulus`) | **none** | — | `test_durable_store.py` (module-level only) | **RED** — rule 1. Zero call sites outside its own docstring (`durable_store.py:28`) and a `hasattr` in `scripts/s2_gate.py:87`. B2's structural fix was not adopted; G1's `.resolve()` is the whole mitigation |
| G4 | the other cross-stage paths are also cwd-independent | — | `s2_run.py:67-68,331-333` | all | none | **RED** — `MANIFEST`, `PROBE`, `ARTIFACT_ROOT`, `ATLAS_CACHE` and `LEDGER_PATH` all default to bare `data/...`. A `cd` between `--preflight` and `--extract-features` loses the frozen atlas (Stage 1 then aborts at `:530`) and starts a second ledger. Same class as B7, on four more paths |

### H — reader factory

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| H1 | `reader_factory` reads the directory it is handed, not a pinned source | `s2_run.py:677-701` (`_reader_for(artifact_dir)` roots on its argument) | `:595`, `:778` | 1+2 | `test_s2_integration.py:872` | **GREEN** — B3's stated defect is fixed |
| H2 | the reader lands on the exca cache it is supposed to read | `s2_run.py:689-694` | as H1 | 1+2 | `test_s2_integration.py:1138` | **RED** — DEMONSTRATED with real exca 0.5.20. `_reader_for` descends **one** directory below `<artifact>/cache`; exca nests **two** (`base.py:143`, `_uid_string = "{method},{version}/{uid}"`, consumed at `base.py:304,340`). So `CacheDict` is pointed at the `<method>,<version>` directory, finds no `*-info.jsonl`, and every lookup raises `KeyError`. Probe P1 wrote an item through the real library at the real layout and then read it back through `_reader_for`: `KeyError: 'item0'`. Consequence: `resolve_artifact` rejects every candidate → `ArtifactNotFound` → "run stage 1 explicitly". A perfect artifact provokes a second 4h45m encode |

### I — weight measurement

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| I1 | the 4.14 GB is hashed, not trusted, exactly once | `provenance.py:552-645` | `s2_run.py:553-555` — `force_hash=True`, a real `WeightIdentity`, `filename="model.safetensors"` | 1 | `test_s2_integration.py:836` | **GREEN** on the argument *shapes*. B1's reported defect (`None` as path, the 57-field dict as `expected`) is gone |
| I2 | …and the path it is given is one `verify_local_weights` can resolve | `provenance.py:524-549` (`_locate`) | `s2_run.py:553` passes `default_hf_cache_dir()` | 1 | **none** | **RED** — DEMONSTRATED. `_locate` accepts the file itself, its snapshot directory, or the **repo** cache directory (`models--facebook--vjepa2-vitg-fpc64-256`). `default_hf_cache_dir()` (`provenance.py:300-313`) returns the **hub root**, one level above that, which has no `model.safetensors` and no `snapshots/`. Probe P2 built a byte-correct hub cache layout — refs, blobs, symlinked snapshot — and got: `WeightFileMissing: model.safetensors not found at .../hub. ... Nothing was verified.` The control call with `_repo_cache_dir(default_hf_cache_dir(), VJEPA2_REPO)` returned `route=full-hash`. **Stage 1 aborts on every run, before it encodes anything.** B1 was closed by exchanging one wrong argument for another |
| I3 | a run that got the free route instead of the measured one is refused | `s2_run.py:556-560` (`route != "full-hash"` → `die`) | as I1 | 1 | `test_s2_integration.py:836` (asserts `force_hash is True`) | **GREEN** (unreachable — behind I2) |
| I4 | the test that covers I1 would fail if the path were wrong | `tests/test_s2_integration.py:341-373` | — | 1 | — | **RED** — the stub's entire path check is `if path_or_cache is None: raise WeightFileMissing`. It does not reimplement `_locate`, so the hub root passes and it returns `route="full-hash"`. `test_stage1_measured_the_weights_and_pinned_the_commit` is green over I2. This is the project's recurring failure mode — a test that goes green over the bug it was written to catch — reappearing inside the harness built to prevent it |

### J — expected_commit pinning

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| J1 | the V-JEPA revision is pinned, never floating `main` | `provenance.py:508-512` (`WeightMismatch` on a differing commit) | `s2_run.py:388-397` (`_weight_identity`, `expected_commit=VJEPA2_COMMIT`), called at `:374`, `:406`, `:538` | preflight + 1 + 2 | `test_s2_integration.py:846-847` (the kwarg is observed through the real call graph); `test_provenance.py` for the raise | **GREEN** — the optional guard is now supplied at every call site. There is no remaining call to `resolve_weight_identity` that omits it |

### K — version "absent" rejection

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| K1 | an environment whose library versions cannot be read must not collapse into a shared identity | `s2_run.py:420-432` (`_versions_or_die`) | `s2_run.py:416` inside `_identity`, reached from `_resolve_identity` (`:487`) in both GPU stages | 1+2 | **none in `tests/`** | **RED** — rule 4. The wiring is correct and the guard is real, but nothing in the suite exercises it: `scripts/s2_kaggle_preflight.py:254-287` tests it and never runs under pytest, and `tests/test_s2_integration.py:379-385` replaces `library_versions` with a stub returning `"1.2.3"` for every distribution, so the integration path cannot reach the branch either. Untested wiring on the one guard that decides whether two different machines share a feature uid |
| K2 | nothing expensive or networked happens after preflight passes | `s2_run.py:348-386` (`preflight`) | `main()` `:846` | preflight | none | **RED** — `preflight()` calls `_check_inputs`, `preflight_atlas` and `_weight_identity`, but never `_identity`. `_identity` (`:404-418`) calls `_processor_config` (`:468-476`), which is an unconditional `hf_hub_download`, and Python evaluates it *before* `_versions_or_die` in the same argument list. So both GPU stages perform a network fetch after the "expensive work is now permitted" banner, and on an offline box they die there rather than at K1. Remediation item 12 ("move `_processor_config`'s download into preflight") is not done |

### L — read-only mode

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| L1 | Stage 2's extractors are configured `mode="read-only"` + `forbid_single_item_computation` | `s2_run.py:406-411` (`model_config_update("consume")`) | `:498` via `_load(cfg, "consume", ...)` at `:770` | 2 | `test_s2_integration.py:1236` (real exca refuses an uncached chunk without encoding it) | **GREEN** (unreachable). SOURCE-TRACED: `exca/map.py:311-312` raises on `mode="read-only"` with any missing item |
| L2 | the config actually **reached** the live extractor | `cuda_guard.py:227-248` (`preflight(model, require_read_only=True)`) | **none** | 2 | `test_cuda_guard.py` (module-level only) | **RED** — rule 1. Zero call sites. Passing a config value is not the same as proving the constructed object carries it, and this is the only thing in the tree that checks the live object graph. Remediation item 5 not done |
| L3 | no exca infra on the live graph can dispatch to a process pool (`infra.cluster is None`) | `cuda_guard.py:236-237`, inside the same uncalled `preflight` | **none** | 1+2 | none | **RED** — round 2 established that `from_pretrained` **pops** `data.study.infra_timelines` (`demo_utils.py:191-196`), resurrecting the class default `MapInfra(cluster="processpool")`, and that root `infra.cluster="slurm"` survives. `model_config_update` sets neither, and nothing asserts either. Remediation item 14 half-done: `arm(sitehook=...)` yes, the `cluster: None` assertion no |

### M — encode counter

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| M1 | an **inactive** counter is refused — zero is not evidence when the instrument is unplugged | `s2_pipeline.py:213-217` | `s2_pipeline.py:208` ← `s2_run.py:784` | 2 | `test_s2_pipeline.py:345`; `test_s2_integration.py:1047` | **GREEN** (unreachable) |
| M2 | a non-zero count in Stage 2 is fatal | `s2_pipeline.py:218-223` | as M1 | 2 | `test_s2_pipeline.py` (`FakeCounter(3)`); `test_s2_integration.py:850` | **GREEN** (unreachable). `EncodeCounter` patches `exca.map.MapInfra._call_and_store`, the single funnel for both the in-process and pool branches (`exca/map.py:468,490`) |

### N — atlas readiness

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| N1 | Stage 1 refuses to start unless the atlas was frozen **for this design** | `atlas_preflight.py:656-678` | `s2_run.py:530` — `assert_atlas_ready(ATLAS_CACHE)`, **`parcels=` omitted** | 1 | none for the Stage-1 call | **RED** — rule 2. Stage 1 accepts a cache frozen for a *different* parcel set, so 4h45m of encoding can proceed against an atlas that Stage 2 will then reject at `:757`. The `parcels=` argument exists, is supplied two stages later, and is dropped here |
| N2 | Stage 2 and the stub refuse a cache frozen for a different parcel set | `atlas_preflight.py:664-673` | `:757` and `:658`, both with `parcels=list(ALL_PARCELS)` | 2 | `test_s2_integration.py:1260` | **GREEN** |
| N3 | the atlas is resolved and frozen before any GPU work is reachable | `atlas_preflight.py:452-551` | `s2_run.py:364-366` | preflight | `test_atlas_preflight.py`; `test_s2_integration.py:986,1001` (no `mne` in `sys.modules` after Stage 2) | **GREEN** — B5's post-GPU mne dependency is gone. `preflight_atlas` re-resolves and re-checks against `PARCEL_VERTEX_SHA256` rather than trusting an existing file |

### O — CUDA firewall layers

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| O1 | **L1** PID sentinel published in the environment, so a re-imported module in a child cannot re-arm as owner | `cuda_guard.py:33-50,66-85` | `s2_run.py:527` (Stage 1), `:754-755` (Stage 2) | 1+2 | `test_cuda_guard.py`; `test_s2_integration.py` (`arm()` runs for real in every scenario) | **GREEN** |
| O2 | **L2** at-fork CUDA poison | `cuda_guard.py:49,105-115` | inside `arm()`, both stages | 1+2 | `test_cuda_guard.py` | **GREEN** for `fork` and `ProcessPoolExecutor`; by construction 0 coverage of spawn/forkserver/subprocess, which is what L4 exists for |
| O3 | **L3** live-object-graph preflight (`num_workers == 0`, no pooled infra, read-only extractors) | `cuda_guard.py:227-248` | **none** | — | `test_cuda_guard.py` (module-level only) | **RED** — rule 1. Same row as L2/L3; counted once here for the firewall and once there for read-only, because the two invariants fail independently |
| O4 | **L4** `sitecustomize` hook — the only layer that survives spawn / forkserver / exec | `tribe_tools/_s2_sitehook/sitecustomize.py:29-34` | `arm(sitehook=...)` at `s2_run.py:754-755` | 2 | none | **RED** — DEMONSTRATED. The hook imports **`s2guard`**; no such module exists anywhere in the repo (the module is `tribe_tools/cuda_guard.py`), and the `except Exception` at line 33 swallows it. Probe P8 armed the firewall exactly as `infer()` does, then launched a child: `PYTHONPATH` was set correctly, `sitecustomize` loaded, and the child printed `is_owner=False`, `CUDA_VISIBLE_DEVICES=None`, with `[s2guard] sitecustomize failed: ModuleNotFoundError("No module named 's2guard'")` on stderr. `install_child_hooks()` never ran, so the child is unguarded and can see the GPU |
| O5 | L4 covers the stage that actually runs V-JEPA | — | `s2_run.py:527` — `cuda_guard.arm()`, **`sitehook=` omitted** | 1 | none | **RED** — rule 2. Even with O4 repaired, Stage 1 arms without the sitehook, so no spawned or exec'd child of the only encoding stage inherits the guard |
| O6 | the extractor sentinel is actually installed | `cuda_guard.py:129-152` | inside `arm()` | 1+2 | `test_cuda_guard.py` | **RED** — `install_extractor_sentinel` returns `[]` silently when exca is not importable (`:138-141`), and `arm()` (`:50`) discards the return value. The firewall reports armed while its only application-level layer is a no-op. Same "an unplugged instrument reads as success" shape that M1 was written to close, one module over |

### P — ledger

| # | Invariant | Guard | Real caller | Stage | Behavioural test | Verified |
|---|---|---|---|---|---|---|
| P1 | a resumed session derives its next action from the ledger rather than trusting a dead session's exca index | `ledger.py:95-128` (`resume_state`) | **none** | — | `test_ledger.py:80-141`; `test_s2_pipeline.py` | **RED** — rule 1. `scripts/s2_run.py` never calls it; the only non-test caller is `scripts/s2_invariant_proof.py:100`. Round 2's "unrepairable tarpit" was a *resumed* Stage 1 trusting the dead session's index, and the module written to answer that question is not consulted by the thing that resumes. `begin_stage1` (A3) covers the `COMPLETE`-marker half of it, which is why this is a RED and not a run-stopper |

---

## 3. Every RED, in one list

| id | one line | kind | cost if shipped |
|---|---|---|---|
| **I2** | `verify_local_weights(default_hf_cache_dir(), ...)` → `WeightFileMissing` | run-stopper | Stage 1 never starts |
| **E1** | `sidecar_digests(None)` → `TypeError` on line 1 of `stage2_infer` | run-stopper | Stage 2 never starts |
| **A1** | every payload read raises → a good artifact is condemned `ArtifactCorrupt` | run-stopper | remedy text orders `rm -rf` + 4h45m |
| **H2** | `_reader_for` is one directory too shallow → `KeyError` per item | run-stopper | `ArtifactNotFound` → 4h45m re-encode |
| **F2** | `np.savez` writes `preds.npz.tmp.npz`, `os.replace` looks for `preds.npz.tmp` | run-stopper | ~86 MB lost immediately after the GPU work |
| **D2** | the resolved artifact location never reaches the extractor | wiring | B2 defeated in a new way |
| **D3** | `require_artifact_location` called without `sidecar_probe=` | optional-arg | provenance laundering undetected during resolution |
| **E3** | `publish` given a 0-arg `sidecar_probe` where 1 arg is passed | optional-arg | Stage 1's durable copy dies after the encode |
| **E4** | that probe also ignores the directory, hashing the source | optional-arg | B3 unfixed for sidecars |
| **N1** | Stage 1's `assert_atlas_ready` called without `parcels=` | optional-arg | 4h45m against an atlas Stage 2 will reject |
| **O5** | Stage 1's `cuda_guard.arm()` called without `sitehook=` | optional-arg | no spawn coverage in the encoding stage |
| **O3** | `cuda_guard.preflight` has no caller | never called | the live object graph is never checked |
| **L2** | …so "read-only reached the extractor" is never asserted | never called | the firewall is a config value, not a fact |
| **L3** | …nor is `infra.cluster is None` | never called | `processpool` resurrection undetected |
| **G3** | `stage_stimulus` has no caller | never called | exca keys stay path-addressed |
| **P1** | `ledger.resume_state` has no caller | never called | resume is unadvised |
| — | `probe_video` imported at `:54`, never called | never called | `has_audio is False` unasserted (item 15) |
| **B2** | `--infer --stub` exercises none of the four things its docstring names | false coverage | the CPU wiring check reports GO |
| **I4** | the integration stub for `verify_local_weights` goes green over I2 | false coverage | B1 looks closed and is not |
| **K1** | the "absent"-version guard has no test in `tests/` | untested | two machines could share one uid |
| **K2** | `_processor_config` downloads from HF inside both GPU stages | ordering | offline abort after preflight passed |
| **G2** | `--stimulus-root` does not `.resolve()` | residue | cwd-dependent exca keys via the CLI knob |
| **G4** | manifest / artifact / atlas / ledger paths are all cwd-relative | residue | a `cd` loses the frozen atlas and the ledger |
| **C2** | `die()`'s `SystemExit` escapes `stage2_infer`'s `except Exception` | residue | no `ABORTED` record for parcel/batch refusals |
| **O4** | the sitecustomize hook imports a module that does not exist | residue → total | L4 is inert; the swallowing `except` hides it |
| **O6** | `install_extractor_sentinel()`'s empty return is discarded | residue | the sentinel can silently not exist |

(26 lines: 25 numbered REDs plus the unnumbered `probe_video` row, which is folded into
the "never called" group rather than counted separately.)

---

## 4. The probes

Each was run on this box. `$X` is
`.../scratchpad/xv/0.5.20/x` (exca 0.5.20) plus `.../scratchpad/.v/lib/python3.12/site-packages`.

**P1 — H2, the reader is one level too shallow.** Wrote one item through the real
`exca.cachedict.CacheDict` at the real `<folder>/<method>,<version>/<uid>` layout, read
it back directly (`[0. 1. 2. 3.]`), then read it back through a verbatim copy of
`_reader_for`:

```
files under uid folder: ['<host>-info.jsonl', 'data']
P1 RESULT: reader_for raised KeyError: 'item0' -> claim CONFIRMED
```

**P2 — I2, Stage 1 cannot verify the weights.** Built a byte-correct HF hub cache
(`refs/main`, `blobs/<sha256>`, `snapshots/<commit>/<file>` symlinks) for
`facebook/vjepa2-vitg-fpc64-256` at the pinned commit, resolved the identity offline,
then made the exact call at `s2_run.py:553`:

```
identity resolved: 875c192b7b70 local-cache
P2 RESULT: raised WeightFileMissing: model.safetensors not found at /tmp/.../hub.
           Pass the file, its snapshot directory, or the repo cache directory.
           Nothing was verified.
P2 CONTROL (repo cache dir): full-hash 7fe15aa1ffcd
```

**P3 — E3, `publish`'s sidecar probe arity.**

```
P3 RESULT: TypeError: <lambda>() takes 0 positional arguments but 1 was given
P3 CONTROL (1-arg probe): published -> /tmp/.../dur2/s2v1-c448e4d0ea1e3859 created= True
```

**P4 — F2, the persistence rename.**

```
P4 RESULT: FileNotFoundError: [Errno 2] No such file or directory:
           '/tmp/.../preds.npz.tmp' -> '/tmp/.../preds.npz'
           files actually written: ['preds.npz.tmp.npz']
```

**P5 — E1, the Stage 2 sidecar probe.**

```
P5 RESULT: TypeError: argument should be a str or an os.PathLike object
           where __fspath__ returns a str, not 'NoneType'
```

**P6 — A1, the Stage 2 payload reader.** Real exca 0.5.20:

```
P6 RESULT: ValueError: At least folder or keep_in_ram should be activated
```

**P8 — O4, the sitecustomize hook.** Armed exactly as `infer()` does
(`arm(sitehook=<repo>/tribe_tools/_s2_sitehook)`), then launched a child interpreter:

```
PYTHONPATH after arm: ['<repo>/tribe_tools', '<repo>/tribe_tools/_s2_sitehook']
CHILD is_owner_pid= False
CHILD CUDA_VISIBLE_DEVICES= None
CHILD sitecustomize loaded= True
CHILD stderr: [s2guard] sitecustomize failed: ModuleNotFoundError("No module named 's2guard'")
```

**P9 — B2, what `--infer --stub` actually calls.** Spies on `stage2_infer`,
`assert_modality_contract`, `verify_artifact`, `require_artifact_location`,
`EncodeCounter` and `_persist_predictions`, then `infer(S2, stub=True)`:

```
P9 RESULT rc= 0
P9 called during --infer --stub: NONE OF THEM
P9 _persist_predictions calls: 0
```

---

## 5. UNPROVEN

`tribev2`, `neuralset` and `torch` are not installed here. These could not be settled:

1. **`_probe_modalities` against a real `get_loaders`.** `tribev2/main.py:160-274`
   returns a `dict` of loaders and *skips* a split whose segment list is empty
   (`main.py:258-260`), so `["all"]` would raise `KeyError` rather than the typed refusal
   at `:722`. Whether the S2 stimulus ever yields zero segments is not decidable here.
2. **`_probe_modalities` cost.** `get_loaders` calls `extractor.prepare(events)`
   (`main.py:215-218`) for every extractor, i.e. the modality probe runs the bulk
   extraction path once and `predict()` runs it again. Warm-cache cost is presumed ~0
   but is not measured.
3. **Whether `library_versions()` returns real versions or `"absent"` on the Kaggle
   image** — K1's guard is only as good as what it sees there.
4. **Whether the real `video_preprocessor_config.json` supplies all 12 keys
   `REQUIRED_PREPROCESSING` demands** — if not, `_identity` aborts in both GPU stages.
5. **Whether `data.<mod>_feature.infra.mode` survives `from_pretrained`'s `config.pop`
   sequence** (`demo_utils.py:191-196`) onto the live extractor. This is exactly what L2
   would answer, and L2 is uncalled.
6. **L1's behaviour under a real DataLoader fork with CUDA initialised.** `arm()` and the
   sentinel are exercised for real in the integration suite, but no GPU is present.

## 6. Notes on the audit itself

* `tests/test_s2_integration.py` is **untracked and being written concurrently** as this
  was produced. Its line citations are as of 2026-08-27 and may move. It independently
  reached the same conclusions on H2, D2/E1, E3 and F2 (its W1–W4), which is
  corroboration, not a shared source: those four were found here from
  `exca 0.5.20`/`tribev2` source and confirmed by probes P1–P6 before that file was read.
* `scripts/s2_gate.py:85,87,105` still grade `resolve_artifact_location`,
  `stage_stimulus` and `load_frozen_parcels` with `hasattr`. Under rule 4 that is a RED
  gate; it is not counted in §1 because it is another agent's file and is under rewrite.
* Test suite at the time of audit:

  | invocation | result |
  |---|---|
  | `python3 -m pytest tests/ -q` | **456 passed, 40 skipped** |
  | same, with `S2_DEV_SITE_PACKAGES` pointing at exca 0.5.20 | **493 passed, 2 skipped** |

  The ~38 tests that move from skip to pass are the `needs_exca` contract set, which
  silently skips without the real library (collection also differs by one item between
  the two runs — 496 vs 495 — so the counts are not a clean subtraction).
  `tests/conftest.py:36-41` turns a skip into a hard failure under `S2_REQUIRE_EXCA=1`;
  the Kaggle run must set it, because a skipped test protects nothing.

  The suite being green is consistent with every RED above: none of these defects is
  currently asserted *against* — four (H2, D2/E1, E3, F2) are asserted *for*, as
  documented workarounds in the integration harness.

## 7. Two claims in `ops/S2-PRE-GPU-REPORT.md` that this audit contradicts

* **§11.1** — *"`scripts/s2_run.py` … cannot execute here — tribev2 and neuralset are
  not installable. It is argued from source, not run."* Partly avoidable. Six of the
  defects above (H2, I2, E3, F2, E1, A1's proximate cause) live in code that needs
  **only exca, numpy and `huggingface_hub`'s cache layout** — no tribev2, no torch. They
  were each found and demonstrated here in minutes (§4). "Cannot execute" was true of
  the whole script and taken to mean it was true of every line in it.
* **§11.12** — *"the free blob-filename route is trusted-not-measured until
  `force_hash=True` runs once at Stage 1, **which it now does**."* It does not. `force_hash=True`
  is passed, but the accompanying path argument makes `verify_local_weights` raise
  `WeightFileMissing` before it reads a byte (I2, probe P2). The 4.14 GB is still never
  measured, and Stage 1 additionally cannot start.
