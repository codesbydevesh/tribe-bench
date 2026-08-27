# S2 pre-GPU report — 2026-08-26

Successor to `ops/S2-INCIDENT-2026-08-25.md`. Covers the reliability architecture built
after the 4h45m loss, and states what is proven, what is argued, and what is neither.

---

## 1. Incident reconstruction

| | |
|---|---|
| Expected feature items | 1050 s x 2 Hz = **2100** |
| Encoded by the prepare pass | 17x120 + 60 = **2100** — the full stimulus, complete |
| Prepare bar | 4 h 31 m @ 7.72 s/item |
| A 19th encode *after* the bar finished | +15 m 33 s |
| Total GPU burned | **4.78 h** |
| Scientific output | **none** |

The encoding stage did its whole job. Everything downstream of it failed.

---

## 2. Root causes, separated

**2.1 Cache failure.** `TribeModel.from_pretrained` assigns its `cache_folder` parameter
straight into `config["data.<mod>_feature.infra.folder"]` (`tribev2/demo_utils.py:206-208`),
*unconditionally*, clobbering the checkpoint's own `folder: CACHEDIR`. Its default is
`None`; we never passed one. We separately set `keep_in_ram: False` to bound RSS. With
folder `None` **and** keep_in_ram `False`, `CacheDict.__init__` raises
(`exca/cachedict/core.py:127-128`) and exca **catches its own exception twice**
(`map.py:290-291`, `:504-507`) as `# no caching`. 4 h 31 m of V-JEPA went into a throwaway
dict and was discarded at `neuralset/extractors/base.py:202`.

The released default is `keep_in_ram: True`. **Our optimisation is the proximate trigger.**

**2.2 Recomputation.** Because nothing was cached, the dataloader legitimately found every
item missing and re-entered the compute path.

**2.3 Multiprocessing / CUDA.** `neuralset/extractors/video.py:265` does
`model.model.to(self.image.device)` where `device` is the literal string `"cuda"` resolved
in the parent. In a forked worker that is `Cannot re-initialize CUDA in forked subprocess`.
`num_workers = 20` is `N_CPUS` from Meta's training cluster (`grids/defaults.py:20,131`),
frozen into the released checkpoint config.

**Causal order matters:** with a warm cache the worker never reaches the compute path and
never touches CUDA. Fixing only `num_workers` would have converted a loud crash into a
**silent, correct-looking ~9.3 h run**.

**2.4 Incorrect configuration key.** The first fix (`323a65c`, pushed) used
`{"num_workers": 0}`. `num_workers` is a field of tribev2's `Data` sub-model
(`main.py:112`), so the correct key is `data.num_workers`. Proven by running exca 0.5.20's
real ConfDict, and by constructing Meta's real `BaseExperiment` from the neuraltrain 0.0.2
wheel: the bare key raises `ValidationError` (loud, ~10 min) rather than being silently
absorbed. Corrected in `7a7ee73`.

**2.5 Weight identity.** neuralset passes no `revision=` anywhere, so V-JEPA came off a
floating branch, and exca's cache uid contains the model NAME but never its identity.

**2.6 Extractor fallback.** `tribev2/main.py:206-212` deletes an extractor with no matching
events; `tribev2/model.py:188-192` zero-fills the gap. Because `time_pos_embedding` defaults
True, a video-less run is finite, non-zero, time-varying and statistically **within 2% of a
real run** — and `modality_dropout: 0.3` puts video-absent in-distribution. Plausible, not
obviously broken.

**2.7 Atlas timing.** The first HCP-MMP1 resolution happened inside `analyse()`, after ~5 h
of GPU. It passed locally only because `~/mne_data` already existed. Worse:
`mne.datasets.sample.data_path()` performs no hash or version check and returns in 1.37 s
for an **empty** directory.

---

## 3. Architecture

```
--prepare      CPU   render, verify, hash
--preflight    CPU   atlas identity + freeze, weight identity, inputs.  ~2 s
                     |  nothing expensive is reachable until this passes
--extract-features   GPU STAGE 1. the ONLY stage permitted to compute
                     |  encode -> READ BACK -> digest -> COMPLETE last -> publish
                     v
             feature artifact  (durable, digest-verified, identity-bound)
                     |
--infer        GPU   STAGE 2. verify BEFORE load_model, then consume.
                     extractors run infra.mode="read-only": cannot encode
```

---

## 4. Cache identity

Two layers, because exca's own key is insufficient and cannot be changed from outside.

**exca's item uid** (`neuralset/extractors/video.py:247`) is
`f"{study_relative_path()}_{offset:.2f}_{duration:.2f}"` — path + offset + duration, **no
content hash**. The stimulus is therefore staged content-addressed at
`<workdir>/s2_stim/<sha16>/`, which makes exca's own keys mount-independent.

**Our identity** is 57 fields via `provenance.feature_uid_fields`, covering: stimulus
sha256 and decode geometry (fps/width/height — `video.py:285` samples by timestamp, so a
re-encode at unchanged duration changes every tensor); the hard-coded
`ChunkEvents("Video", 60, 30)` literals (`demo_utils.py:78`) that decide the 18-item set;
16 extractor fields including `num_frames_effective` resolved to the literal 64 at
`video.py:404-405` that exca's `exclude_defaults=True` hides; 12 preprocessing values as
well as the config digest; the V-JEPA weight sha256; and the versions that move numerics.

Bound into exca via `data.video_feature.infra.version = exca_infra_version(identity)` —
the only `MapInfra` field that participates in its uid (`exca/base.py:191-194`), and one
Meta already sets.

**Deliberately excluded and recorded instead:** `device`, dtype, GPU model, TF32. Keying
on those would invalidate 226 MiB on every Kaggle base-image bump.

---

## 5. Process model

| may touch CUDA | may not |
|---|---|
| the process that called `cuda_guard.arm()` | every fork child |
| | every spawn / forkserver child |
| | every `ProcessPoolExecutor` worker |
| | every exec'd subprocess |

`spawn` was **rejected** as the fix: measured, a spawned worker re-initialises CUDA cleanly
and then honestly re-encodes V-JEPA in every worker in parallel. The crash disappears and
the catastrophe remains. Measured `os.register_at_fork` hits: subprocess 0, fork 1,
ProcessPoolExecutor +2, spawn +0, forkserver +0 — so the at-fork layer can never be the
only layer. The PID sentinel rides the **environment**, which fork, spawn, forkserver and
exec all inherit.

---

## 6. Expensive-operation proof

`scripts/s2_invariant_proof.py`, real exca 0.5.20, Stage 1 and Stage 2 in **separate
interpreters**, encode counter incrementing a file on disk:

```
first run, new identity        -> 3 encodes
consume in a FRESH process     -> 0 encodes,  tally unchanged
consumed value == extracted    -> 97.0        (byte-identical round-trip)
ledger after a crash           -> verify_then_infer   (not "extract")
restart, valid identity        -> 0 encodes,  tally unchanged
new identity                   -> exactly 1 extraction
consume with no artifact       -> raises, encodes nothing
```

**11/11.** Separate processes matter: a same-process test can smuggle features through
exca's in-RAM cache while the disk cache is empty, and would report zero encodes for the
wrong reason.

---

## 7. Restart proof

Covered by rows 3-4 above and by `test_a_crash_after_extraction_does_not_cause_another_extraction`.
The ledger's strongest possible statement is `verify_then_infer` — it never authorises
consumption, only verification. An identity mismatch, or `EXTRACT_STARTED` with no matching
`ARTIFACT_FINALIZED`, both yield `extract`: a half-written cache is not a checkpoint, which
is precisely the state the incident died in.

---

## 8. Failure-injection results

`scripts/s2_failure_injection.py` — **16/16 behaved as specified.**

| # | injection | expected | observed |
|---|---|---|---|
| 1 | corrupt tensor | ArtifactCorrupt | ArtifactCorrupt |
| 2 | truncate artifact payload | ArtifactCorrupt | ArtifactCorrupt |
| 3 | alter manifest metadata | ArtifactCorrupt | ArtifactCorrupt |
| 4 | change stimulus sha256 | ArtifactStale | ArtifactStale |
| 5 | change model revision | ArtifactStale | ArtifactStale |
| 6 | change V-JEPA weight sha256 | ArtifactStale | ArtifactStale |
| 7 | change preprocessing | ArtifactStale | ArtifactStale |
| 8 | remove completion marker | ArtifactIncomplete | ArtifactIncomplete |
| 9 | artifact absent entirely | ArtifactMissing | ArtifactMissing |
| 10 | partial extraction | ExtractionIncomplete | ExtractionIncomplete (+ not certified) |
| 11 | force extraction during consume | ConsumeStageRecomputed | ConsumeStageRecomputed |
| 12 | worker attempts CUDA (real fork) | ChildGPUViolation | ChildGPUViolation |
| 13 | extractor returns failure | RuntimeError | RuntimeError (+ not certified) |
| 14 | required modality zero-filled | ModalityContractViolation | ModalityContractViolation |
| 15 | required extractor missing | ModalityContractViolation | ModalityContractViolation |
| 16 | SIGKILL between stages, restart | verify_then_infer, 0 re-encodes | same |

---

## 9. Test suite

**454 collected, 452 passing** (from 168 before the incident). 286 new behavioural tests:

| file | n |
|---|---|
| test_provenance.py | 122 |
| test_durable_store.py | 50 |
| test_atlas_preflight.py | 42 |
| test_feature_artifact.py | 26 |
| test_s2_pipeline.py | 19 |
| test_ledger.py | 15 |
| test_cuda_guard.py | 9 |
| test_s2_num_workers.py | 3 |

**No source-string assertions were written, and three were deleted.** The cautionary case:
`assert '"num_workers": 0' in src` was green while the key was wrong, and correcting the key
turned it red. A weak test did not fail to protect — it defended the defect.

### Mutation cycles — every one found something

| module | mutations | initially undetected |
|---|---|---|
| feature_artifact | 7 | **M7** item-count check (only a *superset* artifact reaches it) |
| cuda_guard | 5 | — |
| ledger | 6 | **L1** fsync (see below) |
| s2_pipeline | 8 | — |
| provenance | 14 | **M4** float precision (test pair was still separable) |
| durable_store | 15 | **M2** freshness, **M9** dot-directories |
| atlas_preflight | 12 | **M9-M12** parcel guards |

All closed by adding tests, never by deleting a guard. **L1 is a declared limitation, not a
fix:** SIGKILL does not lose the page cache, so no in-process test can observe a missing
`fsync`. The test asserts the syscall; the limit is documented.

---

## 10. GPU cost model

| scenario | cost |
|---|---|
| preflight | ~2 s (0.14 s once frozen) |
| first clean run: extract | ~4.75 h — 2100 items + the structural +1 shape probe |
| first clean run: infer | minutes |
| **first clean run total** | **~5.0-5.5 h** |
| restart after Stage 1 | **0 encodes** — verify (~0.2 s) then infer |
| restart after Stage 2 | 0 encodes |
| cache hit, new session | 0 encodes, artifact pulled from the durable store |
| cache miss (new identity) | exactly one extraction |

The notebook's stated 3.4 h budget is ~40% low; `s2_design.py:312`'s
`compute_s_per_stimulus_s = 11.5` is the stale input. The 19th encode is **structural**
(`extractors/base.py:202-209`: bulk call, then a single-item shape probe whose
`duration=0.001` does not shrink the work — N items yield N+1 encodes) and belongs in the
cost model, not the bug list.

---

## 11. Remaining risks

**Cannot be proven on this box:**

1. **The tribev2 glue is the least-tested code in the repo.** Every module was built and
   mutation-tested in isolation; `scripts/s2_run.py` was written last and *cannot execute
   here* — tribev2 and neuralset are not installable. It is argued from source, not run.
2. Real Kaggle API behaviour: `kaggle datasets status` return codes, `--dir-mode zip`
   round-tripping a nested tree, quotas.
3. Real `/kaggle/input` semantics (simulated with `chmod 555`, which does not deny under root).
4. The real figshare download success branch (only the blocked-socket path is exercised).
5. Cross-mne-version stability (digests confirmed against 1.12.1; `pyproject.toml` allows `>=1.4`).
6. fsync durability under power loss.
7. That torch's DataLoader really *forks* rather than spawns — needs the CPU torch wheel.

**Known and accepted:**

8. `device`, dtype, TF32, cuDNN algorithm choice are outside the uid by design. CPU vs CUDA
   and T4 vs A100 change low-order bits.
9. Run-to-run GPU non-determinism (SDPA selection, cuDNN autotune, atomic reductions) is
   uncoverable by any uid.
10. `ffmpeg` decodes every frame and is not a Python distribution, so `importlib.metadata`
    cannot reach it. Record `ffmpeg -version` separately.
11. `feature_uid_fields` enforces **presence, not truth**. The call site reads values off the
    live extractor object precisely because a literal that disagrees with what neuralset used
    would produce a confidently wrong uid.
12. huggingface_hub does not verify downloaded bytes; the free blob-filename route is
    trusted-not-measured until `force_hash=True` runs once at Stage 1, which it now does.

---

## 12. Exact Kaggle sequence

Deliberately empty until the independent review returns GO. See §13.
