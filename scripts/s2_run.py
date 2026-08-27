"""S2 — execute the frozen design. The implementation, not a reinterpretation.

Every parameter comes from :data:`neurocheck.s2_design.S2`. This script contains no
experimental choices of its own: if a number here disagrees with the frozen design,
the design wins and this script is wrong.

**Four stages, and the expensive boundary is explicit.** Before 2026-08-26 extraction
and consumption happened inside one ``predict()`` call, which is why -- after a run
that burned 4h45m of V-JEPA and produced nothing -- "did the second stage recompute?"
could not be answered. There was no boundary at which to ask.

    --prepare           CPU. Resolve images, render the stimulus, verify, hash.
    --preflight         CPU. Atlas, weight identity, inputs. Seconds. Must pass
                        before ANY GPU work is reachable.
    --extract-features  GPU, STAGE 1. The only stage permitted to compute. Encodes
                        V-JEPA into a digest-verified, durable artifact, then exits.
    --infer             GPU, STAGE 2. Verifies the artifact BEFORE loading the model,
                        then runs the brain model with the extractors in exca
                        read-only mode -- structurally unable to encode.

    # on this box, no GPU:
    python3 scripts/s2_run.py --prepare
    python3 scripts/s2_run.py --preflight
    python3 scripts/s2_run.py --infer --stub        # end-to-end with a fake model

    # on the GPU box, in order:
    python3 scripts/s2_run.py --preflight
    python3 scripts/s2_run.py --extract-features
    python3 scripts/s2_run.py --infer

Refuses to run on any mismatch. A silently substituted input is the failure this
whole phase exists to prevent.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurocheck.s2_design import (  # noqa: E402
    ALL_PARCELS, LAG_CONFLICT, RULES, S2, S2Config, build_manifest, build_schedule,
    check_three_way_consistency, environment_provenance, events_dataframe,
    replication_verdict, resolve_stimulus_images, stop_eligible_parcels,
)
from neurocheck.s2_stimulus import frame_plan, probe_video, render, verify_rendered_frames  # noqa: E402

# The manifest and the probe ship IN GIT, so they arrive with the repo clone and
# stay repo-relative. The images and the video cannot ship (fLoc has no licence),
# so on a GPU box they live wherever the dataset is mounted -- on Kaggle that is
# /kaggle/input/<dataset-slug>, which is READ-ONLY. Hence one knob, resolved once,
# rather than paths guessed inside the session.
# B7: exca keys every cache item on the LITERAL filepath string
# (neuralset/extractors/video.py:247, etypes.py:377-379 returns it verbatim), so a
# relative root makes the keys depend on the working directory. Stage 1 from the
# repo root and Stage 2 from a notebook would then disagree, and the resulting
# error text prescribes a 4h45m re-encode to fix a `cd`. Absolute, once, at import.
STIMULUS_ROOT = Path(os.environ.get("S2_STIMULUS_ROOT", "data")).resolve()
PROBE = Path("data/s2_stimulus_probe.json")
MANIFEST = Path("data/s2_manifest.json")


def video_path(root: Path = None) -> Path:
    return (root or STIMULUS_ROOT) / "s2_stimulus.mp4"


def image_root(root: Path = None) -> Path:
    return (root or STIMULUS_ROOT) / "floc"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def die(msg: str) -> None:
    """Fail loudly. Never substitute, never continue on a mismatch."""
    print(f"\nREFUSING TO RUN: {msg}", file=sys.stderr)
    raise SystemExit(2)


# ---------------------------------------------------------------- preparation

def prepare(cfg: S2Config) -> dict:
    """CPU only: resolve images, render, verify against the spec, hash."""
    print(f"=== S2 prepare === design {cfg.fingerprint()}\n")
    images = resolve_stimulus_images(image_root(), cfg)
    print(f"  images     {len(images)} resolved and hashed from {image_root()}")

    events = build_schedule(cfg)
    r = render(cfg, video_path(), events=events, images=images)
    if r.placeholders:
        die("rendered with placeholder frames; a recorded run needs the real images")

    # verify against the SPEC, not against the renderer's own arithmetic
    plan = frame_plan(cfg, events)
    per_event = np.bincount(plan[plan >= 0], minlength=cfg.n_events)
    problems = []
    if abs(r.duration_s - cfg.stimulus_duration_s) > 1.0 / cfg.fps:
        problems.append(f"duration {r.duration_s} != spec {cfg.stimulus_duration_s}")
    if r.fps != cfg.fps:
        problems.append(f"fps {r.fps} != spec {cfg.fps}")
    if (r.height, r.width) != cfg.frame_size:
        problems.append(f"resolution {r.width}x{r.height} != spec {cfg.frame_size}")
    if r.n_frames != int(round(cfg.stimulus_duration_s * cfg.fps)):
        problems.append(f"frame count {r.n_frames} != spec")
    if len(set(per_event.tolist())) != 1:
        problems.append("events differ in frame count")
    problems += verify_rendered_frames(cfg, video_path(), events, sample=60)
    problems += check_three_way_consistency(cfg, build_manifest(cfg), r.duration_s,
                                            r.fps, events_dataframe(cfg))
    if problems:
        die("rendered stimulus does not match the frozen spec:\n  - " +
            "\n  - ".join(problems[:8]))

    video = {"path": str(video_path()), "sha256": _sha256(video_path()),
             "bytes": video_path().stat().st_size, "n_frames": r.n_frames, "fps": r.fps,
             "duration_s": r.duration_s, "width": r.width, "height": r.height,
             "stimulus_frames": r.stimulus_frames, "grey_frames": r.grey_frames,
             "placeholders": r.placeholders}
    PROBE.write_text(json.dumps(video, indent=2))
    print(f"  video      {r.n_frames} frames @ {r.fps}fps, {r.duration_s:.3f}s, "
          f"{r.width}x{r.height}")
    print(f"             sha256 {video['sha256'][:32]}...")

    idx = json.loads((image_root() / "index.json").read_text())
    man = build_manifest(
        cfg, code_commit=_git("rev-parse", "HEAD"),
        tree_dirty=bool(_git("status", "--porcelain")),
        environment=environment_provenance(), video=video,
        images={"source_repo": idx["source_repo"], "commit": idx["commit"],
                "selection_rule": idx["selection_rule"], "licence": idx["licence"],
                "n": len(images), "files": images})
    from tribe_tools.model import TRIBEV2_CKPT_BYTES, TRIBEV2_CKPT_SHA256, TRIBEV2_REVISION
    man["provenance"]["checkpoint"] = {
        "repo": "facebook/tribev2", "revision": TRIBEV2_REVISION, "file": "best.ckpt",
        "sha256": TRIBEV2_CKPT_SHA256, "bytes": TRIBEV2_CKPT_BYTES}
    MANIFEST.write_text(json.dumps(man, indent=2, default=str))
    print(f"  manifest   {MANIFEST} ({len(json.dumps(man)) / 1024:.0f} KB)")
    print("\nprepare: OK. Inputs verified against the frozen spec.")
    return man


# ------------------------------------------------------------------ inference

def _check_inputs(cfg: S2Config) -> dict:
    """Every input must match what prepare() recorded, or refuse."""
    if not MANIFEST.exists():
        die(f"{MANIFEST} missing — run --prepare first")
    man = json.loads(MANIFEST.read_text())
    if man["design_fingerprint"] != cfg.fingerprint():
        die(f"manifest is for design {man['design_fingerprint']}, code is "
            f"{cfg.fingerprint()} — the design changed since prepare")
    v = (man.get("provenance") or {}).get("video")
    if not v:
        die("manifest records no video — run --prepare first")
    if not video_path().exists():
        die(f"{video_path()} missing — run --prepare first")
    got = _sha256(video_path())
    if got != v["sha256"]:
        die(f"stimulus hash mismatch: manifest {v['sha256'][:16]}..., file "
            f"{got[:16]}... — this is not the video the manifest describes")
    if v.get("placeholders"):
        die("the recorded video was rendered with placeholder frames")
    live = resolve_stimulus_images(image_root(), cfg)
    rec = man["provenance"]["images"]["files"]
    drift = [k for k in rec if k not in live or live[k]["sha256"] != rec[k]["sha256"]]
    if drift:
        die(f"{len(drift)} stimulus images differ from the manifest: {drift[:5]}")
    return man


def _peri_event(preds, row_times, verts, onsets, cfg):
    """Peri-event timecourse over the reported lag grid, one pass, all lags."""
    from tribe_tools.roi_stats import peri_event_timecourse
    pre = -min(cfg.report_lags)
    post = max(cfg.report_lags)
    return peri_event_timecourse(preds, verts, onsets, row_times, pre, post), pre


def analyse(preds, segments, cfg: S2Config, man: dict, parcels: dict) -> dict:
    """Score every parcel at BOTH pre-specified lags from the SAME timecourse.

    `parcels` maps parcel name -> vertex indices and is REQUIRED. It comes from
    atlas_preflight.load_frozen_parcels, which verified a per-parcel sha256 against
    the frozen cache before any GPU work.

    This used to call tribe_tools.atlas.get_vertices, which reaches live MNE -- AFTER
    ~5 h of encoding, on vertices nothing had verified, and only working at all
    because ~/mne_data happened to exist. No implicit lookup remains: if a parcel is
    absent from `parcels` this refuses rather than resolving it.
    """
    from tribe_tools.roi_stats import (
        detection_floor, event_locked_contrast, peak_lag_trs, perm_p,
        row_times_from_segments,
    )

    row_times = row_times_from_segments(segments)
    events = build_schedule(cfg)
    results: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}

    for parcel in ALL_PARCELS:
        if parcel.name not in parcels:
            die(f"parcel {parcel.name} is absent from the frozen atlas artifact. "
                f"Re-run: s2_run.py --preflight. Refusing to resolve it live -- the "
                f"vertices would not be the ones the preflight verified.")
        verts = np.unique(np.asarray(parcels[parcel.name]))
        if verts.size == 0:
            die(f"parcel {parcel.name} is empty in the frozen atlas artifact")

        # ONE timecourse per category; every lag is a column of it.
        courses = {}
        for cat in cfg.categories:
            onsets = [e.onset_s for e in events if e.category == cat]
            tc, pre = _peri_event(preds, row_times, verts, onsets, cfg)
            courses[cat] = tc
        peak = peak_lag_trs([courses[c] for c in cfg.categories], pre_trs=pre)

        by_lag = {}
        for lag in (cfg.primary_lag_trs, cfg.alternative_lag_trs):
            col = pre + lag
            per_cat = {c: np.asarray(courses[c])[:, col] for c in cfg.categories}
            # the parcel's own category is the one the paper pairs with it
            target_cat = PARCEL_CATEGORY[parcel.name]
            tgt = per_cat[target_cat]
            others = [per_cat[c] for c in cfg.categories if c != target_cat]
            effect = event_locked_contrast(tgt, others)
            p = perm_p(list(tgt), list(np.concatenate(others)),
                       n_perm=cfg.n_perm, seed=cfg.perm_seed)
            # the floor is computed from THIS parcel's own noise and n
            noise = float(np.std(np.concatenate(others), ddof=1))
            floor = detection_floor(n_per_group=len(tgt), noise_sd=noise,
                                    alpha=cfg.alpha, seed=cfg.perm_seed)
            by_lag[lag] = {"p": float(p), "effect": float(effect), "floor": float(floor),
                           "n_target": int(len(tgt)), "noise_sd": noise}
        results[parcel.name] = {"by_lag": by_lag, "peak_lag_trs": int(peak),
                                "statistic": cfg.primary_statistic}
        diagnostics[parcel.name] = {
            "n_vertices": int(verts.size), "target_category": PARCEL_CATEGORY[parcel.name],
            "timecourse_by_category": {c: np.asarray(courses[c]).mean(axis=0).tolist()
                                       for c in cfg.categories},
            "lag_grid": list(cfg.report_lags)}
    return {"results": results, "diagnostics": diagnostics}


# Which category each parcel is the paper's target for. Frozen: the contrast is
# category-minus-others, so this decides what "recovered" means per parcel.
PARCEL_CATEGORY = {
    "FFA": "faces", "EBA": "bodies", "PPA": "places", "VWFA": "characters",
    "PPA_literature": "places", "EBA_gate0_union": "bodies", "V1_control": "faces",
}


def _validate_parcel_categories() -> None:
    """Every parcel needs a target category, or analyse() dies mid-run."""
    missing = [p.name for p in ALL_PARCELS if p.name not in PARCEL_CATEGORY]
    if missing:
        die(f"no target category declared for {missing}; the contrast is "
            "category-minus-others, so this decides what 'recovered' means")
    bad = {k: v for k, v in PARCEL_CATEGORY.items() if v not in S2.categories}
    if bad:
        die(f"target categories not in the design: {bad}")


def model_config_update(stage: str = "extract", *, infra_version: str | None = None) -> dict:
    """Dotted config payload for TribeModel.from_pretrained.

    A pure function so the EFFECTIVE config can be asserted in a test without
    tribev2 installed. It exists because the previous attempt at this could only
    be checked by grepping the source, and that check passed over a real bug.

    ``data.num_workers`` MUST carry the ``data.`` prefix. num_workers is a field of
    tribev2's ``Data`` sub-model (main.py:112), consumed at main.py:270; it is not a
    field of the experiment root. exca's ConfDict nests strictly on dots
    (confdict.py:54-58), so a bare ``num_workers`` writes a new root key and leaves
    ``data.num_workers`` at the checkpoint's 20 (= N_CPUS on Meta's training
    cluster, grids/defaults.py:20,131). Meta's own grid spells it the dotted way:
    grids/test_run.py:18. Demonstrated against exca 0.5.20:

        {"num_workers": 0}      -> effective data.num_workers = 20  (+ stray root key)
        {"data.num_workers": 0} -> effective data.num_workers = 0

    ``keep_in_ram: False`` bounds RSS (MASTER-PLAN 3.6). NOTE: it is only safe
    alongside a real feature-cache folder; that half is not fixed here.
    """
    if stage not in ("extract", "consume"):
        raise ValueError(f"stage must be 'extract' or 'consume', got {stage!r}")
    cfg: dict = {"data.num_workers": 0}
    for mod in ("video", "audio", "text"):
        cfg[f"data.{mod}_feature.infra.keep_in_ram"] = False
        if infra_version:
            # Binds the V-JEPA weight sha into exca's OWN cache uid, so the library
            # self-invalidates when the weights move. `version` is the only MapInfra
            # field that participates in the uid (exca/base.py:191-194), and Meta
            # already sets it (grids/defaults.py:94).
            cfg[f"data.{mod}_feature.infra.version"] = infra_version
        if stage == "consume":
            # THE firewall. exca raises before a single frame is encoded:
            #   RuntimeError: self.mode='read-only' but found N missing items
            # Verified to fire even when no cache folder was configured at all, and
            # verified NOT to participate in the cache uid -- so the fix cannot
            # invalidate the thing it protects.
            cfg[f"data.{mod}_feature.infra.mode"] = "read-only"
            # A canary, NOT a firewall: it only fires at len(items)==len(missing)==1,
            # so an 18-item bulk pass sails straight through. What it does catch is
            # prepare()'s single-item shape probe -- the incident's mystery 19th encode.
            cfg[f"data.{mod}_feature.infra.forbid_single_item_computation"] = True
    return cfg


ARTIFACT_ROOT = Path(os.environ.get("S2_ARTIFACT_ROOT", "data/s2_features"))
ATLAS_CACHE = Path(os.environ.get("S2_ATLAS_CACHE", "data/s2_parcels.npz"))
LEDGER_PATH = Path(os.environ.get("S2_LEDGER", "data/s2_ledger.jsonl"))


def _ledger() -> "Ledger":
    from tribe_tools.ledger import Ledger
    return Ledger(LEDGER_PATH)


def preflight(cfg: S2Config) -> int:
    """Everything cheap, before anything expensive. Exits non-zero to stop the run.

    The 2026-08-25 run discovered its atlas dependency AFTER ~5 h of encoding. The
    ordering here is the fix: no GPU work is reachable until every one of these
    passes, and each of them costs seconds.
    """
    from tribe_tools.atlas_preflight import AtlasPreflightError, preflight_atlas
    from tribe_tools.ledger import Event

    led = _ledger()
    led.record(Event.PREFLIGHT_STARTED, design=cfg.fingerprint())
    print(f"=== S2 preflight === design {cfg.fingerprint()}\n")

    try:
        man = _check_inputs(cfg)
        print(f"  [ok] inputs verified          video sha256 "
              f"{man['provenance']['video']['sha256'][:16]}...")
    except SystemExit:
        led.record(Event.ABORTED, stage="preflight", error="inputs")
        raise

    try:
        summary = preflight_atlas(list(ALL_PARCELS), ATLAS_CACHE,
                                  allow_download=os.environ.get("S2_ALLOW_DOWNLOAD") == "1")
        print(f"  [ok] atlas frozen             {summary.get('n_parcels', '?')} parcels "
              f"-> {ATLAS_CACHE}")
    except AtlasPreflightError as e:
        led.record(Event.ABORTED, stage="preflight", error=f"atlas: {e}")
        print(f"  [FAIL] atlas: {e}")
        return 2

    try:
        wid = _weight_identity()
        print(f"  [ok] V-JEPA weights identified commit={getattr(wid, 'commit', '?')[:12]}")
    except Exception as e:
        led.record(Event.ABORTED, stage="preflight", error=f"weights: {e}")
        print(f"  [FAIL] V-JEPA weight identity: {type(e).__name__}: {e}")
        print("         set S2_ALLOW_NETWORK=1 to resolve it from the Hub metadata API")
        return 3

    led.record(Event.PREFLIGHT_PASSED, design=cfg.fingerprint(),
               atlas_cache=str(ATLAS_CACHE))
    print("\n  preflight PASSED -- expensive work is now permitted")
    return 0


def _weight_identity():
    """Resolve V-JEPA's identity with the commit PINNED.

    `expected_commit` is an optional parameter, and an optional guard is not a guard:
    both previous call sites omitted it and resolved floating `main`, while
    VJEPA2_COMMIT sat exported and compared against nothing.
    """
    from tribe_tools.provenance import VJEPA2_COMMIT, resolve_weight_identity
    return resolve_weight_identity(
        allow_network=os.environ.get("S2_ALLOW_NETWORK") == "1",
        expected_commit=VJEPA2_COMMIT)


def _identity(cfg: S2Config, man: dict, extractor) -> dict:
    """The feature identity. Values are read off the resolved objects, never from
    literals: `feature_uid_fields` enforces presence, not truth, so a literal that
    disagrees with what neuralset actually used yields a confidently wrong uid."""
    from tribe_tools.provenance import feature_uid_fields, preprocessing_fields
    wid = _weight_identity()
    v = man["provenance"]["video"]
    return feature_uid_fields(
        stimulus={"sha256": v["sha256"], "size_bytes": v.get("bytes"),
                  "duration_s": cfg.stimulus_duration_s, "fps": cfg.fps,
                  "width": cfg.frame_size[0], "height": cfg.frame_size[1]},
        weights=wid,
        extractor=_extractor_fields(extractor),
        chunking={"event_type": "Video", "max_duration": 60, "min_duration": 30},
        preprocessing=preprocessing_fields(_processor_config(wid)),
        versions=_versions_or_die(),
    )


def _versions_or_die() -> dict:
    """`library_versions` records an unreadable distribution as the literal string
    "absent", which `_require` accepts because it is neither missing nor None. That
    is a valid uid for an environment we could not read -- two different machines
    collapse to the same identity."""
    from tribe_tools.provenance import library_versions
    v = library_versions()
    unknown = sorted(k for k, val in v.items() if val in (None, "absent", ""))
    if unknown:
        die(f"cannot read the version of {unknown}. These affect the produced tensors, "
            f"so an identity that records them as 'absent' would be shared by two "
            f"different environments. Install them or fix the environment.")
    return v


def _extractor_fields(extractor) -> dict:
    """Read off the CONSTRUCTED extractor, never from literals.

    provenance.feature_uid_fields enforces presence, not truth: a literal that
    disagrees with what neuralset actually used produces a confidently wrong uid.
    So every value here comes from the live pydantic object.

    `num_frames_effective` is the resolved value: with `num_frames=None` the operative
    number is the literal 64 at neuralset/extractors/video.py:404-405, which exca's
    `exclude_defaults=True` hides from its own uid entirely.
    """
    img = extractor.image
    nf = getattr(extractor, "num_frames", None)
    return {
        "class": type(extractor).__name__,
        "infra_version": extractor.infra.version,
        "frequency": extractor.frequency,
        "clip_duration": extractor.clip_duration,
        "num_frames_effective": 64 if nf is None else nf,
        "max_imsize": getattr(extractor, "max_imsize", None),
        "layer_type": getattr(extractor, "layer_type", None),
        "use_audio": getattr(extractor, "use_audio", None),
        "model_name": img.model_name,
        "pretrained": getattr(img, "pretrained", None),
        "imsize": getattr(img, "imsize", None),
        "token_aggregation": getattr(img, "token_aggregation", None),
        "cache_all_layers": getattr(img, "cache_all_layers", None),
        "cache_n_layers": getattr(img, "cache_n_layers", None),
        "layers": getattr(img, "layers", None),
        "layer_aggregation": getattr(img, "layer_aggregation", None),
    }


def _processor_config(wid) -> dict:
    """The preprocessing VALUES, not just the config digest. A from_pretrained
    override or an equivalent-but-renamed config would leave the digest untouched
    while changing every tensor."""
    from huggingface_hub import hf_hub_download
    from tribe_tools.provenance import VJEPA2_REPO
    path = hf_hub_download(VJEPA2_REPO, "video_preprocessor_config.json")
    return json.loads(Path(path).read_text())


def _item_uid(event) -> str:
    """neuralset/extractors/video.py:247, reproduced exactly. The item key is
    path + offset + duration with NO content hash, which is why the stimulus is
    staged content-addressed and why the artifact carries its own digests."""
    return f"{event.study_relative_path()}_{event.offset:.2f}_{event.duration:.2f}"


def _video_extractor(model):
    """The constructed HuggingFaceVideo. tribev2 keeps extractors on data.<mod>_feature."""
    for attr in ("video_feature", "video"):
        ex = getattr(model.data, attr, None)
        if ex is not None:
            return ex
    die("no video extractor on model.data -- tribev2's layout changed")


def _load(cfg: S2Config, stage: str, cache_folder=None, infra_version=None):
    from tribe_tools.model import load_model
    return load_model(device="cuda", revision=cfg.model_revision,
                      cache_folder=cache_folder,
                      config_update=model_config_update(stage, infra_version=infra_version))


def _resolve_identity(cfg: S2Config, man: dict, stage: str) -> tuple[dict, str]:
    """Two passes, deliberately.

    exca freezes an extractor the first time its uid is computed, and the uid depends
    on infra.version -- which we want to CONTAIN the weight identity. So pass one
    constructs the model only to read the extractor's real configuration, and pass two
    reconstructs it with the derived version. A model construction is seconds; getting
    the identity from literals instead would be a confidently wrong uid, which costs
    4h45m and a wrong answer.
    """
    from tribe_tools.durable_store import feature_set_uid
    probe = _load(cfg, stage)
    identity = _identity(cfg, man, _video_extractor(probe))
    del probe
    return identity, feature_set_uid(identity)


def extract_features(cfg: S2Config, stub: bool) -> int:
    """STAGE 1. The only stage permitted to compute."""
    import numpy as np
    from tribe_tools import cuda_guard
    from tribe_tools.atlas_preflight import assert_atlas_ready
    from tribe_tools.ledger import Event
    from tribe_tools.provenance import exca_infra_version, verify_local_weights
    from tribe_tools.s2_pipeline import Stage1Deps, stage1_extract

    cuda_guard.arm()
    # Refuses unless --preflight already froze the atlas. This is the ordering that
    # would have stopped the 2026-08-25 run five hours before it actually stopped.
    assert_atlas_ready(ATLAS_CACHE)
    man = _check_inputs(cfg)

    if stub:
        print("=== S2 stage 1 === --stub has no stand-in: this stage exists to run V-JEPA.")
        return 0

    identity, uid = _resolve_identity(cfg, man, "extract")
    wid = _weight_identity()
    artifact_dir = ARTIFACT_ROOT / uid
    cache_folder = artifact_dir / "cache"
    print(f"=== S2 stage 1: extract === design {cfg.fingerprint()}")
    print(f"  identity   {uid}")
    print(f"  artifact   {artifact_dir}")

    # Hash the 4.14 GB once, here. huggingface_hub does NOT verify what it downloaded,
    # so the free blob-filename route is trusted-not-measured until we measure it.
    # B1. The previous call passed None as the path and the 57-field identity dict as
    # `expected`; it raised MissingIdentityField on every run, and it was the ONLY
    # place the 4.14 GB is ever measured -- resolve_weight_identity takes its digest
    # from the blob FILENAME, i.e. what the server asserted, and huggingface_hub does
    # not verify downloads (provenance.py:38-41).
    from tribe_tools.provenance import default_hf_cache_dir
    ver = verify_local_weights(default_hf_cache_dir(), wid,
                               filename="model.safetensors", force_hash=True)
    if getattr(ver, "route", None) != "full-hash":
        die(f"the V-JEPA weights were not measured (route={getattr(ver, 'route', None)!r}). "
            f"Stage 1 must hash the artifact once; the blob filename alone is the "
            f"server's assertion, not evidence.")
    print(f"  weights    {ver.route}: {ver.digest[:16]}... measured over "
          f"{getattr(ver, 'size_bytes', '?')} bytes")

    model = _load(cfg, "extract", cache_folder=cache_folder,
                  infra_version=exca_infra_version(identity))
    extractor = _video_extractor(model)
    events_df = model.get_events_dataframe(video_path=str(video_path()))
    video_events = extractor._event_types_helper.extract(events_df)
    expected = [_item_uid(e) for e in video_events]
    print(f"  items      {len(expected)} video chunks\n")

    def _extract(_model, _events):
        """Return the uids the cache ACTUALLY holds, never the ones we asked for.

        This previously returned `expected`, so stage1_extract's
        `missing = [u for u in expected_uids if u not in set(produced)]` was empty by
        construction and ExtractionIncomplete could never fire on the real path -- a
        guard called with the argument that switches it off, which is the exact
        pattern this whole phase exists to remove.

        The list() is deliberate but is NOT the protection: with a real cache folder
        exca 0.5.20 computes and stores eagerly, so discarding the generator loses
        nothing. The protection is the read-back in s2_pipeline.stage1_extract.
        """
        list(extractor._get_data(video_events))
        cache = extractor.infra.cache_dict
        return [u for u in expected if u in cache]

    def _read(item_uid):
        return np.asarray(extractor.infra.cache_dict[item_uid])

    man_out = stage1_extract(
        cfg, identity, artifact_dir, expected,
        Stage1Deps(load_model=lambda: model, build_events=lambda m: video_events,
                   extract=_extract, read_item=_read,
                   sidecars=lambda: __import__("tribe_tools.feature_artifact",
                                               fromlist=["sidecar_digests"])
                   .sidecar_digests(extractor.infra.uid_folder())),
        _ledger())

    print(f"\n  artifact finalized: {man_out['n_items']} items -> {artifact_dir}")
    dest = os.environ.get("S2_DURABLE_ROOT")
    if dest:
        from tribe_tools.durable_store import ArtifactIdentity, LocalDirectoryBackend, publish
        from tribe_tools.feature_artifact import sidecar_digests
        res = publish(artifact_dir, ArtifactIdentity(identity, tuple(expected)),
                      LocalDirectoryBackend(Path(dest)),
                      reader_factory=_reader_for,   # B3/W1: uses its argument
                      sidecar_probe=_sidecars_for)  # W3: takes the root it is given
        print(f"  published -> {res.location}  (created={res.created})")
    else:
        print("  S2_DURABLE_ROOT unset: the artifact is NOT durable beyond this session")
    return 0


# The frozen design uses a SILENT video, so audio and text are LEGITIMATELY absent.
# Both directions are stated, because "video is required" alone would not notice audio
# appearing -- which would mean the stimulus is not the one the design describes.
REQUIRED_MODALITIES = ("video",)
EXPECTED_ABSENT_MODALITIES = ("audio", "text")


def _write_report(cfg: S2Config, man: dict, out: dict, identity: dict, uid: str,
                  *, stub: bool, elapsed: float = 0.0) -> int:
    from tribe_tools.ledger import Event
    verdict = replication_verdict(out["results"], cfg)
    report = {
        "schema_version": 2,
        "stub": stub,
        "design_fingerprint": cfg.fingerprint(),
        "feature_identity_uid": uid,
        "feature_identity": identity,
        "provenance": {**man["provenance"],
                       "inference_seconds": round(elapsed, 1),
                       "run_environment": environment_provenance()},
        "lag_conflict": LAG_CONFLICT.strip(),
        "decision_rules": {k: v for k, v in vars(RULES).items()
                           if not k.startswith("_")} or None,
        "results": out["results"],
        "diagnostics": out["diagnostics"],
        "verdict": verdict,
    }
    dest = Path("data/s2_report_stub.json" if stub else "data/s2_report.json")
    dest.write_text(json.dumps(report, indent=2, default=str))
    if not stub:
        _ledger().record(Event.REPORT_WRITTEN, path=str(dest), uid=uid)

    print("\n  parcel            status                          p(primary)  effect  floor")
    for name, pp in verdict["per_parcel"].items():
        bl = (pp.get("by_lag") or {}).get(str(cfg.primary_lag_trs), {})
        gate = "*" if pp.get("stop_eligible") else " "
        print(f"  {gate}{name:16s} {pp['status']:30s} "
              f"{bl.get('p', float('nan')):9.4f} {bl.get('effect', float('nan')):8.3f} "
              f"{bl.get('floor', float('nan')):7.3f}")
    print("  (* = may fire the stop rule)")
    for w in verdict["warnings"]:
        print(f"\n  WARNING: {w}")
    print(f"\n  stop = {verdict['stop']}   incomplete = {verdict['incomplete']}")
    print(f"  report -> {dest}")
    return 0


def _infer_stub(cfg: S2Config, man: dict) -> int:
    """CPU stand-in for Stage 2 that walks the SAME orchestration path.

    The point is not the numbers -- they are noise. The point is that the control
    flow, the modality contract, the persistence step and the report writer are the
    real ones, so a wiring regression is visible without a GPU.
    """
    from tribe_tools.atlas_preflight import assert_atlas_ready, load_frozen_parcels
    assert_atlas_ready(ATLAS_CACHE, parcels=list(ALL_PARCELS))
    parcels = load_frozen_parcels(ATLAS_CACHE)

    n_rows = int(round(cfg.stimulus_duration_s))
    rng = np.random.default_rng(0)
    preds = rng.normal(0, 1, (n_rows, 20484)).astype(np.float32)

    class _Seg:
        def __init__(self, t): self.start = float(t)
    segments = [_Seg(t) for t in range(n_rows)]

    print(f"=== S2 stage 2: infer === design {cfg.fingerprint()}  "
          f"[STUB - no GPU, structure only]\n")
    t0 = time.time()
    out = analyse(preds, segments, cfg, man, parcels)
    return _write_report(cfg, man, out, {"stub": True}, "stub", stub=True,
                         elapsed=time.time() - t0)


def _exca_uid_folder(artifact_dir) -> Path:
    """W1. exca nests TWO levels, not one.

    MapInfra._uid_string is "{method},{version}/{uid}" (exca/base.py:143), so the cache
    lives at <folder>/<method,version>/<confighash>. A reader that descends one level
    hands CacheDict the method directory, every lookup KeyErrors, and Stage 2 reports
    ArtifactNotFound for an artifact that is present and correct.
    """
    root = Path(artifact_dir) / "cache"
    if not root.is_dir():
        raise FileNotFoundError(f"no cache directory under {artifact_dir}")
    # bounded, two levels deep by construction -- not a recursive walk
    leaves = [d for lvl1 in root.iterdir() if lvl1.is_dir()
              for d in lvl1.iterdir() if d.is_dir()]
    if len(leaves) != 1:
        raise FileNotFoundError(
            f"expected exactly one exca uid folder two levels under {root}, "
            f"found {len(leaves)}: {[str(x.relative_to(root)) for x in leaves][:4]}. "
            f"Refusing to guess which cache Stage 1 wrote.")
    return leaves[0]


def _reader_for(artifact_dir):
    """A reader ROOTED AT the directory it is handed (B3)."""
    import numpy as _np
    from exca.cachedict import CacheDict

    cache = CacheDict(folder=_exca_uid_folder(artifact_dir), keep_in_ram=False)

    def read(item_uid):
        return _np.asarray(cache[item_uid])
    return read


def _sidecars_for(artifact_dir) -> dict:
    """W3. durable_store.verify_location calls sidecar_probe(root) with ONE argument
    (durable_store.py:181). A zero-arg lambda raises TypeError *after* Stage 1 has
    already finalized the local artifact: the encode survives the process and nothing
    survives the session."""
    from tribe_tools.feature_artifact import sidecar_digests
    return sidecar_digests(_exca_uid_folder(artifact_dir))


def _persist_predictions(preds, segments, dest: Path):
    """B6: the only copy of ~86 MB must not be a local variable across analyse(),
    which has ~58 reachable raise sites and is the first thing to touch the atlas."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    starts = np.asarray([float(getattr(sg, "start", i)) for i, sg in enumerate(segments)])
    # W4. np.savez APPENDS ".npz" unless the name already ends in it, so writing to
    # "preds.npz.tmp" produced "preds.npz.tmp.npz" and the rename below hit a file
    # that never existed -- FileNotFoundError immediately after the GPU work, in the
    # step whose whole purpose is to not lose the predictions.
    tmp = dest.parent / (dest.name + ".tmp.npz")
    np.savez(tmp, preds=np.asarray(preds), segment_starts=starts)
    if not tmp.is_file():
        raise FileNotFoundError(f"np.savez did not produce {tmp}")
    os.replace(tmp, dest)
    return dest


def _probe_modalities(model, events) -> dict:
    """B4: what ACTUALLY reaches the brain model, read off one real batch.

    tribev2/main.py:200-212 deletes an extractor with no matching events from a LOCAL
    dict, so `model.data.video_feature is not None` still passes; model.py:188-192 then
    substitutes torch.zeros. This inspects the batch instead of the config.
    """
    loader = model.data.get_loaders(events=events, split_to_build="all")["all"]
    batch = next(iter(loader))
    data = getattr(batch, "data", None)
    if data is None:
        die("the batch exposes no .data mapping; tribev2's batch layout changed and "
            "the modality contract cannot be evaluated. Refusing to run blind.")
    out = {}
    for name, tensor in dict(data).items():
        try:
            out[name] = tensor.detach().float().cpu().numpy()
        except Exception:
            out[name] = np.asarray(tensor)
    return out


def infer(cfg: S2Config, stub: bool) -> int:
    """STAGE 2. Verify the artifact, then consume it. Cannot encode."""
    from tribe_tools.ledger import Event
    from tribe_tools.s2_pipeline import Stage2Deps, stage2_infer

    _validate_parcel_categories()
    man = _check_inputs(cfg)

    if stub:
        return _infer_stub(cfg, man)

    from tribe_tools import cuda_guard
    from tribe_tools.atlas_preflight import assert_atlas_ready, load_frozen_parcels
    from tribe_tools.durable_store import ArtifactIdentity, require_artifact_location
    from tribe_tools.provenance import exca_infra_version

    cuda_guard.arm(sitehook=str(Path(__file__).resolve().parent.parent
                               / "tribe_tools" / "_s2_sitehook"))
    # B5: parcels are verified and frozen BEFORE the GPU, then passed explicitly.
    assert_atlas_ready(ATLAS_CACHE, parcels=list(ALL_PARCELS))
    parcels = load_frozen_parcels(ATLAS_CACHE)

    identity, uid = _resolve_identity(cfg, man, "consume")
    print(f"=== S2 stage 2: infer === design {cfg.fingerprint()}")
    print(f"  identity   {uid}")

    # W2. The artifact must be RESOLVED BEFORE the consuming model is constructed, and
    # its location must be handed to the extractor as infra.folder. The previous order
    # loaded with cache_folder=None -- which is literally the 2026-08-25 configuration
    # -- resolved the artifact afterwards, and never carried the location back. The
    # extractor pointed at nothing, and sidecar_digests(None) raised TypeError on the
    # first line of stage2_infer.
    #
    # Getting the expected uids needs the events, and the events need a model, so a
    # throwaway probe supplies them. A model construction is seconds against a 4.5 h
    # encode, and deriving the uids from literals instead is how you get a confidently
    # wrong identity.
    probe = _load(cfg, "consume", cache_folder=None,
                  infra_version=exca_infra_version(identity))
    probe_extractor = _video_extractor(probe)
    events_df = probe.get_events_dataframe(video_path=str(video_path()))
    video_events = probe_extractor._event_types_helper.extract(events_df)
    expected = [_item_uid(e) for e in video_events]
    del probe, probe_extractor

    # B2: search every durable location. A Stage 1 that finished in a previous session
    # left its artifact somewhere this search covers; reconstructing the path
    # guarantees a miss and an error whose remedy prescribes another 4h45m.
    ident = ArtifactIdentity(identity, tuple(expected))
    artifact_dir = require_artifact_location(
        ident, search_paths=_search_paths(), reader_factory=_reader_for,
        sidecar_probe=_sidecars_for)
    print(f"  artifact   {artifact_dir}")

    # NOW load the real consuming model, pointed at the artifact we resolved.
    model = _load(cfg, "consume", cache_folder=artifact_dir / "cache",
                  infra_version=exca_infra_version(identity))
    extractor = _video_extractor(model)
    if extractor.infra.folder is None:
        die("the consuming extractor has infra.folder=None -- that is the "
            "configuration that discarded 4h45m on 2026-08-25. Refusing to run.")

    def _read(item_uid):
        return np.asarray(extractor.infra.cache_dict[item_uid])

    out = stage2_infer(
        cfg, identity, artifact_dir, expected,
        Stage2Deps(
            load_model=lambda: model,
            build_events=lambda m: events_df,
            predict=lambda m, e: _predict(m, e),
            read_item=_read,
            analyse=lambda preds, segments: analyse(preds, segments, cfg, man, parcels),
            probe_modalities=_probe_modalities,
            persist=lambda preds, segs: _persist_predictions(
                preds, segs, ARTIFACT_ROOT / uid / "preds.npz"),
            sidecar_probe=lambda: _sidecars_for(artifact_dir),
        ),
        _ledger(),
        required_modalities=REQUIRED_MODALITIES,
        expected_absent=EXPECTED_ABSENT_MODALITIES)

    return _write_report(cfg, man, out, identity, uid, stub=False)


def _predict(model, events):
    from tribe_tools.inference import predict_single
    return predict_single(model, video_path())


def _search_paths() -> list:
    """Every place a durable artifact could be. Read-only mounts included."""
    paths = [ARTIFACT_ROOT]
    for env in ("S2_DURABLE_ROOT", "S2_ARTIFACT_SEARCH"):
        for part in (os.environ.get(env, "") or "").split(os.pathsep):
            if part:
                paths.append(Path(part))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="CPU: render and verify")
    ap.add_argument("--preflight", action="store_true",
                    help="CPU: atlas, identity and inputs. Must pass before any GPU work")
    ap.add_argument("--extract-features", dest="extract", action="store_true",
                    help="STAGE 1 (GPU): encode V-JEPA features into a verified, durable "
                         "artifact, then exit. The ONLY stage permitted to compute")
    ap.add_argument("--infer", action="store_true",
                    help="STAGE 2 (GPU): verify the artifact and run the brain model. "
                         "Extractors run in exca read-only mode, so this stage is "
                         "structurally unable to encode")
    ap.add_argument("--stimulus-root", default=None,
                    help="where floc/ and s2_stimulus.mp4 live. Default 'data'; on "
                         "Kaggle pass /kaggle/input/<dataset-slug>. Env: S2_STIMULUS_ROOT")
    ap.add_argument("--stub", action="store_true",
                    help="with --infer: fake model, CPU only, structure check")
    args = ap.parse_args()
    if args.stimulus_root:
        global STIMULUS_ROOT
        STIMULUS_ROOT = Path(args.stimulus_root)
    if args.prepare and not os.access(STIMULUS_ROOT, os.W_OK):
        die(f"--prepare writes the rendered video into {STIMULUS_ROOT}, which is not "
            "writable. On Kaggle the dataset mount is READ-ONLY: prepare on a machine "
            "you control, upload the result, and use --infer only.")
    if not (args.prepare or args.preflight or args.extract or args.infer):
        ap.error("choose --prepare, --preflight, --extract-features and/or --infer")
    if args.preflight:
        rc = preflight(S2)
        if rc:
            return rc
    if args.prepare:
        prepare(S2)
    if args.extract:
        rc = extract_features(S2, stub=args.stub)
        if rc:
            return rc
    if args.infer:
        return infer(S2, stub=args.stub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
