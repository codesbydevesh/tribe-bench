#!/usr/bin/env python3
"""S2 Kaggle preflight — settle the four Kaggle-only unknowns BEFORE any extraction.

Four facts about the S2 run cannot be established on the dev box, and each one, if
wrong, is discovered only *after* GPU time has been spent:

  1. ``provenance.library_versions()`` renders an unreadable distribution as the
     literal string ``"absent"``.  ``_require`` accepts it (it is neither missing
     nor None), so two different machines collapse to one identity.  On the dev box
     seven of the eight UID distributions are "absent"; nobody knows what Kaggle
     returns until Kaggle is asked.
  2. ``feature_uid_fields`` demands all twelve ``REQUIRED_PREPROCESSING`` values, and
     ``preprocessing_fields`` maps a key the real ``video_preprocessor_config.json``
     does not carry to ``None`` -- which raises.  That raise happens inside
     ``_resolve_identity`` (``scripts/s2_run.py:513``), i.e. AFTER ``_load`` has put
     the brain model on the GPU.
  3. ``ledger.EncodeCounter`` patches ``exca.map.MapInfra._call_and_store``.  If the
     installed exca's signature differs, the only instrument that can tell a cache
     hit from a 4h45m recompute turns from a no-op into a run-killer -- or, worse,
     stays silent at zero, which is also the success value.
  4. Filesystem: the durable root must be writable, ``/kaggle/input`` mounts readable,
     and one filesystem must hold 4.14 GB of V-JEPA weights + 709 MB of TRIBE
     checkpoint + ~250 MB of features.

Plus one more that is cheap while we are here: that ``num_workers`` is still rejected
at the ``TribeExperiment`` root on the installed build, which is what makes the
``data.`` prefix in ``model_config_update()`` load-bearing rather than cosmetic.

    python3 scripts/s2_kaggle_preflight.py            # on Kaggle, before --preflight
    python3 scripts/s2_kaggle_preflight.py --json out.json

No GPU.  No model load.  No multi-gigabyte hashing.  Seconds.

**Three outcomes, not two.**  A check is PASS, FAIL, or SKIP, and SKIP exits non-zero.
That is deliberate: `s2_gate.py` reported GO partly because a boolean short-circuit
treated "the test did not run" as "the test passed", and this project has already
shipped a test that went green over the bug it was written to catch.  An unrun check
is not a passed check.

    exit 0   every check passed
    exit 1   at least one check FAILED
    exit 2   nothing failed but something could not be run here (pass --allow-skip
             to accept that on a dev box; never pass it on the GPU box)

**Every environment check is paired with a control that runs everywhere.**  The
controls break the production code path on purpose and require the guard to notice --
twelve separate mutations of the preprocessing identity, a distribution forced to
"absent", a signature that no longer matches.  A green control on this box proves the
detector detects; a green environment check on Kaggle proves the environment is sound.
Only the pair is evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import typing
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Same convention as tests/conftest.py: a dev box cannot pip install exca (PEP 668),
# so it points at a venv that has the pinned 0.5.20 instead.
for _p in reversed([p for p in os.environ.get("S2_DEV_SITE_PACKAGES", "").split(os.pathsep) if p]):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
# Sizes.  Every one is a measured constant that already lives in the repo, not a
# guess: a headroom check built on a rounded number is a headroom check that lies
# by a few hundred megabytes at the exact moment it matters.
# --------------------------------------------------------------------------- #

#: ``tribe_tools.provenance.VJEPA2_WEIGHTS_BYTES`` -- 4.14 GB, model.safetensors.
VJEPA_WEIGHT_BYTES = 4_138_311_608
#: ``tribe_tools.model.TRIBEV2_CKPT_BYTES`` -- 709 MB, best.ckpt.
TRIBE_CKPT_BYTES = 708_856_138
#: The S2 feature artifact.  Measured at 226 MiB (S2-PRE-GPU-REPORT §4); budgeted at
#: the round 250 MB the operator plans against.
FEATURE_BYTES = 250 * 1024 * 1024
#: Free space that must survive the run.  A filesystem at exactly 0 bytes free does
#: not fail cleanly -- it fails inside a 4.14 GB download, hours in.
DEFAULT_MARGIN = 0.15

VJEPA_REPO = "facebook/vjepa2-vitg-fpc64-256"
TRIBE_REPO = "facebook/tribev2"
PROC_CONFIG = "video_preprocessor_config.json"

#: The twelve flat identity keys, each with the raw-config path that feeds it.
#: ``preprocessing_fields`` never omits a key -- it maps an absent source to None,
#: and None is what ``_require`` rejects.  So a control that must make the guard
#: fire has to remove the SOURCE, which is why this mapping exists.
PREPROCESSING_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("do_resize", ("do_resize",)),
    ("shortest_edge", ("size", "shortest_edge")),
    ("resample", ("resample",)),
    ("do_center_crop", ("do_center_crop",)),
    ("crop_height", ("crop_size", "height")),
    ("crop_width", ("crop_size", "width")),
    ("do_rescale", ("do_rescale",)),
    ("rescale_factor", ("rescale_factor",)),
    ("do_normalize", ("do_normalize",)),
    ("image_mean", ("image_mean",)),
    ("image_std", ("image_std",)),
    ("video_processor_type", ("video_processor_type",)),
)

#: Shape-complete stand-in used ONLY by the controls, so the twelve mutation probes
#: can run on a box that has no copy of the real file.  It is never used to answer
#: unknown 2 -- a control proves the guard bites, it does not prove what Kaggle ships.
CONTROL_RAW_CONFIG = {
    "do_resize": True, "size": {"shortest_edge": 292}, "resample": 2,
    "do_center_crop": True, "crop_size": {"height": 256, "width": 256},
    "do_rescale": True, "rescale_factor": 0.00392156862745098,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225],
    "video_processor_type": "VJEPA2VideoProcessor",
}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Report:
    """Flat list of checks grouped under numbered items."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._item = "0"
        self._title = ""

    def item(self, number: str, title: str) -> None:
        self._item, self._title = number, title
        print(f"\n--- {number}. {title}")

    def _add(self, status: str, name: str, detail: str, env: bool) -> None:
        self.rows.append({"item": self._item, "item_title": self._title,
                          "check": name, "status": status, "detail": detail,
                          "environment_dependent": env})
        tag = "  (environment)" if env else ""
        print(f"  [{status}] {name}{tag}" + (f"\n         {detail}" if detail else ""))

    def ok(self, name: str, detail: str = "", *, env: bool = False) -> None:
        self._add(PASS, name, detail, env)

    def bad(self, name: str, detail: str = "", *, env: bool = False) -> None:
        self._add(FAIL, name, detail, env)

    def skip(self, name: str, detail: str = "") -> None:
        self._add(SKIP, name, detail, True)

    def run(self, name: str, fn, *, env: bool = False) -> bool:
        """Run one behavioural assertion.  Any exception is a FAIL, with its type."""
        try:
            detail = fn()
        except Exception as ex:  # noqa: BLE001 -- an unexpected type is still a failure
            self.bad(name, f"{type(ex).__name__}: {ex}".replace("\n", " ")[:400], env=env)
            return False
        self.ok(name, "" if detail is None else str(detail), env=env)
        return True

    def counts(self) -> dict[str, int]:
        return {s: sum(1 for r in self.rows if r["status"] == s) for s in (PASS, FAIL, SKIP)}


def _raises(exc_types, fn, *, must_mention: str = "") -> str:
    """Assert fn() raises.  Returns the message, so the caller can show it."""
    try:
        fn()
    except exc_types as ex:
        msg = str(ex)
        if must_mention and must_mention not in msg:
            raise AssertionError(
                f"raised {type(ex).__name__} but the message never names {must_mention!r}: "
                f"{msg[:200]}") from None
        return f"{type(ex).__name__}: {msg[:120]}"
    raise AssertionError(
        f"expected {getattr(exc_types, '__name__', exc_types)} and nothing was raised -- "
        "the guard is not live on this code path")


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return str(n)


# --------------------------------------------------------------------------- #
# 1. library_versions() must not answer "absent"
# --------------------------------------------------------------------------- #

def check_versions(rep: Report) -> None:
    rep.item("1", "library_versions() returns real versions, not the literal \"absent\"")
    from tribe_tools.provenance import ABSENT, UID_DISTRIBUTIONS, library_versions

    versions = library_versions()

    def keys_are_the_uid_set() -> str:
        got, want = sorted(versions), sorted(UID_DISTRIBUTIONS)
        if got != want:
            raise AssertionError(f"library_versions() covers {got}, uid needs {want}")
        return f"{len(want)} distributions on the tensor's critical path"

    rep.run("library_versions() covers exactly UID_DISTRIBUTIONS", keys_are_the_uid_set)

    def no_absent() -> str:
        unreadable = sorted(k for k, v in versions.items() if v in (None, "", ABSENT))
        if unreadable:
            raise AssertionError(
                f"{unreadable} render as {ABSENT!r}. Two different environments would "
                "produce the SAME feature identity, so a cache built elsewhere would be "
                "accepted here. Install them before extracting.")
        return "; ".join(f"{k}={v}" for k, v in sorted(versions.items()))

    rep.run("no UID distribution renders as \"absent\"", no_absent, env=True)

    # Control: the string really is what an unreadable distribution produces, and the
    # production guard really does refuse it.  Both run everywhere.
    def control_absent_is_produced() -> str:
        class Blind:
            def version(self, name):
                raise ModuleNotFoundError(name)

            def distribution(self, name):
                raise ModuleNotFoundError(name)

        out = library_versions(("torch",), metadata=Blind())
        if out != {"torch": ABSENT}:
            raise AssertionError(f"an unreadable distribution rendered as {out!r}, not {ABSENT!r}")
        return f"an unreadable distribution renders as {ABSENT!r} -- the value this check hunts"

    rep.run("control: an unreadable distribution really does render as \"absent\"",
            control_absent_is_produced)

    def control_guard_refuses_absent() -> str:
        """`_versions_or_die` is the guard s2_run actually calls (scripts/s2_run.py:419).
        Calling it here rather than reimplementing its test is the whole point: a
        preflight that agrees with a copy of the rule proves nothing about the rule."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("_s2run_probe", REPO / "scripts" / "s2_run.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import tribe_tools.provenance as prov
        real = prov.library_versions
        poisoned = {**{d: "1.0" for d in UID_DISTRIBUTIONS}, "torch": ABSENT}
        prov.library_versions = lambda *a, **k: dict(poisoned)
        # s2_run.die() writes "REFUSING TO RUN" to stderr. That banner is the guard
        # working, but printed loose it reads as this preflight aborting, so capture it
        # and prove it actually named the offender instead.
        noise = io.StringIO()
        try:
            with contextlib.redirect_stderr(noise):
                msg = _raises(SystemExit, mod._versions_or_die)
        finally:
            prov.library_versions = real
        banner = noise.getvalue()
        if "REFUSING TO RUN" not in banner or "torch" not in banner:
            raise AssertionError(
                "the guard exited but never said why on stderr; an operator would see a "
                f"bare exit code. stderr was {banner[:200]!r}")
        clean = {d: "1.0" for d in UID_DISTRIBUTIONS}
        prov.library_versions = lambda *a, **k: dict(clean)
        try:
            got = mod._versions_or_die()
        finally:
            prov.library_versions = real
        if got != clean:
            raise AssertionError(f"the guard mangled a clean version set: {got!r}")
        return f"s2_run._versions_or_die() aborts on one \"absent\" ({msg}) and passes a clean set"

    rep.run("control: the run's own guard aborts on an \"absent\" version",
            control_guard_refuses_absent)


# --------------------------------------------------------------------------- #
# 2. the real video_preprocessor_config.json supplies all 12 required keys
# --------------------------------------------------------------------------- #

def _hf_cache_dir() -> Path:
    try:
        from huggingface_hub import constants  # noqa: PLC0415
        return Path(constants.HF_HUB_CACHE)
    except Exception:
        pass
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_cache_dir(repo_id: str) -> Path:
    return _hf_cache_dir() / ("models--" + repo_id.replace("/", "--"))


def _locate_processor_config(*, offline: bool) -> tuple[dict | None, str]:
    """Cheapest route first.  Returns (config, provenance-string).

    The network route is ON by default.  The file is ~1.3 kB and the whole point of
    this script is to SETTLE unknown 2 before the GPU is touched -- a preflight that
    defaults to not looking hands back the same open question it was written to close.
    ``--offline`` turns it off; a failure to reach the hub is reported as an unrun
    check (SKIP), never as an answer.
    """
    override = os.environ.get("S2_PREPROCESSOR_CONFIG")
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"S2_PREPROCESSOR_CONFIG={p} is not a file")
        return json.loads(p.read_text()), f"S2_PREPROCESSOR_CONFIG={p}"

    snaps = sorted(_repo_cache_dir(VJEPA_REPO).glob(f"snapshots/*/{PROC_CONFIG}"))
    if snaps:
        return json.loads(snaps[-1].read_text()), f"HF cache {snaps[-1]}"

    if offline:
        return None, ("no local copy on disk and --offline was passed. Re-run without it, "
                      "or point S2_PREPROCESSOR_CONFIG at the file.")

    # A dead network must cost seconds, not the default 10-minute socket wait.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from tribe_tools.provenance import VJEPA2_COMMIT  # noqa: PLC0415
        path = Path(hf_hub_download(VJEPA_REPO, PROC_CONFIG, revision=VJEPA2_COMMIT))
    except Exception as ex:  # noqa: BLE001 -- unreachable hub is "unknown", not "bad"
        return None, (f"no local copy, and fetching it failed: {type(ex).__name__}: "
                      f"{str(ex).splitlines()[0][:180]}. Unknown 2 stays open.")
    return json.loads(path.read_text()), f"hub @ {VJEPA2_COMMIT[:12]} -> {path}"


def _drop(raw: dict, path: tuple[str, ...]) -> dict:
    out = json.loads(json.dumps(raw))
    node = out
    for key in path[:-1]:
        node = node.get(key)
        if not isinstance(node, dict):
            return out
    node.pop(path[-1], None)
    return out


def _probe_identity(raw_config: dict) -> dict:
    """Build the identity the way ``s2_run._identity`` does, through the real public
    entry point, with every section EXCEPT preprocessing filled from a fixed probe
    fixture.  So the only thing that can make this raise is the preprocessing config --
    which is precisely unknown 2.  Calling ``feature_uid_fields`` rather than the
    private ``_require`` is deliberate: the question is whether the guard is wired
    into the path Stage 1 takes, not whether the guard exists."""
    from tribe_tools.provenance import (  # noqa: PLC0415
        MODEL_CONFIG_FILENAME, PROCESSOR_CONFIG_FILENAME, UID_DISTRIBUTIONS,
        VJEPA2_COMMIT, VJEPA2_REPO, VJEPA2_WEIGHTS_SHA256, WEIGHTS_FILENAME,
        FileIdentity, WeightIdentity, feature_uid_fields, preprocessing_fields,
    )
    files = {
        WEIGHTS_FILENAME: FileIdentity(WEIGHTS_FILENAME, VJEPA2_WEIGHTS_SHA256,
                                       "sha256", VJEPA_WEIGHT_BYTES, "preflight-probe"),
        MODEL_CONFIG_FILENAME: FileIdentity(MODEL_CONFIG_FILENAME, "0" * 40,
                                            "git-blob-sha1", 1, "preflight-probe"),
        PROCESSOR_CONFIG_FILENAME: FileIdentity(PROCESSOR_CONFIG_FILENAME, "1" * 40,
                                                "git-blob-sha1", 1, "preflight-probe"),
    }
    return feature_uid_fields(
        stimulus={"sha256": "2" * 64, "size_bytes": 1, "duration_s": 1050.0,
                  "fps": 30.0, "width": 256, "height": 256},
        weights=WeightIdentity(VJEPA2_REPO, VJEPA2_COMMIT, files, "preflight-probe"),
        extractor={k: 1 for k in (
            "class", "infra_version", "frequency", "clip_duration",
            "num_frames_effective", "max_imsize", "layer_type", "use_audio",
            "model_name", "pretrained", "imsize", "token_aggregation",
            "cache_all_layers", "cache_n_layers", "layers", "layer_aggregation")},
        chunking={"event_type": "Video", "max_duration": 60, "min_duration": 30},
        preprocessing=preprocessing_fields(raw_config),
        versions={d: "probe" for d in UID_DISTRIBUTIONS},
    )


def check_preprocessing(rep: Report, offline: bool = False) -> None:
    rep.item("2", "video_preprocessor_config.json supplies all 12 REQUIRED_PREPROCESSING keys")
    from tribe_tools.provenance import MissingIdentityField, REQUIRED_PREPROCESSING

    if len(REQUIRED_PREPROCESSING) != 12 or set(REQUIRED_PREPROCESSING) != {
            k for k, _ in PREPROCESSING_SOURCES}:
        rep.bad("this preflight's key map matches REQUIRED_PREPROCESSING",
                f"provenance requires {sorted(REQUIRED_PREPROCESSING)}; this file maps "
                f"{sorted(k for k, _ in PREPROCESSING_SOURCES)}. The controls below would "
                "be testing a different contract than the run enforces.")
        return
    rep.ok("this preflight's key map matches REQUIRED_PREPROCESSING",
           f"all {len(REQUIRED_PREPROCESSING)} keys, each traced to its raw-config source")

    # ---- controls: every one of the twelve is individually load-bearing ----
    def control_all_twelve_bite() -> str:
        for flat, path in PREPROCESSING_SOURCES:
            mutated = _drop(CONTROL_RAW_CONFIG, path)
            _raises(MissingIdentityField, lambda m=mutated: _probe_identity(m),
                    must_mention=flat)
        return (f"removing any one of the {len(PREPROCESSING_SOURCES)} raw sources aborts "
                "feature_uid_fields and the error names the key")

    rep.run("control: each of the 12 keys, removed, aborts identity construction",
            control_all_twelve_bite)

    def control_complete_config_passes() -> str:
        ident = _probe_identity(CONTROL_RAW_CONFIG)
        got = sorted(k.split(".", 1)[1] for k in ident if k.startswith("preprocessing."))
        if got != sorted(REQUIRED_PREPROCESSING):
            raise AssertionError(f"identity carries preprocessing keys {got}")
        return "a shape-complete config yields all 12 preprocessing.* identity fields"

    rep.run("control: a complete config is accepted (the probe is not failing for "
            "an unrelated reason)", control_complete_config_passes)

    # ---- the environment answer ----
    try:
        raw, where = _locate_processor_config(offline=offline)
    except Exception as ex:  # noqa: BLE001
        rep.bad("the real config was located", f"{type(ex).__name__}: {ex}", env=True)
        return
    if raw is None:
        rep.skip("the real config was located", where)
        rep.skip("the real config satisfies all 12 keys",
                 "cannot be answered without the file; this is unknown 2 and it stays open")
        return
    rep.ok("the real config was located", where, env=True)

    def real_config_builds_an_identity() -> str:
        ident = _probe_identity(raw)
        vals = {k.split(".", 1)[1]: v for k, v in ident.items() if k.startswith("preprocessing.")}
        return "; ".join(f"{k}={vals[k]}" for k in sorted(vals))

    rep.run("the real config satisfies all 12 keys (identity construction does not abort)",
            real_config_builds_an_identity, env=True)

    # s2_run._processor_config downloads WITHOUT revision=, i.e. off the floating
    # branch, while the identity claims a pinned commit. Report the drift if we can
    # see both; do not fail on it -- that file is not this one's to change.
    if offline:
        rep.skip("s2_run's unpinned download currently agrees with the pinned commit",
                 "--offline: the floating branch cannot be read without the network")
        return
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        floating = json.loads(Path(hf_hub_download(VJEPA_REPO, PROC_CONFIG)).read_text())
    except Exception as ex:  # noqa: BLE001
        rep.skip("s2_run's unpinned download currently agrees with the pinned commit",
                 f"could not read the floating branch: {type(ex).__name__}: "
                 f"{str(ex).splitlines()[0][:160]}")
        return

    def floating_matches_pin() -> str:
        if floating != raw:
            raise AssertionError(
                "the floating-branch copy differs from the pinned one. "
                "s2_run._processor_config() (scripts/s2_run.py:474) downloads with no "
                "revision=, so the identity would record values the pin does not imply.")
        return "floating branch == pinned commit today"

    rep.run("s2_run's unpinned download currently agrees with the pinned commit",
            floating_matches_pin, env=True)


# --------------------------------------------------------------------------- #
# 3. the installed exca's _call_and_store matches what EncodeCounter patches
# --------------------------------------------------------------------------- #

def _assert_patch_fits(library_fn, patch_fn) -> str:
    """Does ``patch_fn`` accept everything ``library_fn``'s callers pass?

    exca calls ``_call_and_store`` three ways: in-process with positional items and a
    keyword ``use_cache_dict`` (``exca/map.py:467-469``), through a Process/ThreadPool as
    a bound-method submit with the same keywords (``exca/map.py:488-492``), and through
    submitit (``exca/map.py:392-395``).  A live run only exercises the first, and would sail
    straight past a parameter a newer exca ADDED -- while a real caller passing it hits
    TypeError inside our patch, mid-extraction.  So the parameters are compared name by
    name, and then every call shape exca's own source uses is bound against the patch.

    Raises AssertionError on any mismatch; returns a description when they fit.
    """
    sig_lib = inspect.signature(library_fn)
    sig_patch = inspect.signature(patch_fn)
    p_lib = list(sig_lib.parameters.values())[1:]      # drop self
    p_patch = list(sig_patch.parameters.values())[1:]
    if [p.name for p in p_lib] != [p.name for p in p_patch]:
        raise AssertionError(
            f"the library takes {[p.name for p in p_lib]} but EncodeCounter's replacement "
            f"takes {[p.name for p in p_patch]}. A caller using a name the patch does not "
            "accept raises TypeError mid-run -- the counter stops being a harmless no-op "
            "and becomes the thing that kills a 4h45m extraction.")
    for a, b in zip(p_lib, p_patch):
        if a.kind != b.kind:
            raise AssertionError(f"parameter {a.name}: library kind {a.kind}, patch {b.kind}")
        if a.default is not inspect.Parameter.empty and a.default != b.default:
            raise AssertionError(
                f"parameter {a.name}: library default {a.default!r}, patch {b.default!r}")
        if a.default is inspect.Parameter.empty and b.default is not inspect.Parameter.empty:
            raise AssertionError(
                f"parameter {a.name} is required by the library but optional in the patch")
    me = object()
    for args, kwargs in (((me, ["x"]), {"use_cache_dict": True}),
                         ((me, ["x"]), {"use_cache_dict": False}),
                         ((me, ["x"]), {})):
        try:
            sig_patch.bind(*args, **kwargs)
        except TypeError as ex:
            raise AssertionError(
                f"exca calls it as (items{', ' + ', '.join(kwargs) if kwargs else ''}) and "
                f"the patch will not take that call: {ex}") from None
    return f"{sig_lib} -- all three exca call shapes bind to the patch"


def check_encode_counter(rep: Report) -> None:
    rep.item("3", "installed exca MapInfra._call_and_store matches EncodeCounter's patch")
    from tribe_tools.ledger import EncodeCounter

    try:
        from exca import map as emap  # noqa: PLC0415
    except Exception as ex:  # noqa: BLE001
        rep.skip("exca is importable",
                 f"{type(ex).__name__}: {ex}. EncodeCounter.active would be False and its "
                 "count would sit at 0 -- which is ALSO the success value, so the counter "
                 "would certify a recompute as a cache hit. Set S2_DEV_SITE_PACKAGES or "
                 "install exca==0.5.20.")
        for name in ("EncodeCounter reports active",
                     "signature accepts every call shape exca itself uses",
                     "a cold run is counted item-for-item",
                     "a warm cache is counted as zero",
                     "the patch is removed on exit"):
            rep.skip(name, "requires the real exca")
        return

    try:
        import importlib.metadata as _md  # noqa: PLC0415
        ver = _md.version("exca")
    except Exception:
        ver = "unknown"
    rep.ok("exca is importable", f"exca {ver} at {emap.__file__}", env=True)

    orig = emap.MapInfra._call_and_store

    def counter_is_active() -> str:
        with EncodeCounter() as c:
            if not c.active:
                raise AssertionError(
                    "EncodeCounter.active is False with exca importable -- the patch did "
                    "not take. Its count of 0 would be indistinguishable from success.")
            if emap.MapInfra._call_and_store is orig:
                raise AssertionError("MapInfra._call_and_store was never replaced")
        return "the counter binds to the installed MapInfra"

    rep.run("EncodeCounter reports active", counter_is_active, env=True)

    def signature_matches() -> str:
        with EncodeCounter() as c:
            if not c.active:
                raise AssertionError("counter inactive")
            patched = emap.MapInfra._call_and_store
        return _assert_patch_fits(orig, patched)

    rep.run("signature accepts every call shape exca itself uses", signature_matches, env=True)

    # A live end-to-end run.  The "encoder" increments a counter instead of doing 7.7 s
    # of V-JEPA, but MapInfra, its uid, its cache dict and its index are the real library.
    def live_counts() -> str:
        import numpy as np  # noqa: PLC0415
        import pydantic  # noqa: PLC0415
        import exca  # noqa: PLC0415

        tally = {"n": 0}

        def _encode(self, items):
            for i in items:
                tally["n"] += 1
                yield np.full((2, 2), float(ord(i)), dtype=np.float32)

        # This module uses `from __future__ import annotations`, which makes every
        # annotation a STRING -- and exca/map.py:574 reads the raw annotation object and
        # rejects a string outright. Setting the real objects is the fix; there is no
        # behavioural difference for exca, which only checks the origin type.
        _encode.__annotations__ = {"items": typing.Sequence[str],
                                   "return": typing.Iterator[object]}
        _encode.__name__ = "encode"

        class Stand(pydantic.BaseModel):
            tag: str = "kaggle-preflight"
            infra: exca.MapInfra = exca.MapInfra(version="preflight")
            encode = infra.apply(item_uid=lambda x: str(x))(_encode)

        with tempfile.TemporaryDirectory(prefix="s2-preflight-exca-") as td:
            cfg = {"folder": td, "keep_in_ram": False, "version": "preflight"}
            uids = ["a", "b", "c"]

            with EncodeCounter() as cold:
                out = [np.asarray(x) for x in Stand(infra=dict(cfg)).encode(uids)]
            if tally["n"] != len(uids):
                raise AssertionError(f"the stand-in encoded {tally['n']} times, expected {len(uids)}")
            if cold.items != len(uids):
                raise AssertionError(
                    f"EncodeCounter saw {cold.items} items while {tally['n']} were really "
                    "encoded. The counter is not measuring what it claims to.")
            if [float(x.ravel()[0]) for x in out] != [float(ord(u)) for u in uids]:
                raise AssertionError("the patch corrupted the values exca returned")

            before = tally["n"]
            with EncodeCounter() as warm:
                out2 = [np.asarray(x) for x in Stand(infra=dict(cfg)).encode(uids)]
            if tally["n"] != before:
                raise AssertionError(f"a warm cache re-encoded {tally['n'] - before} items")
            if warm.items != 0:
                raise AssertionError(
                    f"EncodeCounter reported {warm.items} on a warm cache. It is counting "
                    "calls, not encodes -- every restart would look like a recompute.")
            if any(np.asarray(a).tolist() != np.asarray(b).tolist() for a, b in zip(out, out2)):
                raise AssertionError("warm read-back differs from the cold values")
        return f"cold={cold.items} encodes (real={before}), warm={warm.items}, values round-trip"

    rep.run("a cold run is counted item-for-item and a warm cache is counted as zero",
            live_counts, env=True)

    def restored() -> str:
        with EncodeCounter():
            pass
        if emap.MapInfra._call_and_store is not orig:
            raise AssertionError(
                "MapInfra._call_and_store was not restored on exit. A leaked patch stacks "
                "on every subsequent context and eventually recurses.")
        return "MapInfra._call_and_store is the library's own function again"

    rep.run("the patch is removed on exit", restored, env=True)

    # Control: the comparison above is the same function, fed a library whose signature
    # has drifted in each of the three ways that actually break EncodeCounter. If any of
    # these passes, the check above is decoration.
    def control_signature_drift_detected() -> str:
        with EncodeCounter() as c:
            if not c.active:
                raise AssertionError("counter inactive")
            patched = emap.MapInfra._call_and_store

        def renamed(self_infra, items, cache=True):
            """the parameter EncodeCounter forwards by keyword, renamed"""

        def added(self_infra, items, use_cache_dict=True, progress=False):
            """a newer exca grows a parameter the patch cannot accept"""

        def reordered(self_infra, use_cache_dict, items=None):
            """the two swapped -- the patch would count use_cache_dict as the items"""

        seen = []
        for drifted in (renamed, added, reordered):
            seen.append(_raises(AssertionError,
                                lambda d=drifted: _assert_patch_fits(d, patched)))
        if _assert_patch_fits(orig, patched) is None:
            raise AssertionError("the comparison returned nothing for a matching pair")
        return f"{len(seen)} drift shapes rejected (rename, added param, reorder)"

    rep.run("control: a drifted signature is detected", control_signature_drift_detected)


# --------------------------------------------------------------------------- #
# 4. filesystem and storage
# --------------------------------------------------------------------------- #

def _existing_ancestor(p: Path) -> Path:
    p = Path(p).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _cached_blob_of_size(repo_id: str, size: int) -> Path | None:
    blobs = _repo_cache_dir(repo_id) / "blobs"
    if not blobs.is_dir():
        return None
    for b in blobs.iterdir():
        try:
            if b.is_file() and b.stat().st_size == size:
                return b
        except OSError:
            continue
    return None


def check_filesystem(rep: Report, margin: float) -> None:
    rep.item("4", "filesystem: durable root writable, /kaggle/input readable, headroom")

    artifact_root = Path(os.environ.get("S2_ARTIFACT_ROOT", REPO / "data" / "s2_features"))
    durable_roots = [Path(part)
                     for env in ("S2_DURABLE_ROOT", "S2_ARTIFACT_SEARCH")
                     for part in (os.environ.get(env, "") or "").split(os.pathsep) if part]

    # ---- writability, proven by writing ----
    def writable(root: Path):
        def _f() -> str:
            # Remember what we had to create, and undo it: a preflight that leaves a
            # directory behind changes the state that the next check reads.
            created: list[Path] = []
            node = root.resolve()
            while not node.exists() and node != node.parent:
                created.append(node)
                node = node.parent
            root.mkdir(parents=True, exist_ok=True)
            probe = root / f".s2-preflight-{os.getpid()}"
            tmp = root / f".s2-preflight-{os.getpid()}.tmp"
            try:
                with open(tmp, "wb") as fh:
                    fh.write(b"s2")
                    fh.flush()
                    os.fsync(fh.fileno())
                # The publish protocol's primitive. A filesystem that cannot rename
                # within itself cannot be written to atomically, and durable_store
                # publishes by rename.
                os.replace(tmp, probe)
                if probe.read_bytes() != b"s2":
                    raise AssertionError("read-back after rename returned different bytes")
                dfd = os.open(root, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
                st = os.statvfs(root)
                return (f"wrote, fsynced and renamed inside {root} "
                        f"(dev={os.stat(root).st_dev}, {human(st.f_bavail * st.f_frsize)} free)")
            finally:
                for f in (tmp, probe):
                    try:
                        f.unlink()
                    except OSError:
                        pass
                for d in created:              # deepest first
                    try:
                        d.rmdir()
                    except OSError:
                        break                  # not empty: something else owns it now
        return _f

    rep.run(f"artifact root is writable and supports atomic rename ({artifact_root})",
            writable(artifact_root), env=True)
    if durable_roots:
        for d in durable_roots:
            rep.run(f"durable root is writable and supports atomic rename ({d})",
                    writable(d), env=True)
    else:
        rep.skip("a durable root is configured",
                 "neither S2_DURABLE_ROOT nor S2_ARTIFACT_SEARCH is set, so a Stage-1 "
                 "artifact survives only as long as the session. On Kaggle set "
                 "S2_DURABLE_ROOT=/kaggle/working/s2 (20 GB, needs Save Version) or use "
                 "the Kaggle-Dataset backend.")

    # ---- cross-filesystem rename: EXDEV is the documented Kaggle failure ----
    def cross_device() -> str:
        root = _existing_ancestor(artifact_root)
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            src = Path(fh.name)
            fh.write(b"s2")
        dst = root / f".s2-preflight-xdev-{os.getpid()}"
        try:
            os.replace(src, dst)
            dst.unlink()
            return f"the default tempdir and {root} are on one filesystem; os.replace works"
        except OSError as ex:
            if ex.errno == errno.EXDEV:
                return (f"EXDEV: the default tempdir is on a DIFFERENT filesystem from {root}. "
                        "Any atomic publish must stage its tempfile inside the destination "
                        "tree, not in /tmp. Informational -- durable_store is expected to.")
            raise
        finally:
            for f in (src, dst):
                try:
                    f.unlink()
                except OSError:
                    pass

    rep.run("os.replace from the default tempdir into the artifact root (EXDEV probe)",
            cross_device, env=True)

    # ---- /kaggle/input mounts ----
    # The override exists so this check can be exercised against a real directory tree
    # off Kaggle. It is announced in the detail line: a check that can be redirected
    # silently is a check that can be satisfied silently.
    kin = Path(os.environ.get("S2_KAGGLE_INPUT", "/kaggle/input"))
    where = "" if kin == Path("/kaggle/input") else f" [S2_KAGGLE_INPUT={kin}]"
    if not kin.is_dir():
        rep.skip("/kaggle/input mounts are readable",
                 f"{kin} does not exist on this box. Simulating it with chmod 555 "
                 "does not reproduce it (and does not deny under root), so this stays "
                 "open until the check runs on Kaggle.")
    else:
        def mounts_readable() -> str:
            mounts = sorted(p for p in kin.iterdir())
            if not mounts:
                raise AssertionError("/kaggle/input exists but is empty -- no dataset is attached")
            notes = []
            for m in mounts:
                if not m.is_dir():
                    raise AssertionError(f"{m} is not a directory")
                if not os.access(m, os.R_OK | os.X_OK):
                    raise AssertionError(f"{m} is not readable/traversable")
                first = None
                for dirpath, _dirs, files in os.walk(m):
                    if files:
                        first = Path(dirpath) / files[0]
                        break
                if first is None:
                    raise AssertionError(f"{m} contains no regular file -- an empty mount reads "
                                         "as 'present' to every existence check")
                with open(first, "rb") as fh:  # actually read a byte
                    if not fh.read(1):
                        raise AssertionError(f"{first} is empty")
                notes.append(f"{m.name} (read {first.name})")
            return ", ".join(notes)

        rep.run(f"/kaggle/input mounts are readable{where}", mounts_readable, env=True)

    # ---- the stimulus root, wherever it points ----
    stim_root = Path(os.environ.get("S2_STIMULUS_ROOT", REPO / "data"))

    def stimulus_readable() -> str:
        if not stim_root.is_dir():
            raise AssertionError(f"{stim_root} is not a directory (S2_STIMULUS_ROOT)")
        if not os.access(stim_root, os.R_OK | os.X_OK):
            raise AssertionError(f"{stim_root} is not readable")
        video = stim_root / "s2_stimulus.mp4"
        if video.is_file():
            with open(video, "rb") as fh:
                fh.read(1)
            return f"{stim_root} readable; s2_stimulus.mp4 present ({human(video.stat().st_size)})"
        return (f"{stim_root} readable; s2_stimulus.mp4 NOT present here -- --prepare has "
                "not run against this root")

    rep.run(f"stimulus root is readable ({stim_root})", stimulus_readable, env=True)

    # ---- headroom, grouped by filesystem ----
    hf_cache = _hf_cache_dir()
    # (download-id, description, bytes, where it lands).  download-id is None for
    # things this run produces rather than fetches.
    needs: list[tuple[tuple | None, str, int, Path]] = [
        ((VJEPA_REPO, VJEPA_WEIGHT_BYTES),
         f"V-JEPA 2 model.safetensors ({VJEPA_REPO})", VJEPA_WEIGHT_BYTES, hf_cache),
        ((TRIBE_REPO, TRIBE_CKPT_BYTES),
         f"TRIBE v2 best.ckpt ({TRIBE_REPO})", TRIBE_CKPT_BYTES, hf_cache),
        (None, "S2 feature artifact", FEATURE_BYTES, artifact_root),
    ]
    for d in durable_roots:
        needs.append((None, "published copy of the feature artifact", FEATURE_BYTES, d))

    # A file already in the cache is not downloaded again, so requiring room for it a
    # second time would refuse a session that is in fact fine.  Size-only matching is
    # deliberately weak evidence -- it decides a budget, never an identity; the bytes
    # are hashed for real by verify_local_weights at Stage 1.
    waived: list[str] = []
    for key in [n[0] for n in needs if n[0] is not None]:
        repo, size = key
        blob = _cached_blob_of_size(repo, size)
        if blob is not None:
            waived.append(f"{repo} has a cached blob of exactly {size:,} B ({blob.name})")
            needs = [n for n in needs if n[0] != key]
    if waived:
        rep.ok("already-downloaded artifacts are not double-counted", "; ".join(waived))

    by_fs: dict[int, dict] = {}
    for _key, what, size, target in needs:
        anc = _existing_ancestor(target)
        try:
            dev = os.stat(anc).st_dev
        except OSError as ex:
            rep.bad(f"headroom target {target} is reachable", f"{type(ex).__name__}: {ex}", env=True)
            continue
        slot = by_fs.setdefault(dev, {"mount": anc, "need": 0, "items": []})
        slot["need"] += size
        slot["items"].append(f"{what} -> {target} ({human(size)})")

    for dev, slot in sorted(by_fs.items()):
        need = int(slot["need"] * (1 + margin))
        mount = slot["mount"]

        def enough(mount=mount, need=need, slot=slot) -> str:
            free = shutil.disk_usage(mount).free
            detail = (f"free {human(free)}, need {human(need)} "
                      f"(+{int(margin * 100)}% margin) :: " + "; ".join(slot["items"]))
            if free < need:
                raise AssertionError(
                    f"only {human(free)} free on the filesystem at {mount}, need "
                    f"{human(need)}. Short by {human(need - free)}. " + "; ".join(slot["items"]))
            return detail

        rep.run(f"headroom on the filesystem at {mount}", enough, env=True)


# --------------------------------------------------------------------------- #
# 5. num_workers is still rejected at the TribeExperiment root
# --------------------------------------------------------------------------- #

def check_num_workers(rep: Report) -> None:
    rep.item("5", "num_workers is still rejected at the TribeExperiment root")

    import importlib.util
    spec = importlib.util.spec_from_file_location("_s2run_probe_nw", REPO / "scripts" / "s2_run.py")
    s2run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s2run)
    payload = s2run.model_config_update()

    def payload_is_dotted() -> str:
        stray = [k for k in payload if not k.startswith("data.")]
        if stray:
            raise AssertionError(
                f"{stray} address the config ROOT. TribeExperiment sets extra='forbid', so "
                "from_pretrained aborts with a pydantic ValidationError.")
        return f"{len(payload)} keys, all under data.*"

    rep.run("model_config_update() addresses only the data subtree", payload_is_dotted)

    def effective_via_real_confdict() -> str:
        """exca's ConfDict is what actually performs the merge inside from_pretrained."""
        from exca import confdict  # noqa: PLC0415
        # tribev2/grids/defaults.py:20,131 -- N_CPUS on Meta's training cluster, frozen
        # into the released checkpoint config.
        merged = confdict.ConfDict({"data": {"num_workers": 20, "batch_size": 8}})
        merged.update(payload)
        got = dict(merged.flat())["data.num_workers"]
        if got != 0:
            raise AssertionError(f"effective data.num_workers is {got}, not 0")
        bare = confdict.ConfDict({"data": {"num_workers": 20, "batch_size": 8}})
        bare.update({"num_workers": 0})
        if dict(bare.flat())["data.num_workers"] != 20:
            raise AssertionError("the bare-key control did not reproduce the 323a65c defect")
        return ("data.num_workers merges to 0; the bare key leaves it at 20 and adds a stray "
                "root key -- the 323a65c defect, reproduced against the real ConfDict")

    try:
        import exca.confdict  # noqa: F401,PLC0415
    except Exception:
        rep.skip("the merge is evaluated against the real exca ConfDict",
                 "exca not importable; set S2_DEV_SITE_PACKAGES")
    else:
        rep.run("the merge is evaluated against the real exca ConfDict",
                effective_via_real_confdict)

    try:
        from tribev2.main import Data, TribeExperiment  # noqa: PLC0415
    except Exception as ex:  # noqa: BLE001
        rep.skip("TribeExperiment.model_fields rejects num_workers at the root",
                 f"tribev2 is not installed here ({type(ex).__name__}). This is the "
                 "installed-build question and it can only be answered on the GPU box.")
        rep.skip("Data.model_fields carries num_workers", "requires tribev2")
        return

    def root_rejects() -> str:
        if "num_workers" in TribeExperiment.model_fields:
            raise AssertionError(
                "TribeExperiment now declares num_workers at the ROOT. The data. prefix in "
                "model_config_update() would then set the wrong field, and the checkpoint's "
                "data.num_workers=20 would still fork 20 CUDA-initialising workers.")
        if TribeExperiment.model_config.get("extra") != "forbid":
            raise AssertionError(
                f"TribeExperiment extra policy is {TribeExperiment.model_config.get('extra')!r}, "
                "not 'forbid' -- a stray root key would now be silently absorbed instead of "
                "aborting loudly.")
        import pydantic  # noqa: PLC0415
        try:
            TribeExperiment.model_validate({"num_workers": 0})
        except pydantic.ValidationError as ve:
            hits = [e for e in ve.errors()
                    if e.get("loc") == ("num_workers",) and e.get("type") == "extra_forbidden"]
            if not hits:
                raise AssertionError(
                    "validation failed, but not because num_workers is forbidden: "
                    f"{[ (e.get('loc'), e.get('type')) for e in ve.errors() ][:6]}") from None
        else:
            raise AssertionError("TribeExperiment accepted a bare num_workers key")
        return "root: extra='forbid' and a bare num_workers raises extra_forbidden"

    rep.run("TribeExperiment.model_fields rejects num_workers at the root", root_rejects, env=True)

    def data_carries_it() -> str:
        if "num_workers" not in Data.model_fields:
            raise AssertionError(
                f"Data no longer declares num_workers; it has {sorted(Data.model_fields)}. "
                "data.num_workers would then be an extra key on a model that forbids extras.")
        import pydantic  # noqa: PLC0415
        try:
            Data.model_validate({"num_workers": 0})
        except pydantic.ValidationError as ve:
            bad = [e for e in ve.errors()
                   if e.get("loc") == ("num_workers",) and e.get("type") == "extra_forbidden"]
            if bad:
                raise AssertionError("Data rejects num_workers as an extra key") from None
        return "Data.model_fields declares num_workers -- the data. prefix reaches a real field"

    rep.run("Data.model_fields carries num_workers", data_carries_it, env=True)


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="also write the full result as JSON")
    ap.add_argument("--allow-skip", action="store_true",
                    help="exit 0 when nothing failed but checks could not be run. For a dev "
                         "box only -- on the GPU box a skipped check is an open unknown")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                    help=f"fractional free-space margin over the raw requirement "
                         f"(default {DEFAULT_MARGIN})")
    ap.add_argument("--offline", action="store_true",
                    help="never touch the network. Item 2 then reports SKIP unless the "
                         "config is already on disk -- it does not guess")
    args = ap.parse_args(argv)

    # Kaggle captures this through a pipe, which switches stdout to block buffering and
    # lets a subprocess's stderr overtake it. An out-of-order log of a go/no-go decision
    # is how a banner from a deliberate control gets read as the tool aborting.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001 -- not a TextIOWrapper; ordering is cosmetic
        pass

    t0 = time.time()
    rep = Report()
    print("=== S2 Kaggle preflight === no GPU, no model load, no large hashing")
    print(f"    python {sys.version.split()[0]}  cwd {Path.cwd()}  repo {REPO}")

    for name, fn in (("library versions", lambda: check_versions(rep)),
                     ("preprocessing config", lambda: check_preprocessing(rep, args.offline)),
                     ("encode counter", lambda: check_encode_counter(rep)),
                     ("filesystem", lambda: check_filesystem(rep, args.margin)),
                     ("num_workers", lambda: check_num_workers(rep))):
        try:
            fn()
        except Exception:  # noqa: BLE001 -- a crashed item must not look like a clean run
            rep.bad(f"the {name} item ran to completion",
                    traceback.format_exc().strip().splitlines()[-1][:300])

    c = rep.counts()
    envfails = [r for r in rep.rows if r["status"] == FAIL and r["environment_dependent"]]
    print("\n" + "=" * 76)
    print(f"  {c[PASS]} passed   {c[FAIL]} FAILED   {c[SKIP]} skipped "
          f"   {time.time() - t0:.1f}s")
    if c[FAIL]:
        print("\n  FAILED:")
        for r in rep.rows:
            if r["status"] == FAIL:
                print(f"    {r['item']}. {r['check']}"
                      + ("   (environment-dependent)" if r["environment_dependent"] else ""))
    if c[SKIP]:
        print("\n  NOT RUN HERE -- each one is still an open unknown:")
        for r in rep.rows:
            if r["status"] == SKIP:
                print(f"    {r['item']}. {r['check']}")
    if envfails and all(r["environment_dependent"] for r in rep.rows if r["status"] == FAIL):
        print("\n  Every failure above is environment-dependent: this box is not the GPU box.")
        print("  The controls, which do not depend on the environment, are the part that")
        print("  proves these checks can fail. Read those first.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"generated": time.time(), "counts": c, "checks": rep.rows}, indent=2))
        print(f"\n  json -> {args.json}")

    if c[FAIL]:
        rc = 1
    elif c[SKIP] and not args.allow_skip:
        rc = 2
    else:
        rc = 0
    print(f"\n  exit {rc}  " + {0: "GO for extraction",
                                1: "NO-GO: a check failed",
                                2: "NO-GO: an unknown is still open (--allow-skip to "
                                   "accept on a dev box)"}[rc])
    print("=" * 76)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
