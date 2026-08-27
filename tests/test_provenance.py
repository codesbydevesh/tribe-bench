"""F5 -- behavioural tests for weight identity and feature-uid binding.

Every test here runs with **no network and no huggingface_hub**.  The Hub is
represented by a fake module injected into ``sys.modules``; the HF cache is
represented by a real directory tree in the layout huggingface_hub actually
writes (``blobs/<digest>`` + ``snapshots/<sha>/<name>`` symlink + ``refs/<branch>``),
so the cache-reading path is exercised for real rather than mocked away.

Nothing here inspects source text.  The properties under test are:

* a change to any tensor-affecting input changes the uid (F5's whole point);
* an *absent* input is an error, never a quietly narrower uid;
* ``allow_network=False`` performs no network operation at all -- proven by
  making every socket constructor and the whole Hub API explode;
* the free blob-filename route and the paid full-hash route agree on honest
  bytes and disagree on dishonest ones (that disagreement IS the residual risk,
  and it is pinned here so nobody claims the fast path is a proof of content).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import sys
import types

import numpy as np
import pytest

from tribe_tools import provenance as P
from tribe_tools.feature_artifact import (
    ArtifactStale, begin_stage1, verify_artifact, write_artifact,
)

HAS_HFH = importlib.util.find_spec("huggingface_hub") is not None

WEIGHTS = P.WEIGHTS_FILENAME
CFG = P.MODEL_CONFIG_FILENAME
PROC = P.PROCESSOR_CONFIG_FILENAME


# --------------------------------------------------------------------------- #
# A real HF cache tree, and a fake Hub
# --------------------------------------------------------------------------- #

def build_cache(root, repo_id=P.VJEPA2_REPO, commit=P.VJEPA2_COMMIT, *,
                files=None, branch="main", symlinks=True, write_refs=True):
    """Materialise the cache layout huggingface_hub writes.

    ``files`` maps filename -> (bytes, digest_name).  ``digest_name`` is the
    blob filename, i.e. the digest the Hub asserted; it is passed separately
    from the content so a test can build a cache whose blob name LIES.
    """
    root = _p(root)
    repo = root / ("models--" + repo_id.replace("/", "--"))
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    snap = repo / "snapshots" / commit
    snap.mkdir(parents=True, exist_ok=True)
    if write_refs:
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / branch).write_text(commit)
    for name, (content, digest) in (files or {}).items():
        blob = repo / "blobs" / digest
        blob.write_bytes(content)
        target = snap / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if symlinks:
            target.symlink_to(blob)
        else:
            # HF_HUB_DISABLE_SYMLINKS=1, or a Kaggle Dataset copy: the digest is
            # no longer recoverable from the filename.
            target.write_bytes(content)
    return root


def _p(x):
    from pathlib import Path
    return Path(x)


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


CFG_BYTES = b'{"hidden_size": 1408, "num_hidden_layers": 40}'
PROC_BYTES = json.dumps({
    "do_resize": True, "size": {"shortest_edge": 292}, "resample": 2,
    "do_center_crop": True, "crop_size": {"height": 256, "width": 256},
    "do_rescale": True, "rescale_factor": 0.00392156862745098,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225],
    "video_processor_type": "VJEPA2VideoProcessor",
}, indent=2).encode()
WEIGHT_BYTES = b"\x00" * 4096  # stand-in; the real one is 4,138,311,608 bytes

CFG_SHA1 = blob_sha1(CFG_BYTES)
PROC_SHA1 = blob_sha1(PROC_BYTES)
WEIGHT_SHA256 = hashlib.sha256(WEIGHT_BYTES).hexdigest()

ALL_FILES = {
    CFG: (CFG_BYTES, CFG_SHA1),
    PROC: (PROC_BYTES, PROC_SHA1),
    WEIGHTS: (WEIGHT_BYTES, WEIGHT_SHA256),
}


class FakeSibling:
    def __init__(self, rfilename, size, blob_id=None, lfs_sha256=None):
        self.rfilename = rfilename
        self.size = size
        self.blob_id = blob_id
        self.lfs = {"sha256": lfs_sha256, "size": size} if lfs_sha256 else None


class FakeModelInfo:
    def __init__(self, sha, siblings):
        self.sha = sha
        self.siblings = siblings


def install_fake_hub(monkeypatch, *, sha=P.VJEPA2_COMMIT, files=ALL_FILES,
                     raises=None, calls=None):
    """Inject a huggingface_hub whose only working entry point is model_info.

    ``hf_hub_download`` and ``snapshot_download`` blow up: no test is allowed to
    pull the 4 GB artifact, and a regression that starts doing so must fail loudly.
    """
    calls = [] if calls is None else calls
    mod = types.ModuleType("huggingface_hub")

    def _no_download(*a, **k):
        raise AssertionError(
            "resolve_weight_identity must never download payload bytes; it called "
            f"a download API with {a!r} {k!r}")

    class FakeApi:
        def model_info(self, repo_id, files_metadata=False, revision=None, **kw):
            calls.append(("model_info", repo_id, files_metadata, revision))
            if raises is not None:
                raise raises
            sibs = []
            for name, (content, digest) in files.items():
                if len(digest) == 64:
                    sibs.append(FakeSibling(name, len(content), lfs_sha256=digest))
                else:
                    sibs.append(FakeSibling(name, len(content), blob_id=digest))
            return FakeModelInfo(sha, sibs)

    mod.HfApi = FakeApi
    mod.hf_hub_download = _no_download
    mod.snapshot_download = _no_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return calls


@pytest.fixture
def no_network(monkeypatch):
    """Make every outbound socket operation raise."""
    def boom(*a, **k):
        raise AssertionError("a network call was attempted")
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    return True


@pytest.fixture(autouse=True)
def _isolate_hf_env(monkeypatch, tmp_path):
    """Never let a developer's real ~/.cache/huggingface answer a test."""
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf-home"))


# --------------------------------------------------------------------------- #
# Identity fixtures
# --------------------------------------------------------------------------- #

def weight_identity(**over) -> P.WeightIdentity:
    files = {
        CFG: P.FileIdentity(CFG, CFG_SHA1, "git-blob-sha1", len(CFG_BYTES)),
        PROC: P.FileIdentity(PROC, PROC_SHA1, "git-blob-sha1", len(PROC_BYTES)),
        WEIGHTS: P.FileIdentity(WEIGHTS, WEIGHT_SHA256, "sha256", len(WEIGHT_BYTES)),
    }
    kw = {"repo_id": P.VJEPA2_REPO, "commit": P.VJEPA2_COMMIT,
          "files": files, "source": "local-cache"}
    kw.update(over)
    return P.WeightIdentity(**kw)


STIMULUS = {"sha256": "a" * 64, "size_bytes": 16_927_130,
            "duration_s": 60.0, "fps": 8.0, "width": 256, "height": 256}
CHUNKING = {"event_type": "Video", "max_duration": 60.0, "min_duration": 30.0}
EXTRACTOR = {
    "class": "neuralset.extractors.video.HuggingFaceVideo",
    "infra_version": "release", "frequency": 2.0, "clip_duration": 4.0,
    "num_frames_effective": 64, "max_imsize": None, "layer_type": "",
    "use_audio": True, "model_name": P.VJEPA2_REPO, "pretrained": True,
    "imsize": None, "token_aggregation": "mean", "cache_all_layers": False,
    "cache_n_layers": 20, "layers": None, "layer_aggregation": None,
}
PREPROCESSING = P.preprocessing_fields(json.loads(PROC_BYTES))
VERSIONS = {
    "tribev2": "0.1.0+gaf58661791a351a448a489042a28f6c37e1c14b7",
    "neuralset": "0.0.2", "exca": "0.5.20", "torch": "2.6.0+cu124",
    "torchvision": "0.21.0", "transformers": "4.53.0", "moviepy": "2.2.1",
    "numpy": "2.2.6",
}


def fields(**over) -> dict:
    kw = dict(stimulus=dict(STIMULUS), weights=weight_identity(),
              extractor=dict(EXTRACTOR), chunking=dict(CHUNKING),
              preprocessing=dict(PREPROCESSING), versions=dict(VERSIONS))
    kw.update(over)
    return P.feature_uid_fields(**kw)


def _perturb(value):
    """A different value of a compatible kind, for any rendered field."""
    if value in ("true", "false"):
        return "false" if value == "true" else "true"
    if value == "null":
        return "0"
    try:
        return str(int(value) + 1)
    except ValueError:
        pass
    try:
        return repr(float(value) + 1.0)
    except ValueError:
        pass
    return value + "-CHANGED"


# =========================================================================== #
# 1. The uid contract -- these must NEVER skip
# =========================================================================== #

def test_baseline_identity_is_complete_and_stable():
    a, b = fields(), fields()
    assert a == b
    assert P.feature_set_uid(a) == P.feature_set_uid(b)
    assert P.feature_set_uid(a).startswith("s2v1-")


def test_identity_change_produces_a_different_uid():
    base = fields()
    moved = fields(stimulus={**STIMULUS, "sha256": "b" * 64})
    assert base != moved
    assert P.feature_set_uid(base) != P.feature_set_uid(moved)


def test_every_single_field_independently_changes_the_uid():
    """The property that makes the uid worth having.

    A field that can move without moving the uid is a field the cache cannot see
    -- which is precisely F5 (the weights), G3 (num_frames), G5 (preprocessing)
    and G11 (chunking).  So: perturb each key on its own, and demand a new uid.
    """
    base = fields()
    base_uid = P.feature_set_uid(base)
    unmoved = []
    for key, value in base.items():
        mutated = dict(base)
        mutated[key] = _perturb(value)
        assert mutated[key] != value, f"test bug: {key} was not perturbed"
        if P.feature_set_uid(mutated) == base_uid:
            unmoved.append(key)
    assert not unmoved, f"these fields do not affect the uid: {unmoved}"


def test_the_uid_covers_the_specific_gaps_f5_was_about():
    """Named, so a future refactor that drops one of them fails by name."""
    base = fields()
    for key in (f"weights.file.{WEIGHTS}.digest",
                f"weights.file.{PROC}.digest",
                f"weights.file.{CFG}.digest",
                "weights.commit",
                "stimulus.sha256",
                "extractor.num_frames_effective",
                "chunking.max_duration", "chunking.min_duration",
                "preprocessing.image_mean", "preprocessing.crop_height",
                "preprocessing.shortest_edge", "preprocessing.rescale_factor",
                "versions.transformers", "versions.neuralset", "versions.tribev2"):
        assert key in base, f"{key} is not in the identity"


def test_key_order_does_not_change_the_uid():
    base = fields()
    shuffled = dict(reversed(list(base.items())))
    assert list(shuffled) != list(base)
    assert P.feature_set_uid(shuffled) == P.feature_set_uid(base)


@pytest.mark.parametrize("section,key", (
    [("stimulus", k) for k in P.REQUIRED_STIMULUS]
    + [("chunking", k) for k in P.REQUIRED_CHUNKING]
    + [("extractor", k) for k in P.REQUIRED_EXTRACTOR]
    + [("preprocessing", k) for k in P.REQUIRED_PREPROCESSING]
    + [("versions", k) for k in P.UID_DISTRIBUTIONS]
))
def test_a_missing_field_is_an_error_not_a_silent_omission(section, key):
    src = {"stimulus": STIMULUS, "chunking": CHUNKING, "extractor": EXTRACTOR,
           "preprocessing": PREPROCESSING, "versions": VERSIONS}[section]
    pruned = {k: v for k, v in src.items() if k != key}
    with pytest.raises(P.MissingIdentityField) as ei:
        fields(**{section: pruned})
    assert key in str(ei.value)
    assert section in str(ei.value)


@pytest.mark.parametrize("section,key", [
    ("stimulus", "sha256"), ("stimulus", "fps"),
    ("extractor", "num_frames_effective"), ("extractor", "infra_version"),
    ("preprocessing", "image_std"), ("chunking", "max_duration"),
    ("versions", "transformers"),
])
def test_none_in_a_non_nullable_field_is_an_error(section, key):
    src = {"stimulus": STIMULUS, "chunking": CHUNKING, "extractor": EXTRACTOR,
           "preprocessing": PREPROCESSING, "versions": VERSIONS}[section]
    with pytest.raises(P.MissingIdentityField) as ei:
        fields(**{section: {**src, key: None}})
    assert key in str(ei.value)


def test_nullable_fields_are_allowed_and_still_key_the_uid():
    """``max_imsize=None`` is legitimate; it must still be distinguishable from 224."""
    a = fields()
    b = fields(extractor={**EXTRACTOR, "max_imsize": 224})
    assert a["extractor.max_imsize"] == "null"
    assert P.feature_set_uid(a) != P.feature_set_uid(b)


@pytest.mark.parametrize("section,key,other", [
    # 1/255 vs 1/255.0001 -- both render as "0.003922" under %.6f
    ("preprocessing", "rescale_factor", 1.0 / 255.0001),
    # both render as "60.000000" under %.6f
    ("stimulus", "duration_s", 60.0000001),
    ("stimulus", "fps", 8.0000001),
])
def test_float_precision_is_not_collapsed(section, key, other):
    """Rounding a float into the uid loses distinctions the tensors keep.

    Each pair below is chosen so that ``"%.6f"`` renders both sides identically:
    a formatter that rounds would fuse two genuinely different preprocessing
    settings into one cache entry.
    """
    src = {"stimulus": STIMULUS, "preprocessing": PREPROCESSING}[section]
    a = fields()
    b = fields(**{section: {**src, key: other}})
    assert "%.6f" % src[key] == "%.6f" % other, "test bug: %.6f already separates these"
    assert a[f"{section}.{key}"] != b[f"{section}.{key}"]
    assert P.feature_set_uid(a) != P.feature_set_uid(b)
    assert P.exca_infra_version(a) != P.exca_infra_version(b)


def test_a_repo_id_string_is_not_an_acceptable_weight_identity():
    """The repo *name* is what exca already has, and it is what F5 is about."""
    with pytest.raises(TypeError):
        fields(weights=P.VJEPA2_REPO)


def test_weight_identity_missing_the_processor_config_is_an_error():
    partial = weight_identity(files={
        CFG: P.FileIdentity(CFG, CFG_SHA1, "git-blob-sha1", len(CFG_BYTES)),
        WEIGHTS: P.FileIdentity(WEIGHTS, WEIGHT_SHA256, "sha256", len(WEIGHT_BYTES)),
    })
    with pytest.raises(P.MissingIdentityField) as ei:
        fields(weights=partial)
    assert PROC in str(ei.value)


def test_caller_supplied_extra_fields_are_kept_not_dropped():
    extra = fields(extractor={**EXTRACTOR, "future_knob": "on"})
    assert extra["extractor.future_knob"] == "on"
    assert P.feature_set_uid(extra) != P.feature_set_uid(fields())


def test_a_commit_that_is_not_a_sha_is_rejected():
    """``main`` is a pointer, not an identity."""
    with pytest.raises(ValueError):
        weight_identity(commit="main")


def test_unrenderable_value_is_refused_rather_than_stringified():
    with pytest.raises(TypeError):
        fields(extractor={**EXTRACTOR, "layers": {"a": 1}})


def test_nan_is_refused():
    with pytest.raises(ValueError):
        fields(stimulus={**STIMULUS, "fps": float("nan")})


# =========================================================================== #
# 2. The consumer contract: feature_artifact.verify_artifact
# =========================================================================== #

def _artifact(tmp_path, identity):
    arrays = {"stim_0.00_60.00": np.arange(6, dtype=np.float32).reshape(2, 3)}
    begin_stage1(tmp_path)
    write_artifact(tmp_path, identity, arrays)
    return arrays


def test_identity_dict_round_trips_through_the_artifact_manifest(tmp_path):
    ident = fields()
    arrays = _artifact(tmp_path, ident)
    man = verify_artifact(tmp_path, ident, list(arrays), lambda u: arrays[u])
    assert man["identity"] == ident


def test_a_weight_change_makes_a_prior_artifact_stale(tmp_path):
    """The end-to-end point of F5: new weights must not be served old features."""
    arrays = _artifact(tmp_path, fields())
    moved = weight_identity(files={
        CFG: P.FileIdentity(CFG, CFG_SHA1, "git-blob-sha1", len(CFG_BYTES)),
        PROC: P.FileIdentity(PROC, PROC_SHA1, "git-blob-sha1", len(PROC_BYTES)),
        WEIGHTS: P.FileIdentity(WEIGHTS, "f" * 64, "sha256", len(WEIGHT_BYTES)),
    })
    with pytest.raises(ArtifactStale) as ei:
        verify_artifact(tmp_path, fields(weights=moved), list(arrays), lambda u: arrays[u])
    assert f"weights.file.{WEIGHTS}.digest" in str(ei.value)


def test_a_preprocessing_change_makes_a_prior_artifact_stale(tmp_path):
    arrays = _artifact(tmp_path, fields())
    zoomed = {**PREPROCESSING, "shortest_edge": 256}
    with pytest.raises(ArtifactStale) as ei:
        verify_artifact(tmp_path, fields(preprocessing=zoomed),
                        list(arrays), lambda u: arrays[u])
    assert "preprocessing.shortest_edge" in str(ei.value)


def test_dropping_a_field_from_the_checker_is_refused_by_the_artifact(tmp_path):
    """Belt and braces: even if a future caller narrows the identity, the
    artifact's extra keys make it stale rather than a false accept."""
    arrays = _artifact(tmp_path, fields())
    narrowed = {k: v for k, v in fields().items() if k != "versions.transformers"}
    with pytest.raises(ArtifactStale) as ei:
        verify_artifact(tmp_path, narrowed, list(arrays), lambda u: arrays[u])
    assert "versions.transformers" in str(ei.value)


# =========================================================================== #
# 3. exca_infra_version
# =========================================================================== #

def test_exca_infra_version_is_stable_for_identical_inputs():
    assert P.exca_infra_version(fields()) == P.exca_infra_version(fields())


def test_exca_infra_version_differs_for_any_change():
    base = fields()
    v0 = P.exca_infra_version(base)
    unmoved = []
    for key, value in base.items():
        mutated = dict(base)
        mutated[key] = _perturb(value)
        if P.exca_infra_version(mutated) == v0:
            unmoved.append(key)
    assert not unmoved, f"infra.version blind to: {unmoved}"


def test_exca_infra_version_names_the_weight_sha_in_clear():
    """An operator reading the cache path must be able to see which weights it is."""
    v = P.exca_infra_version(fields())
    assert WEIGHT_SHA256[:12] in v
    assert v.startswith("release+vjepa2-")


def test_exca_infra_version_base_is_configurable_and_changes_the_value():
    a = P.exca_infra_version(fields())
    b = P.exca_infra_version(fields(), base="release")
    c = P.exca_infra_version(fields(), base="rc1")
    assert a == b != c


def test_exca_infra_version_is_short_and_path_safe():
    """exca formats it into a directory name (exca/base.py:143)."""
    v = P.exca_infra_version(fields())
    assert len(v) <= 64
    assert not set(v) & set("/\\ \t\n:*?\"<>|")


def test_exca_infra_version_accepts_a_bare_weight_identity():
    a = P.exca_infra_version(weight_identity())
    b = P.exca_infra_version(weight_identity())
    c = P.exca_infra_version(weight_identity(commit="0" * 40))
    assert a == b != c
    assert WEIGHT_SHA256[:12] in a


def test_exca_infra_version_refuses_a_mapping_that_is_not_an_identity():
    with pytest.raises(P.MissingIdentityField):
        P.exca_infra_version({"whatever": "1"})
    with pytest.raises(TypeError):
        P.exca_infra_version(["not", "a", "mapping"])


# =========================================================================== #
# 4. resolve_weight_identity
# =========================================================================== #

def test_local_cache_yields_every_digest_for_free(tmp_path, no_network):
    cache = build_cache(tmp_path / "hub", files=ALL_FILES)
    ident = P.resolve_weight_identity(allow_network=False, cache_dir=cache)
    assert ident.commit == P.VJEPA2_COMMIT
    assert ident.source == "local-cache"
    assert ident.weights_sha256 == WEIGHT_SHA256
    assert ident.processor_config_digest == PROC_SHA1
    assert ident.model_config_digest == CFG_SHA1
    assert ident.file(WEIGHTS).size_bytes == len(WEIGHT_BYTES)
    assert ident.file(WEIGHTS).algo == "sha256"
    assert ident.file(CFG).algo == "git-blob-sha1"


def test_allow_network_false_makes_no_network_call(tmp_path, monkeypatch, no_network):
    """Not 'unlikely to' -- *cannot*.  The Hub API is booby-trapped and every
    socket constructor raises; the call must still succeed off the cache."""
    calls = install_fake_hub(monkeypatch, raises=AssertionError("network used"))
    cache = build_cache(tmp_path / "hub", files=ALL_FILES)
    ident = P.resolve_weight_identity(allow_network=False, cache_dir=cache)
    assert ident.weights_sha256 == WEIGHT_SHA256
    assert calls == []


def _watch_hub_imports(monkeypatch):
    """Record every attempt to import huggingface_hub, however it is spelled."""
    import importlib as _il

    seen = []
    real = _il.import_module

    def watch(name, package=None):
        if name.split(".")[0] == "huggingface_hub":
            seen.append(name)
        return real(name, package)

    monkeypatch.setattr(_il, "import_module", watch)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    return seen


def test_allow_network_false_does_not_even_import_huggingface_hub(tmp_path, monkeypatch,
                                                                  no_network):
    seen = _watch_hub_imports(monkeypatch)
    cache = build_cache(tmp_path / "hub", files=ALL_FILES)
    P.resolve_weight_identity(allow_network=False, cache_dir=cache)
    assert seen == []
    assert "huggingface_hub" not in sys.modules


def test_the_import_watcher_is_not_vacuous(tmp_path, monkeypatch):
    """Positive control for the test above: with network permitted, the very
    same watcher DOES see the import.  Without this, a watcher wired to the
    wrong hook would make the offline claim meaningless."""
    seen = _watch_hub_imports(monkeypatch)
    install_fake_hub(monkeypatch)
    P.resolve_weight_identity(allow_network=True, cache_dir=tmp_path / "empty")
    assert seen == ["huggingface_hub"]


def test_a_cache_miss_without_network_is_a_typed_error_not_a_guess(tmp_path, no_network):
    cache = build_cache(tmp_path / "hub", files={CFG: ALL_FILES[CFG]})
    with pytest.raises(P.WeightIdentityUnavailable) as ei:
        P.resolve_weight_identity(allow_network=False, cache_dir=cache)
    assert WEIGHTS in str(ei.value)
    assert "allow_network=True" in str(ei.value)


def test_an_empty_cache_without_network_is_a_typed_error(tmp_path, no_network):
    with pytest.raises(P.WeightIdentityUnavailable):
        P.resolve_weight_identity(allow_network=False, cache_dir=tmp_path / "nothing")


def test_metadata_api_supplies_identity_and_never_downloads(tmp_path, monkeypatch):
    calls = install_fake_hub(monkeypatch)
    ident = P.resolve_weight_identity(allow_network=True, cache_dir=tmp_path / "empty")
    assert ident.commit == P.VJEPA2_COMMIT
    assert ident.source == "hub-api"
    assert ident.weights_sha256 == WEIGHT_SHA256
    assert ident.file(WEIGHTS).size_bytes == len(WEIGHT_BYTES)
    assert [c[0] for c in calls] == ["model_info"]
    assert calls[0][2] is True, "files_metadata must be requested, else no digests"


def test_a_partially_cached_repo_uses_the_cache_for_what_it_has(tmp_path, monkeypatch):
    install_fake_hub(monkeypatch)
    cache = build_cache(tmp_path / "hub", files={CFG: ALL_FILES[CFG], PROC: ALL_FILES[PROC]})
    ident = P.resolve_weight_identity(allow_network=True, cache_dir=cache)
    assert ident.source == "hub-api+local-cache"
    assert ident.file(CFG).source == "local-cache"
    assert ident.file(WEIGHTS).source == "hub-api"


def test_expected_commit_mismatch_is_fatal(tmp_path, no_network):
    cache = build_cache(tmp_path / "hub", files=ALL_FILES)
    with pytest.raises(P.WeightMismatch) as ei:
        P.resolve_weight_identity(allow_network=False, cache_dir=cache,
                                  expected_commit="0" * 40)
    assert P.VJEPA2_COMMIT in str(ei.value)


def test_expected_commit_match_passes(tmp_path, no_network):
    cache = build_cache(tmp_path / "hub", files=ALL_FILES)
    ident = P.resolve_weight_identity(allow_network=False, cache_dir=cache,
                                      expected_commit=P.VJEPA2_COMMIT)
    assert ident.commit == P.VJEPA2_COMMIT


def test_branch_moving_under_a_partially_cached_repo_is_detected(tmp_path, monkeypatch):
    """The live F5 failure mode: cache holds one commit, `main` now points elsewhere."""
    install_fake_hub(monkeypatch, sha="1" * 40)
    cache = build_cache(tmp_path / "hub", files={CFG: ALL_FILES[CFG]})
    with pytest.raises(P.WeightMismatch) as ei:
        P.resolve_weight_identity(allow_network=True, cache_dir=cache)
    assert "branch moved" in str(ei.value)


def test_revision_by_sha_resolves_without_a_refs_entry(tmp_path, no_network):
    """`snapshot_download(revision=<sha>)` writes snapshots/ but no refs/
    (file_download.py:706), so resolution by sha must not depend on refs."""
    cache = build_cache(tmp_path / "hub", files=ALL_FILES, write_refs=False)
    ident = P.resolve_weight_identity(allow_network=False, cache_dir=cache,
                                      revision=P.VJEPA2_COMMIT)
    assert ident.commit == P.VJEPA2_COMMIT
    with pytest.raises(P.WeightIdentityUnavailable):
        P.resolve_weight_identity(allow_network=False, cache_dir=cache)  # 'main' absent


def test_a_dereferenced_cache_loses_the_free_digest_and_says_so(tmp_path, no_network):
    """Copying a cache through something that follows symlinks (a Kaggle Dataset)
    destroys the digest.  That must be a typed error, not a fabricated digest."""
    cache = build_cache(tmp_path / "hub", files=ALL_FILES, symlinks=False)
    with pytest.raises(P.WeightIdentityUnavailable) as ei:
        P.resolve_weight_identity(allow_network=False, cache_dir=cache)
    assert WEIGHTS in str(ei.value)


def test_a_dereferenced_cache_recovers_via_the_metadata_api(tmp_path, monkeypatch):
    install_fake_hub(monkeypatch)
    cache = build_cache(tmp_path / "hub", files=ALL_FILES, symlinks=False)
    ident = P.resolve_weight_identity(allow_network=True, cache_dir=cache)
    assert ident.weights_sha256 == WEIGHT_SHA256


def test_a_hub_error_is_a_typed_error_with_a_remedy(tmp_path, monkeypatch):
    install_fake_hub(monkeypatch, raises=OSError("connection reset"))
    with pytest.raises(P.WeightIdentityUnavailable) as ei:
        P.resolve_weight_identity(allow_network=True, cache_dir=tmp_path / "empty")
    assert "allow_network=False" in str(ei.value)


def test_huggingface_hub_absent_and_no_cache_is_a_typed_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(P.WeightIdentityUnavailable) as ei:
        P.resolve_weight_identity(allow_network=True, cache_dir=tmp_path / "empty")
    assert "huggingface_hub" in str(ei.value)


def test_a_hub_listing_without_digests_is_refused(tmp_path, monkeypatch):
    install_fake_hub(monkeypatch, files={CFG: ALL_FILES[CFG]})
    with pytest.raises(P.WeightIdentityUnavailable) as ei:
        P.resolve_weight_identity(allow_network=True, cache_dir=tmp_path / "empty")
    assert WEIGHTS in str(ei.value)


def test_cache_dir_defaults_to_the_hf_environment(tmp_path, monkeypatch, no_network):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    build_cache(tmp_path / "hub", files=ALL_FILES)
    ident = P.resolve_weight_identity(allow_network=False)
    assert ident.weights_sha256 == WEIGHT_SHA256


# =========================================================================== #
# 5. verify_local_weights -- the two routes
# =========================================================================== #

def _snapshot(tmp_path, files=ALL_FILES, **kw):
    cache = build_cache(tmp_path / "hub", files=files, **kw)
    return cache / ("models--" + P.VJEPA2_REPO.replace("/", "--"))


def test_blob_filename_route_reads_no_payload_bytes(tmp_path, monkeypatch):
    """Proven by making hashing itself impossible."""
    repo = _snapshot(tmp_path)
    ident = weight_identity()

    def explode(*a, **k):
        raise AssertionError("the fast path must not hash the 4 GB artifact")

    monkeypatch.setattr(P, "digest_file", explode)
    res = P.verify_local_weights(repo, ident)
    assert res.route == "blob-filename"
    assert res.hashed is False
    assert res.digest == WEIGHT_SHA256


def test_the_two_routes_agree_on_honest_bytes(tmp_path):
    repo = _snapshot(tmp_path)
    ident = weight_identity()
    for name in (WEIGHTS, CFG, PROC):
        fast = P.verify_local_weights(repo, ident, filename=name)
        slow = P.verify_local_weights(repo, ident, filename=name, force_hash=True)
        assert fast.route == "blob-filename" and slow.route == "full-hash"
        assert fast.digest == slow.digest == ident.file(name).digest
        assert fast.size_bytes == slow.size_bytes


def test_full_hash_catches_what_the_fast_path_cannot(tmp_path):
    """THE residual risk, pinned.

    A blob whose *name* is the expected digest but whose *content* is not: the
    Hub never verified it and neither does huggingface_hub.  The fast path
    accepts it (it is trusting the name); only force_hash measures.
    """
    lying = dict(ALL_FILES)
    lying[WEIGHTS] = (b"\xff" * len(WEIGHT_BYTES), WEIGHT_SHA256)  # same size, wrong bytes
    repo = _snapshot(tmp_path, files=lying)
    ident = weight_identity()

    assert P.verify_local_weights(repo, ident).route == "blob-filename"
    with pytest.raises(P.WeightMismatch) as ei:
        P.verify_local_weights(repo, ident, force_hash=True)
    assert "does not check this for you" in str(ei.value)


def test_a_wrong_blob_name_is_caught_without_hashing(tmp_path, monkeypatch):
    wrong = dict(ALL_FILES)
    wrong[WEIGHTS] = (WEIGHT_BYTES, "9" * 64)
    repo = _snapshot(tmp_path, files=wrong)
    monkeypatch.setattr(P, "digest_file",
                        lambda *a, **k: pytest.fail("should not hash"))
    with pytest.raises(P.WeightMismatch) as ei:
        P.verify_local_weights(repo, weight_identity())
    assert "9" * 64 in str(ei.value)


def test_a_size_mismatch_short_circuits_before_any_hashing(tmp_path, monkeypatch):
    truncated = dict(ALL_FILES)
    truncated[WEIGHTS] = (WEIGHT_BYTES[:10], WEIGHT_SHA256)
    repo = _snapshot(tmp_path, files=truncated)
    monkeypatch.setattr(P, "digest_file",
                        lambda *a, **k: pytest.fail("size check must come first"))
    with pytest.raises(P.WeightMismatch) as ei:
        P.verify_local_weights(repo, weight_identity(), force_hash=True)
    assert "bytes" in str(ei.value)


def test_a_dereferenced_file_falls_back_to_hashing_and_still_verifies(tmp_path):
    repo = _snapshot(tmp_path, symlinks=False)
    res = P.verify_local_weights(repo, weight_identity())
    assert res.route == "full-hash"
    assert res.digest == WEIGHT_SHA256


def test_a_dereferenced_file_with_wrong_content_is_caught(tmp_path):
    bad = dict(ALL_FILES)
    bad[WEIGHTS] = (b"\x01" * len(WEIGHT_BYTES), WEIGHT_SHA256)
    repo = _snapshot(tmp_path, files=bad, symlinks=False)
    with pytest.raises(P.WeightMismatch):
        P.verify_local_weights(repo, weight_identity())


def test_verify_accepts_the_file_path_directly(tmp_path):
    repo = _snapshot(tmp_path)
    path = repo / "snapshots" / P.VJEPA2_COMMIT / WEIGHTS
    res = P.verify_local_weights(path, weight_identity())
    assert res.route == "blob-filename"


def test_verify_accepts_a_snapshot_directory(tmp_path):
    repo = _snapshot(tmp_path)
    snap = repo / "snapshots" / P.VJEPA2_COMMIT
    assert P.verify_local_weights(snap, weight_identity()).digest == WEIGHT_SHA256


def test_a_missing_artifact_is_a_typed_error(tmp_path):
    with pytest.raises(P.WeightFileMissing) as ei:
        P.verify_local_weights(tmp_path / "nowhere", weight_identity())
    assert "Nothing was verified" in str(ei.value)


def test_verify_accepts_a_plain_mapping_but_demands_a_size(tmp_path):
    repo = _snapshot(tmp_path)
    ok = P.verify_local_weights(
        repo, {"filename": WEIGHTS, "digest": WEIGHT_SHA256,
               "size_bytes": len(WEIGHT_BYTES)})
    assert ok.digest == WEIGHT_SHA256
    with pytest.raises(P.MissingIdentityField):
        P.verify_local_weights(repo, {"filename": WEIGHTS, "digest": WEIGHT_SHA256})


def test_verify_of_an_unknown_filename_names_what_it_has(tmp_path):
    repo = _snapshot(tmp_path)
    with pytest.raises(P.MissingIdentityField) as ei:
        P.verify_local_weights(repo, weight_identity(), filename="pytorch_model.bin")
    assert WEIGHTS in str(ei.value)


def test_git_blob_sha1_matches_the_real_hf_blob_naming():
    """Cross-checked against a real cached V-JEPA config.json: 801 bytes ->
    3534852408cef7f5c0c54dfed6e0842c24492863."""
    assert P.git_blob_sha1_file.__doc__
    data = b"hello\n"
    import tempfile, os as _os
    fd, p = tempfile.mkstemp()
    with _os.fdopen(fd, "wb") as f:
        f.write(data)
    try:
        assert P.git_blob_sha1_file(p) == hashlib.sha1(b"blob 6\0" + data).hexdigest()
        assert P.sha256_file(p) == hashlib.sha256(data).hexdigest()
        assert P.digest_file(p, "sha256") == P.sha256_file(p)
        with pytest.raises(ValueError):
            P.digest_file(p, "md5")
    finally:
        _os.unlink(p)


# =========================================================================== #
# 6. library_versions
# =========================================================================== #

class FakeDist:
    def __init__(self, direct_url=None):
        self._du = direct_url

    def read_text(self, name):
        return json.dumps(self._du) if (name == "direct_url.json" and self._du) else None


class FakeMetadata:
    """Stands in for importlib.metadata."""
    def __init__(self, versions, dists=None):
        self._v = versions
        self._d = dists or {}

    def version(self, name):
        try:
            return self._v[name]
        except KeyError:
            raise ModuleNotFoundError(name) from None

    def distribution(self, name):
        if name not in self._v:
            raise ModuleNotFoundError(name)
        return self._d.get(name, FakeDist())


def test_library_versions_reads_metadata_not_dunder_version():
    """`neuralset` has no ``__version__`` -- its module ``__getattr__`` raises --
    so the ``getattr(mod, '__version__', 'unknown')`` idiom yields 'unknown'."""
    md = FakeMetadata({"neuralset": "0.0.2", "exca": "0.5.20"})
    out = P.library_versions(("neuralset", "exca"), metadata=md)
    assert out == {"neuralset": "0.0.2", "exca": "0.5.20"}
    assert "unknown" not in out.values()


def test_a_vcs_install_contributes_its_commit():
    """``importlib.metadata.version('tribev2')`` is the permanent constant
    '0.1.0'; only the PEP 610 commit distinguishes two builds."""
    sha_a = "af58661791a351a448a489042a28f6c37e1c14b7"
    sha_b = "34f52344e5ba96660fac877393e1954e399d3ef3"
    def md(sha):
        return FakeMetadata({"tribev2": "0.1.0"},
                            {"tribev2": FakeDist({"vcs_info": {"vcs": "git", "commit_id": sha}})})
    a = P.library_versions(("tribev2",), metadata=md(sha_a))
    b = P.library_versions(("tribev2",), metadata=md(sha_b))
    assert a["tribev2"] == f"0.1.0+g{sha_a}"
    assert a != b
    assert (P.feature_set_uid(fields(versions={**VERSIONS, **a}))
            != P.feature_set_uid(fields(versions={**VERSIONS, **b})))


def test_an_absent_distribution_is_recorded_not_dropped():
    md = FakeMetadata({"exca": "0.5.20"})
    out = P.library_versions(("exca", "moviepy"), metadata=md)
    assert out["moviepy"] == P.ABSENT
    installed = P.library_versions(("exca", "moviepy"),
                                   metadata=FakeMetadata({"exca": "0.5.20", "moviepy": "2.2.1"}))
    assert out != installed


def test_distribution_provenance_records_revision_and_editable():
    md = FakeMetadata(
        {"tribev2": "0.1.0", "tribe-bench": "0.1.0"},
        {"tribev2": FakeDist({"url": "git+https://github.com/facebookresearch/tribev2.git",
                              "vcs_info": {"vcs": "git", "commit_id": "a" * 40,
                                           "requested_revision": "a" * 40}}),
         "tribe-bench": FakeDist({"url": "file:///repo", "dir_info": {"editable": True}})})
    out = P.distribution_provenance(("tribev2", "tribe-bench", "nope"), metadata=md)
    assert out["tribev2"]["vcs_commit"] == "a" * 40
    assert out["tribev2"]["requested_revision"] == "a" * 40
    assert out["tribe-bench"]["editable"] is True
    assert out["nope"] is None


def test_library_versions_default_path_runs_against_the_real_environment():
    out = P.library_versions()
    assert set(out) == set(P.UID_DISTRIBUTIONS)
    assert all(isinstance(v, str) and v for v in out.values())


def test_feature_uid_fields_defaults_versions_to_the_live_environment():
    ident = P.feature_uid_fields(stimulus=STIMULUS, weights=weight_identity(),
                                 extractor=EXTRACTOR, chunking=CHUNKING,
                                 preprocessing=PREPROCESSING)
    for name in P.UID_DISTRIBUTIONS:
        assert f"versions.{name}" in ident


def test_incomplete_versions_dict_is_an_error():
    with pytest.raises(P.MissingIdentityField) as ei:
        fields(versions={"numpy": "2.2.6"})
    assert "transformers" in str(ei.value)


# =========================================================================== #
# 7. Helpers
# =========================================================================== #

def test_preprocessing_fields_flattens_the_real_vjepa_config():
    out = P.preprocessing_fields(json.loads(PROC_BYTES))
    assert out["shortest_edge"] == 292
    assert out["crop_height"] == 256 and out["crop_width"] == 256
    assert out["image_mean"] == [0.485, 0.456, 0.406]
    assert out["rescale_factor"] == 0.00392156862745098
    assert out["video_processor_type"] == "VJEPA2VideoProcessor"
    assert set(P.REQUIRED_PREPROCESSING) <= set(out)


def test_stimulus_fields_keys_on_content_not_path(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"MOOV" * 100)
    b.write_bytes(b"MOOV" * 100)
    fa = P.stimulus_fields(a, duration_s=60.0, fps=8.0, width=256, height=256)
    fb = P.stimulus_fields(b, duration_s=60.0, fps=8.0, width=256, height=256)
    assert fa == fb, "the same bytes at two paths must have one identity"
    b.write_bytes(b"XXXX" * 100)
    fc = P.stimulus_fields(b, duration_s=60.0, fps=8.0, width=256, height=256)
    assert fc["sha256"] != fa["sha256"]
    assert set(P.REQUIRED_STIMULUS) <= set(fa)


def test_stimulus_fields_on_a_missing_file_is_a_typed_error(tmp_path):
    with pytest.raises(P.WeightFileMissing):
        P.stimulus_fields(tmp_path / "nope.mp4", duration_s=1.0, fps=1.0,
                          width=1, height=1)


def test_pinned_constants_match_the_resolved_artifact():
    """Guards against a well-meaning edit that quietly repoints the pin."""
    assert P.VJEPA2_COMMIT == "875c192b7b704b87d1e1d99345769632dd5f739a"
    assert P.VJEPA2_WEIGHTS_SHA256 == (
        "f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b")
    assert P.VJEPA2_WEIGHTS_BYTES == 4_138_311_608
    assert P.PROCESSOR_CONFIG_FILENAME == "video_preprocessor_config.json"


# =========================================================================== #
# 8. Only this one needs the real huggingface_hub
# =========================================================================== #

@pytest.mark.skipif(not HAS_HFH, reason="huggingface_hub is not installed here")
def test_real_hfapi_accepts_the_arguments_we_pass():
    """No network: just proves the call shape we depend on still exists."""
    import inspect
    import huggingface_hub

    sig = inspect.signature(huggingface_hub.HfApi.model_info)
    for arg in ("repo_id", "revision", "files_metadata"):
        assert arg in sig.parameters, f"HfApi.model_info lost {arg}"


def test_a_stale_sibling_snapshot_does_not_shadow_the_pinned_one(tmp_path):
    """Two snapshots in one repo cache: verification must look in the pinned
    commit, not in whichever directory sorts first."""
    other = "0" * 40
    repo = _snapshot(tmp_path)
    stale = repo / "snapshots" / other
    stale.mkdir(parents=True)
    (repo / "blobs" / ("1" * 64)).write_bytes(b"\x02" * len(WEIGHT_BYTES))
    (stale / WEIGHTS).symlink_to(repo / "blobs" / ("1" * 64))
    assert other < P.VJEPA2_COMMIT, "test bug: the stale snapshot must sort first"

    # Without the commit preference this resolves the stale snapshot and raises
    # WeightMismatch, even though the pinned artifact is present and correct.
    res = P.verify_local_weights(repo, weight_identity())
    assert res.digest == WEIGHT_SHA256
    assert not res.path.endswith("1" * 64)
