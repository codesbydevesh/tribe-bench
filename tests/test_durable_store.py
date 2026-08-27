"""Durable cross-session storage (F7), behaviourally.

Every test builds real bytes on disk, damages or moves them in a specific way,
and asserts what the API does about it.  Nothing here reads source text, nothing
touches the network: the Kaggle backend's only outside contact is an injected
runner, and every other test uses `tmp_path` and a fake backend.

The invariant under test is the operator's: for a given unique feature identity,
encoding happens AT MOST ONCE across sessions.  `_Sessions` below models that
literally -- it counts encodes across simulated session boundaries, and the
target-behaviour tests assert the count.
"""
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from tribe_tools import durable_store as ds
from tribe_tools.feature_artifact import (
    COMPLETE,
    MANIFEST,
    ArtifactCorrupt,
    ArtifactIncomplete,
    ArtifactStale,
    FeatureArtifactError,
    begin_stage1,
    write_artifact,
)

UIDS = (
    "/w/s2_stim/0123456789abcdef/s2_stimulus.mp4_0.00_60.00",
    "/w/s2_stim/0123456789abcdef/s2_stimulus.mp4_60.00_60.00",
    "/w/s2_stim/0123456789abcdef/s2_stimulus.mp4_120.00_30.00",
)

IDENTITY_A = {
    "schema": 1,
    "stimulus": {"sha256": "a" * 64, "duration_s": 60.0},
    "extractor": {"class": "HuggingFaceVideo", "clip_duration": 4.0},
    "weights": {"revision": "875c192b" + "0" * 32},
}
IDENTITY_B = dict(IDENTITY_A, weights={"revision": "deadbeef" + "0" * 32})


def ident_a():
    return ds.ArtifactIdentity(IDENTITY_A, UIDS)


def ident_b():
    return ds.ArtifactIdentity(IDENTITY_B, UIDS)


# --------------------------------------------------------------------------
# a self-contained artifact: payloads live INSIDE the directory, so a publish
# that copies the directory carries everything verification needs.
# --------------------------------------------------------------------------

def _payload_file(root: Path, uid: str) -> Path:
    return Path(root) / "payload" / (hashlib.sha256(uid.encode()).hexdigest() + ".npy")


def reader_factory(root):
    def read_item(uid):
        return np.load(_payload_file(root, uid))
    return read_item


def build_artifact(root: Path, ident: ds.ArtifactIdentity, *, salt: float = 1.0) -> Path:
    root = Path(root)
    (root / "payload").mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, uid in enumerate(ident.expected_item_uids):
        a = np.full((4, 3), (i + 1) * salt, dtype=np.float32)
        np.save(_payload_file(root, uid), a)
        arrays[uid] = a
    begin_stage1(root)
    write_artifact(root, dict(ident.identity), arrays)
    return root


def corrupt_payload(root: Path, uid: str) -> None:
    """Same size, different bytes -- the tarpit exca serves without complaint."""
    f = _payload_file(root, uid)
    a = np.load(f)
    np.save(f, np.full(a.shape, 999.0, dtype=a.dtype))


def chmod_tree(root: Path, mode: int) -> None:
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for fn in filenames:
            os.chmod(os.path.join(dirpath, fn), mode & 0o666 if mode == 0o555 else mode)
        for dn in dirnames:
            os.chmod(os.path.join(dirpath, dn), mode)
    os.chmod(root, mode)


@pytest.fixture
def readonly_tree():
    made = []

    def _mk(root: Path) -> Path:
        chmod_tree(root, 0o555)
        made.append(root)
        return root

    yield _mk
    for root in made:
        chmod_tree(root, 0o755)


# --------------------------------------------------------------------------
# fake backend -- durable across "sessions", no network, no CLI
# --------------------------------------------------------------------------

class FakeBackend(ds.DurableBackend):
    """A durable store that is really just another directory, plus a ledger of
    what it was asked to do."""

    name = "fake"

    def __init__(self, root: Path, *, damage: str | None = None):
        self.root = Path(root)
        self.damage = damage
        self.calls = []
        self.verified = []

    def search_roots(self):
        return [self.root]

    def store(self, artifact_dir, ident, verify):
        self.calls.append((Path(artifact_dir), ident.uid))
        dest = self.root / ident.uid
        if dest.exists():
            verify(dest)
            self.verified.append(dest)
            return ds.StoreOutcome(str(dest), created=False)
        import shutil
        shutil.copytree(artifact_dir, dest)
        if self.damage:
            corrupt_payload(dest, self.damage)
        verify(dest)
        self.verified.append(dest)
        return ds.StoreOutcome(str(dest), created=True)


class _Sessions:
    """Simulates the operator's world across session boundaries.

    `run(...)` is one session: resolve, and encode ONLY if nothing verified.
    `encodes` is the number the invariant is measured against.
    """

    def __init__(self, workdir: Path, durable: Path):
        self.workdir = Path(workdir)
        self.durable = Path(durable)
        self.encodes = 0
        self.inferences = 0
        self.backend = FakeBackend(self.durable)

    def search_paths(self):
        return [self.durable, self.workdir / "s2_features"]

    def run(self, ident, *, crash_after_extract=False):
        found = ds.resolve_artifact_location(
            ident, search_paths=self.search_paths(), reader_factory=reader_factory)
        if found is None:
            # the ONLY route to an encode: the caller decides, explicitly
            self.encodes += 1
            local = self.workdir / "s2_features" / ident.uid
            build_artifact(local, ident)
            if crash_after_extract:
                raise KeyboardInterrupt("session died after extraction")
            ds.publish(local, ident, self.backend, reader_factory=reader_factory)
            found = ds.resolve_artifact_location(
                ident, search_paths=self.search_paths(), reader_factory=reader_factory)
        assert found is not None
        self.inferences += 1
        return found


# ==========================================================================
# target behaviour 1 -- first run: extract, verify, finalize, persist, infer
# ==========================================================================

def test_first_run_encodes_once_and_persists(tmp_path):
    s = _Sessions(tmp_path / "work", tmp_path / "durable")
    ident = ident_a()

    assert ds.resolve_artifact_location(
        ident, search_paths=s.search_paths(), reader_factory=reader_factory) is None

    where = s.run(ident)

    assert s.encodes == 1
    assert s.inferences == 1
    assert Path(where).is_dir()
    assert (tmp_path / "durable" / ident.uid / MANIFEST).is_file()
    assert (tmp_path / "durable" / ident.uid / COMPLETE).is_file()


def test_second_session_reuses_and_never_re_encodes(tmp_path):
    """At most once ACROSS SESSIONS: the durable store is the only thing that
    survives, so the second session must find it there."""
    s = _Sessions(tmp_path / "work", tmp_path / "durable")
    ident = ident_a()
    s.run(ident)

    # a new session: the previous session's scratch is gone
    import shutil
    shutil.rmtree(tmp_path / "work")
    s.run(ident)

    assert s.encodes == 1
    assert s.inferences == 2


# ==========================================================================
# target behaviour 2 -- crash after extraction: verify, skip extraction, infer
# ==========================================================================

def test_crash_after_extraction_skips_the_second_encode(tmp_path):
    s = _Sessions(tmp_path / "work", tmp_path / "durable")
    ident = ident_a()

    with pytest.raises(KeyboardInterrupt):
        s.run(ident, crash_after_extract=True)
    assert s.encodes == 1
    assert s.inferences == 0
    assert not (tmp_path / "durable" / ident.uid).exists()   # never published

    s.run(ident)          # resume: the local artifact is still valid
    assert s.encodes == 1
    assert s.inferences == 1


def test_crash_before_marker_is_not_reused(tmp_path):
    """A cache directory with no COMPLETE marker is not an artifact.  It must be
    passed over, not resumed into (B2 R3: a partial folder can be poisoned)."""
    local = build_artifact(tmp_path / "work" / "s2_features" / ident_a().uid, ident_a())
    (local / COMPLETE).unlink()

    s = _Sessions(tmp_path / "work", tmp_path / "durable")
    assert ds.resolve_artifact_location(
        ident_a(), search_paths=s.search_paths(), reader_factory=reader_factory) is None


# ==========================================================================
# target behaviour 3 -- new identity: extract exactly once, verify, persist
# ==========================================================================

def test_new_identity_encodes_exactly_once_more(tmp_path):
    s = _Sessions(tmp_path / "work", tmp_path / "durable")
    s.run(ident_a())
    s.run(ident_b())
    s.run(ident_a())
    s.run(ident_b())

    assert s.encodes == 2
    assert s.inferences == 4
    assert (tmp_path / "durable" / ident_a().uid).is_dir()
    assert (tmp_path / "durable" / ident_b().uid).is_dir()
    assert ident_a().uid != ident_b().uid


def test_stale_artifact_for_another_identity_is_not_returned(tmp_path):
    """A leftover from a previous stimulus sitting in a search path is a trap:
    same shape, same file names, wrong tensors."""
    durable = tmp_path / "durable"
    build_artifact(durable / "some-old-run", ident_b())

    assert ds.resolve_artifact_location(
        ident_a(), search_paths=[durable], reader_factory=reader_factory) is None

    res = ds.resolve_artifact(
        ident_a(), search_paths=[durable], reader_factory=reader_factory)
    assert [type(e) for _p, e in res.rejected] == [ArtifactStale]


def test_identity_uid_is_stable_and_order_independent(tmp_path):
    a = ds.ArtifactIdentity(dict(reversed(list(IDENTITY_A.items()))), UIDS)
    assert a.uid == ident_a().uid
    assert ident_a().uid.startswith(ds.UID_PREFIX)
    assert ident_a().uid != ident_b().uid


def test_float_identity_fields_do_not_drift(tmp_path):
    """Raw float repr drifts; a drifting uid is a 4h45m re-encode."""
    one = ds.ArtifactIdentity({"clip_duration": 4.0}, UIDS).uid
    two = ds.ArtifactIdentity({"clip_duration": 4.0000000001}, UIDS).uid
    three = ds.ArtifactIdentity({"clip_duration": 2.0}, UIDS).uid
    assert one == two
    assert one != three


def test_non_canonicalisable_identity_field_is_refused():
    with pytest.raises(TypeError) as ex:
        ds.ArtifactIdentity({"stimulus": Path("/w/x.mp4")}, UIDS)
    assert "ArtifactIdentity" in str(ex.value)


def test_empty_item_set_is_refused():
    with pytest.raises(ValueError):
        ds.ArtifactIdentity(IDENTITY_A, ())


# ==========================================================================
# target behaviour 4 -- corrupted artifact: reject, never infer
# ==========================================================================

def test_corrupt_artifact_is_rejected_and_inference_never_runs(tmp_path):
    durable = tmp_path / "durable"
    art = build_artifact(durable / ident_a().uid, ident_a())
    corrupt_payload(art, UIDS[1])

    assert ds.resolve_artifact_location(
        ident_a(), search_paths=[durable], reader_factory=reader_factory) is None

    with pytest.raises(ds.ArtifactNotFound) as ex:
        ds.require_artifact_location(
            ident_a(), search_paths=[durable], reader_factory=reader_factory)
    assert "ArtifactCorrupt" in str(ex.value)


def test_unreadable_payload_is_rejected(tmp_path):
    durable = tmp_path / "durable"
    art = build_artifact(durable / ident_a().uid, ident_a())
    _payload_file(art, UIDS[0]).write_bytes(b"")

    res = ds.resolve_artifact(
        ident_a(), search_paths=[durable], reader_factory=reader_factory)
    assert res.path is None
    assert [type(e) for _p, e in res.rejected] == [ArtifactCorrupt]


def test_corrupt_candidate_is_skipped_in_favour_of_a_valid_one(tmp_path):
    durable = tmp_path / "durable"
    good = build_artifact(durable / "good", ident_a())
    bad = build_artifact(durable / "bad", ident_a())
    corrupt_payload(bad, UIDS[0])
    # make the CORRUPT one the newest, so freshness alone would pick it
    now = time.time()
    os.utime(good / COMPLETE, (now - 500, now - 500))
    os.utime(bad / COMPLETE, (now, now))

    found = ds.resolve_artifact_location(
        ident_a(), search_paths=[durable], reader_factory=reader_factory)
    assert found == good

    res = ds.resolve_artifact(
        ident_a(), search_paths=[durable], reader_factory=reader_factory)
    assert [p for p, _e in res.rejected] == [bad]


@pytest.mark.parametrize("newer,older", [("aaa", "zzz"), ("zzz", "aaa")])
def test_newest_valid_wins_between_two_valid_candidates(tmp_path, newer, older):
    """Both orderings, so a name-ordered (or arrival-ordered) search cannot pass
    by luck: only the COMPLETE marker's mtime may decide."""
    durable = tmp_path / "durable"
    fresh = build_artifact(durable / newer, ident_a())
    stale = build_artifact(durable / older, ident_a(), salt=7.0)
    now = time.time()
    os.utime(stale / COMPLETE, (now - 900, now - 900))
    os.utime(fresh / COMPLETE, (now, now))

    assert ds.resolve_artifact_location(
        ident_a(), search_paths=[durable], reader_factory=reader_factory) == fresh


def test_missing_manifest_directory_is_not_a_candidate(tmp_path):
    durable = tmp_path / "durable"
    (durable / "not-an-artifact").mkdir(parents=True)
    (durable / "not-an-artifact" / "payload").mkdir()

    res = ds.resolve_artifact(
        ident_a(), search_paths=[durable, tmp_path / "nope"], reader_factory=reader_factory)
    assert res.path is None and res.rejected == []


# ==========================================================================
# target behaviour 5 -- "artifact missing" NEVER means "extract quietly"
# ==========================================================================

FORBIDDEN_ROUTE = ("extract", "compute", "encode", "on_missing", "fallback",
                   "recompute", "if_missing", "auto")


def test_public_api_offers_no_route_from_missing_to_extraction():
    """There must be no function, and no parameter of any function, through
    which a failed lookup can turn into a GPU pass."""
    offenders = []
    for name, obj in vars(ds).items():
        if name.startswith("_") or not callable(obj):
            continue
        if getattr(obj, "__module__", None) != ds.__name__:
            continue
        if any(word in name.lower() for word in FORBIDDEN_ROUTE):
            offenders.append(name)
        members = [obj] if not isinstance(obj, type) else [
            m for _n, m in inspect.getmembers(obj, callable)
            if getattr(m, "__module__", None) == ds.__name__]
        for member in members:
            try:
                params = inspect.signature(member).parameters
            except (TypeError, ValueError):
                continue
            for pname in params:
                if any(word in pname.lower() for word in FORBIDDEN_ROUTE):
                    offenders.append(f"{name}({pname})")
    assert offenders == []


def test_a_missed_lookup_returns_none_and_writes_nothing(tmp_path):
    """Read path on a miss: no directory created, no marker written, no repair.
    (A resolve that mkdir'd would also blow up on /kaggle/input, chmod 555.)"""
    durable = tmp_path / "durable"
    durable.mkdir()
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    assert ds.resolve_artifact_location(
        ident_a(), search_paths=[durable, tmp_path / "absent"],
        reader_factory=reader_factory) is None

    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_require_artifact_location_raises_with_a_remedy(tmp_path):
    with pytest.raises(ds.ArtifactNotFound) as ex:
        ds.require_artifact_location(
            ident_a(), search_paths=[tmp_path / "durable"], reader_factory=reader_factory)
    msg = str(ex.value)
    assert "--extract-features" in msg          # the remedy is explicit and human-run
    assert ident_a().uid in msg


def test_verification_reads_every_payload(tmp_path):
    """No fast path: presence is not proof (FINDINGS2 D2)."""
    art = build_artifact(tmp_path / "d" / ident_a().uid, ident_a())
    seen = []

    def counting_factory(root):
        inner = reader_factory(root)

        def read_item(uid):
            seen.append(uid)
            return inner(uid)
        return read_item

    ds.resolve_artifact_location(
        ident_a(), search_paths=[tmp_path / "d"], reader_factory=counting_factory)
    assert sorted(seen) == sorted(UIDS)


# ==========================================================================
# read-only mounts (/kaggle/input is chmod 555)
# ==========================================================================

def test_complete_artifact_on_a_readonly_tree_resolves(tmp_path, readonly_tree):
    durable = tmp_path / "durable"
    art = build_artifact(durable / ident_a().uid, ident_a())
    readonly_tree(durable)

    assert ds.resolve_artifact_location(
        ident_a(), search_paths=[durable], reader_factory=reader_factory) == art


def test_publish_from_a_readonly_source_directory(tmp_path, readonly_tree):
    src = build_artifact(tmp_path / "ro" / ident_a().uid, ident_a())
    readonly_tree(tmp_path / "ro")
    backend = ds.LocalDirectoryBackend(tmp_path / "durable")

    result = ds.publish(src, ident_a(), backend, reader_factory=reader_factory)

    assert result.created is True
    assert Path(result.location) == tmp_path / "durable" / ident_a().uid
    assert ds.resolve_artifact_location(
        ident_a(), search_paths=[tmp_path / "durable"],
        reader_factory=reader_factory) == Path(result.location)


def test_publish_to_a_readonly_root_names_the_remedy(tmp_path, readonly_tree):
    src = build_artifact(tmp_path / "src", ident_a())
    root = tmp_path / "ro-root"
    root.mkdir()
    readonly_tree(root)
    backend = ds.LocalDirectoryBackend(root / "nested")

    with pytest.raises(ds.BackendUnavailable) as ex:
        ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    assert "/kaggle/working" in str(ex.value)


# ==========================================================================
# publish: refusal, atomicity, idempotence
# ==========================================================================

def test_publish_refuses_an_artifact_without_a_marker(tmp_path):
    src = build_artifact(tmp_path / "src", ident_a())
    (src / COMPLETE).unlink()
    backend = FakeBackend(tmp_path / "durable")

    with pytest.raises(ds.PublishRefused) as ex:
        ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    assert isinstance(ex.value.__cause__, ArtifactIncomplete)
    assert backend.calls == []
    assert not (tmp_path / "durable").exists()


def test_publish_refuses_a_corrupt_artifact(tmp_path):
    src = build_artifact(tmp_path / "src", ident_a())
    corrupt_payload(src, UIDS[2])
    backend = FakeBackend(tmp_path / "durable")

    with pytest.raises(ds.PublishRefused) as ex:
        ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    assert isinstance(ex.value.__cause__, ArtifactCorrupt)
    assert backend.calls == []


def test_publish_refuses_an_artifact_built_for_another_identity(tmp_path):
    src = build_artifact(tmp_path / "src", ident_b())
    backend = FakeBackend(tmp_path / "durable")

    with pytest.raises(ds.PublishRefused):
        ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    assert backend.calls == []


def test_publish_verifies_the_copy_not_only_the_source(tmp_path, monkeypatch):
    """A transfer that truncates a payload must not leave a durable tarpit."""
    src = build_artifact(tmp_path / "src", ident_a())
    backend = ds.LocalDirectoryBackend(tmp_path / "durable")
    real_copytree = ds.shutil.copytree

    def damaging_copytree(a, b, *args, **kw):
        out = real_copytree(a, b, *args, **kw)
        if Path(b).name.startswith(".incoming-"):        # the top-level call only
            _payload_file(Path(b), UIDS[0]).write_bytes(b"")
        return out

    monkeypatch.setattr(ds.shutil, "copytree", damaging_copytree)

    with pytest.raises(ds.PublishRefused) as ex:
        ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    assert isinstance(ex.value.__cause__, ArtifactCorrupt)
    assert not (tmp_path / "durable" / ident_a().uid).exists()
    assert list((tmp_path / "durable").iterdir()) == []      # no scratch left behind


def test_republishing_the_same_identity_is_a_no_op(tmp_path):
    src = build_artifact(tmp_path / "src", ident_a())
    backend = ds.LocalDirectoryBackend(tmp_path / "durable")

    first = ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    dest = Path(first.location)
    stamp = (dest / MANIFEST).stat().st_mtime_ns

    second = ds.publish(src, ident_a(), backend, reader_factory=reader_factory)

    assert first.created is True and second.created is False
    assert (dest / MANIFEST).stat().st_mtime_ns == stamp


def test_publish_replaces_a_corrupt_destination_and_keeps_it_for_forensics(tmp_path):
    src = build_artifact(tmp_path / "src", ident_a())
    backend = ds.LocalDirectoryBackend(tmp_path / "durable")
    ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    corrupt_payload(tmp_path / "durable" / ident_a().uid, UIDS[0])

    result = ds.publish(src, ident_a(), backend, reader_factory=reader_factory)

    assert result.created is True
    assert ds.resolve_artifact_location(
        ident_a(), search_paths=[tmp_path / "durable"],
        reader_factory=reader_factory) == Path(result.location)
    aside = [p for p in (tmp_path / "durable").iterdir() if p.name.startswith(".rejected-")]
    assert len(aside) == 1


def test_interrupted_publish_leaves_nothing_resolvable(tmp_path):
    """A half-copied `.incoming-*` directory must never be mistaken for the
    artifact, and must be prunable."""
    src = build_artifact(tmp_path / "src", ident_a())
    backend = ds.LocalDirectoryBackend(tmp_path / "durable")
    half = tmp_path / "durable" / f".incoming-{ident_a().uid}-999-1"
    half.mkdir(parents=True)
    (half / MANIFEST).write_text(json.dumps({"schema_version": 1}))

    res = ds.resolve_artifact(
        ident_a(), search_paths=[tmp_path / "durable"], reader_factory=reader_factory)
    assert res.path is None
    assert res.rejected == []          # not merely rejected: never even a candidate

    assert backend.prune_incomplete() == [half]
    assert not half.exists()

    ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    assert ds.resolve_artifact_location(
        ident_a(), search_paths=[tmp_path / "durable"], reader_factory=reader_factory) is not None


def test_publish_result_reports_the_identity_it_persisted(tmp_path):
    src = build_artifact(tmp_path / "src", ident_a())
    backend = ds.LocalDirectoryBackend(tmp_path / "durable")
    r = ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    assert r.feature_set_uid == ident_a().uid
    assert r.item_count == len(UIDS)
    assert r.backend == "local"


def test_backend_search_roots_feed_the_read_path(tmp_path):
    src = build_artifact(tmp_path / "src", ident_a())
    backend = ds.LocalDirectoryBackend(tmp_path / "durable")
    ds.publish(src, ident_a(), backend, reader_factory=reader_factory)

    roots = ds.describe_search_paths([backend, backend])
    assert roots == [tmp_path / "durable"]
    assert ds.resolve_artifact_location(
        ident_a(), search_paths=roots, reader_factory=reader_factory) is not None


# ==========================================================================
# stimulus staging -- content-addressed, idempotent, verified
# ==========================================================================

def _mp4(path: Path, payload: bytes = b"\x00\x00\x00 ftypisom fake mp4 bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_stage_stimulus_is_content_addressed(tmp_path):
    src = _mp4(tmp_path / "mnt" / "s2.mp4")
    digest = ds.file_sha256(src)

    dest = ds.stage_stimulus(src, tmp_path / "work", digest)

    assert dest == tmp_path / "work" / "s2_stim" / digest[:16] / "s2_stimulus.mp4"
    assert dest.read_bytes() == src.read_bytes()


def test_stage_stimulus_is_mount_independent(tmp_path):
    """The whole point: the same bytes under two different mounts stage to the
    same path, so exca's absolute-path item uids match across sessions."""
    a = _mp4(tmp_path / "mnt-a" / "s2.mp4")
    b = _mp4(tmp_path / "kaggle" / "input" / "datasets" / "me" / "s2" / "movie.mp4")
    digest = ds.file_sha256(a)

    assert ds.stage_stimulus(a, tmp_path / "work", digest) == \
           ds.stage_stimulus(b, tmp_path / "work", digest)


def test_stage_stimulus_is_idempotent(tmp_path):
    src = _mp4(tmp_path / "mnt" / "s2.mp4")
    digest = ds.file_sha256(src)
    first = ds.stage_stimulus(src, tmp_path / "work", digest)
    stamp = first.stat().st_mtime_ns

    second = ds.stage_stimulus(src, tmp_path / "work", digest)

    assert second == first
    assert second.stat().st_mtime_ns == stamp      # not rewritten


def test_already_staged_stimulus_needs_no_write_permission(tmp_path, readonly_tree):
    src = _mp4(tmp_path / "mnt" / "s2.mp4")
    digest = ds.file_sha256(src)
    staged = ds.stage_stimulus(src, tmp_path / "work", digest)
    readonly_tree(tmp_path / "work")

    assert ds.stage_stimulus(src, tmp_path / "work", digest) == staged


def test_stage_stimulus_detects_a_mismatched_existing_copy(tmp_path):
    src = _mp4(tmp_path / "mnt" / "s2.mp4")
    digest = ds.file_sha256(src)
    staged = ds.stage_stimulus(src, tmp_path / "work", digest)
    staged.write_bytes(b"a completely different film")

    with pytest.raises(ds.StimulusDigestMismatch) as ex:
        ds.stage_stimulus(src, tmp_path / "work", digest)
    assert "rm -rf" in str(ex.value)


def test_stage_stimulus_refuses_a_source_that_is_not_the_named_bytes(tmp_path):
    src = _mp4(tmp_path / "mnt" / "s2.mp4")
    other = _mp4(tmp_path / "mnt" / "other.mp4", b"different film entirely")

    with pytest.raises(ds.StimulusDigestMismatch):
        ds.stage_stimulus(other, tmp_path / "work", ds.file_sha256(src))
    assert not (tmp_path / "work").exists()


def test_stage_stimulus_detects_a_truncated_copy(tmp_path, monkeypatch):
    src = _mp4(tmp_path / "mnt" / "s2.mp4")
    digest = ds.file_sha256(src)

    def truncating_copyfile(a, b, **kw):
        Path(b).write_bytes(Path(a).read_bytes()[:4])
        return b

    monkeypatch.setattr(ds.shutil, "copyfile", truncating_copyfile)

    with pytest.raises(ds.StimulusDigestMismatch):
        ds.stage_stimulus(src, tmp_path / "work", digest)
    assert not ds.staged_stimulus_path(tmp_path / "work", digest).exists()
    assert list((tmp_path / "work" / "s2_stim" / digest[:16]).iterdir()) == []


def test_stage_stimulus_reports_an_absent_source(tmp_path):
    with pytest.raises(ds.StimulusStagingError):
        ds.stage_stimulus(tmp_path / "nope.mp4", tmp_path / "work", "a" * 64)


def test_staged_path_rejects_a_non_digest(tmp_path):
    with pytest.raises(ValueError):
        ds.staged_stimulus_path(tmp_path, "not-a-digest")


# ==========================================================================
# Kaggle Dataset backend -- decision logic, offline
# ==========================================================================

class FakeKaggleCLI:
    def __init__(self, *, exists: bool, fail_on: str | None = None):
        self.exists = exists
        self.fail_on = fail_on
        self.argv = []
        self.envs = []

    def __call__(self, argv, env):
        self.argv.append(list(argv))
        self.envs.append(dict(env))
        verb = argv[2] if len(argv) > 2 else ""
        if verb == "status":
            return (0, "ready", "") if self.exists else (1, "", "404 - Not Found")
        if self.fail_on == verb:
            return 1, "", "403 - Forbidden"
        return 0, f"{verb} ok", ""


KAGGLE_ENV = {"KAGGLE_USERNAME": "operator", "KAGGLE_KEY": "k" * 32}


def test_kaggle_backend_creates_the_dataset_the_first_time(tmp_path):
    src = build_artifact(tmp_path / "src", ident_a())
    cli = FakeKaggleCLI(exists=False)
    backend = ds.KaggleDatasetBackend("operator/s2-features", runner=cli, env=KAGGLE_ENV)

    r = ds.publish(src, ident_a(), backend, reader_factory=reader_factory)

    assert r.created is True
    assert cli.argv[-1][:3] == ["kaggle", "datasets", "create"]
    assert json.loads((src / ds.KAGGLE_METADATA).read_text())["id"] == "operator/s2-features"


def test_kaggle_backend_versions_an_existing_dataset(tmp_path):
    src = build_artifact(tmp_path / "src", ident_a())
    cli = FakeKaggleCLI(exists=True)
    backend = ds.KaggleDatasetBackend("operator/s2-features", runner=cli, env=KAGGLE_ENV)

    r = ds.publish(src, ident_a(), backend, reader_factory=reader_factory)

    assert r.created is False
    argv = cli.argv[-1]
    assert argv[:3] == ["kaggle", "datasets", "version"]
    assert ident_a().uid in argv[argv.index("-m") + 1]
    assert "--dir-mode" in argv


def test_kaggle_backend_never_uploads_an_unverified_artifact(tmp_path):
    src = build_artifact(tmp_path / "src", ident_a())
    corrupt_payload(src, UIDS[0])
    cli = FakeKaggleCLI(exists=True)
    backend = ds.KaggleDatasetBackend("operator/s2-features", runner=cli, env=KAGGLE_ENV)

    with pytest.raises(ds.PublishRefused):
        ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    assert cli.argv == []


def test_kaggle_backend_demands_credentials_before_touching_the_network(tmp_path):
    src = build_artifact(tmp_path / "src", ident_a())
    cli = FakeKaggleCLI(exists=True)
    backend = ds.KaggleDatasetBackend("operator/s2-features", runner=cli, env={})

    with pytest.raises(ds.BackendUnavailable) as ex:
        ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    assert "Secrets" in str(ex.value)
    assert cli.argv == []


def test_kaggle_backend_surfaces_a_cli_failure_with_a_remedy(tmp_path):
    src = build_artifact(tmp_path / "src", ident_a())
    cli = FakeKaggleCLI(exists=True, fail_on="version")
    backend = ds.KaggleDatasetBackend("operator/s2-features", runner=cli, env=KAGGLE_ENV)

    with pytest.raises(ds.BackendUnavailable) as ex:
        ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    assert "Internet" in str(ex.value)


def test_kaggle_backend_reports_where_to_attach_it(tmp_path):
    src = build_artifact(tmp_path / "src", ident_a())
    cli = FakeKaggleCLI(exists=True)
    backend = ds.KaggleDatasetBackend("operator/s2-features", runner=cli, env=KAGGLE_ENV)
    r = ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    assert "/kaggle/input/s2-features" in r.details["attach"]
    assert backend.search_roots() == [Path("/kaggle/input/s2-features")]


def test_kaggle_backend_rejects_a_malformed_dataset_id():
    with pytest.raises(ValueError):
        ds.KaggleDatasetBackend("s2-features")


def test_kaggle_metadata_cannot_be_written_to_a_readonly_artifact(tmp_path, readonly_tree):
    src = build_artifact(tmp_path / "src", ident_a())
    readonly_tree(tmp_path / "src")
    cli = FakeKaggleCLI(exists=True)
    backend = ds.KaggleDatasetBackend("operator/s2-features", runner=cli, env=KAGGLE_ENV)

    with pytest.raises(ds.BackendUnavailable) as ex:
        ds.publish(src, ident_a(), backend, reader_factory=reader_factory)
    assert "/kaggle/working" in str(ex.value)
    assert cli.argv == []


# ==========================================================================
# error taxonomy
# ==========================================================================

def test_every_durable_error_is_catchable_as_one_type():
    for cls in (ds.ArtifactNotFound, ds.PublishRefused, ds.BackendUnavailable,
                ds.StimulusStagingError, ds.StimulusDigestMismatch):
        assert issubclass(cls, ds.DurableStoreError)
    assert issubclass(ds.StimulusDigestMismatch, ds.StimulusStagingError)
    assert not issubclass(ds.DurableStoreError, FeatureArtifactError)
