"""Durable, cross-session home for the V-JEPA feature artifact (F7).

The invariant this file exists to enforce, in the operator's words:

    for a given unique feature identity, encoding happens AT MOST ONCE
    across sessions.

That is a *storage* property, not a compute property, so everything here is
about finding, proving and persisting bytes -- and about NOT offering any route
back to the GPU.  There is deliberately no `extract`, no `compute`, no
`on_missing` callback and no fallback anywhere in this module's surface: when a
valid artifact cannot be found the only two outcomes are `None` and a typed
`ArtifactNotFound`.  A caller who wants to encode must go and say so out loud in
stage 1.

Verification is NOT reimplemented here.  `tribe_tools.feature_artifact` already
reads every payload and compares a per-item sha256 recorded at write time
(presence-based checks are defeated -- see FINDINGS2 D2); this module calls
`verify_artifact` and never second-guesses it.  Payload reads arrive through an
injected `reader_factory`, because only the caller knows how to open an exca
CacheDict.

Read path constraints, from B2's proofs:
  * `/kaggle/input` is `chmod 555`.  Resolution therefore never mkdirs, never
    writes a marker and never repairs -- a complete artifact on a read-only
    mount is consumable with 0 recompute, and an incomplete one must simply be
    passed over.
  * exca item uids embed the ABSOLUTE stimulus path, so `stage_stimulus` puts
    the stimulus at `<workdir>/s2_stim/<sha16>/s2_stimulus.mp4` in BOTH stages,
    making exca's own keys content-addressed and mount-independent.
  * `/kaggle/temp` is per-session and is not a durable backend.  It is not
    offered.
"""
from __future__ import annotations

import abc
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from tribe_tools.feature_artifact import (
    COMPLETE,
    MANIFEST,
    FeatureArtifactError,
    verify_artifact,
)

IDENTITY_SCHEMA = 1
UID_PREFIX = "s2v1-"
STIM_DIRNAME = "s2_stim"
STIM_FILENAME = "s2_stimulus.mp4"
KAGGLE_METADATA = "dataset-metadata.json"

#: Prefixes for directories this module owns and that are NEVER candidates.
_INCOMING = ".incoming-"
_REJECTED = ".rejected-"


# --------------------------------------------------------------------------
# typed errors -- every message names a remedy
# --------------------------------------------------------------------------

class DurableStoreError(RuntimeError):
    """Base.  Every subclass message must name a remedy."""


class ArtifactNotFound(DurableStoreError):
    """No artifact for this identity verified in any search path."""


class PublishRefused(DurableStoreError):
    """The thing you asked to persist is not a finalized, verified artifact."""


class BackendUnavailable(DurableStoreError):
    """The durable backend cannot be used from here (credentials, mount, net)."""


class StimulusStagingError(DurableStoreError):
    """The content-addressed stimulus could not be staged."""


class StimulusDigestMismatch(StimulusStagingError):
    """Bytes at hand are not the bytes the identity names."""


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------

def _canonicalise(obj: Any, _where: str = "identity") -> Any:
    """Canonical JSON-able form.  Floats become `%.6f` STRINGS -- raw float repr
    drifts between Python versions and a drifting uid is a 4h45m re-encode."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError(
                f"{_where} contains a non-finite float ({obj!r}); it cannot be "
                "part of a stable identity. Replace it with a finite value or a string.")
        return f"{obj:.6f}"
    if isinstance(obj, Mapping):
        return {str(k): _canonicalise(v, f"{_where}.{k}") for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalise(v, f"{_where}[{i}]") for i, v in enumerate(obj)]
    raise TypeError(
        f"{_where} holds a {type(obj).__name__}, which has no canonical JSON form. "
        "Identity fields must be str/int/float/bool/None/list/dict -- convert it "
        "(e.g. str(path), or a sha256 hexdigest) before building ArtifactIdentity.")


def canonical_identity_json(identity: Mapping[str, Any], *, schema: int = IDENTITY_SCHEMA) -> str:
    return json.dumps(
        {"schema": schema, "identity": _canonicalise(identity)},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def feature_set_uid(identity: Mapping[str, Any], *, schema: int = IDENTITY_SCHEMA) -> str:
    """The semantic identity of an encode.  Names the artifact directory, and is
    what "at most once across sessions" is measured against."""
    payload = canonical_identity_json(identity, schema=schema).encode("utf8")
    return UID_PREFIX + hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class ArtifactIdentity:
    """Everything needed to recognise, prove and name one encode."""

    identity: Mapping[str, Any]
    expected_item_uids: tuple[str, ...]
    schema: int = IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if not self.expected_item_uids:
            raise ValueError(
                "expected_item_uids is empty; an artifact with no declared items can "
                "never be proven complete. Pass the full chunk uid set from stage 1.")
        object.__setattr__(self, "expected_item_uids", tuple(self.expected_item_uids))
        # fail loudly at construction, not at uid time
        _canonicalise(self.identity)

    @property
    def uid(self) -> str:
        return feature_set_uid(self.identity, schema=self.schema)

    @property
    def item_count(self) -> int:
        return len(self.expected_item_uids)


# --------------------------------------------------------------------------
# verification seam
# --------------------------------------------------------------------------

#: given an artifact directory, return `read_item(item_uid) -> array`
ReaderFactory = Callable[[Path], Callable[[str], Any]]
#: given an artifact directory, return exca's sidecar digest dict (or None)
SidecarProbe = Callable[[Path], Mapping[str, Any] | None]


def verify_location(
    artifact_dir: Path | str,
    ident: ArtifactIdentity,
    *,
    reader_factory: ReaderFactory,
    sidecar_probe: SidecarProbe | None = None,
) -> dict:
    """Prove one directory holds THIS identity's finalized artifact.

    Delegates entirely to `feature_artifact.verify_artifact`, which reads and
    hashes every payload.  Raises the same typed `FeatureArtifactError`
    subclasses; returns the manifest on success.  Writes nothing.
    """
    root = Path(artifact_dir)
    sidecars = sidecar_probe(root) if sidecar_probe is not None else None
    return verify_artifact(
        root,
        dict(ident.identity),
        ident.expected_item_uids,
        reader_factory(root),
        sidecars=sidecars,
    )


# --------------------------------------------------------------------------
# read path -- must work unchanged on a chmod 555 mount
# --------------------------------------------------------------------------

@dataclass
class Resolution:
    """What the search found, and why every rejected candidate was rejected."""

    path: Path | None
    manifest: dict | None = None
    rejected: list[tuple[Path, Exception]] = field(default_factory=list)
    searched: list[Path] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.path is not None

    def why(self) -> str:
        if not self.rejected:
            return "no candidate directory carried a manifest in any search path"
        return "; ".join(f"{p}: {type(e).__name__}: {e}" for p, e in self.rejected)


def _is_candidate(p: Path) -> bool:
    try:
        return p.is_dir() and (p / MANIFEST).is_file()
    except OSError:
        return False


def _iter_candidates(root: Path, max_depth: int) -> list[Path]:
    """Bounded, non-recursive-by-default scan.  Never descends into a directory
    that is itself an artifact, never touches dot-directories (our own
    `.incoming-*`/`.rejected-*` scratch), never writes."""
    out: list[Path] = []
    try:
        if not root.is_dir():
            return out
    except OSError:
        return out
    if _is_candidate(root):
        return [root]
    level = [root]
    for _ in range(max(0, max_depth)):
        nxt: list[Path] = []
        try:
            children = sorted(p for lvl in level for p in lvl.iterdir())
        except OSError:
            children = []
        for child in children:
            if child.name.startswith("."):
                continue
            if _is_candidate(child):
                out.append(child)
            elif child.is_dir():
                nxt.append(child)
        level = nxt
        if not level:
            break
    return out


def _freshness(p: Path) -> float:
    """Newest-valid-wins ordering key: the COMPLETE marker's mtime, because that
    is the moment the artifact became finalized.  Falls back to the manifest."""
    for name in (COMPLETE, MANIFEST):
        try:
            return (p / name).stat().st_mtime
        except OSError:
            continue
    return 0.0


def resolve_artifact(
    ident: ArtifactIdentity,
    *,
    search_paths: Sequence[Path | str],
    reader_factory: ReaderFactory,
    sidecar_probe: SidecarProbe | None = None,
    max_depth: int = 2,
) -> Resolution:
    """Newest-valid-wins search across candidate roots, read-only throughout.

    Candidates are verified newest-first and the FIRST that verifies is
    returned, so a corrupt or stale artifact sitting next to a good one costs a
    rejection, not the answer.  A candidate that raises anything from
    `feature_artifact` is recorded and skipped -- it is never returned.
    """
    seen: set[Path] = set()
    candidates: list[Path] = []
    searched: list[Path] = []
    for raw in search_paths:
        root = Path(raw)
        searched.append(root)
        for cand in _iter_candidates(root, max_depth):
            key = cand.absolute()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)

    candidates.sort(key=_freshness, reverse=True)

    rejected: list[tuple[Path, Exception]] = []
    for cand in candidates:
        try:
            man = verify_location(
                cand, ident, reader_factory=reader_factory, sidecar_probe=sidecar_probe)
        except FeatureArtifactError as ex:
            rejected.append((cand, ex))
            continue
        except OSError as ex:  # unreadable mount, permission, vanished
            rejected.append((cand, ex))
            continue
        return Resolution(path=cand, manifest=man, rejected=rejected, searched=searched)
    return Resolution(path=None, manifest=None, rejected=rejected, searched=searched)


def resolve_artifact_location(
    ident: ArtifactIdentity,
    *,
    search_paths: Sequence[Path | str],
    reader_factory: ReaderFactory,
    sidecar_probe: SidecarProbe | None = None,
    max_depth: int = 2,
) -> Path | None:
    """The location of a PROVEN artifact for `ident`, or None.

    None means "you do not have this artifact".  It does not mean "go and make
    one" -- this module has no such route; stage 1 must be invoked explicitly.
    """
    return resolve_artifact(
        ident,
        search_paths=search_paths,
        reader_factory=reader_factory,
        sidecar_probe=sidecar_probe,
        max_depth=max_depth,
    ).path


def require_artifact_location(
    ident: ArtifactIdentity,
    *,
    search_paths: Sequence[Path | str],
    reader_factory: ReaderFactory,
    sidecar_probe: SidecarProbe | None = None,
    max_depth: int = 2,
) -> Path:
    """`resolve_artifact_location`, but a miss is a typed failure.

    Use this on the inference path: it makes "artifact missing" a stop, never a
    silent 4h45m encode inside a DataLoader worker.
    """
    res = resolve_artifact(
        ident,
        search_paths=search_paths,
        reader_factory=reader_factory,
        sidecar_probe=sidecar_probe,
        max_depth=max_depth,
    )
    if res.path is None:
        raise ArtifactNotFound(
            f"no verified feature artifact for {ident.uid} in "
            f"{[str(p) for p in res.searched]}. Cause: {res.why()}. "
            "Remedy: attach the Kaggle dataset holding it (Add Data -> your "
            "s2-features dataset), or run stage 1 explicitly: "
            "s2_run.py --extract-features. Inference will NOT encode for you.")
    return res.path


# --------------------------------------------------------------------------
# write path -- backends
# --------------------------------------------------------------------------

@dataclass
class StoreOutcome:
    """What a backend did.  `created=False` means the identity was already
    published there -- the at-most-once invariant, observed."""

    location: str
    created: bool
    details: dict = field(default_factory=dict)


@dataclass
class PublishResult:
    backend: str
    location: str
    feature_set_uid: str
    item_count: int
    created: bool
    details: dict = field(default_factory=dict)


class DurableBackend(abc.ABC):
    """Somewhere an artifact survives the death of this session.

    `store` receives `verify`, a callable that proves a directory holds the
    identity's artifact.  Backends that materialise a COPY must call it on the
    copy: a publish that silently corrupts is worse than no publish.
    """

    name: str = "backend"

    @abc.abstractmethod
    def store(
        self,
        artifact_dir: Path,
        ident: ArtifactIdentity,
        verify: Callable[[Path], dict],
    ) -> StoreOutcome:
        ...

    def search_roots(self) -> list[Path]:
        """Where a later session should look for what this backend stored."""
        return []


def publish(
    artifact_dir: Path | str,
    ident: ArtifactIdentity,
    backend: DurableBackend,
    *,
    reader_factory: ReaderFactory,
    sidecar_probe: SidecarProbe | None = None,
) -> PublishResult:
    """Persist a FINALIZED artifact.  Refuses anything that does not verify.

    The source is proven BEFORE the backend is touched (never ship a tarpit to a
    durable store, where it would outlive the session that made it), and the
    backend proves the copy it made.
    """
    src = Path(artifact_dir)
    try:
        verify_location(src, ident, reader_factory=reader_factory, sidecar_probe=sidecar_probe)
    except FeatureArtifactError as ex:
        raise PublishRefused(
            f"refusing to publish {src}: it is not a finalized, verified artifact "
            f"({type(ex).__name__}: {ex}). Remedy: finish stage 1 so it writes "
            f"{MANIFEST} then {COMPLETE}, or `rm -rf` the directory and re-run "
            "s2_run.py --extract-features. Publishing an unverified artifact would "
            "make a poisoned cache durable.") from ex

    def _verify(p: Path) -> dict:
        return verify_location(
            p, ident, reader_factory=reader_factory, sidecar_probe=sidecar_probe)

    outcome = backend.store(src, ident, _verify)
    return PublishResult(
        backend=backend.name,
        location=outcome.location,
        feature_set_uid=ident.uid,
        item_count=ident.item_count,
        created=outcome.created,
        details=dict(outcome.details),
    )


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            try:
                fd = os.open(os.path.join(dirpath, fn), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.fsync(fd)
            except OSError:
                pass
            finally:
                os.close(fd)
        _fsync_dir(Path(dirpath))


class LocalDirectoryBackend(DurableBackend):
    """A directory that outlives the session: a mounted dataset staging area,
    `/kaggle/working`, an HF cache, a laptop disk.

    Publish is a copy into `.incoming-*` (same filesystem -- D2 hit EXDEV from a
    default `/tmp` tempfile), a verification of that copy, then a single
    `os.replace` of the DIRECTORY.  `os.replace` onto a non-empty directory is
    ENOTEMPTY (D2 again), so an occupied destination is resolved first: if what
    is there already verifies, nothing is copied over it and `created=False`; if
    it does not, it is moved aside to `.rejected-*` for the post-mortem rather
    than deleted.
    """

    name = "local"

    def __init__(self, root: Path | str, *, fsync: bool = True) -> None:
        self.root = Path(root)
        self.fsync = fsync

    def search_roots(self) -> list[Path]:
        return [self.root]

    def target(self, ident: ArtifactIdentity) -> Path:
        return self.root / ident.uid

    def prune_incomplete(self) -> list[Path]:
        """Remove leftovers of an interrupted publish.  Safe by construction:
        `.incoming-*` is never a candidate and never the live artifact."""
        removed = []
        try:
            children = sorted(self.root.iterdir())
        except OSError:
            return removed
        for child in children:
            if child.is_dir() and child.name.startswith(_INCOMING):
                shutil.rmtree(child, ignore_errors=True)
                removed.append(child)
        return removed

    def store(self, artifact_dir, ident, verify) -> StoreOutcome:
        src = Path(artifact_dir)
        dest = self.target(ident)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            raise BackendUnavailable(
                f"cannot create the publish root {self.root} ({ex}). Remedy: point the "
                "backend at a WRITABLE directory -- /kaggle/input is chmod 555 and is a "
                "read path only; publish to /kaggle/working.") from ex

        if dest.exists():
            try:
                verify(dest)
            except FeatureArtifactError:
                aside = self.root / f"{_REJECTED}{ident.uid}-{int(time.time() * 1000)}"
                os.rename(dest, aside)
            else:
                return StoreOutcome(str(dest), created=False,
                                    details={"reason": "identity already published"})

        tmp = self.root / f"{_INCOMING}{ident.uid}-{os.getpid()}-{int(time.time() * 1000)}"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            shutil.copytree(src, tmp, symlinks=False)
            if self.fsync:
                _fsync_tree(tmp)
            try:
                verify(tmp)
            except FeatureArtifactError as ex:
                raise PublishRefused(
                    f"the copy of {src} at {tmp} does not verify ({type(ex).__name__}: "
                    f"{ex}) -- the transfer corrupted or truncated it. Nothing was "
                    f"published. Remedy: check free space on {self.root} and publish "
                    "again.") from ex
            os.replace(tmp, dest)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        if self.fsync:
            _fsync_dir(self.root)
        return StoreOutcome(str(dest), created=True, details={"root": str(self.root)})


class KaggleDatasetBackend(DurableBackend):
    """B2's first-ranked durable home: a private Kaggle Dataset, versioned from
    inside the notebook via the API.  200 GB cap, fully automatic, no human
    press -- unlike Save Version, which cannot be trusted for a 4h45m job.

    Every network/CLI call goes through `runner`, injected, so the decision
    logic above it is testable offline.  `runner(argv, env) -> (returncode,
    stdout, stderr)`.
    """

    name = "kaggle-dataset"

    def __init__(
        self,
        dataset_id: str,
        *,
        runner: Callable[[Sequence[str], Mapping[str, str]], tuple[int, str, str]] | None = None,
        env: Mapping[str, str] | None = None,
        title: str | None = None,
        licence: str = "CC0-1.0",
        dir_mode: str = "zip",
    ) -> None:
        if dataset_id.count("/") != 1 or not all(dataset_id.split("/")):
            raise ValueError(
                f"dataset_id must be '<kaggle-username>/<dataset-slug>', got {dataset_id!r}. "
                "Remedy: pass e.g. 'myuser/s2-features'.")
        self.dataset_id = dataset_id
        self.title = title or dataset_id.split("/")[1]
        self.licence = licence
        self.dir_mode = dir_mode
        self._runner = runner if runner is not None else _subprocess_runner
        self._env = dict(env if env is not None else os.environ)

    def search_roots(self) -> list[Path]:
        return [Path("/kaggle/input") / self.dataset_id.split("/")[1]]

    def _credentials(self) -> dict:
        missing = [k for k in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not self._env.get(k)]
        if missing:
            raise BackendUnavailable(
                f"Kaggle API credentials absent from the environment: {missing}. Remedy: "
                "Notebook -> Add-ons -> Secrets, add KAGGLE_USERNAME and KAGGLE_KEY from "
                "kaggle.com/settings -> Create New Token, then export them before "
                "publishing. Also enable Internet on the notebook.")
        return dict(self._env)

    def _run(self, argv: Sequence[str], env: Mapping[str, str]) -> tuple[int, str, str]:
        try:
            return self._runner(list(argv), env)
        except FileNotFoundError as ex:
            raise BackendUnavailable(
                f"the `kaggle` CLI is not installed ({ex}). Remedy: "
                "`pip -q install kaggle` in the notebook, with Internet enabled.") from ex

    def exists(self, env: Mapping[str, str]) -> bool:
        rc, _out, _err = self._run(
            ["kaggle", "datasets", "status", self.dataset_id], env)
        return rc == 0

    def write_metadata(self, artifact_dir: Path) -> Path:
        """`kaggle datasets version -p <dir>` requires this file INSIDE <dir>."""
        meta = Path(artifact_dir) / KAGGLE_METADATA
        payload = json.dumps(
            {"title": self.title, "id": self.dataset_id,
             "licenses": [{"name": self.licence}]},
            indent=2, sort_keys=True).encode()
        try:
            meta.write_bytes(payload)
        except OSError as ex:
            raise BackendUnavailable(
                f"cannot write {meta} ({ex}); `kaggle datasets version -p` requires "
                f"{KAGGLE_METADATA} inside the folder it uploads. Remedy: build the "
                "artifact under /kaggle/working (writable), not on a /kaggle/input "
                "mount (chmod 555).") from ex
        return meta

    def store(self, artifact_dir, ident, verify) -> StoreOutcome:
        src = Path(artifact_dir)
        env = self._credentials()
        self.write_metadata(src)
        message = f"s2 features {ident.uid} ({ident.item_count} items)"
        if self.exists(env):
            argv = ["kaggle", "datasets", "version", "-p", str(src),
                    "-m", message, "--dir-mode", self.dir_mode]
            created = False
        else:
            argv = ["kaggle", "datasets", "create", "-p", str(src),
                    "--dir-mode", self.dir_mode]
            created = True
        rc, out, err = self._run(argv, env)
        if rc != 0:
            raise BackendUnavailable(
                f"`{' '.join(argv)}` failed with exit {rc}: {(err or out).strip()[:400]}. "
                "Remedy: confirm Internet is enabled on the notebook, that the Secret "
                "token is valid (kaggle.com/settings -> Create New Token), and that "
                f"{self.dataset_id} is owned by KAGGLE_USERNAME.")
        return StoreOutcome(
            f"kaggle://{self.dataset_id}", created=created,
            details={"argv": list(argv), "stdout": out.strip()[:400],
                     "attach": "next session: Add Data -> your dataset -> mounts "
                               f"read-only at /kaggle/input/{self.dataset_id.split('/')[1]}"})


def _subprocess_runner(argv: Sequence[str], env: Mapping[str, str]) -> tuple[int, str, str]:
    import subprocess  # local: keeps the module import-safe and the seam obvious

    proc = subprocess.run(list(argv), env=dict(env), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


# --------------------------------------------------------------------------
# content-addressed stimulus staging (closes G1/G2 at the exca layer)
# --------------------------------------------------------------------------

def file_sha256(path: Path | str, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def staged_stimulus_path(workdir: Path | str, sha256: str) -> Path:
    """The one true location.  Both stages must build events from THIS path, or
    exca's item uids differ between stage 1 and stage 2 for identical bytes."""
    digest = sha256.strip().lower()
    if len(digest) < 16 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(
            f"sha256 must be a hex digest of at least 16 chars, got {sha256!r}. "
            "Remedy: pass durable_store.file_sha256(stimulus).")
    return Path(workdir) / STIM_DIRNAME / digest[:16] / STIM_FILENAME


def stage_stimulus(src: Path | str, workdir: Path | str, sha256: str) -> Path:
    """Copy the stimulus to `<workdir>/s2_stim/<sha16>/s2_stimulus.mp4`.

    Idempotent: an already-staged file whose bytes hash to `sha256` is returned
    untouched, so this is safe on a read-only tree that already holds it.  The
    copy is verified by re-hashing what actually landed -- a truncated copy that
    kept the right path would otherwise mint wrong features under a right-looking
    item uid.
    """
    dest = staged_stimulus_path(workdir, sha256)
    want = sha256.strip().lower()

    if dest.exists():
        got = file_sha256(dest)
        if got.startswith(want) or want.startswith(got):
            return dest
        raise StimulusDigestMismatch(
            f"{dest} already holds different bytes (sha256 {got}, expected {want}). "
            "exca keys items by this path, so reusing it would serve the OLD features "
            f"under the new identity. Remedy: `rm -rf {dest.parent}` and stage again.")

    source = Path(src)
    if not source.is_file():
        raise StimulusStagingError(
            f"stimulus {source} does not exist. Remedy: attach the stimulus dataset, or "
            "pass --stimulus pointing at the mp4.")
    src_digest = file_sha256(source)
    if not (src_digest.startswith(want) or want.startswith(src_digest)):
        raise StimulusDigestMismatch(
            f"{source} hashes to {src_digest}, not the {want} this identity names. "
            "Staging it would silently encode the wrong film. Remedy: fetch the correct "
            "stimulus, or rebuild the identity from the file you actually have.")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        raise StimulusStagingError(
            f"cannot create {dest.parent} ({ex}). Remedy: stage into a WRITABLE workdir "
            "(/kaggle/working), not onto a /kaggle/input mount (chmod 555).") from ex

    tmp = dest.parent / f".{STIM_FILENAME}.{os.getpid()}.tmp"
    try:
        shutil.copyfile(source, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        landed = file_sha256(tmp)
        if not (landed.startswith(want) or want.startswith(landed)):
            raise StimulusDigestMismatch(
                f"the copy at {tmp} hashes to {landed}, not {want} -- the transfer "
                f"corrupted or truncated it. Nothing was staged. Remedy: check free "
                f"space on {dest.parent} and stage again.")
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(dest.parent)
    return dest


def describe_search_paths(backends: Iterable[DurableBackend]) -> list[Path]:
    """Roots to hand to `resolve_artifact_location`, in backend order."""
    out: list[Path] = []
    for b in backends:
        for p in b.search_roots():
            if p not in out:
                out.append(p)
    return out
