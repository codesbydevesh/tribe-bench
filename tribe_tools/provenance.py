"""F5 -- V-JEPA weight identity, and binding it into the feature uid.

The defect this file closes
---------------------------
`neuralset` passes **no** ``revision=`` anywhere.  Both V-JEPA load sites --
``Model.from_pretrained`` (``neuralset/extractors/video.py:394``) and
``Processor.from_pretrained`` (``neuralset/extractors/video.py:399``); likewise
``neuralset/extractors/image.py:74,87,94`` -- resolve against the floating
``main`` branch of ``facebook/vjepa2-vitg-fpc64-256``.

And exca's cache uid records the model's **name**, never its **identity**:
``HuggingFaceMixin._exclude_from_cache_uid`` returns ``["device"]``
(``neuralset/extractors/base.py:566-570``) so ``image.model_name`` -- the *string*
``"facebook/vjepa2-vitg-fpc64-256"`` -- is in the uid, and that string is exactly
the thing that stays constant while the weights underneath it change.

So: a re-push to ``main`` changes every tensor at an unchanged cache uid, with no
error, no warning, and no log line.  This module makes the weights, the
preprocessing config and the tensor-affecting library versions part of the
identity, so a change to any of them is a *miss*, not a silent wrong answer.

What it does not do
-------------------
It never downloads ``model.safetensors`` (4,138,311,608 bytes).  Identity comes
from the local HF cache (0 bytes, 0 network -- huggingface_hub is not even
imported) or, when explicitly permitted, from one ``model_info`` metadata call
(~2.6 kB).  A missing identity is a typed error, never a guess.

Design notes that cost someone a day each
-----------------------------------------
* Once a file is cached, huggingface_hub names the blob by its digest
  (``file_download.py:1173``, etag taken from ``X-Linked-ETag`` at
  ``file_download.py:1629-1631``), so ``basename(realpath(path))`` **is** the
  LFS sha256 for LFS files and the git blob sha1 for the rest.  Verified here
  against a real cache: ``config.json`` (801 B) hashes to
  ``3534852408cef7f5c0c54dfed6e0842c24492863`` under
  ``sha1(b"blob <n>\\0" + data)``, which is precisely its blob filename.
* huggingface_hub does **not** verify downloaded bytes against that digest, so
  the blob filename records what the server said, not what we measured.
  :func:`verify_local_weights` with ``force_hash=True`` is the one place that
  measures.
* Monkeypatching ``from_pretrained`` to inject ``revision=`` survives ``fork``
  and is **silently lost** under ``spawn``.  Not used here, deliberately.
* ``neuralset.__version__`` does not exist -- ``neuralset/__init__.py:33-79``
  installs a module ``__getattr__`` that *raises* -- so the widespread
  ``getattr(mod, "__version__", "unknown")`` idiom yields ``"unknown"``.  Use
  ``importlib.metadata``.  And ``importlib.metadata.version("tribev2")`` is the
  permanent constant ``"0.1.0"``; the only usable identity is the PEP 610
  ``direct_url.json`` commit.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "ProvenanceError", "WeightIdentityUnavailable", "WeightMismatch",
    "WeightFileMissing", "MissingIdentityField",
    "FileIdentity", "WeightIdentity", "WeightVerification",
    "VJEPA2_REPO", "VJEPA2_COMMIT", "VJEPA2_WEIGHTS_SHA256", "VJEPA2_WEIGHTS_BYTES",
    "WEIGHTS_FILENAME", "MODEL_CONFIG_FILENAME", "PROCESSOR_CONFIG_FILENAME",
    "VJEPA2_FILENAMES", "IDENTITY_SCHEMA", "CRITICAL_DISTRIBUTIONS",
    "sha256_file", "git_blob_sha1_file", "digest_file",
    "resolve_weight_identity", "verify_local_weights",
    "library_versions", "distribution_provenance",
    "stimulus_fields", "preprocessing_fields",
    "feature_uid_fields", "feature_set_uid", "exca_infra_version",
]

# --------------------------------------------------------------------------- #
# Pinned facts.  Every value below was resolved from the Hub metadata API and
# cross-checked against the incident log's byte counts (801 / 4.14 GB / 1.30 kB).
# --------------------------------------------------------------------------- #

VJEPA2_REPO = "facebook/vjepa2-vitg-fpc64-256"
VJEPA2_COMMIT = "875c192b7b704b87d1e1d99345769632dd5f739a"
VJEPA2_WEIGHTS_SHA256 = "f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b"
VJEPA2_WEIGHTS_BYTES = 4_138_311_608

WEIGHTS_FILENAME = "model.safetensors"
MODEL_CONFIG_FILENAME = "config.json"
#: V-JEPA 2 ships its preprocessing under this name, NOT ``preprocessor_config.json``
#: -- ``neuralset/extractors/video.py:391-392`` selects ``AutoVideoProcessor``.
PROCESSOR_CONFIG_FILENAME = "video_preprocessor_config.json"

VJEPA2_FILENAMES = (MODEL_CONFIG_FILENAME, PROCESSOR_CONFIG_FILENAME, WEIGHTS_FILENAME)

#: Bump when the *shape* of the identity dict changes, so an artifact built by an
#: older layout is rejected as stale rather than compared field-by-field.
IDENTITY_SCHEMA = 1

#: Distributions whose version can change the produced tensor, plus the ones that
#: decide cache identity (``exca``) or the download path (``huggingface_hub``).
CRITICAL_DISTRIBUTIONS = (
    "tribev2", "neuralset", "neuraltrain", "exca",
    "torch", "torchvision", "transformers", "huggingface_hub", "safetensors",
    "moviepy", "numpy", "pillow", "x-transformers", "scikit-learn",
)

#: Distributions whose version must appear in the feature uid.  Narrower than
#: CRITICAL_DISTRIBUTIONS: these are the ones on the tensor's critical path.
UID_DISTRIBUTIONS = (
    "tribev2", "neuralset", "exca",
    "torch", "torchvision", "transformers", "moviepy", "numpy",
)

_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

_ALGO_BY_DIGEST_LEN = {64: "sha256", 40: "git-blob-sha1"}


# --------------------------------------------------------------------------- #
# Typed errors.  Every message names a remedy; none of these is ever a bool.
# --------------------------------------------------------------------------- #

class ProvenanceError(RuntimeError):
    """Base for everything in this module."""


class WeightIdentityUnavailable(ProvenanceError):
    """Identity could not be established, and guessing is not an option."""


class WeightMismatch(ProvenanceError):
    """A local artifact does not match the identity it was supposed to have."""


class WeightFileMissing(ProvenanceError):
    """The artifact whose identity we were asked to verify is not on disk."""


class MissingIdentityField(ProvenanceError):
    """A tensor-affecting input was not supplied.

    Raised instead of omitting the field, because an omitted field produces a
    uid that silently ignores whatever it was supposed to cover -- which is the
    exact defect (F5/G3/G4/G5) this module exists to close.
    """


# --------------------------------------------------------------------------- #
# Digests
# --------------------------------------------------------------------------- #

def sha256_file(path, chunk_bytes: int = 1 << 20) -> str:
    """Streaming sha256.  Constant memory; never loads a 4 GB file into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1_file(path, chunk_bytes: int = 1 << 20) -> str:
    """The git object id of a blob: ``sha1(b"blob <size>\\0" + content)``.

    This is what huggingface_hub names non-LFS blobs by, so it is the digest for
    ``config.json`` and ``video_preprocessor_config.json``.  Verified against a
    real cache: 801-byte ``config.json`` -> ``3534852408cef7f5c0c54dfed6e0842c24492863``.
    """
    size = os.path.getsize(path)
    h = hashlib.sha1()
    h.update(b"blob %d\0" % size)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()


def digest_file(path, algo: str) -> str:
    if algo == "sha256":
        return sha256_file(path)
    if algo == "git-blob-sha1":
        return git_blob_sha1_file(path)
    raise ValueError(f"unknown digest algorithm {algo!r}; expected 'sha256' or 'git-blob-sha1'")


def _algo_for(digest: str) -> str:
    try:
        return _ALGO_BY_DIGEST_LEN[len(digest)]
    except KeyError:
        raise ValueError(
            f"digest {digest!r} is {len(digest)} hex chars; expected 64 (LFS sha256) "
            "or 40 (git blob sha1)"
        ) from None


def _is_digest_name(name: str) -> bool:
    return bool(_HEX64.match(name) or _HEX40.match(name))


# --------------------------------------------------------------------------- #
# Identity records
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FileIdentity:
    """One file's identity as the Hub names it."""
    filename: str
    digest: str
    algo: str
    size_bytes: int
    #: "local-cache" | "hub-api" -- how this particular row was obtained.
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or not _is_digest_name(self.digest):
            raise ValueError(
                f"{self.filename}: digest {self.digest!r} is not 40 or 64 lowercase hex chars")
        if self.algo != _algo_for(self.digest):
            raise ValueError(
                f"{self.filename}: algo {self.algo!r} disagrees with a "
                f"{len(self.digest)}-char digest (expected {_algo_for(self.digest)!r})")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError(f"{self.filename}: size_bytes must be a non-negative int, got {self.size_bytes!r}")

    def as_dict(self) -> dict:
        return {"filename": self.filename, "digest": self.digest,
                "algo": self.algo, "bytes": self.size_bytes, "source": self.source}


@dataclass(frozen=True)
class WeightIdentity:
    """A model repo pinned to a commit, with a digest and size per file.

    The commit sha transitively commits to the LFS oid (the tree entry for an LFS
    file is a ~135-byte pointer containing ``oid sha256:<digest>``), so pinning
    the commit does pin *which digest is expected*.  It does not guarantee the
    bytes you received match it -- see :func:`verify_local_weights`.
    """
    repo_id: str
    commit: str
    files: Mapping[str, FileIdentity]
    #: How the identity as a whole was obtained, e.g. "local-cache",
    #: "hub-api", "local-cache+hub-api".
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not _HEX40.match(self.commit or ""):
            raise ValueError(
                f"{self.repo_id}: commit {self.commit!r} is not a 40-char lowercase hex sha. "
                "A branch name is not an identity -- resolve it first.")
        object.__setattr__(self, "files", dict(sorted(self.files.items())))

    def file(self, filename: str) -> FileIdentity:
        try:
            return self.files[filename]
        except KeyError:
            raise MissingIdentityField(
                f"{self.repo_id}@{self.commit[:12]} carries no identity for {filename!r}; "
                f"it has {sorted(self.files)}. Re-run resolve_weight_identity() with "
                f"filenames including {filename!r}."
            ) from None

    @property
    def weights_sha256(self) -> str:
        return self.file(WEIGHTS_FILENAME).digest

    @property
    def processor_config_digest(self) -> str:
        return self.file(PROCESSOR_CONFIG_FILENAME).digest

    @property
    def model_config_digest(self) -> str:
        return self.file(MODEL_CONFIG_FILENAME).digest

    def as_dict(self) -> dict:
        return {"repo_id": self.repo_id, "commit": self.commit, "source": self.source,
                "files": {k: v.as_dict() for k, v in sorted(self.files.items())}}


@dataclass(frozen=True)
class WeightVerification:
    """What :func:`verify_local_weights` measured, and how it measured it."""
    filename: str
    path: str
    digest: str
    algo: str
    size_bytes: int
    #: "blob-filename" (the cache blob name already proves the digest, 0 bytes
    #: read) or "full-hash" (the file was streamed and hashed).
    route: str

    @property
    def hashed(self) -> bool:
        return self.route == "full-hash"


# --------------------------------------------------------------------------- #
# Resolution: local cache first (free, offline), metadata API only if permitted
# --------------------------------------------------------------------------- #

def default_hf_cache_dir() -> Path:
    """The hub cache directory, resolved the way huggingface_hub resolves it.

    Reimplemented rather than imported so this works with huggingface_hub absent
    -- which is the Kaggle-offline case we most need it in.
    """
    v = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if v:
        return Path(v)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "huggingface" / "hub"


def _repo_cache_dir(cache_dir: Path, repo_id: str) -> Path:
    return Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))


def _resolve_cached_commit(repo_root: Path, revision: str | None) -> str | None:
    """Turn a revision (or None -> 'main') into a commit sha using only the cache."""
    rev = revision or "main"
    if _HEX40.match(rev) and (repo_root / "snapshots" / rev).is_dir():
        return rev
    ref = repo_root / "refs" / rev
    try:
        sha = ref.read_text().strip()
    except OSError:
        return None
    return sha if _HEX40.match(sha) else None


def _scan_local_cache(cache_dir: Path, repo_id: str, filenames: Sequence[str],
                      revision: str | None) -> tuple[str | None, dict[str, FileIdentity]]:
    """Read identity straight off the cache layout.  Zero bytes of payload read.

    Returns ``(commit, {filename: FileIdentity})`` for whatever resolved; callers
    decide whether a partial result is good enough.

    Returns nothing for a file whose ``realpath`` is not a digest-shaped name --
    that happens under ``HF_HUB_DISABLE_SYMLINKS=1`` or when the cache was copied
    through something that dereferences symlinks (a Kaggle Dataset does).  Losing
    the digest is not an error here; it just means this route cannot serve.
    """
    root = _repo_cache_dir(cache_dir, repo_id)
    commit = _resolve_cached_commit(root, revision)
    if commit is None:
        return None, {}
    snap = root / "snapshots" / commit
    out: dict[str, FileIdentity] = {}
    for fn in filenames:
        p = snap / fn
        if not p.exists():
            continue
        real = os.path.realpath(p)
        base = os.path.basename(real)
        if base == os.path.basename(fn) or not _is_digest_name(base):
            continue  # symlinks unavailable -> the digest is not recoverable for free
        try:
            size = os.path.getsize(real)
        except OSError:
            continue
        out[fn] = FileIdentity(filename=fn, digest=base, algo=_algo_for(base),
                               size_bytes=size, source="local-cache")
    return commit, out


def _sibling_identity(fn: str, sib: Any) -> FileIdentity | None:
    """Pull (digest, algo, size) out of one huggingface_hub RepoSibling.

    ``lfs`` is a ``BlobLfsInfo`` in current versions and was a plain dict in
    older ones; both are accepted so this does not break on a hub bump.
    """
    lfs = getattr(sib, "lfs", None)
    sha256 = None
    lfs_size = None
    if isinstance(lfs, Mapping):
        sha256, lfs_size = lfs.get("sha256"), lfs.get("size")
    elif lfs is not None:
        sha256, lfs_size = getattr(lfs, "sha256", None), getattr(lfs, "size", None)
    size = getattr(sib, "size", None)
    if size is None:
        size = lfs_size
    digest = sha256 or getattr(sib, "blob_id", None)
    if not digest or size is None or not _is_digest_name(str(digest)):
        return None
    return FileIdentity(filename=fn, digest=str(digest), algo=_algo_for(str(digest)),
                        size_bytes=int(size), source="hub-api")


def _hub_metadata(repo_id: str, filenames: Sequence[str],
                  revision: str | None) -> tuple[str, dict[str, FileIdentity]]:
    """One ``model_info(files_metadata=True)`` call.  ~2.6 kB, no payload."""
    import importlib

    try:
        hfh = importlib.import_module("huggingface_hub")
    except Exception as ex:  # pragma: no cover - exercised via a fake module
        raise WeightIdentityUnavailable(
            f"huggingface_hub is not importable ({ex}), so the metadata API cannot be "
            "reached. Either pip install huggingface_hub, or pre-populate the HF cache "
            "and call resolve_weight_identity(allow_network=False)."
        ) from ex
    try:
        info = hfh.HfApi().model_info(repo_id, files_metadata=True, revision=revision)
    except Exception as ex:
        raise WeightIdentityUnavailable(
            f"model_info({repo_id!r}, revision={revision!r}) failed: {ex}. "
            "Check network/auth, or pre-populate the HF cache and call with "
            "allow_network=False."
        ) from ex
    commit = getattr(info, "sha", None)
    if not commit or not _HEX40.match(str(commit)):
        raise WeightIdentityUnavailable(
            f"model_info({repo_id!r}) returned sha={commit!r}, which is not a commit. "
            "Refusing to record an identity that is not one.")
    wanted = set(filenames)
    out: dict[str, FileIdentity] = {}
    for sib in getattr(info, "siblings", None) or ():
        fn = getattr(sib, "rfilename", None)
        if fn in wanted:
            ident = _sibling_identity(fn, sib)
            if ident is not None:
                out[fn] = ident
    return str(commit), out


def resolve_weight_identity(
    repo_id: str = VJEPA2_REPO,
    *,
    allow_network: bool,
    filenames: Sequence[str] = VJEPA2_FILENAMES,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
    expected_commit: str | None = None,
) -> WeightIdentity:
    """Identity of a model repo -- commit sha, per-file digest, per-file size.

    Cheapest sufficient route wins:

    1. **Local HF cache** -- 0 bytes read, 0 network, ``huggingface_hub`` is not
       even imported.  ``basename(realpath(...))`` is the digest.
    2. **Metadata API** -- one ``model_info(files_metadata=True)``, ~2.6 kB,
       measured at 0.086 s.  Only when ``allow_network=True``, and only for the
       files the cache could not answer for.

    ``model.safetensors`` is **never** downloaded by this function.

    Args:
        allow_network: keyword-only and mandatory, because "did this touch the
            network" is a property callers must decide, not inherit from a
            default.  With ``False`` this function performs no import of
            ``huggingface_hub`` and no socket operation whatsoever.
        revision: branch, tag or sha.  ``None`` means ``"main"`` for the cache
            lookup and the Hub's default for the API -- i.e. it reproduces what
            unpinned ``from_pretrained`` actually does.
        expected_commit: if given, a resolved commit that differs is a
            :class:`WeightMismatch`.  This is how you pin.

    Raises:
        WeightIdentityUnavailable: the identity is not knowable under the
            permissions given.  Never returns a partial or guessed identity.
        WeightMismatch: ``expected_commit`` was supplied and does not match.
    """
    filenames = tuple(filenames)
    if not filenames:
        raise ValueError("filenames is empty; there is no identity to resolve")

    cache_root = Path(cache_dir) if cache_dir is not None else default_hf_cache_dir()
    commit, files = _scan_local_cache(cache_root, repo_id, filenames, revision)
    sources = {"local-cache"} if files else set()

    absent = [fn for fn in filenames if fn not in files]
    if absent:
        if not allow_network:
            raise WeightIdentityUnavailable(
                f"{repo_id}: no cached identity for {absent} under {cache_root} "
                f"(revision={revision or 'main'}). "
                "Either pre-populate the cache with snapshot_download(revision=<sha>) "
                "and write refs/main (snapshot_download by sha writes no refs/ -- "
                "offline resolution of 'main' then fails with LocalEntryNotFoundError), "
                "or call resolve_weight_identity(..., allow_network=True). "
                "Refusing to guess an identity.")
        hub_commit, hub_files = _hub_metadata(repo_id, absent, revision)
        if commit is not None and hub_commit != commit:
            raise WeightMismatch(
                f"{repo_id}: the local cache holds revision {revision or 'main'} at "
                f"{commit} but the Hub now reports {hub_commit}. The branch moved under "
                "you -- exactly the F5 defect. Pin with revision=<sha> and re-populate "
                "the cache.")
        commit = hub_commit
        files = {**files, **hub_files}
        sources.add("hub-api")
        still_absent = [fn for fn in filenames if fn not in files]
        if still_absent:
            raise WeightIdentityUnavailable(
                f"{repo_id}@{commit}: the Hub listing carries no usable digest for "
                f"{still_absent}. Refusing to record a partial identity.")

    if commit is None:
        raise WeightIdentityUnavailable(
            f"{repo_id}: could not resolve revision {revision or 'main'} to a commit "
            f"from {cache_root} and network was not permitted.")
    if expected_commit is not None and commit != expected_commit:
        raise WeightMismatch(
            f"{repo_id}: resolved commit {commit} != pinned {expected_commit}. "
            "The weights are not the ones this experiment was designed against; "
            "refusing to continue.")

    ident = WeightIdentity(repo_id=repo_id, commit=commit,
                           files={fn: files[fn] for fn in filenames},
                           source="+".join(sorted(sources)) or "unknown")
    logger.info("weight identity %s@%s via %s", repo_id, commit[:12], ident.source)
    return ident


# --------------------------------------------------------------------------- #
# Verification: the one place that measures bytes instead of trusting the Hub
# --------------------------------------------------------------------------- #

def _locate(path_or_cache: str | os.PathLike, filename: str,
            commit: str | None = None) -> Path:
    """Accept the file itself, a snapshot dir, or a repo cache dir.

    When several snapshots are present the pinned ``commit`` is preferred, so a
    stale sibling snapshot cannot turn a correct artifact into a spurious
    :class:`WeightMismatch`.
    """
    p = Path(path_or_cache)
    if p.is_file():
        return p
    if p.is_dir():
        direct = p / filename
        if direct.exists():
            return direct
        # a repo cache dir: models--org--name/snapshots/<sha>/<filename>
        snaps = p / "snapshots"
        if snaps.is_dir():
            preferred = [snaps / commit] if commit else []
            for snap in preferred + sorted(snaps.iterdir()):
                cand = snap / filename
                if cand.exists():
                    return cand
    raise WeightFileMissing(
        f"{filename} not found at {p}. Pass the file, its snapshot directory, or the "
        "repo cache directory. Nothing was verified.")


def verify_local_weights(
    path_or_cache: str | os.PathLike,
    expected: WeightIdentity | FileIdentity | Mapping[str, Any],
    *,
    filename: str | None = None,
    force_hash: bool = False,
) -> WeightVerification:
    """Prove that the artifact on disk is the one ``expected`` names.

    Two routes, and the returned :class:`WeightVerification` says which was used:

    * ``"blob-filename"`` -- the cache blob is *named* by its digest, so a
      matching name plus a matching size is proof at the same trust level the
      Hub gave us, for 0 bytes read.  This is what lets a 4 GB artifact be
      re-checked every run for free.
    * ``"full-hash"`` -- the file is streamed and hashed.  This is the only route
      that measures rather than trusts, because huggingface_hub does **not**
      verify downloaded bytes against the sha256 it was given.  Pay for it once,
      at Stage-1 artifact build (``force_hash=True``); measured ~2.6 s for
      4.14 GB warm here, estimated 20-60 s cold on a Kaggle VM.

    A size disagreement short-circuits both routes -- it is one ``stat`` and it
    catches a truncated download before anything expensive happens.

    Returns:
        WeightVerification -- the digest, the size and the ``route``.  (The route
        is returned rather than only logged because "which route did you take"
        is the difference between *measured* and *trusted*, and a caller writing
        a provenance manifest has to record which one it got.)

    Raises:
        WeightFileMissing: nothing to verify at that path.
        WeightMismatch: size or digest disagreement.  Fatal, never a warning:
            a silently different set of weights is the failure this prevents.
    """
    if isinstance(expected, WeightIdentity):
        filename = filename or WEIGHTS_FILENAME
        exp = expected.file(filename)
    elif isinstance(expected, FileIdentity):
        exp = expected
        filename = filename or exp.filename
    elif isinstance(expected, Mapping):
        fn = filename or expected.get("filename")
        digest = expected.get("digest")
        if not fn or not digest:
            raise MissingIdentityField(
                "expected mapping needs at least 'filename' and 'digest' "
                f"(got keys {sorted(expected)}). Refusing to verify against nothing.")
        size = expected.get("size_bytes", expected.get("bytes"))
        if size is None:
            raise MissingIdentityField(
                f"expected mapping for {fn!r} has no size_bytes/bytes. A size-free "
                "check cannot short-circuit a truncated download; supply it.")
        exp = FileIdentity(filename=fn, digest=str(digest),
                           algo=str(expected.get("algo") or _algo_for(str(digest))),
                           size_bytes=int(size))
        filename = fn
    else:
        raise TypeError(
            f"expected must be a WeightIdentity, FileIdentity or mapping, got {type(expected).__name__}")

    commit = expected.commit if isinstance(expected, WeightIdentity) else None
    path = _locate(path_or_cache, filename, commit)
    real = Path(os.path.realpath(path))
    try:
        size = real.stat().st_size
    except OSError as ex:
        raise WeightFileMissing(
            f"{filename}: {path} resolves to {real}, which cannot be stat'd ({ex}). "
            "Nothing was verified.") from ex
    if size != exp.size_bytes:
        raise WeightMismatch(
            f"{filename}: {real} is {size} bytes, expected {exp.size_bytes} "
            f"({exp.algo} {exp.digest[:12]}...). A truncated or substituted artifact; "
            "delete it and re-download at the pinned revision.")

    base = real.name
    if not force_hash and _is_digest_name(base) and len(base) == len(exp.digest):
        if base != exp.digest:
            raise WeightMismatch(
                f"{filename}: the cache blob is named {base}, expected {exp.digest}. "
                "The cache holds a different revision of this file; delete the repo "
                "cache and re-download at the pinned revision.")
        logger.info("%s verified via blob-filename (%s, %d bytes, 0 read)",
                    filename, base[:12], size)
        return WeightVerification(filename=filename, path=str(real), digest=base,
                                  algo=exp.algo, size_bytes=size, route="blob-filename")

    got = digest_file(real, exp.algo)
    if got != exp.digest:
        raise WeightMismatch(
            f"{filename}: {exp.algo} of {real} is {got}, expected {exp.digest}. "
            "The bytes on disk are NOT the artifact this experiment was designed "
            "against -- huggingface_hub does not check this for you. Delete the cache "
            "and re-download at the pinned revision.")
    logger.info("%s verified via full-hash (%s, %d bytes read)", filename, got[:12], size)
    return WeightVerification(filename=filename, path=str(real), digest=got,
                              algo=exp.algo, size_bytes=size, route="full-hash")


# --------------------------------------------------------------------------- #
# Library versions
# --------------------------------------------------------------------------- #

#: What a distribution that is not installed renders as.  A literal is used
#: rather than omitting the key, so that installing the package later CHANGES
#: the uid instead of leaving a hole in it.
ABSENT = "absent"


def _vcs_commit(dist) -> str | None:
    """PEP 610 ``direct_url.json`` commit, or None.

    Required for ``tribev2``: ``importlib.metadata.version("tribev2")`` is the
    permanent constant ``"0.1.0"`` hard-coded in its pyproject, so the version
    string carries no information at all.
    """
    try:
        text = dist.read_text("direct_url.json")
    except Exception:
        return None
    if not text:
        return None
    try:
        info = (json.loads(text).get("vcs_info") or {})
    except Exception:
        return None
    commit = info.get("commit_id")
    return str(commit) if commit else None


def distribution_provenance(names: Iterable[str] = CRITICAL_DISTRIBUTIONS,
                            *, metadata=None) -> dict[str, dict | None]:
    """Rich per-distribution record: version, VCS commit, editable flag, url.

    Drop-in for ``neurocheck/s2_design.py:461-484`` ``environment_provenance``,
    whose ``__import__(mod).__version__`` at line 468 cannot work for
    ``neuralset`` (module ``__getattr__`` raises) or ``tribev2`` (no attribute).
    """
    md = metadata
    if md is None:
        import importlib.metadata as md  # noqa: PLC0415
    out: dict[str, dict | None] = {}
    for name in names:
        try:
            version = md.version(name)
        except Exception:
            out[name] = None
            continue
        rec: dict[str, Any] = {"version": version}
        try:
            dist = md.distribution(name)
        except Exception:
            dist = None
        if dist is not None:
            commit = _vcs_commit(dist)
            if commit:
                rec["vcs_commit"] = commit
            try:
                text = dist.read_text("direct_url.json")
                if text:
                    du = json.loads(text)
                    if du.get("url"):
                        rec["url"] = du["url"]
                    rr = (du.get("vcs_info") or {}).get("requested_revision")
                    if rr:
                        rec["requested_revision"] = rr
                    if (du.get("dir_info") or {}).get("editable"):
                        rec["editable"] = True
            except Exception:
                pass
        out[name] = rec
    return out


def library_versions(names: Iterable[str] = UID_DISTRIBUTIONS, *, metadata=None) -> dict[str, str]:
    """One deterministic string per distribution, suitable for a uid.

    A VCS install renders as ``"<version>+g<40-hex commit>"``.  That matters for
    ``tribev2``, whose declared version is a constant.
    """
    rich = distribution_provenance(names, metadata=metadata)
    out: dict[str, str] = {}
    for name, rec in rich.items():
        if rec is None:
            out[name] = ABSENT
            continue
        v = str(rec["version"])
        commit = rec.get("vcs_commit")
        out[name] = f"{v}+g{commit}" if commit else v
    return out


# --------------------------------------------------------------------------- #
# Canonical rendering
# --------------------------------------------------------------------------- #

def _render(value: Any, where: str) -> str:
    """Every uid value becomes a string, deterministically.

    Floats are rendered with ``repr``, which since Python 3.1 is the shortest
    string that round-trips exactly -- so ``0.00392156862745098`` stays
    distinguishable from ``0.00390625``.  ``"%.6f"`` would collapse them.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{where}: {value!r} is not a finite number")
        return repr(value + 0.0)  # normalises -0.0 -> 0.0
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_render(v, f"{where}[{i}]") for i, v in enumerate(value)) + "]"
    raise TypeError(
        f"{where}: {type(value).__name__} is not renderable into a uid. Convert it to "
        "a str/int/float/bool/None or a list of those.")


def _canonical_bytes(fields: Mapping[str, str]) -> bytes:
    return json.dumps(dict(sorted(fields.items())), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# The identity dict
# --------------------------------------------------------------------------- #

#: Content and container of the stimulus.  ``sha256`` closes B2/G1 (the exca item
#: uid keys on the *path*: overwrite the mp4 in place and every key still matches,
#: so the cache serves the OLD features, in read-only mode too).  fps/resolution
#: close G6 -- ``video.py:285`` samples by timestamp, so a re-encode at a
#: different fps changes which frames are seen at an unchanged duration.
REQUIRED_STIMULUS = ("sha256", "size_bytes", "duration_s", "fps", "width", "height")

#: ``ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30)``
#: -- hard-coded literals at ``tribev2/demo_utils.py:78``.  They decide the 18-item
#: set and every item's offset; nothing records them today.  Closes G11.
REQUIRED_CHUNKING = ("event_type", "max_duration", "min_duration")

#: Mirrors exca's own extractor uid, but *total*: ``num_frames_effective`` is the
#: resolved value, never ``None``.  With ``num_frames=None`` the operative value
#: is the literal ``64`` at ``video.py:404-405``, which ``exclude_defaults=True``
#: hides from exca's uid entirely -- so a neuralset that changes that literal
#: changes every tensor at an unchanged uid.  Closes G3 and G10.
REQUIRED_EXTRACTOR = (
    "class", "infra_version", "frequency", "clip_duration", "num_frames_effective",
    "max_imsize", "layer_type", "use_audio",
    "model_name", "pretrained", "imsize", "token_aggregation",
    "cache_all_layers", "cache_n_layers", "layers", "layer_aggregation",
)

#: From ``video_preprocessor_config.json``.  Recorded as *values*, not only as a
#: file digest, so that an equivalent-but-renamed config or a
#: ``from_pretrained(**overrides)`` kwarg is still covered.  Closes G5.
REQUIRED_PREPROCESSING = (
    "do_resize", "shortest_edge", "resample", "do_center_crop",
    "crop_height", "crop_width", "do_rescale", "rescale_factor",
    "do_normalize", "image_mean", "image_std", "video_processor_type",
)

#: Fields that may legitimately be ``None``.  Anything else that is ``None`` is a
#: caller bug, and a caller bug here silently widens the cache.
NULLABLE = frozenset({
    "extractor.max_imsize", "extractor.imsize", "extractor.cache_n_layers",
    "extractor.layers", "extractor.layer_aggregation",
})

#: Files whose identity must be present in the WeightIdentity.
REQUIRED_WEIGHT_FILES = (MODEL_CONFIG_FILENAME, PROCESSOR_CONFIG_FILENAME, WEIGHTS_FILENAME)


def _require(section: str, values: Mapping[str, Any], required: Sequence[str]) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"{section} must be a mapping, got {type(values).__name__}")
    missing = [k for k in required if k not in values]
    nulls = [k for k in required
             if k in values and values[k] is None and f"{section}.{k}" not in NULLABLE]
    problems = []
    if missing:
        problems.append(f"absent: {missing}")
    if nulls:
        problems.append(f"None (not nullable): {nulls}")
    if problems:
        raise MissingIdentityField(
            f"{section} is incomplete -- " + "; ".join(problems) + ". "
            "These inputs change the tensor, so omitting one would produce a uid that "
            "cannot tell two different runs apart. Supply them, or if a value is "
            "genuinely unobtainable, say so explicitly rather than dropping the key.")


def stimulus_fields(path, *, duration_s: float, fps: float,
                    width: int, height: int) -> dict[str, Any]:
    """Content identity of the stimulus: sha256 of the bytes plus its container.

    The sha256 is the fix for the single most dangerous gap in the pipeline --
    exca's item uid is ``f"{path}_{offset:.2f}_{duration:.2f}"``
    (``neuralset/extractors/video.py:247``) and never looks at the content.
    """
    p = Path(path)
    if not p.is_file():
        raise WeightFileMissing(f"stimulus {p} does not exist; cannot identify it by content.")
    return {"sha256": sha256_file(p), "size_bytes": p.stat().st_size,
            "duration_s": float(duration_s), "fps": float(fps),
            "width": int(width), "height": int(height)}


def preprocessing_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a ``video_preprocessor_config.json`` into the required keys.

    ``size``/``crop_size`` are nested dicts on the hub; flattened here so each
    number is an independent uid field.
    """
    size = config.get("size") or {}
    crop = config.get("crop_size") or {}
    return {
        "do_resize": config.get("do_resize"),
        "shortest_edge": size.get("shortest_edge"),
        "resample": config.get("resample"),
        "do_center_crop": config.get("do_center_crop"),
        "crop_height": crop.get("height"),
        "crop_width": crop.get("width"),
        "do_rescale": config.get("do_rescale"),
        "rescale_factor": config.get("rescale_factor"),
        "do_normalize": config.get("do_normalize"),
        "image_mean": list(config.get("image_mean") or []) or None,
        "image_std": list(config.get("image_std") or []) or None,
        "video_processor_type": config.get("video_processor_type"),
    }


def feature_uid_fields(
    *,
    stimulus: Mapping[str, Any],
    weights: WeightIdentity,
    extractor: Mapping[str, Any],
    chunking: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    versions: Mapping[str, str] | None = None,
    schema: int = IDENTITY_SCHEMA,
) -> dict[str, str]:
    """The identity dict handed to ``feature_artifact.verify_artifact``.

    Flat, dotted keys, string values.  Flat because ``verify_artifact`` diffs
    ``identity`` key by key (``tribe_tools/feature_artifact.py:132-135``) and
    names each drifted key in its error -- a nested dict would collapse an entire
    subtree into one unreadable diff.  Strings because the dict is JSON
    round-tripped through the manifest and float repr is not something to bet a
    4h45m recompute on.

    Every value supplied is included.  A required key that is absent -- or
    ``None`` where ``None`` is not meaningful -- raises
    :class:`MissingIdentityField`.  Extra keys are accepted and *kept*, so they
    still affect the uid; nothing a caller supplies is silently dropped.

    Note ``verify_artifact`` also rejects an artifact whose identity carries keys
    the current run does not check, so adding a field is safe in both directions:
    old artifact vs new checker is ``ArtifactStale``, not a false accept.
    """
    if not isinstance(weights, WeightIdentity):
        raise TypeError(
            f"weights must be a WeightIdentity from resolve_weight_identity(), got "
            f"{type(weights).__name__}. A bare string repo id is exactly the thing "
            "that is already in exca's uid and already insufficient.")
    _require("stimulus", stimulus, REQUIRED_STIMULUS)
    _require("chunking", chunking, REQUIRED_CHUNKING)
    _require("extractor", extractor, REQUIRED_EXTRACTOR)
    _require("preprocessing", preprocessing, REQUIRED_PREPROCESSING)

    for fn in REQUIRED_WEIGHT_FILES:
        weights.file(fn)  # raises MissingIdentityField naming the file

    vers = dict(library_versions() if versions is None else versions)
    _require("versions", vers, UID_DISTRIBUTIONS)

    out: dict[str, str] = {"schema": _render(int(schema), "schema")}
    for section, values in (("stimulus", stimulus), ("chunking", chunking),
                            ("extractor", extractor), ("preprocessing", preprocessing),
                            ("versions", vers)):
        for k in sorted(values):
            key = f"{section}.{k}"
            out[key] = _render(values[k], key)

    out["weights.repo_id"] = _render(weights.repo_id, "weights.repo_id")
    out["weights.commit"] = _render(weights.commit, "weights.commit")
    for fn, fid in sorted(weights.files.items()):
        out[f"weights.file.{fn}.digest"] = fid.digest
        out[f"weights.file.{fn}.algo"] = fid.algo
        out[f"weights.file.{fn}.bytes"] = str(fid.size_bytes)
    return out


def feature_set_uid(fields: Mapping[str, str], *, prefix: str = "s2v1-",
                    length: int = 16) -> str:
    """Stable short name for one identity.  Names the artifact directory."""
    return prefix + hashlib.sha256(_canonical_bytes(fields)).hexdigest()[:length]


def exca_infra_version(identity: Mapping[str, str] | WeightIdentity,
                       *, base: str = "release") -> str:
    """The string for ``data.video_feature.infra.version``.

    ``exca.base.BaseInfra`` declares ``version: str = "0"`` (``exca/base.py:138``)
    and formats it into both the cache path segment and the uid hash
    (``_uid_string = "{method},{version}/{uid}"``, ``exca/base.py:143``, applied at
    ``exca/base.py:293-304``).  ``@infra.apply`` in ``video.py:246-249`` passes no
    ``version=``, so the config value is respected -- and the channel is already
    live in this pipeline: ``tribev2/grids/defaults.py:94`` sets it to
    ``"release"``.  So this needs **no** change to neuralset or exca.

    The returned string carries the weight sha in clear (an operator can read the
    cache path and see which weights it belongs to) plus a hash over the *whole*
    identity, so exca's own cache self-invalidates on any tensor-affecting change
    -- not only on the weights.

    Deliberate consequence: adopting this changes the cache path from
    ``...,release/...`` to ``...,release+vjepa2-<sha12>+id-<hash>/...`` and so
    invalidates any existing cache.  That is correct: a cache built against
    unrecorded weights has no provable identity to keep.
    """
    if isinstance(identity, WeightIdentity):
        weight12 = identity.weights_sha256[:12]
        payload = {k: v for k, v in _flatten_weight_identity(identity).items()}
    else:
        if not isinstance(identity, Mapping):
            raise TypeError(
                f"identity must be a WeightIdentity or the feature_uid_fields() dict, "
                f"got {type(identity).__name__}")
        key = f"weights.file.{WEIGHTS_FILENAME}.digest"
        try:
            weight12 = identity[key][:12]
        except KeyError:
            raise MissingIdentityField(
                f"identity has no {key!r}; it is not a feature_uid_fields() dict. "
                "Refusing to emit an infra.version that does not name the weights."
            ) from None
        payload = identity
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()[:12]
    return f"{base}+vjepa2-{weight12}+id-{digest}"


def _flatten_weight_identity(identity: WeightIdentity) -> dict[str, str]:
    out = {"weights.repo_id": identity.repo_id, "weights.commit": identity.commit}
    for fn, fid in sorted(identity.files.items()):
        out[f"weights.file.{fn}.digest"] = fid.digest
        out[f"weights.file.{fn}.algo"] = fid.algo
        out[f"weights.file.{fn}.bytes"] = str(fid.size_bytes)
    return out
