"""The V-JEPA feature artifact: write it, then prove it before consuming it.

The point of this file is that verification READS EVERY PAYLOAD and compares a
per-item sha256 recorded at write time.  Presence-based verification (key count,
COMPLETE marker, exca `missing == 0`) is defeated by D2's tarpit; this is not.

Typed errors only.  verify() NEVER returns a boolean.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

MANIFEST = "S2_FEATURES.json"
COMPLETE = "S2_FEATURES.COMPLETE"
SCHEMA_VERSION = 1
SCHEMA = SCHEMA_VERSION  # back-compat alias


class FeatureArtifactError(RuntimeError):
    """Base.  Every subclass message must name a remedy."""

class ArtifactMissing(FeatureArtifactError): pass
class ArtifactIncomplete(FeatureArtifactError): pass
class ArtifactCorrupt(FeatureArtifactError): pass
class ArtifactStale(FeatureArtifactError): pass


def item_digest(arr) -> str:
    """sha256 over dtype+shape+C-contiguous bytes.  Shape/dtype are inside the
    hash so a reshape-preserving corruption still changes the digest."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(b"|")
    h.update(repr(tuple(a.shape)).encode()); h.update(b"|")
    h.update(a.tobytes())
    return h.hexdigest()


def _atomic_write(dest: Path, payload: bytes) -> None:
    """Temp file INSIDE the destination directory (D4/EXDEV), then os.replace.
    Never os.replace a directory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def sidecar_digests(uid_folder) -> dict:
    """exca's own provenance files.  D2 showed all four can be deleted and the
    cache still serves, AND that a read-only Stage 2 RE-CREATES uid.yaml and
    config.yaml from its own config -- provenance laundering.  Recording them at
    write time and checking them at read time is the only defence."""
    uf = Path(uid_folder)
    out = {}
    for name in ("uid.yaml", "full-uid.yaml", "config.yaml"):
        f = uf / name
        out[name] = hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None
    return out


def begin_stage1(cache_root) -> None:
    """Stage 1's FIRST act.  An artifact under construction must not carry a
    COMPLETE marker from an earlier session -- otherwise a Stage 1 that dies
    during read-back leaves the previous run's certificate over poisoned bytes.
    (Found by executing e2_tarpit.py: COMPLETE survived a failed read-back.)"""
    c = Path(cache_root) / COMPLETE
    if c.exists():
        c.unlink()


def write_artifact(cache_root, identity: dict, materialised: dict,
                   sidecars: dict | None = None) -> dict:
    """Stage 1's LAST act.  `materialised` maps item_uid -> the array that was
    READ BACK from the cache (not the array that was computed in RAM).

    Callers must pass values obtained by iterating the generator exca returned
    -- neuralset/extractors/base.py:201 discards it, which is exactly the hole.
    """
    root = Path(cache_root)
    man = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "n_items": len(materialised),
        "items": {uid: item_digest(a) for uid, a in sorted(materialised.items())},
        "exca_sidecars": sidecars or {},
    }
    _atomic_write(root / MANIFEST, json.dumps(man, indent=2, sort_keys=True).encode())
    _atomic_write(root / COMPLETE, b"")          # LAST, atomically, same filesystem
    return man


def verify_artifact(cache_root, identity: dict, expected_uids, read_item,
                    sidecars: dict | None = None):
    """Raises exactly one typed error, or returns the manifest.

    `read_item(uid) -> array` is the ONLY way payload bytes enter this function.
    Every expected uid is read and hashed.  There is no fast path.
    """
    root = Path(cache_root)
    if not root.is_dir():
        raise ArtifactMissing(
            f"no feature artifact at {root}. Re-run: s2_run.py --extract-features")
    if not (root / MANIFEST).is_file():
        raise ArtifactMissing(
            f"{MANIFEST} absent in {root}. Re-run: s2_run.py --extract-features")
    if not (root / COMPLETE).is_file():
        raise ArtifactIncomplete(
            f"{COMPLETE} absent: --extract-features did not finish. "
            f"Delete {root} and re-run: s2_run.py --extract-features")
    try:
        man = json.loads((root / MANIFEST).read_text())
    except Exception as ex:
        raise ArtifactCorrupt(
            f"{MANIFEST} is unreadable ({ex}). Delete {root} and re-run "
            "s2_run.py --extract-features") from ex
    if man.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactStale(f"manifest schema {man.get('schema_version')} != {SCHEMA_VERSION}; "
                            "delete and re-run s2_run.py --extract-features")

    drift = {k: (man["identity"].get(k), v) for k, v in identity.items()
             if man["identity"].get(k) != v}
    if drift:
        raise ArtifactStale(
            "the artifact was built from different inputs; it cannot be reused. "
            + "; ".join(f"{k}: artifact={a!r} now={b!r}" for k, (a, b) in sorted(drift.items()))
            + f". Delete {root} and re-run s2_run.py --extract-features")
    extra_keys = set(man["identity"]) - set(identity)
    if extra_keys:
        raise ArtifactStale(
            f"the artifact declares identity fields this run does not check: "
            f"{sorted(extra_keys)}. Refusing rather than ignoring them.")

    expected = list(expected_uids)
    if man["n_items"] != len(man["items"]) or man["n_items"] != len(expected):
        raise ArtifactIncomplete(
            f"manifest declares {man['n_items']} items, holds {len(man['items'])} "
            f"digests, design expects {len(expected)}. Delete {root} and re-run "
            "s2_run.py --extract-features")
    absent = [u for u in expected if u not in man["items"]]
    if absent:
        raise ArtifactIncomplete(
            f"{len(absent)} expected item(s) have no digest: {absent[:3]}. "
            f"Delete {root} and re-run s2_run.py --extract-features")

    recorded = man.get("exca_sidecars") or {}
    if sidecars is not None:
        for name, want in sorted(recorded.items()):
            got = sidecars.get(name)
            if want is not None and got is None:
                raise ArtifactCorrupt(
                    f"exca provenance file {name} was present when the artifact was "
                    f"built and is now ABSENT. exca serves the data anyway and a "
                    f"read-only stage will RE-CREATE it from its own config, "
                    f"laundering the provenance. Delete {root} and re-run "
                    "s2_run.py --extract-features")
            if want != got:
                raise ArtifactStale(
                    f"exca provenance file {name} changed: artifact={want} now={got}. "
                    f"Delete {root} and re-run s2_run.py --extract-features")

    bad, unreadable = [], []
    for uid in expected:
        try:
            arr = read_item(uid)
            got = item_digest(arr)
        except Exception as ex:
            unreadable.append((uid, f"{type(ex).__name__}: {ex}"))
            continue
        if got != man["items"][uid]:
            bad.append(uid)
    if unreadable:
        raise ArtifactCorrupt(
            f"{len(unreadable)} item(s) are indexed but UNREADABLE -- this is a "
            f"poisoned cache and there is no repair: {unreadable[:2]}. "
            f"You MUST `rm -rf {root}` and re-run s2_run.py --extract-features")
    if bad:
        raise ArtifactCorrupt(
            f"{len(bad)} item(s) do not match their recorded sha256: {bad[:3]}. "
            f"You MUST `rm -rf {root}` and re-run s2_run.py --extract-features")
    return man
