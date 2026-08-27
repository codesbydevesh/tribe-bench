"""F4 -- the atlas preflight.  Fail at second 2, not at hour 5.

Today the first HCP-MMP1 parcel resolution happens inside ``analyse()``, AFTER
the GPU work: ``scripts/s2_run.py`` -> ``tribe_tools/atlas.py:127`` ->
``tribev2/utils.py:219-222`` (``mne.datasets.sample.data_path()`` +
``fetch_hcp_mmp_parcellation``).  A missing annot therefore costs ~5 h of GPU
and writes nothing.

This module moves that resolution to the front of the run and freezes its
result, so that:

* the whole thing is decided in ~2 s, before any model is loaded;
* the four assets are checked by IDENTITY, not by existence -- see below;
* ``analyse()`` reads a ~10 KB ``.npz`` and never imports mne at all.

Three measured facts this design rests on (Agent A, report_A2.md §2):

1.  Only four files are needed, 14 431 583 bytes total::

        <MNE_DATA>/MNE-sample-data/subjects/fsaverage/label/{lh,rh}.HCPMMP1.annot
        <MNE_DATA>/MNE-sample-data/subjects/fsaverage/surf/{lh,rh}.white

    out of the 2.8 GB sample dataset.  ``combine=False`` at
    ``tribev2/utils.py:269`` means ``HCPMMP1_combined.annot`` is never touched
    (``mne/datasets/utils.py:509-515`` is the ``if combine:`` branch).

2.  ``mne.datasets.sample.data_path()`` performs NO hash check and NO version
    check.  ``RELEASES.get("sample")`` is ``None``, so ``outdated`` is always
    ``False`` (``mne/datasets/_fetch.py:178-185``) and the early bail at
    ``mne/datasets/_fetch.py:201-208`` returns as soon as
    ``op.isdir(final_path)``.  Measured: an EMPTY ``MNE-sample-data/``
    directory returns in 1.37 s with no network.  **"The directory exists" is
    not evidence of anything.**  Hence the md5 check below.

3.  The resolved vertex sets depend on the two ``.annot`` files and NOTHING
    else.  ``Label.vertices`` comes from ``np.where(annot == label_id)[0]``
    (``mne/label.py:2337``); the ``.white`` surfaces are read only to fill
    ``Label.pos`` (``mne/label.py:2326-2333``).  Proven by overwriting both
    surfaces with random geometry: identical indices.  The surfaces are still
    REQUIRED -- ``read_labels_from_annot`` opens them -- but they do not
    participate in the answer.  So the answer is freezable.

Vertex conventions, matching ``tribev2/utils.py:213-247`` exactly (which
``tribe_tools/atlas.py:_get_hcp_labels_mne`` mirrors):

* mesh ``fsaverage5``, 10242 vertices per hemisphere, 20484 total;
* left hemisphere occupies 0..10241, right hemisphere 10242..20483
  (``index_offset = expected_size if hemi == "right" else 0``);
* a hemisphere's labels are the fsaverage (163842-vertex) label restricted to
  ``v < 10242``, i.e. fsaverage5 is the first 10242 vertices of fsaverage.

Typed errors only.  Nothing here ever returns a bare boolean, and nothing here
ever falls back to mne on a bad cache -- a silent fallback would reintroduce
the 5-hour failure it exists to prevent.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "AtlasPreflightError",
    "AtlasAssetsMissing",
    "AtlasIdentityMismatch",
    "AtlasUnresolvable",
    "AtlasDownloadForbidden",
    "AtlasCacheMissing",
    "AtlasCacheCorrupt",
    "HCP_ANNOT_MD5",
    "PARCEL_VERTEX_SHA256",
    "FSAVERAGE5_SIZE",
    "N_VERTICES",
    "SCHEMA_VERSION",
    "required_assets",
    "sample_data_root",
    "preflight_atlas",
    "load_frozen_parcels",
    "assert_atlas_ready",
    "atlas_manifest",
]

SCHEMA_VERSION = 1

#: Vertices per hemisphere on fsaverage5 (``FSAVERAGE_SIZES["fsaverage5"]``,
#: ``neuralset/extractors/neuro.py``).
FSAVERAGE5_SIZE = 10242
#: Model output width: left hemi first, then right.
N_VERTICES = 2 * FSAVERAGE5_SIZE
#: Vertices per hemisphere on the full fsaverage surface the annot describes
#: (asserted bare at ``tribev2/utils.py:240``).
FSAVERAGE_SIZE = 163842

#: MNE's OWN pinned md5s for the two HCP-MMP1 annots, copied from
#: ``mne/datasets/utils.py:487-489`` (mne 1.12.1), where they are the
#: ``known_hash`` passed to ``pooch.retrieve`` for two immutable figshare files
#: (``ndownloader.figshare.com/files/5528816`` and ``/5528819``).  They are
#: function-local there, so they cannot be imported; they must be duplicated
#: and cited.  Verified byte-for-byte against ``~/mne_data`` on this box.
HCP_ANNOT_MD5 = {
    "lh": "46a102b59b2fb1bb4bd62d51bf02e975",
    "rh": "75e96b331940227bbcb07c1c791c2463",
}

#: The frozen answer for S2's seven parcels: sha256 over
#: ``np.unique(verts).astype("<i8").tobytes()``.  Measured by Agent A on
#: mne 1.12.1, reproduced from two independent trees (the full 2.8 GB
#: ``~/mne_data`` and a hand-built 4-file tree) in three separate processes.
#: A mismatch means the HCP-MMP1 annot moved under the study; it is a hard
#: error, not a warning, because every published number is indexed by these.
PARCEL_VERTEX_SHA256 = {
    "FFA": "39e81595f8566401b57567cbb8d1f2fd9c7a8f647c9ccd362656018abd91c830",
    "EBA": "72f16de86344a9d9d3e5eaf6a2b9eae23e63b2747ba6c6199af5bfb180a229c6",
    "PPA": "36b4934983d3b54317b7a72ea01a16e242517134538ae82313b05d8bc71731d1",
    "VWFA": "4841b83621c46c16ac81a6457caa176f9a58900b0dc3494d74eef2b81bb2dc64",
    "PPA_literature": "f5b25b0f45e6256c535dde8148f4f1b0a9b99ee4e63cd0e9e6e1839685694ae9",
    "EBA_gate0_union": "dd5d37bb1f2b76a108aa76a03ed2b7b6edd820a0cc079d490432b183539d5ca4",
    "V1_control": "bb19f5acf6a69595e910e6e85e9cf56dd34c32ccbf66c01d03f5dde9ff263e90",
}

_MANIFEST_KEY = "__manifest__"
_PARCEL_PREFIX = "parcel::"


# --------------------------------------------------------------------- errors

class AtlasPreflightError(RuntimeError):
    """Base.  Every subclass message must name a remedy."""


class AtlasAssetsMissing(AtlasPreflightError):
    """A required file is not on disk.  Ship it, or allow the download."""


class AtlasIdentityMismatch(AtlasPreflightError):
    """A file is present but is not the file we mean, or the annots resolve to
    different vertices than the frozen study answer."""


class AtlasUnresolvable(AtlasPreflightError):
    """The assets are right but a parcel does not resolve to a usable vertex
    set (unknown label, empty, or out of the 20484 range)."""


class AtlasDownloadForbidden(AtlasPreflightError):
    """Something is absent and ``allow_download=False``.  Never downloaded."""


class AtlasCacheMissing(AtlasPreflightError):
    """The frozen cache is not there.  Run the preflight."""


class AtlasCacheCorrupt(AtlasPreflightError):
    """The frozen cache is there and is wrong.  We refuse to guess, and we do
    NOT fall back to mne -- that fallback is the 5-hour bug."""


# ----------------------------------------------------------------- asset side

def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _mne_config(key: str) -> str | None:
    """Read one key from ``~/.mne/mne-python.json`` WITHOUT importing mne.

    Importing mne costs ~1.4 s and 180 MB RSS; the whole point of the asset
    stage is to fail before paying that.  ``mne.get_config`` consults
    ``os.environ`` first, then this file (``mne/utils/config.py``), and
    ``_get_path`` (``mne/datasets/utils.py:112-140``) tries
    ``MNE_DATASETS_SAMPLE_PATH``, then ``MNE_DATA``, then ``~/mne_data``.
    """
    if key in os.environ:
        return os.environ[key]
    cfg = Path(os.getenv("_MNE_FAKE_HOME_DIR", "~")).expanduser() / ".mne" / "mne-python.json"
    try:
        return json.loads(cfg.read_text()).get(key)
    except (OSError, ValueError):
        return None


def sample_data_root(mne_root: str | os.PathLike | None = None) -> Path:
    """Where ``MNE-sample-data`` lives, resolved the way mne resolves it.

    ``mne_root`` is the MNE_DATA directory (the PARENT of ``MNE-sample-data``),
    matching ``MNE_DATASETS_SAMPLE_PATH``.  No mne import, no directory
    creation, no network.
    """
    if mne_root is not None:
        base = Path(mne_root).expanduser()
    else:
        found = _mne_config("MNE_DATASETS_SAMPLE_PATH") or _mne_config("MNE_DATA")
        if found:
            base = Path(found).expanduser()
        else:
            base = Path(os.getenv("_MNE_FAKE_HOME_DIR", "~")).expanduser() / "mne_data"
    return base / "MNE-sample-data"


def required_assets(mne_root: str | os.PathLike | None = None) -> dict[str, Path]:
    """The four files, by role.  13.76 MiB out of a 2.8 GB dataset."""
    fs = sample_data_root(mne_root) / "subjects" / "fsaverage"
    assets = {}
    for hemi in ("lh", "rh"):
        assets[f"{hemi}.annot"] = fs / "label" / f"{hemi}.HCPMMP1.annot"
        assets[f"{hemi}.white"] = fs / "surf" / f"{hemi}.white"
    return assets


def _check_assets(mne_root, *, allow_download: bool) -> dict[str, str]:
    """Existence AND identity, before mne is imported.

    Returns the measured annot md5s.  Raises before any expensive work.
    """
    assets = required_assets(mne_root)
    root = sample_data_root(mne_root)

    missing_annots = [h for h in ("lh", "rh") if not assets[f"{h}.annot"].is_file()]
    missing_surfs = [h for h in ("lh", "rh") if not assets[f"{h}.white"].is_file()]

    # The surfaces are NOT fetchable by fetch_hcp_mmp_parcellation -- only the
    # 2.8 GB sample tarball carries them -- so no amount of allow_download
    # helps here.  Say so.
    if missing_surfs:
        raise AtlasAssetsMissing(
            f"fsaverage white surface(s) absent: "
            f"{[str(assets[h + '.white']) for h in missing_surfs]}. "
            f"mne.read_labels_from_annot opens them (mne/label.py:2236) so they are "
            f"required even though they do not affect the vertex indices. Remedy: ship "
            f"subjects/fsaverage/surf/{{lh,rh}}.white (5 898 808 B each) under {root}, "
            f"or point mne_root/MNE_DATASETS_SAMPLE_PATH at a tree that has them."
        )

    if missing_annots:
        if not allow_download:
            raise AtlasDownloadForbidden(
                f"HCP-MMP1 annot(s) absent: "
                f"{[str(assets[h + '.annot']) for h in missing_annots]}, and "
                f"allow_download=False. NOTHING was downloaded. This is the exact file "
                f"whose absence today costs ~5 h of GPU before it is noticed "
                f"(scripts/s2_run.py -> tribe_tools/atlas.py -> tribev2/utils.py:219-222). "
                f"Remedy: ship the two annots (1 316 983 / 1 316 984 B) into "
                f"{root / 'subjects' / 'fsaverage' / 'label'}, or re-run the preflight "
                f"with allow_download=True on a box that has network."
            )
        # Only now, and only for the two 1.3 MB annots.  We deliberately do NOT
        # call mne.datasets.sample.data_path() to repair a missing tree: that
        # is a 2.8 GB download for 13.8 MiB of payload.
        import mne  # noqa: PLC0415  -- lazy on purpose: ~1.4 s, 180 MB RSS

        try:
            mne.datasets.fetch_hcp_mmp_parcellation(
                subjects_dir=root / "subjects", accept=True, verbose=False, combine=False
            )
        except Exception as exc:
            raise AtlasAssetsMissing(
                f"HCP-MMP1 annot(s) {missing_annots} absent under {root} and the "
                f"figshare fetch failed ({type(exc).__name__}: {exc}). Remedy: ship the "
                f"two annots with the inputs instead of relying on the network."
            ) from exc
        still = [h for h in missing_annots if not assets[f"{h}.annot"].is_file()]
        if still:
            raise AtlasAssetsMissing(
                f"fetch_hcp_mmp_parcellation returned but {still} still absent under "
                f"{root}. Remedy: fetch the files by hand from "
                f"https://ndownloader.figshare.com/files/5528816 and /5528819."
            )

    measured = {}
    for hemi in ("lh", "rh"):
        path = assets[f"{hemi}.annot"]
        got = _md5(path)
        measured[hemi] = got
        if got != HCP_ANNOT_MD5[hemi]:
            raise AtlasIdentityMismatch(
                f"{path} is present but is NOT the HCP-MMP1 annot: md5 {got} != "
                f"{HCP_ANNOT_MD5[hemi]} (mne's own pinned hash, mne/datasets/utils.py:"
                f"487-489). Note that mne.datasets.sample.data_path() would have "
                f"accepted this tree without complaint -- it does no hash or version "
                f"check at all. Remedy: delete {path} and re-fetch it "
                f"(allow_download=True), or ship the correct file."
            )
        if assets[f"{hemi}.white"].stat().st_size == 0:
            raise AtlasIdentityMismatch(
                f"{assets[hemi + '.white']} is empty. Remedy: re-ship the fsaverage "
                f"surfaces (5 898 808 B each)."
            )
    return measured


# ------------------------------------------------------------ live resolution

def _label_to_vertices(subjects_dir: Path, mesh_size: int, hemi: str) -> dict[str, np.ndarray]:
    """Verbatim re-implementation of ``tribev2/utils.py:213-247``.

    The two bare ``assert``s there (``:240`` and ``:246``) become typed errors
    with a diagnosis; that is the only difference, and it is the point.
    """
    if hemi not in ("left", "right", "both"):
        raise AtlasUnresolvable(f"invalid hemisphere {hemi!r}; use left/right/both")

    if hemi == "both":
        left = _label_to_vertices(subjects_dir, mesh_size, "left")
        right = _label_to_vertices(subjects_dir, mesh_size, "right")
        return {k: np.concatenate([left[k], right[k]]) for k in left}

    import mne  # noqa: PLC0415

    labels = mne.read_labels_from_annot(
        "fsaverage", "HCPMMP1", hemi="both", subjects_dir=subjects_dir, verbose="ERROR"
    )
    out: dict[str, np.ndarray] = {}
    for label in labels:
        name, vertices = label.name, np.array(label.vertices)
        name = name[2:]                      # strip the "L_"/"R_" prefix
        name = name.replace("_ROI", "")
        if (hemi == "right" and "-lh" in name) or (hemi == "left" and "-rh" in name):
            continue
        name = name.replace("-rh", "").replace("-lh", "")
        out[name] = np.array(vertices)

    total = sum(len(v) for v in out.values())
    if total != FSAVERAGE_SIZE:
        raise AtlasIdentityMismatch(
            f"the {hemi} HCP-MMP1 annots cover {total} fsaverage vertices, expected "
            f"{FSAVERAGE_SIZE} (tribev2/utils.py:240 asserts this bare). The annot "
            f"files parse but are not the parcellation this study was built on. "
            f"Remedy: restore the pinned annots (md5 {HCP_ANNOT_MD5})."
        )

    index_offset = mesh_size if hemi == "right" else 0
    out = {k: v[v < mesh_size] + index_offset for k, v in out.items()}
    total = sum(len(v) for v in out.values())
    if total != mesh_size:
        raise AtlasIdentityMismatch(
            f"downsampling to the first {mesh_size} vertices left {total} vertices in "
            f"the {hemi} hemisphere, expected {mesh_size} (tribev2/utils.py:246). "
            f"Remedy: restore the pinned annots."
        )
    return out


def _select(labels: Mapping[str, np.ndarray], roi: str) -> list[str]:
    """``get_hcp_roi_indices``' wildcard rules (``tribev2/utils.py:273-281``)."""
    if roi.endswith("*"):
        sel = [name for name in labels if name.startswith(roi[:-1])]
    elif roi.startswith("*"):
        sel = [name for name in labels if name.endswith(roi[1:])]
    else:
        sel = [name for name in labels if name == roi]
    return sel


def _resolve_parcel(parcel, cache: dict[str, dict[str, np.ndarray]],
                    subjects_dir: Path, mesh_size: int) -> np.ndarray:
    """One parcel -> its unique vertex indices, exactly as ``analyse()`` does:
    ``np.unique(np.concatenate([get_vertices(l, hemi=p.hemi) for l in p.labels]))``
    (``scripts/s2_run.py:198-202``)."""
    hemi = parcel.hemi
    if hemi not in cache:
        cache[hemi] = _label_to_vertices(subjects_dir, mesh_size, hemi)
    labels = cache[hemi]

    chunks = []
    for roi in parcel.labels:
        sel = _select(labels, roi)
        if not sel:
            raise AtlasUnresolvable(
                f"parcel {parcel.name}: HCP-MMP1 label {roi!r} does not exist in this "
                f"annot ({len(labels)} labels available, e.g. "
                f"{sorted(labels)[:6]}). Today this raises inside analyse() at hour 5 "
                f"(scripts/s2_run.py:200-201 -> die()). Remedy: fix the label in "
                f"neurocheck/s2_design.py, or restore the pinned annots."
            )
        chunks.extend(labels[name] for name in sel)

    verts = np.unique(np.concatenate(chunks))
    if verts.size == 0:
        raise AtlasUnresolvable(
            f"parcel {parcel.name} ({list(parcel.labels)}, hemi={hemi}) resolves to ZERO "
            f"vertices. Remedy: fix the parcel definition in neurocheck/s2_design.py."
        )
    if verts.min() < 0 or verts.max() >= N_VERTICES:
        raise AtlasUnresolvable(
            f"parcel {parcel.name} resolves to vertices outside 0..{N_VERTICES - 1} "
            f"(min {verts.min()}, max {verts.max()}); the model output is "
            f"(n_segments, {N_VERTICES}). Remedy: check the mesh -- this preflight "
            f"only supports fsaverage5."
        )
    if hemi == "left" and verts.max() >= mesh_size:
        raise AtlasUnresolvable(
            f"parcel {parcel.name} is declared hemi='left' but holds right-hemisphere "
            f"vertices (max {verts.max()} >= {mesh_size}). Left hemi is 0..{mesh_size - 1}."
        )
    if hemi == "right" and verts.min() < mesh_size:
        raise AtlasUnresolvable(
            f"parcel {parcel.name} is declared hemi='right' but holds left-hemisphere "
            f"vertices (min {verts.min()} < {mesh_size}). Right hemi is "
            f"{mesh_size}..{N_VERTICES - 1}."
        )
    return verts.astype("<i8", copy=False)


def parcel_digest(verts: np.ndarray) -> str:
    """sha256 over ``np.unique(v).astype("<i8").tobytes()``.  Same recipe as the
    frozen table, so the two are directly comparable."""
    a = np.unique(np.asarray(verts)).astype("<i8")
    return hashlib.sha256(a.tobytes()).hexdigest()


# ------------------------------------------------------------------ the cache

def _atomic_write(dest: Path, payload: bytes) -> None:
    """Temp file INSIDE the destination directory (EXDEV across Kaggle mounts
    makes a ``/tmp`` tempfile unrenameable), fsync, then ``os.replace``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _pack(manifest: dict, indices: Mapping[str, np.ndarray]) -> bytes:
    import io

    buf = io.BytesIO()
    payload = {_PARCEL_PREFIX + k: np.asarray(v).astype("<i8") for k, v in indices.items()}
    payload[_MANIFEST_KEY] = np.array(json.dumps(manifest, sort_keys=True))
    np.savez(buf, **payload)
    return buf.getvalue()


def preflight_atlas(parcels: Iterable, cache_path: str | os.PathLike, *,
                    allow_download: bool,
                    mne_root: str | os.PathLike | None = None,
                    mesh: str = "fsaverage5") -> dict:
    """Prove the atlas, freeze the answer.  Run this BEFORE ``load_model``.

    Verifies the four assets are present AND identity-correct, resolves every
    parcel to its vertex indices through mne, checks each against the frozen
    study answer, and writes a self-verifying ~10 KB cache atomically.

    Args:
        parcels: iterable of objects with ``.name``, ``.labels``, ``.hemi``
            (``neurocheck.s2_design.ALL_PARCELS``).
        cache_path: where to write the frozen ``.npz``.
        allow_download: if False, NOTHING may touch the network -- a missing
            annot raises ``AtlasDownloadForbidden`` instead.  If True, only the
            two 1.3 MB figshare annots may be fetched; the 2.8 GB sample
            tarball is never downloaded, because 13.76 MiB of it is all we need.
        mne_root: the MNE_DATA directory (parent of ``MNE-sample-data``).
            Defaults to mne's own resolution order.
        mesh: fsaverage5 only.

    Returns:
        A summary dict (also embedded in the cache as its manifest).

    Raises:
        AtlasAssetsMissing, AtlasDownloadForbidden, AtlasIdentityMismatch,
        AtlasUnresolvable -- each naming a remedy.  Asset errors are raised
        before mne is imported.
    """
    t0 = time.time()
    parcels = list(parcels)
    if not parcels:
        raise AtlasUnresolvable(
            "no parcels given; there is nothing to prove. Remedy: pass "
            "neurocheck.s2_design.ALL_PARCELS."
        )
    if mesh != "fsaverage5":
        raise AtlasUnresolvable(
            f"mesh={mesh!r}: this preflight is pinned to fsaverage5 "
            f"({FSAVERAGE5_SIZE} per hemi, {N_VERTICES} total), which is what the "
            f"model emits. Remedy: do not change the mesh."
        )

    # --- stage 1: assets.  Cheap, and it happens before `import mne`.
    annot_md5 = _check_assets(mne_root, allow_download=allow_download)

    # --- stage 2: the real resolution.  ~0.2 s once mne is loaded.
    subjects_dir = sample_data_root(mne_root) / "subjects"
    cache: dict[str, dict[str, np.ndarray]] = {}
    indices: dict[str, np.ndarray] = {}
    for parcel in parcels:
        indices[parcel.name] = _resolve_parcel(parcel, cache, subjects_dir, FSAVERAGE5_SIZE)

    # --- stage 3: is it the SAME atlas the study's numbers were computed on?
    digests = {name: parcel_digest(v) for name, v in indices.items()}
    drift = {n: (d, PARCEL_VERTEX_SHA256[n]) for n, d in digests.items()
             if n in PARCEL_VERTEX_SHA256 and d != PARCEL_VERTEX_SHA256[n]}
    if drift:
        raise AtlasIdentityMismatch(
            "the annots parse and the md5s match, but the resolved vertex sets differ "
            "from the frozen study answer -- every published S2 number is indexed by "
            "these: "
            + "; ".join(f"{n}: got {g[:16]}... want {w[:16]}..." for n, (g, w) in sorted(drift.items()))
            + ". Remedy: this is an atlas version change, not a config problem. Restore "
            "the pinned annots, or re-freeze PARCEL_VERTEX_SHA256 deliberately and "
            "re-run every affected analysis."
        )

    import mne  # noqa: PLC0415  -- already imported by the resolution above

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mesh": mesh,
        "n_vertices": N_VERTICES,
        "hemi_size": FSAVERAGE5_SIZE,
        "mne_version": mne.__version__,
        "annot_md5": annot_md5,
        "subjects_dir": str(subjects_dir),
        "parcels": {
            name: {
                "n": int(indices[name].size),
                "sha256": digests[name],
                "hemi": p.hemi,
                "labels": list(p.labels),
                "min": int(indices[name].min()),
                "max": int(indices[name].max()),
            }
            for name, p in ((p.name, p) for p in parcels)
        },
    }

    cache_path = Path(cache_path)
    _atomic_write(cache_path, _pack(manifest, indices))

    summary = dict(manifest)
    summary["cache_path"] = str(cache_path)
    summary["n_parcels"] = len(indices)
    summary["total_vertices"] = int(sum(v.size for v in indices.values()))
    summary["elapsed_s"] = round(time.time() - t0, 3)
    return summary


def atlas_manifest(cache_path: str | os.PathLike) -> dict:
    """The cache's manifest, fully verified.  No mne import, no network.

    This is the cheap "is the atlas ready?" assertion: it reads ~10 KB, checks
    a sha256 per parcel, and raises a typed error otherwise.
    """
    return _read(cache_path)[0]


def load_frozen_parcels(cache_path: str | os.PathLike) -> dict[str, np.ndarray]:
    """The frozen vertex sets, verified.  Never imports mne, never falls back.

    Raises:
        AtlasCacheMissing if the cache is not there;
        AtlasCacheCorrupt if it is there and does not verify.

    There is deliberately no ``except: resolve_live()`` branch.  A silent
    fallback would put ``import mne`` and ``fetch_hcp_mmp_parcellation`` back on
    the post-GPU path, which is the entire bug this module exists to remove.
    """
    return _read(cache_path)[1]


def _read(cache_path):
    path = Path(cache_path)
    if not path.is_file():
        raise AtlasCacheMissing(
            f"no frozen atlas at {path}. Remedy: run the atlas preflight "
            f"(preflight_atlas(ALL_PARCELS, {path!s}, allow_download=...)) before "
            f"the GPU stage. This module will NOT fall back to mne."
        )
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.files)
            if _MANIFEST_KEY not in keys:
                raise AtlasCacheCorrupt(
                    f"{path} carries no manifest. Remedy: delete it and re-run the "
                    f"atlas preflight."
                )
            manifest = json.loads(str(z[_MANIFEST_KEY]))
            arrays = {k[len(_PARCEL_PREFIX):]: np.asarray(z[k])
                      for k in keys if k.startswith(_PARCEL_PREFIX)}
    except AtlasPreflightError:
        raise
    except Exception as exc:
        raise AtlasCacheCorrupt(
            f"{path} is unreadable ({type(exc).__name__}: {exc}). Remedy: delete it "
            f"and re-run the atlas preflight."
        ) from exc

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise AtlasCacheCorrupt(
            f"{path} has schema {manifest.get('schema_version')!r}, this code writes "
            f"{SCHEMA_VERSION}. Remedy: delete it and re-run the atlas preflight."
        )
    if manifest.get("n_vertices") != N_VERTICES or manifest.get("hemi_size") != FSAVERAGE5_SIZE:
        raise AtlasCacheCorrupt(
            f"{path} was frozen for a {manifest.get('n_vertices')}-vertex mesh "
            f"({manifest.get('hemi_size')} per hemi); this run expects {N_VERTICES} "
            f"({FSAVERAGE5_SIZE}). Remedy: delete it and re-run the atlas preflight."
        )

    recorded = manifest.get("parcels")
    if not isinstance(recorded, dict) or not recorded:
        raise AtlasCacheCorrupt(
            f"{path} records no parcels. Remedy: delete it and re-run the preflight.")
    if set(recorded) != set(arrays):
        raise AtlasCacheCorrupt(
            f"{path}: manifest names {sorted(recorded)} but the file holds "
            f"{sorted(arrays)}. Remedy: delete it and re-run the atlas preflight."
        )

    for name in sorted(recorded):
        arr = arrays[name]
        entry = recorded[name]
        if arr.ndim != 1 or arr.size == 0:
            raise AtlasCacheCorrupt(
                f"{path}: parcel {name} is not a non-empty 1-D index array "
                f"(shape {arr.shape}). Remedy: delete it and re-run the preflight."
            )
        if arr.size != entry.get("n"):
            raise AtlasCacheCorrupt(
                f"{path}: parcel {name} holds {arr.size} vertices, manifest says "
                f"{entry.get('n')}. Remedy: delete it and re-run the preflight."
            )
        got = parcel_digest(arr)
        if got != entry.get("sha256"):
            raise AtlasCacheCorrupt(
                f"{path}: parcel {name} does not match its recorded sha256 "
                f"({got[:16]}... != {str(entry.get('sha256'))[:16]}...). The frozen "
                f"vertex set has been altered. Remedy: delete {path} and re-run the "
                f"atlas preflight. We will NOT silently re-resolve through mne."
            )
        if arr.min() < 0 or arr.max() >= N_VERTICES:
            raise AtlasCacheCorrupt(
                f"{path}: parcel {name} indexes outside 0..{N_VERTICES - 1}. Remedy: "
                f"delete it and re-run the atlas preflight."
            )
    return manifest, {k: np.asarray(v) for k, v in arrays.items()}


def assert_atlas_ready(cache_path: str | os.PathLike,
                       parcels: Sequence | None = None) -> dict:
    """Cheap gate for the caller: "the atlas is ready", or a typed error.

    Reads ~10 KB and hashes it.  Does NOT import mne (so it costs neither the
    1.4 s nor the 180 MB, and cannot touch the network).  When ``parcels`` is
    given, the frozen set must cover exactly those names -- a cache frozen for
    a different design is a stale cache, not a usable one.

    Returns the verified manifest.
    """
    manifest = atlas_manifest(cache_path)
    if parcels is not None:
        want = {p.name for p in parcels}
        have = set(manifest["parcels"])
        if want != have:
            raise AtlasCacheCorrupt(
                f"{cache_path} was frozen for {sorted(have)}, this design needs "
                f"{sorted(want)} (missing {sorted(want - have)}, extra "
                f"{sorted(have - want)}). Remedy: re-run the atlas preflight with the "
                f"current ALL_PARCELS."
            )
    return manifest
