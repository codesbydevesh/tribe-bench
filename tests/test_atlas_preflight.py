"""The atlas preflight must fail at second 2, and it must not lie.

Every test here is behavioural: it builds a real MNE tree (by symlink, so
``~/mne_data`` is only ever READ), damages it in one specific way, and asserts
the typed error.  Nothing inspects source text.

Two traps are pinned explicitly, because both defeat the obvious design:

* ``test_empty_sample_directory_is_accepted_by_mne_but_not_by_us`` -- an EMPTY
  ``MNE-sample-data/`` satisfies ``mne.datasets.sample.data_path()``, which
  does no hash and no version check.  "The directory exists" proves nothing.
* ``test_corrupt_cache_raises_and_never_falls_back_to_mne`` -- a preflight that
  quietly re-resolves through mne on a bad cache has reintroduced the 5-hour
  failure it exists to prevent.

``~/mne_data`` is read-only here: the fixtures symlink into it and nothing in
this file writes, deletes, or downloads anything under it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from neurocheck.s2_design import ALL_PARCELS
from tribe_tools.atlas_preflight import (
    FSAVERAGE5_SIZE,
    HCP_ANNOT_MD5,
    N_VERTICES,
    PARCEL_VERTEX_SHA256,
    SCHEMA_VERSION,
    AtlasAssetsMissing,
    AtlasCacheCorrupt,
    AtlasCacheMissing,
    AtlasDownloadForbidden,
    AtlasIdentityMismatch,
    AtlasPreflightError,
    AtlasUnresolvable,
    assert_atlas_ready,
    load_frozen_parcels,
    preflight_atlas,
    required_assets,
    sample_data_root,
)

REPO = Path(__file__).resolve().parents[1]
REAL_ASSETS = required_assets(None)
HAVE_REAL = all(p.is_file() for p in REAL_ASSETS.values())

pytestmark = pytest.mark.skipif(
    not HAVE_REAL,
    reason=f"needs a read-only HCP-MMP1 tree; missing "
           f"{[str(p) for p in REAL_ASSETS.values() if not p.is_file()]}",
)


# ------------------------------------------------------------------ fixtures

class FakeParcel:
    """Duck-types ``neurocheck.s2_design.Parcel`` for the negative cases."""

    def __init__(self, name, labels, hemi):
        self.name, self.labels, self.hemi = name, labels, hemi


def _tree(tmp_path, *, annots=True, surfaces=True, name="mne_root") -> Path:
    """A minimal MNE_DATA root, built from symlinks into the real tree.

    Returns the MNE_DATA directory (the parent of ``MNE-sample-data``), which
    is what ``mne_root`` / ``MNE_DATASETS_SAMPLE_PATH`` expect.
    """
    root = tmp_path / name
    fs = root / "MNE-sample-data" / "subjects" / "fsaverage"
    (fs / "label").mkdir(parents=True, exist_ok=True)
    (fs / "surf").mkdir(parents=True, exist_ok=True)
    for hemi in ("lh", "rh"):
        if annots:
            (fs / "label" / f"{hemi}.HCPMMP1.annot").symlink_to(REAL_ASSETS[f"{hemi}.annot"])
        if surfaces:
            (fs / "surf" / f"{hemi}.white").symlink_to(REAL_ASSETS[f"{hemi}.white"])
    return root


def _sub(body: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a snippet in a clean interpreter, so that `import mne` and network
    use are observable facts rather than claims."""
    e = dict(os.environ)
    e.pop("MNE_DATASETS_SAMPLE_PATH", None)
    e.pop("MNE_DATA", None)
    e.update(env or {})
    return subprocess.run([sys.executable, "-c", textwrap.dedent(body)],
                          capture_output=True, text=True, cwd=REPO, env=e, timeout=300)


# Indented to 8 spaces so it concatenates cleanly with the call sites' f-strings
# before textwrap.dedent sees them.
NO_NETWORK = """
        import socket, ssl                      # ssl first: it subclasses socket.socket
        def _blocked(*a, **k):
            raise AssertionError("NETWORK ACCESS ATTEMPTED")
        class _NoNet(socket.socket):            # a class, so ssl/http stay importable
            def connect(self, *a, **k): _blocked()
            def connect_ex(self, *a, **k): _blocked()
        socket.socket = _NoNet
        socket.create_connection = _blocked
        socket.getaddrinfo = _blocked
"""


# -------------------------------------------------- 1. fail before the GPU...
# ...and, more strictly, before mne is even imported.

def test_absent_assets_raise_before_mne_is_imported(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    r = _sub(NO_NETWORK + f"""
        import sys
        from neurocheck.s2_design import ALL_PARCELS
        from tribe_tools.atlas_preflight import preflight_atlas, AtlasPreflightError
        try:
            preflight_atlas(ALL_PARCELS, {str(tmp_path / 'c.npz')!r},
                            allow_download=False, mne_root={str(empty)!r})
        except AtlasPreflightError as exc:
            print("TYPED", type(exc).__name__)
        else:
            print("NO ERROR")
        print("MNE_IMPORTED", "mne" in sys.modules)
    """)
    assert r.returncode == 0, r.stderr
    assert "TYPED AtlasAssetsMissing" in r.stdout, r.stdout
    # The whole point of F4 is that the cheap check happens first.  mne costs
    # ~1.4 s and ~180 MB; a missing file must not have to pay for it.
    assert "MNE_IMPORTED False" in r.stdout, r.stdout


def test_absent_assets_write_no_cache(tmp_path):
    cache = tmp_path / "S2_ATLAS.npz"
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(AtlasPreflightError):
        preflight_atlas(ALL_PARCELS, cache, allow_download=False, mne_root=empty)
    assert not cache.exists()
    assert list(tmp_path.glob(".*tmp*")) == []


# ------------------------------------------------------------- 2. THE TRAP
# mne.datasets.sample.data_path() does NO hash and NO version check: an empty
# MNE-sample-data/ directory is accepted.  A preflight that trusts it lies.

def test_empty_sample_directory_is_accepted_by_mne_but_not_by_us(tmp_path):
    root = tmp_path / "mne_root"
    (root / "MNE-sample-data").mkdir(parents=True)
    assert list((root / "MNE-sample-data").iterdir()) == []

    r = _sub(NO_NETWORK + f"""
        import os
        os.environ["MNE_DATASETS_SAMPLE_PATH"] = {str(root)!r}
        # third-party behaviour, executed rather than assumed:
        import mne
        mne.set_log_level("ERROR")
        got = mne.datasets.sample.data_path(update_path=False)
        print("DATA_PATH_RETURNED", repr(str(got)))
        print("DIR_IS_EMPTY", os.listdir(str(got)) == [])

        from neurocheck.s2_design import ALL_PARCELS
        from tribe_tools.atlas_preflight import preflight_atlas, AtlasPreflightError
        try:
            preflight_atlas(ALL_PARCELS, {str(tmp_path / 'c.npz')!r},
                            allow_download=False, mne_root={str(root)!r})
        except AtlasPreflightError as exc:
            print("TYPED", type(exc).__name__)
        else:
            print("NO ERROR")
    """)
    assert r.returncode == 0, r.stderr
    # mne is happy with the empty directory -- no network, no complaint.
    assert "MNE-sample-data" in r.stdout, r.stdout
    assert "DIR_IS_EMPTY True" in r.stdout, r.stdout
    # We are not.
    assert "TYPED AtlasAssetsMissing" in r.stdout, r.stdout


def test_half_populated_tree_still_fails(tmp_path):
    """Annots present, surfaces absent: passes data_path(), fails us."""
    root = _tree(tmp_path, annots=True, surfaces=False)
    with pytest.raises(AtlasAssetsMissing):
        preflight_atlas(ALL_PARCELS, tmp_path / "c.npz",
                        allow_download=False, mne_root=root)


# --------------------------------------------------------- 3. wrong identity

def test_annot_present_but_wrong_bytes_raises_identity_mismatch(tmp_path):
    root = _tree(tmp_path)
    bad = root / "MNE-sample-data" / "subjects" / "fsaverage" / "label" / "lh.HCPMMP1.annot"
    bad.unlink()
    bad.write_bytes(b"not an annot" * 4096)
    with pytest.raises(AtlasIdentityMismatch) as exc:
        preflight_atlas(ALL_PARCELS, tmp_path / "c.npz",
                        allow_download=False, mne_root=root)
    assert "md5" in str(exc.value)


def test_the_other_hcp_annot_is_rejected(tmp_path):
    """HCPMMP1_combined is a real, valid, WRONG file.  Size and parseability
    are not identity; md5 is."""
    combined = REAL_ASSETS["lh.annot"].with_name("lh.HCPMMP1_combined.annot")
    if not combined.is_file():
        pytest.skip("no HCPMMP1_combined.annot to impersonate with")
    root = _tree(tmp_path)
    target = root / "MNE-sample-data" / "subjects" / "fsaverage" / "label" / "lh.HCPMMP1.annot"
    target.unlink()
    target.symlink_to(combined)
    with pytest.raises(AtlasIdentityMismatch):
        preflight_atlas(ALL_PARCELS, tmp_path / "c.npz",
                        allow_download=False, mne_root=root)


def test_wrong_identity_is_not_repaired_by_downloading(tmp_path):
    """A present-but-wrong annot must never be silently overwritten from the
    network -- that would hide a tampered input behind a green run."""
    root = _tree(tmp_path)
    bad = root / "MNE-sample-data" / "subjects" / "fsaverage" / "label" / "rh.HCPMMP1.annot"
    bad.unlink()
    bad.write_bytes(b"x" * 1316984)
    before = bad.read_bytes()
    r = _sub(NO_NETWORK + f"""
        from neurocheck.s2_design import ALL_PARCELS
        from tribe_tools.atlas_preflight import preflight_atlas, AtlasPreflightError
        try:
            preflight_atlas(ALL_PARCELS, {str(tmp_path / 'c.npz')!r},
                            allow_download=True, mne_root={str(root)!r})
        except AtlasPreflightError as exc:
            print("TYPED", type(exc).__name__)
        else:
            print("NO ERROR")
    """)
    assert r.returncode == 0, r.stderr
    assert "TYPED AtlasIdentityMismatch" in r.stdout, r.stdout
    assert bad.read_bytes() == before


def test_empty_surface_file_raises_identity_mismatch(tmp_path):
    root = _tree(tmp_path, surfaces=False)
    surf = root / "MNE-sample-data" / "subjects" / "fsaverage" / "surf"
    (surf / "lh.white").write_bytes(b"")
    (surf / "rh.white").symlink_to(REAL_ASSETS["rh.white"])
    with pytest.raises(AtlasIdentityMismatch):
        preflight_atlas(ALL_PARCELS, tmp_path / "c.npz",
                        allow_download=False, mne_root=root)


def test_vertex_drift_from_the_frozen_answer_is_fatal(tmp_path):
    """Same file names, different parcellation content -> different vertices.
    Simulated here by resolving a KNOWN parcel name from different labels,
    which is exactly what an atlas version bump looks like downstream."""
    impostor = [FakeParcel("V1_control", ("V2",), "both")]
    with pytest.raises(AtlasIdentityMismatch) as exc:
        preflight_atlas(impostor, tmp_path / "c.npz", allow_download=False)
    assert "V1_control" in str(exc.value)


# ----------------------------------------------------------- 4. unresolvable

def test_unknown_label_raises_unresolvable(tmp_path):
    bogus = [FakeParcel("nonsense", ("NO_SUCH_LABEL",), "both")]
    with pytest.raises(AtlasUnresolvable) as exc:
        preflight_atlas(bogus, tmp_path / "c.npz", allow_download=False)
    assert "NO_SUCH_LABEL" in str(exc.value)
    assert not (tmp_path / "c.npz").exists()


def test_empty_parcel_list_is_refused(tmp_path):
    with pytest.raises(AtlasUnresolvable):
        preflight_atlas([], tmp_path / "c.npz", allow_download=False)


def test_non_fsaverage5_mesh_is_refused(tmp_path):
    with pytest.raises(AtlasUnresolvable):
        preflight_atlas(ALL_PARCELS, tmp_path / "c.npz",
                        allow_download=False, mesh="fsaverage6")


# ------------------------------------------------ 5. the freeze is faithful

@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """One real preflight against the read-only tree, reused by the readers."""
    cache = tmp_path_factory.mktemp("atlas") / "S2_ATLAS.npz"
    summary = preflight_atlas(ALL_PARCELS, cache, allow_download=False)
    return cache, summary


def test_frozen_cache_round_trips(frozen):
    cache, summary = frozen
    loaded = load_frozen_parcels(cache)
    assert set(loaded) == {p.name for p in ALL_PARCELS}
    for name, arr in loaded.items():
        assert arr.ndim == 1 and arr.size == summary["parcels"][name]["n"]
        assert np.array_equal(arr, np.unique(arr))          # sorted, deduplicated


def test_frozen_cache_matches_live_mne_resolution(frozen):
    """The freeze must equal what analyse() computes today, index for index.

    Live path: scripts/s2_run.py:198-201, i.e.
    ``np.unique(np.concatenate([get_vertices(l, hemi=p.hemi) for l in p.labels]))``.
    """
    from tribe_tools.atlas import get_vertices

    cache, _ = frozen
    loaded = load_frozen_parcels(cache)
    for parcel in ALL_PARCELS:
        live = np.unique(np.concatenate(
            [get_vertices(label, hemi=parcel.hemi) for label in parcel.labels]))
        assert np.array_equal(loaded[parcel.name], live), parcel.name


def test_frozen_cache_obeys_the_fsaverage5_vertex_conventions(frozen):
    cache, _ = frozen
    loaded = load_frozen_parcels(cache)
    by_name = {p.name: p for p in ALL_PARCELS}
    for name, arr in loaded.items():
        assert arr.size > 0
        assert arr.min() >= 0 and arr.max() < N_VERTICES
        hemi = by_name[name].hemi
        if hemi == "left":                       # left hemisphere comes first
            assert arr.max() < FSAVERAGE5_SIZE, name
        elif hemi == "right":
            assert arr.min() >= FSAVERAGE5_SIZE, name


def test_frozen_cache_is_small_and_the_reader_never_imports_mne(frozen):
    cache, _ = frozen
    assert cache.stat().st_size < 64 * 1024
    r = _sub(NO_NETWORK + f"""
        import sys
        from tribe_tools.atlas_preflight import load_frozen_parcels, assert_atlas_ready
        d = load_frozen_parcels({str(cache)!r})
        m = assert_atlas_ready({str(cache)!r})
        print("N", len(d), "TOTAL", sum(v.size for v in d.values()))
        print("MNE_IMPORTED", "mne" in sys.modules)
    """)
    assert r.returncode == 0, r.stderr
    assert "MNE_IMPORTED False" in r.stdout, r.stdout
    assert f"N {len(ALL_PARCELS)}" in r.stdout, r.stdout


def test_summary_records_the_atlas_identity(frozen):
    _, summary = frozen
    assert summary["annot_md5"] == HCP_ANNOT_MD5
    assert summary["mesh"] == "fsaverage5"
    assert summary["n_vertices"] == N_VERTICES
    for name, entry in summary["parcels"].items():
        if name in PARCEL_VERTEX_SHA256:
            assert entry["sha256"] == PARCEL_VERTEX_SHA256[name], name


def test_preflight_is_idempotent(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    sa = preflight_atlas(ALL_PARCELS, a, allow_download=False)
    sb = preflight_atlas(ALL_PARCELS, b, allow_download=False)
    assert sa["parcels"] == sb["parcels"]
    la, lb = load_frozen_parcels(a), load_frozen_parcels(b)
    assert all(np.array_equal(la[k], lb[k]) for k in la)


def test_a_failing_preflight_does_not_destroy_a_good_cache(tmp_path):
    cache = tmp_path / "S2_ATLAS.npz"
    preflight_atlas(ALL_PARCELS, cache, allow_download=False)
    good = load_frozen_parcels(cache)
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(AtlasPreflightError):
        preflight_atlas(ALL_PARCELS, cache, allow_download=False, mne_root=empty)
    again = load_frozen_parcels(cache)
    assert all(np.array_equal(good[k], again[k]) for k in good)
    assert list(tmp_path.glob(".*tmp*")) == []


def test_randomised_white_surfaces_do_not_change_the_indices(tmp_path, frozen):
    """The premise of freezing: the answer depends on the two .annot files and
    nothing else.  Label.vertices is np.where(annot == id)[0]; the surfaces
    only fill Label.pos.  So corrupt the geometry completely and demand the
    same indices."""
    import mne

    cache, _ = frozen
    root = _tree(tmp_path, surfaces=False, name="fakegeom")
    surf = root / "MNE-sample-data" / "subjects" / "fsaverage" / "surf"
    rng = np.random.default_rng(7)
    for hemi in ("lh", "rh"):
        rr, tris = mne.read_surface(REAL_ASSETS[f"{hemi}.white"], verbose="ERROR")
        mne.write_surface(surf / f"{hemi}.white",
                          rng.normal(size=rr.shape).astype(rr.dtype), tris,
                          overwrite=True, verbose="ERROR")
    other = tmp_path / "fake.npz"
    preflight_atlas(ALL_PARCELS, other, allow_download=False, mne_root=root)
    a, b = load_frozen_parcels(cache), load_frozen_parcels(other)
    assert set(a) == set(b)
    assert all(np.array_equal(a[k], b[k]) for k in a)


# --------------------------------------------- 6. the cache defends itself

def test_missing_cache_raises_typed(tmp_path):
    with pytest.raises(AtlasCacheMissing):
        load_frozen_parcels(tmp_path / "never_written.npz")
    with pytest.raises(AtlasCacheMissing):
        assert_atlas_ready(tmp_path / "never_written.npz")


def _rewrite(cache: Path, dest: Path, *, mutate_arrays=None, mutate_manifest=None,
             drop=None):
    """Rebuild an npz from `cache`, optionally damaged.  Used to forge the
    corruptions a torn write or a careless edit would produce."""
    with np.load(cache, allow_pickle=False) as z:
        payload = {k: np.asarray(z[k]) for k in z.files}
    manifest = json.loads(str(payload.pop("__manifest__")))
    arrays = {k[len("parcel::"):]: v for k, v in payload.items()}
    if mutate_arrays:
        mutate_arrays(arrays)
    if mutate_manifest:
        mutate_manifest(manifest)
    if drop:
        arrays.pop(drop, None)
    out = {"parcel::" + k: v for k, v in arrays.items()}
    out["__manifest__"] = np.array(json.dumps(manifest, sort_keys=True))
    np.savez(dest, **out)
    return dest


def test_corrupt_cache_raises_and_never_falls_back_to_mne(frozen, tmp_path):
    """Altered vertices must be rejected, and rejection must not be repaired
    by quietly re-resolving through mne at hour 5."""
    cache, _ = frozen
    bad = _rewrite(cache, tmp_path / "bad.npz",
                   mutate_arrays=lambda a: a.__setitem__(
                       "FFA", a["FFA"] + 1))
    r = _sub(NO_NETWORK + f"""
        import sys
        from tribe_tools.atlas_preflight import load_frozen_parcels, AtlasCacheCorrupt
        try:
            load_frozen_parcels({str(bad)!r})
        except AtlasCacheCorrupt as exc:
            print("TYPED AtlasCacheCorrupt")
        else:
            print("NO ERROR")
        print("MNE_IMPORTED", "mne" in sys.modules)
    """)
    assert r.returncode == 0, r.stderr
    assert "TYPED AtlasCacheCorrupt" in r.stdout, r.stdout
    assert "MNE_IMPORTED False" in r.stdout, r.stdout


def test_single_flipped_vertex_is_caught(frozen, tmp_path):
    cache, _ = frozen

    def flip(arrays):
        v = arrays["V1_control"].copy()
        v[3] = (v[3] + 1) % N_VERTICES
        arrays["V1_control"] = np.unique(v)

    bad = _rewrite(cache, tmp_path / "flip.npz", mutate_arrays=flip)
    with pytest.raises(AtlasCacheCorrupt):
        load_frozen_parcels(bad)


def test_manifest_digest_edited_to_match_damaged_array_still_fails(frozen, tmp_path):
    """Damage the array AND the recorded size, leaving the sha256 stale: the
    per-parcel digest is the check that has teeth."""
    cache, _ = frozen

    def chop(arrays):
        arrays["PPA"] = arrays["PPA"][:-1]

    def fix_n(manifest):
        manifest["parcels"]["PPA"]["n"] -= 1

    bad = _rewrite(cache, tmp_path / "chop.npz", mutate_arrays=chop, mutate_manifest=fix_n)
    with pytest.raises(AtlasCacheCorrupt) as exc:
        load_frozen_parcels(bad)
    assert "PPA" in str(exc.value)


def test_dropped_parcel_array_is_caught(frozen, tmp_path):
    cache, _ = frozen
    bad = _rewrite(cache, tmp_path / "drop.npz", drop="EBA")
    with pytest.raises(AtlasCacheCorrupt):
        load_frozen_parcels(bad)


def test_truncated_cache_file_is_caught(frozen, tmp_path):
    cache, _ = frozen
    bad = tmp_path / "trunc.npz"
    bad.write_bytes(cache.read_bytes()[: cache.stat().st_size // 2])
    with pytest.raises(AtlasCacheCorrupt):
        load_frozen_parcels(bad)


def test_garbage_cache_file_is_caught(tmp_path):
    bad = tmp_path / "garbage.npz"
    bad.write_bytes(b"this is not an npz")
    with pytest.raises(AtlasCacheCorrupt):
        load_frozen_parcels(bad)


def test_wrong_schema_version_is_caught(frozen, tmp_path):
    cache, _ = frozen
    bad = _rewrite(cache, tmp_path / "schema.npz",
                   mutate_manifest=lambda m: m.__setitem__(
                       "schema_version", SCHEMA_VERSION + 1))
    with pytest.raises(AtlasCacheCorrupt):
        load_frozen_parcels(bad)


def test_cache_frozen_for_a_different_mesh_is_caught(frozen, tmp_path):
    cache, _ = frozen
    bad = _rewrite(cache, tmp_path / "mesh.npz",
                   mutate_manifest=lambda m: m.__setitem__("n_vertices", 81924))
    with pytest.raises(AtlasCacheCorrupt):
        load_frozen_parcels(bad)


def test_assert_atlas_ready_rejects_a_cache_frozen_for_a_different_design(tmp_path):
    subset = [p for p in ALL_PARCELS if p.name != "V1_control"]
    cache = tmp_path / "subset.npz"
    preflight_atlas(subset, cache, allow_download=False)
    assert assert_atlas_ready(cache, subset)["parcels"].keys() == {p.name for p in subset}
    with pytest.raises(AtlasCacheCorrupt) as exc:
        assert_atlas_ready(cache, ALL_PARCELS)
    assert "V1_control" in str(exc.value)


# ------------------------------------------- 7. allow_download=False is real

def test_allow_download_false_raises_and_touches_no_socket(tmp_path):
    """The annots are absent and the network is armed to explode.  The typed
    error must arrive with the socket untouched."""
    root = _tree(tmp_path, annots=False, surfaces=True)
    r = _sub(NO_NETWORK + f"""
        from neurocheck.s2_design import ALL_PARCELS
        from tribe_tools.atlas_preflight import preflight_atlas, AtlasPreflightError
        try:
            preflight_atlas(ALL_PARCELS, {str(tmp_path / 'c.npz')!r},
                            allow_download=False, mne_root={str(root)!r})
        except AssertionError as exc:
            print("NETWORK", exc)
        except AtlasPreflightError as exc:
            print("TYPED", type(exc).__name__)
        else:
            print("NO ERROR")
    """)
    assert r.returncode == 0, r.stderr
    assert "TYPED AtlasDownloadForbidden" in r.stdout, r.stdout
    assert "NETWORK" not in r.stdout
    # and nothing was created where the annots would have landed
    label = root / "MNE-sample-data" / "subjects" / "fsaverage" / "label"
    assert list(label.iterdir()) == []


def test_allow_download_false_in_process_raises_download_forbidden(tmp_path):
    root = _tree(tmp_path, annots=False, surfaces=True)
    with pytest.raises(AtlasDownloadForbidden) as exc:
        preflight_atlas(ALL_PARCELS, tmp_path / "c.npz",
                        allow_download=False, mne_root=root)
    assert "allow_download=False" in str(exc.value)


def test_the_happy_path_needs_no_network_either(tmp_path):
    """A complete tree plus allow_download=False must fully succeed with the
    network armed to explode -- otherwise Stage 2 has a hidden dependency."""
    root = _tree(tmp_path)
    cache = tmp_path / "ok.npz"
    r = _sub(NO_NETWORK + f"""
        from neurocheck.s2_design import ALL_PARCELS
        from tribe_tools.atlas_preflight import preflight_atlas
        s = preflight_atlas(ALL_PARCELS, {str(cache)!r},
                            allow_download=False, mne_root={str(root)!r})
        print("OK", s["n_parcels"], s["total_vertices"])
    """)
    assert r.returncode == 0, r.stderr
    assert f"OK {len(ALL_PARCELS)} 1257" in r.stdout, r.stdout
    assert set(load_frozen_parcels(cache)) == {p.name for p in ALL_PARCELS}


def test_allow_download_true_is_the_only_path_that_may_reach_the_network(tmp_path):
    """Symmetry check on the previous test: with the same tree and the same
    blocked socket, allow_download=True DOES try to reach figshare.  That is
    what makes allow_download=False a real switch rather than decoration."""
    root = _tree(tmp_path, annots=False, surfaces=True)
    r = _sub(NO_NETWORK + f"""
        from neurocheck.s2_design import ALL_PARCELS
        from tribe_tools.atlas_preflight import preflight_atlas, AtlasPreflightError
        try:
            preflight_atlas(ALL_PARCELS, {str(tmp_path / 'c.npz')!r},
                            allow_download=True, mne_root={str(root)!r})
        except AtlasPreflightError as exc:
            print("TYPED", type(exc).__name__)
            print("REACHED_NETWORK", "NETWORK ACCESS ATTEMPTED" in str(exc))
        else:
            print("NO ERROR")
    """)
    assert r.returncode == 0, r.stderr
    assert "TYPED AtlasAssetsMissing" in r.stdout, r.stdout
    assert "REACHED_NETWORK True" in r.stdout, r.stdout


# ----------------------------------------------------- 8. path resolution

def test_sample_data_root_follows_mne_without_importing_it(tmp_path):
    explicit = sample_data_root(tmp_path / "explicit")
    assert explicit == tmp_path / "explicit" / "MNE-sample-data"
    r = _sub(f"""
        import sys
        from tribe_tools.atlas_preflight import sample_data_root, required_assets
        print("ROOT", sample_data_root({str(tmp_path)!r}))
        print("N_ASSETS", len(required_assets({str(tmp_path)!r})))
        print("MNE_IMPORTED", "mne" in sys.modules)
    """, env={"MNE_DATASETS_SAMPLE_PATH": str(tmp_path / "ignored")})
    assert r.returncode == 0, r.stderr
    assert "N_ASSETS 4" in r.stdout
    assert "MNE_IMPORTED False" in r.stdout, r.stdout


def test_env_var_is_honoured_when_no_root_is_passed(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setenv("MNE_DATASETS_SAMPLE_PATH", str(root))
    assert sample_data_root(None) == root / "MNE-sample-data"
    cache = tmp_path / "env.npz"
    preflight_atlas(ALL_PARCELS, cache, allow_download=False)
    assert len(load_frozen_parcels(cache)) == len(ALL_PARCELS)


# ------------------------------- 9. the range guards, reached by injection
# The real annots can never yield an out-of-range index -- the downsampling at
# tribev2/utils.py:243-245 guarantees it -- so these guards are unreachable
# through the filesystem.  They still have to work, because they are what
# stands between a future atlas/mesh mismatch and a silent out-of-bounds index
# into a (n_segments, 20484) array.  Inject the atlas to reach them.

def _inject(monkeypatch, table):
    import tribe_tools.atlas_preflight as ap
    monkeypatch.setattr(ap, "_label_to_vertices",
                        lambda subjects_dir, mesh_size, hemi: table)


def test_out_of_range_vertices_are_refused(tmp_path, monkeypatch):
    _inject(monkeypatch, {"X": np.array([0, 5, N_VERTICES])})
    with pytest.raises(AtlasUnresolvable) as exc:
        preflight_atlas([FakeParcel("p", ("X",), "both")], tmp_path / "c.npz",
                        allow_download=False)
    assert str(N_VERTICES) in str(exc.value)
    assert not (tmp_path / "c.npz").exists()


def test_negative_vertices_are_refused(tmp_path, monkeypatch):
    _inject(monkeypatch, {"X": np.array([-1, 5, 9])})
    with pytest.raises(AtlasUnresolvable):
        preflight_atlas([FakeParcel("p", ("X",), "both")], tmp_path / "c.npz",
                        allow_download=False)


def test_a_parcel_that_resolves_to_nothing_is_refused(tmp_path, monkeypatch):
    _inject(monkeypatch, {"X": np.array([], dtype=np.int64)})
    with pytest.raises(AtlasUnresolvable) as exc:
        preflight_atlas([FakeParcel("p", ("X",), "both")], tmp_path / "c.npz",
                        allow_download=False)
    assert "ZERO" in str(exc.value)


def test_left_hemi_parcel_holding_right_hemi_vertices_is_refused(tmp_path, monkeypatch):
    _inject(monkeypatch, {"X": np.array([3, FSAVERAGE5_SIZE + 3])})
    with pytest.raises(AtlasUnresolvable) as exc:
        preflight_atlas([FakeParcel("p", ("X",), "left")], tmp_path / "c.npz",
                        allow_download=False)
    assert "left" in str(exc.value)


def test_right_hemi_parcel_holding_left_hemi_vertices_is_refused(tmp_path, monkeypatch):
    _inject(monkeypatch, {"X": np.array([3, FSAVERAGE5_SIZE + 3])})
    with pytest.raises(AtlasUnresolvable) as exc:
        preflight_atlas([FakeParcel("p", ("X",), "right")], tmp_path / "c.npz",
                        allow_download=False)
    assert "right" in str(exc.value)


def test_wildcard_label_selection_matches_the_live_convention(tmp_path, monkeypatch):
    """``get_hcp_roi_indices`` accepts ``V1*`` and ``*_ROI``-style suffixes
    (tribev2/utils.py:273-281); the preflight must resolve them the same way."""
    _inject(monkeypatch, {"V1": np.array([1, 2]), "V1b": np.array([3]),
                          "zzV1": np.array([4]), "other": np.array([9])})
    cache = tmp_path / "w.npz"
    preflight_atlas([FakeParcel("prefix", ("V1*",), "both"),
                     FakeParcel("suffix", ("*V1",), "both"),
                     FakeParcel("exact", ("V1",), "both")], cache, allow_download=False)
    got = load_frozen_parcels(cache)
    assert np.array_equal(got["prefix"], [1, 2, 3])
    assert np.array_equal(got["suffix"], [1, 2, 4])
    assert np.array_equal(got["exact"], [1, 2])
