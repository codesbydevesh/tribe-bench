"""End-to-end integration of the REAL S2 call graph.

    s2_run.main(argv) -> preflight -> Stage 1 -> artifact -> Stage 2 -> report

Nothing here re-implements the orchestration.  ``scripts/s2_run.py`` is loaded with
importlib (as the other test files do) and driven through its own ``main()`` with a
real ``sys.argv``; ``tribe_tools.s2_pipeline.stage1_extract`` / ``stage2_infer`` run
inside it for real, as do ``feature_artifact``, ``ledger``, ``durable_store``,
``atlas_preflight``, ``provenance``, ``cuda_guard`` and ``roi_stats``.

Exactly six things are stubbed, and each is stubbed because it is expensive or absent
on this box, never because it is inconvenient to satisfy:

  1. **V-JEPA encode.**  The fake extractor is a real ``pydantic.BaseModel`` carrying a
     real ``exca.MapInfra`` with a real ``@infra.apply`` method, so the cache layout,
     the read-only firewall, the sidecar files and ``EncodeCounter``'s hook on
     ``MapInfra._call_and_store`` are all the genuine library.  Only the
     tensor-producing body is fake, and every invocation of it appends to a list --
     "how many items did V-JEPA encode" is a measured integer, not an inference from a
     config value.
  2. **The tribev2 brain model.**  ``tribe_tools.model.load_model`` is replaced by a
     stand-in that reproduces the two behaviours of ``TribeModel.from_pretrained``
     that matter here: ``tribev2/demo_utils.py:190-191`` mkdirs ``cache_folder``, and
     ``:206-207`` writes that argument into ``data.<mod>_feature.infra.folder``
     UNCONDITIONALLY (so ``None`` really does mean "no folder"), then applies
     ``config_update`` on top.
  3. **CUDA** -- never touched; ``cuda_guard.arm()`` itself runs for real.
  4. **The network and the 4.14 GB weight hash** -- ``resolve_weight_identity``,
     ``verify_local_weights``, ``library_versions``, ``_processor_config``.
  5. **The mne atlas resolution.**  ``preflight_atlas`` needs the HCP-MMP1 annots,
     which are not on this box.  The stub installs a cache built with
     ``atlas_preflight``'s OWN ``_pack``/``_atomic_write``/``parcel_digest``, so
     ``assert_atlas_ready`` and ``load_frozen_parcels`` -- the two functions the
     wiring is about -- run for real against a real file.
  6. **``roi_stats.detection_floor``**, a 13.4 s Monte-Carlo power simulation per
     call.  Every other statistic in ``analyse()`` is the real one.

Every scenario runs in a FRESH interpreter (``subprocess``), so "Stage 2 in a
different process than Stage 1" is literally true, module-level state cannot leak
between cases, and ``"mne" in sys.modules`` is a meaningful question.

No test asserts on source text and none uses ``hasattr`` as evidence.

FOUR DOCUMENTED WORKAROUNDS.  The harness has to inject four things the shipped
wiring does not do, or Stage 1 cannot publish and Stage 2 cannot run at all.  None of
them is a convenience: each is a defect in the code under test, and each has a test of
its own that removes the workaround and proves the shipped path fails.

  W1  _reader_for descended one level under <artifact>/cache; exca nests two
  W2  infer() loaded with cache_folder=None and never carried the resolved location
  W3  publish got a zero-argument sidecar_probe; durable_store calls it with one
  W4  np.savez appended .npz, so the rename targeted a file that never existed

All four are FIXED in the product. The harness carries no workaround for any of
them -- every scenario below exercises scripts/s2_run.py as shipped.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from neurocheck.s2_design import (  # noqa: E402
    ALL_PARCELS, S2, build_manifest, environment_provenance, resolve_stimulus_images,
)
from tribe_tools.atlas_preflight import (  # noqa: E402
    FSAVERAGE5_SIZE, HCP_ANNOT_MD5, N_VERTICES, SCHEMA_VERSION as ATLAS_SCHEMA,
    _atomic_write, _pack, parcel_digest,
)
from tribe_tools.feature_artifact import COMPLETE, MANIFEST as ART_MANIFEST  # noqa: E402

# Only this parcel is frozen into the atlas fixture.  analyse() is identical wiring
# for one parcel and for seven, and s2_run.ALL_PARCELS is narrowed to match so the
# design, the atlas artifact and analyse() all agree.
HARNESS_PARCELS = ("FFA",)
#: 4 chunks instead of the real 18: the wiring per chunk is identical.
N_ITEMS = 4

try:                      # conftest has already prepended S2_DEV_SITE_PACKAGES
    import exca as _exca  # noqa: F401
    EXCA = True
except Exception:         # pragma: no cover - depends on the box
    EXCA = False

pytestmark = [
    pytest.mark.needs_exca,
    pytest.mark.skipif(not EXCA, reason="the real exca 0.5.20 is required: this file "
                                        "measures its cache layout and its firewall"),
]


# --------------------------------------------------------------------------- #
# the harness that runs inside the fresh interpreter
# --------------------------------------------------------------------------- #

HARNESS_SRC = r'''
"""Executed in a fresh interpreter by tests/test_s2_integration.py."""
import hashlib, json, os, sys, traceback
from pathlib import Path

SCEN = json.loads(Path(os.environ["S2_SCENARIO"]).read_text())
TEL_PATH = Path(os.environ["S2_TELEMETRY"])
REPO = Path(os.environ["S2_REPO"])

T = {
    "pid": os.getpid(),
    "mne_at_start": "mne" in sys.modules,
    "vjepa_encoded_items": [],
    "load_model_calls": [],
    "resolve_weight_identity_calls": [],
    "verify_local_weights_calls": [],
    "require_artifact_location_calls": [],
    "publish_calls": [],
    "artifact_dir": None,
    "reader_factory_calls": [],
    "sidecar_digest_calls": [],
    "preflight_atlas_calls": 0,
    "loader_calls": 0,
    "predict_calls": 0,
    "predict_items_read": 0,
    "persist_calls": [],
    "analyse_calls": 0,
    "analyse_parcels": None,
    "preds_npz_at_analyse": None,
    "mne_at_analyse_entry": None,
    "mne_after_analyse": None,
    "get_vertices_calls": 0,
    "detection_floor_calls": 0,
    "rc": None,
    "exc": None,
    "msg": "",
    "traceback": "",
}


def _dump():
    T["mne_at_end"] = "mne" in sys.modules
    TEL_PATH.write_text(json.dumps(T, indent=1, default=str))


for _p in [p for p in os.environ.get("S2_DEV_SITE_PACKAGES", "").split(os.pathsep) if p][::-1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(REPO))

import typing as tp

import numpy as np
import pydantic

import exca

# --------------------------------------------------------------- fake V-JEPA

ENCODED = []            # one entry per item the fake encoder actually produced


def _event_uid(ev):
    """neuralset/extractors/video.py:247, the same string s2_run._item_uid builds."""
    return f"{ev.study_relative_path()}_{ev.offset:.2f}_{ev.duration:.2f}"


class _Event:
    def __init__(self, path, offset, duration):
        self._path, self.offset, self.duration = path, float(offset), float(duration)

    def study_relative_path(self):
        return self._path


EVENTS = [_Event(SCEN.get("stim_rel_path", "s2_stimulus.mp4"), 60.0 * i, 60.0)
          for i in range(SCEN.get("n_items", 4))]


class _Helper:
    def extract(self, events_df):
        T["helper_extract_calls"] = T.get("helper_extract_calls", 0) + 1
        return list(EVENTS)


_HELPER = _Helper()


class _ImageCfg(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    model_name: str = "facebook/vjepa2-vitg-fpc64-256"
    pretrained: bool = True
    imsize: int | None = None
    token_aggregation: str = "mean"
    cache_all_layers: bool = False
    cache_n_layers: int | None = None
    layers: str | None = None
    layer_aggregation: str | None = None


class HuggingFaceVideo(pydantic.BaseModel):
    """A REAL exca-backed extractor with a fake tensor body.

    The class name matches neuralset's so provenance._extractor_fields records the
    same "class" field it would on the GPU box.
    """
    model_config = pydantic.ConfigDict(extra="forbid", ignored_types=(property,))
    image: _ImageCfg = _ImageCfg()
    frequency: float = 2.0
    clip_duration: float = 2.0
    num_frames: int | None = None
    max_imsize: int | None = None
    layer_type: str = "hidden"
    use_audio: bool = False
    infra: exca.MapInfra = exca.MapInfra(version="0")

    @property
    def _event_types_helper(self):
        return _HELPER

    @infra.apply(item_uid=_event_uid)
    def _get_data(self, items: tp.Sequence[tp.Any]) -> tp.Iterator[np.ndarray]:
        for ev in items:
            uid = _event_uid(ev)
            ENCODED.append(uid)
            seed = int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)
            yield np.random.default_rng(seed).normal(size=(4, 8)).astype(np.float32)


# ------------------------------------------------------------ fake brain model

class _Batch:
    def __init__(self):
        rng = np.random.default_rng(7)
        self.data = {}
        for name, kind in (SCEN.get("modalities") or {"video": "real"}).items():
            if kind == "absent":
                continue
            if kind == "zero":
                # tribev2/model.py:188-192 substitutes torch.zeros for a modality
                # missing from the batch; _missing_default is exactly 0.0.
                self.data[name] = np.zeros((6, 1408), dtype=np.float32)
            elif kind == "one_dead_timestep":
                arr = rng.normal(size=(6, 1408)).astype(np.float32)
                arr[3, :] = 0.0
                self.data[name] = arr
            else:
                self.data[name] = rng.normal(size=(6, 1408)).astype(np.float32)


class _Data:
    def __init__(self, extractor):
        self.video_feature = extractor
        self.audio_feature = None
        self.text_feature = None

    def get_loaders(self, events=None, split_to_build=None):
        T["loader_calls"] += 1
        if SCEN.get("break_exca_at_loader"):
            # "exca is not importable in the process that runs Stage 2" -- the exact
            # condition EncodeCounter.active is supposed to detect.  Nothing else in
            # this scenario imports exca again.
            sys.modules["exca"] = None
        if SCEN.get("batch_without_data"):
            class _Bare:
                pass
            return {"all": [_Bare()]}
        return {"all": [_Batch()]}


class _TribeModel:
    def __init__(self, extractor):
        self.data = _Data(extractor)

    def get_events_dataframe(self, video_path=None):
        T["events_df_video_path"] = str(video_path)
        return {"video_path": str(video_path)}


_INFRA_PREFIX = "data.video_feature.infra."


def _fake_load_model(device="cuda", cache_folder=None, config_update=None,
                     revision=None, checkpoint_dir=None):
    """Stand-in for tribe_tools.model.load_model -> TribeModel.from_pretrained.

    Faithful to demo_utils.py on the two points that decide this test:
      :190-191  `if cache_folder is not None: Path(cache_folder).mkdir(parents=True)`
      :206-207  infra.folder = cache_folder, UNCONDITIONALLY, then config_update.
    """
    cu = dict(config_update or {})
    T["load_model_calls"].append({
        "device": device,
        "cache_folder": None if cache_folder is None else str(cache_folder),
        "revision": revision,
        "config_update": cu,
    })
    if cache_folder is not None:
        Path(cache_folder).mkdir(parents=True, exist_ok=True)
    infra = {"folder": None if cache_folder is None else str(cache_folder),
             "cluster": None}
    for key, val in cu.items():
        if key.startswith(_INFRA_PREFIX):
            infra[key[len(_INFRA_PREFIX):]] = val
    for key, val in (SCEN.get("infra_override") or {}).items():
        infra[key] = val
    return _TribeModel(HuggingFaceVideo(infra=infra))


# ------------------------------------------------------------------ provenance

import tribe_tools.provenance as prov  # noqa: E402

_FILES = {
    prov.MODEL_CONFIG_FILENAME: prov.FileIdentity(
        prov.MODEL_CONFIG_FILENAME, "1" * 64, "sha256", 1234, "hub-api"),
    prov.PROCESSOR_CONFIG_FILENAME: prov.FileIdentity(
        prov.PROCESSOR_CONFIG_FILENAME, "2" * 64, "sha256", 567, "hub-api"),
    prov.WEIGHTS_FILENAME: prov.FileIdentity(
        prov.WEIGHTS_FILENAME, prov.VJEPA2_WEIGHTS_SHA256, "sha256",
        prov.VJEPA2_WEIGHTS_BYTES, "hub-api"),
}
_WID = prov.WeightIdentity(prov.VJEPA2_REPO, prov.VJEPA2_COMMIT, _FILES, "hub-api")


def _fake_resolve_weight_identity(*, allow_network=False, expected_commit=None, **kw):
    T["resolve_weight_identity_calls"].append(
        {"allow_network": bool(allow_network), "expected_commit": expected_commit})
    if expected_commit is not None and expected_commit != prov.VJEPA2_COMMIT:
        raise prov.WeightMismatch(f"commit {expected_commit} != {prov.VJEPA2_COMMIT}")
    return _WID


def _fake_verify_local_weights(path_or_cache, expected, *, filename=None,
                               force_hash=False):
    """Mirrors the real dispatch (provenance.py:552-600) minus the 4.14 GB read.

    A caller that passes the 57-field identity DICT as `expected` -- blocker B1 --
    still gets MissingIdentityField here, a caller that passes None as the path still
    gets WeightFileMissing, and a caller that omits force_hash still gets
    route="blob-filename", which s2_run rejects.
    """
    T["verify_local_weights_calls"].append({
        "path": None if path_or_cache is None else str(path_or_cache),
        "expected_type": type(expected).__name__,
        "filename": filename,
        "force_hash": bool(force_hash),
    })
    if path_or_cache is None:
        raise prov.WeightFileMissing("no path given")
    if isinstance(expected, prov.WeightIdentity):
        fid = expected.file(filename or prov.WEIGHTS_FILENAME)
    elif isinstance(expected, prov.FileIdentity):
        fid = expected
    else:
        fn = filename or (expected or {}).get("filename")
        digest = (expected or {}).get("digest")
        if not fn or not digest:
            raise prov.MissingIdentityField(
                "expected mapping needs at least 'filename' and 'digest' "
                f"(got keys {sorted(expected or {})})")
        fid = prov.FileIdentity(fn, digest, "sha256",
                                int((expected or {}).get("size_bytes") or 0), "caller")
    return prov.WeightVerification(fid.filename, str(path_or_cache), fid.digest,
                                   fid.algo, fid.size_bytes,
                                   "full-hash" if force_hash else "blob-filename")


_VERSIONS = {name: "1.2.3" for name in prov.UID_DISTRIBUTIONS}


def _fake_library_versions(names=None, *, metadata=None):
    return dict(_VERSIONS)


prov.resolve_weight_identity = _fake_resolve_weight_identity
prov.verify_local_weights = _fake_verify_local_weights
prov.library_versions = _fake_library_versions

import tribe_tools.model as tmodel  # noqa: E402
tmodel.load_model = _fake_load_model

import tribe_tools.inference as tinference  # noqa: E402

# ------------------------------------------------------- atlas (no mne on this box)

import tribe_tools.atlas_preflight as _ap  # noqa: E402

_real_preflight_atlas = _ap.preflight_atlas


def _fake_preflight_atlas(parcels, cache_path, *, allow_download, **kw):
    """Install the pre-frozen cache through atlas_preflight's OWN atomic writer.

    The only thing skipped is `import mne` + the two HCP-MMP1 annots, which are not
    on this box.  Everything the WIRING depends on -- assert_atlas_ready,
    load_frozen_parcels, the per-parcel sha256 -- reads this file for real.
    """
    T["preflight_atlas_calls"] += 1
    _ap._atomic_write(Path(cache_path), Path(os.environ["S2_ATLAS_SEED"]).read_bytes())
    man = _ap.atlas_manifest(cache_path)
    return {**man, "cache_path": str(cache_path), "n_parcels": len(man["parcels"])}


_ap.preflight_atlas = _fake_preflight_atlas

# ------------------------------------- the 13.4 s Monte-Carlo power simulation

import tribe_tools.roi_stats as _rs  # noqa: E402


def _fake_detection_floor(*, n_per_group, noise_sd, alpha, seed, **kw):
    T["detection_floor_calls"] += 1
    return 0.5 * float(noise_sd)


_rs.detection_floor = _fake_detection_floor

# ------------------------------------------------------------------- load s2_run

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("s2_run_harness", REPO / "scripts" / "s2_run.py")
s2run = _ilu.module_from_spec(_spec)
sys.modules["s2_run_harness"] = s2run
_spec.loader.exec_module(s2run)

from neurocheck.s2_design import ALL_PARCELS as _ALL  # noqa: E402

_want = set(SCEN.get("parcels") or [p.name for p in _ALL])
s2run.ALL_PARCELS = tuple(p for p in _ALL if p.name in _want)

_PROC_CFG = {
    "do_resize": True, "size": {"shortest_edge": 256}, "resample": 3,
    "do_center_crop": True, "crop_size": {"height": 256, "width": 256},
    "do_rescale": True, "rescale_factor": 0.00392156862745098,
    "do_normalize": True, "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225], "video_processor_type": "VJEPA2VideoProcessor",
}
s2run._processor_config = lambda wid: dict(_PROC_CFG)


def _fake_predict_single(model, video_path, features_to_mask=None):
    T["predict_calls"] += 1
    if SCEN.get("read_cache_in_predict", True):
        # what the real dataloader does: pull every feature through the extractor,
        # which in the consume stage is in exca read-only mode.  Without this, "zero
        # encodes" would be zero because nothing asked for anything.
        items = list(EVENTS)
        if SCEN.get("extra_uncached_item"):
            # a chunk the artifact does not hold.  read-only must refuse it BEFORE
            # anything is computed; without read-only exca would just encode it.
            items.append(_Event(SCEN.get("stim_rel_path", "s2_stimulus.mp4"),
                                99999.0, 60.0))
        got = list(model.data.video_feature._get_data(items))
        T["predict_items_read"] = len(got)
    n = int(round(SCEN.get("n_rows", 1050)))
    preds = np.random.default_rng(3).standard_normal((n, 20484), dtype=np.float32)

    class _Seg:
        def __init__(self, t):
            self.start = float(t)

    return preds, [_Seg(t) for t in range(n)]


tinference.predict_single = _fake_predict_single

# ------------------------------------------------------------------- spies

import tribe_tools.feature_artifact as _fa  # noqa: E402
_real_sidecars = _fa.sidecar_digests


def _counting_sidecars(uid_folder):
    out = _real_sidecars(uid_folder)
    T["sidecar_digest_calls"].append({"folder": str(uid_folder), "digests": out})
    return out


_fa.sidecar_digests = _counting_sidecars

import tribe_tools.durable_store as _ds  # noqa: E402
_real_require = _ds.require_artifact_location


def _spy_require(ident, **kw):
    T["require_artifact_location_calls"].append(
        {"uid": ident.uid, "n_items": ident.item_count,
         "search_paths": [str(p) for p in kw.get("search_paths", [])]})
    out = _real_require(ident, **kw)
    T["artifact_dir"] = str(out)
    return out


_ds.require_artifact_location = _spy_require

_real_publish = _ds.publish


def _spy_publish(artifact_dir, ident, backend, *, reader_factory, sidecar_probe=None):
    """WORKAROUND W3 -- see test_publish_calls_the_sidecar_probe_with_a_directory.

    durable_store.verify_location calls `sidecar_probe(root)`; s2_run hands publish a
    ZERO-argument lambda.  The fix also has to change the semantics, not just the
    arity: the probe must describe the directory it is given (the COPY), the same
    requirement B3 fixed for reader_factory.
    """
    import inspect
    n = None
    if sidecar_probe is not None:
        try:
            n = len(inspect.signature(sidecar_probe).parameters)
        except (TypeError, ValueError):
            n = -1
    T["publish_calls"].append({"src": str(artifact_dir), "probe_params": n})
    probe = sidecar_probe          # no workaround: W3 is fixed in the product
    return _real_publish(artifact_dir, ident, backend,
                         reader_factory=reader_factory, sidecar_probe=probe)


_ds.publish = _spy_publish

import tribe_tools.atlas as _atlas  # noqa: E402


def _no_live_atlas(*a, **k):
    T["get_vertices_calls"] += 1
    raise AssertionError("tribe_tools.atlas.get_vertices reached live mne AFTER the "
                         "GPU stage -- blocker B5 is back")


_atlas.get_vertices = _no_live_atlas

_real_item_uid = s2run._item_uid
_UID_CALLS = []


def _spy_item_uid(event):
    """Optionally make ONE expected uid name an item the cache will never hold.

    Used by test_stage1_does_not_finalize_an_artifact_it_cannot_read_back.  It is
    the only way to reach that state from the real Stage 1, because
    ``extract_features`` builds `expected` and then RETURNS it as `produced`
    (s2_run.py:574-576) -- so `stage1_extract`'s own completeness check can never
    see a difference, and the read-back is the only thing left standing.
    """
    uid = _real_item_uid(event)
    _UID_CALLS.append(uid)
    if SCEN.get("orphan_expected_uid") == len(_UID_CALLS):
        return uid + "-never-encoded"
    return uid


s2run._item_uid = _spy_item_uid

_real_reader_for = s2run._reader_for


def _spy_reader_for(artifact_dir):
    """Records the call and delegates. NO workaround.

    This used to substitute a corrected reader when SCEN["fix_reader"] was set, which
    meant every green test was exercising the harness rather than s2_run._reader_for.
    W1 is fixed in the product now (s2_run._exca_uid_folder descends the two levels
    exca actually nests), so the spy must not stand in for it -- a harness that
    silently repairs the code under test is the same failure as a guard that is only
    on when the caller remembers to ask.
    """
    T["reader_factory_calls"].append(str(artifact_dir))
    return _real_reader_for(artifact_dir)


s2run._reader_for = _spy_reader_for

_real_persist = s2run._persist_predictions


def _spy_persist(preds, segments, dest):
    """WORKAROUND W4 -- see test_persist_predictions_renames_a_file_numpy_never_wrote.

    Repairs the shipped writer's own output rather than replacing it: np.savez still
    does the writing, the shim only completes the rename it could not.
    """
    dest = Path(dest)
    out = _real_persist(preds, segments, dest)   # no workaround: W4 is fixed
    T["persist_calls"].append(str(out))
    return out


s2run._persist_predictions = _spy_persist

_real_analyse = s2run.analyse


def _spy_analyse(preds, segments, cfg, man, parcels):
    T["analyse_calls"] += 1
    T["mne_at_analyse_entry"] = "mne" in sys.modules
    root = Path(os.environ["S2_ARTIFACT_ROOT"])
    found = sorted(root.glob("*/preds.npz"))
    T["preds_npz_at_analyse"] = [[str(p), p.stat().st_size] for p in found]
    T["analyse_parcels"] = {
        k: [int(np.asarray(v).size),
            hashlib.sha256(np.unique(np.asarray(v)).astype("<i8").tobytes()).hexdigest()]
        for k, v in parcels.items()}
    out = _real_analyse(preds, segments, cfg, man, parcels)
    T["mne_after_analyse"] = "mne" in sys.modules
    return out


s2run.analyse = _spy_analyse

# --------------------------------------------------------------------- dispatch

sys.argv = ["s2_run.py", *SCEN["argv"]]

try:
    T["rc"] = s2run.main()
except BaseException as exc:            # SystemExit from die() included, on purpose
    T["exc"] = type(exc).__name__
    T["msg"] = str(exc)
    T["traceback"] = traceback.format_exc()[-4000:]
finally:
    T["vjepa_encoded_items"] = list(ENCODED)
    _dump()

raise SystemExit(1 if T["exc"] is not None else int(T["rc"] or 0))
'''


# --------------------------------------------------------------------------- #
# fixtures: a workdir the REAL _check_inputs accepts, and a frozen atlas
# --------------------------------------------------------------------------- #

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atlas_seed_bytes() -> bytes:
    """A frozen-parcel cache built with atlas_preflight's OWN packer and digest.

    Not a hand-rolled npz: ``_pack`` and ``parcel_digest`` are the shipped
    functions, so ``assert_atlas_ready`` / ``load_frozen_parcels`` verify this file
    exactly as they verify one written by the real ``preflight_atlas``.
    """
    verts = np.unique(np.random.default_rng(11).choice(N_VERTICES, size=96,
                                                       replace=False)).astype("<i8")
    manifest = {
        "schema_version": ATLAS_SCHEMA,
        "mesh": "fsaverage5",
        "n_vertices": N_VERTICES,
        "hemi_size": FSAVERAGE5_SIZE,
        "mne_version": "1.8.0",
        "annot_md5": dict(HCP_ANNOT_MD5),
        "subjects_dir": "frozen-by-tests/test_s2_integration.py",
        "parcels": {
            "FFA": {"n": int(verts.size), "sha256": parcel_digest(verts),
                    "hemi": "both", "labels": ["L_FFC_ROI-lh", "R_FFC_ROI-rh"],
                    "min": int(verts.min()), "max": int(verts.max())},
        },
    }
    return _pack(manifest, {"FFA": verts})


#: (n_vertices, sha256) of the FFA vertex set frozen into the fixture above.
ATLAS_SEED = _atlas_seed_bytes()
FFA_N, FFA_SHA = (lambda m: (m["parcels"]["FFA"]["n"], m["parcels"]["FFA"]["sha256"]))(
    json.loads(str(np.load(__import__("io").BytesIO(ATLAS_SEED),
                           allow_pickle=False)["__manifest__"]))
)


def _expected_uids(n: int = N_ITEMS) -> list[str]:
    """The item uids s2_run._item_uid builds for the harness's events."""
    return [f"s2_stimulus.mp4_{60.0 * i:.2f}_60.00" for i in range(n)]


def _build_template(root: Path) -> Path:
    """A directory the REAL ``_check_inputs`` accepts, with nothing rendered.

    ``prepare()`` is the one stage that needs moviepy and 1050 s of real frames;
    everything it would have written is constructed here from the same frozen
    design object, so ``--preflight``/``--extract-features``/``--infer`` run
    against a manifest they cannot tell from a rendered one.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    floc = root / "stim" / "floc"
    for cat in S2.categories:
        d = floc / cat
        d.mkdir(parents=True, exist_ok=True)
        for i in range(S2.exemplars_per_category):
            (d / f"{cat}_{i:03d}.png").write_bytes(f"{cat}/{i:03d}".encode())
    images = resolve_stimulus_images(floc, S2)

    vid = root / "stim" / "s2_stimulus.mp4"
    vid.write_bytes(b"S2 integration harness stimulus placeholder\n" * 64)
    n_frames = int(round(S2.stimulus_duration_s * S2.fps))
    video = {
        "path": str(vid), "sha256": _sha256(vid), "bytes": vid.stat().st_size,
        "n_frames": n_frames, "fps": S2.fps, "duration_s": S2.stimulus_duration_s,
        "width": S2.frame_size[0], "height": S2.frame_size[1],
        "stimulus_frames": n_frames, "grey_frames": 0, "placeholders": 0,
    }
    man = build_manifest(
        S2, code_commit="0" * 40, tree_dirty=False,
        environment=environment_provenance(), video=video,
        images={"source_repo": "tests/test_s2_integration.py", "commit": "0" * 40,
                "selection_rule": S2.stimulus_selection_rule, "licence": "synthetic",
                "n": len(images), "files": images})
    (root / "data" / "s2_manifest.json").write_text(json.dumps(man, indent=2,
                                                               default=str))
    (root / "atlas_seed.npz").write_bytes(ATLAS_SEED)
    (root / "harness.py").write_text(HARNESS_SRC)
    return root


class Run:
    """One fresh-interpreter execution of s2_run.main()."""

    def __init__(self, rc, stdout, stderr, tel, work):
        self.rc, self.stdout, self.stderr, self.tel, self.work = \
            rc, stdout, stderr, tel, work

    def __repr__(self):  # pragma: no cover - diagnostics only
        return (f"<Run rc={self.rc} exc={self.tel.get('exc')} "
                f"msg={self.tel.get('msg')!r}>\n{self.tel.get('traceback', '')}"
                f"\nSTDERR:\n{self.stderr[-2000:]}")


def _run(work: Path, argv, *, scen=None, durable=None, artifact_root=None,
         tag="run", timeout=900) -> Run:
    scen = dict(scen or {})
    scen["argv"] = list(argv)
    scen.setdefault("n_items", N_ITEMS)
    scen.setdefault("parcels", list(HARNESS_PARCELS))
    spath = work / f"scenario-{tag}.json"
    spath.write_text(json.dumps(scen, indent=1))
    tpath = work / f"telemetry-{tag}.json"
    if tpath.exists():
        tpath.unlink()

    env = dict(os.environ)
    env.update({
        "S2_SCENARIO": str(spath),
        "S2_TELEMETRY": str(tpath),
        "S2_REPO": str(REPO),
        "S2_ATLAS_SEED": str(work / "atlas_seed.npz"),
        "S2_ARTIFACT_ROOT": str(artifact_root or (work / "features")),
        "S2_ATLAS_CACHE": str(work / "data" / "s2_parcels.npz"),
        "S2_LEDGER": str(work / "data" / "s2_ledger.jsonl"),
        "S2_STIMULUS_ROOT": str(work / "stim"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    env.pop("S2_DURABLE_ROOT", None)
    if durable is not None:
        env["S2_DURABLE_ROOT"] = str(durable)
    proc = subprocess.run([sys.executable, str(work / "harness.py")], cwd=str(work),
                          env=env, capture_output=True, text=True, timeout=timeout)
    tel = json.loads(tpath.read_text()) if tpath.is_file() else {}
    return Run(proc.returncode, proc.stdout, proc.stderr, tel, work)


@pytest.fixture(scope="session")
def stage1_session(tmp_path_factory):
    """Stage 1 executed ONCE, in its own process, into a session-scoped template.

    Every Stage-2 test starts from a byte copy of the directory this left behind,
    so "Stage 2 runs in a different process than Stage 1" is not simulated.
    """
    root = tmp_path_factory.mktemp("s2-template")
    work = _build_template(root / "work")
    run = _run(work, ["--preflight", "--extract-features"],
               scen={},
               durable=work / "durable", tag="stage1")
    return run


@pytest.fixture
def work(stage1_session, tmp_path) -> Path:
    """A private byte copy of the completed Stage-1 workdir."""
    dest = tmp_path / "work"
    shutil.copytree(stage1_session.work, dest)
    for stale in dest.glob("telemetry-*.json"):
        stale.unlink()
    return dest


def _artifact_dirs(work: Path) -> tuple[Path, Path]:
    """(local artifact dir, durable artifact dir) written by the session Stage 1."""
    local = sorted(p for p in (work / "features").iterdir() if p.is_dir())
    durable = sorted(p for p in (work / "durable").iterdir()
                     if p.is_dir() and not p.name.startswith("."))
    assert len(local) == 1 and len(durable) == 1, (local, durable)
    return local[0], durable[0]


def _consume(work: Path, artifact_dir: Path, **scen) -> dict:
    """Scenario for a Stage-2 run. No workarounds: W1-W4 are fixed in the product,
    and a harness that stands in for the code under test proves nothing."""
    out: dict = {}
    out.update(scen)
    return out


def _uid_folder(artifact_dir: Path) -> Path:
    """exca's <folder>/<method,version>/<confighash> leaf inside an artifact."""
    leaves = [d for d in sorted((artifact_dir / "cache").rglob("*"))
              if (d / "data").is_dir()]
    assert len(leaves) == 1, leaves
    return leaves[0]


# --------------------------------------------------------------------------- #
# 1. a clean first run encodes exactly once; Stage 2 encodes ZERO
# --------------------------------------------------------------------------- #

def test_stage1_encodes_every_item_exactly_once(stage1_session):
    r = stage1_session
    assert r.rc == 0 and r.tel.get("exc") is None, r
    # measured at the tensor-producing body, one append per item produced
    assert sorted(r.tel["vjepa_encoded_items"]) == sorted(_expected_uids())
    assert len(r.tel["vjepa_encoded_items"]) == N_ITEMS      # exactly once each
    # and the artifact really was finalized, by the real writer
    local, durable = _artifact_dirs(r.work)
    for d in (local, durable):
        assert (d / ART_MANIFEST).is_file() and (d / COMPLETE).is_file()
    man = json.loads((local / ART_MANIFEST).read_text())
    assert man["n_items"] == N_ITEMS
    assert sorted(man["items"]) == sorted(_expected_uids())


def test_stage1_measured_the_weights_and_pinned_the_commit(stage1_session):
    """B1: the call that used to raise MissingIdentityField on every run."""
    r = stage1_session
    calls = r.tel["verify_local_weights_calls"]
    assert calls, r
    assert all(c["path"] is not None for c in calls)
    assert all(c["expected_type"] == "WeightIdentity" for c in calls)
    assert all(c["force_hash"] is True for c in calls)
    assert all(c["filename"] == "model.safetensors" for c in calls)
    # and every identity resolution pinned the commit rather than floating main
    assert r.tel["resolve_weight_identity_calls"]
    assert all(c["expected_commit"] for c in r.tel["resolve_weight_identity_calls"])


def test_extract_features_refuses_before_preflight_and_encodes_nothing(tmp_path):
    """The ordering that would have stopped the 2026-08-25 run five hours early.

    ``--extract-features`` is invoked on a workdir where ``--preflight`` has never
    run, so no frozen atlas exists.  ``assert_atlas_ready(ATLAS_CACHE)`` is the first
    thing Stage 1 does (s2_run.py:530) and it must be a typed refusal, not a
    discovery made after V-JEPA.  The measurement is that the fake encoder was never
    entered and the brain model was never constructed.
    """
    work = _build_template(tmp_path / "work")
    r = _run(work, ["--extract-features"], scen={}, tag="s1-order")
    assert r.rc != 0
    assert r.tel["exc"] == "AtlasCacheMissing", r
    assert r.tel["vjepa_encoded_items"] == []
    assert r.tel["load_model_calls"] == []
    assert r.tel["verify_local_weights_calls"] == []
    assert not (work / "features").exists()


def test_stage1_does_not_finalize_an_artifact_it_cannot_read_back(tmp_path):
    """One expected uid names an item the cache does not hold.

    OLD BUG: ``extract_features._extract`` returned its own ``expected`` list as
    ``produced``, so ``stage1_extract``'s ``missing`` was empty by construction and
    the typed ``ExtractionIncomplete`` could never fire on the real path -- a guard
    called with the argument that switches it off. The run was stopped instead by a
    bare ``KeyError`` from the read-back, several frames deeper.

    FIXED: ``_extract`` now returns the uids the cache ACTUALLY holds, so the typed
    guard fires first and names the missing item.

    What still matters most is unchanged and still asserted: no COMPLETE marker, no
    manifest, nothing a later Stage 2 could mistake for a finished artifact -- and
    the failure arrives with the encode already paid, which is why the marker, not
    the encode, is the thing being protected.
    """
    work = _build_template(tmp_path / "work")
    r = _run(work, ["--preflight", "--extract-features"],
             scen={"orphan_expected_uid": 2}, tag="s1-orphan")
    assert r.rc != 0
    assert r.tel["exc"] == "ExtractionIncomplete", r
    assert "never-encoded" in r.tel["msg"]      # the typed error NAMES the item
    dirs = [p for p in (work / "features").iterdir() if p.is_dir()]
    assert len(dirs) == 1, dirs
    assert not (dirs[0] / COMPLETE).exists()
    assert not (dirs[0] / ART_MANIFEST).exists()
    # the encode DID happen -- so "nothing was finalized" is the whole protection
    assert len(r.tel["vjepa_encoded_items"]) == N_ITEMS


def test_stage2_encodes_zero_after_reading_every_item(work):
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2")
    assert r.rc == 0 and r.tel.get("exc") is None, r
    # the dataloader really did pull every feature through the extractor ...
    assert r.tel["predict_items_read"] == N_ITEMS
    # ... and nothing was computed doing it
    assert r.tel["vjepa_encoded_items"] == []
    assert (work / "data" / "s2_report.json").is_file()


def test_stage2_runs_in_a_different_process_than_stage1(work, stage1_session):
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2")
    assert r.rc == 0, r
    assert r.tel["pid"] != stage1_session.tel["pid"]


# --------------------------------------------------------------------------- #
# 2. Stage 2 consumes a DURABLE artifact resolved via require_artifact_location
# --------------------------------------------------------------------------- #

def test_stage2_resolves_a_durable_artifact_after_the_local_one_is_gone(work):
    """B2.  The local copy is deleted between the two processes, exactly as a
    dead Kaggle session deletes /kaggle/temp.  Stage 2 must find the published
    copy by SEARCHING, not by rebuilding a path it happens to know."""
    local, durable = _artifact_dirs(work)
    shutil.rmtree(work / "features")

    r = _run(work, ["--infer"], scen=_consume(work, durable),
             durable=work / "durable", tag="s2-durable")
    assert r.rc == 0 and r.tel.get("exc") is None, r

    calls = r.tel["require_artifact_location_calls"]
    assert len(calls) == 1, calls
    assert calls[0]["n_items"] == N_ITEMS
    assert str(work / "durable") in calls[0]["search_paths"]
    assert Path(r.tel["artifact_dir"]) == durable
    assert r.tel["vjepa_encoded_items"] == []
    assert (work / "data" / "s2_report.json").is_file()


def test_the_report_names_the_artifact_it_actually_consumed(work):
    """The report is the only thing that leaves the machine, so the tie between it
    and the verified artifact has to be in the artifact's own words: the uid the
    report records is the directory name Stage 1 minted, and the 57-field identity
    it embeds is byte-equal to the one inside that artifact's manifest."""
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2-uid")
    assert r.rc == 0, r
    report = json.loads((work / "data" / "s2_report.json").read_text())
    art = json.loads((local / ART_MANIFEST).read_text())
    assert report["stub"] is False
    assert report["feature_identity_uid"] == local.name
    assert report["feature_identity"] == art["identity"]
    assert set(report["results"]) == set(HARNESS_PARCELS)


def test_stage2_refuses_when_no_durable_artifact_is_reachable(work):
    """The miss must be a typed stop, never a silent 4h45m re-encode."""
    local, durable = _artifact_dirs(work)
    shutil.rmtree(work / "features")
    shutil.rmtree(work / "durable")

    r = _run(work, ["--infer"], scen=_consume(work, durable), tag="s2-nothing")
    assert r.rc != 0
    assert r.tel["exc"] == "ArtifactNotFound", r
    assert r.tel["vjepa_encoded_items"] == []
    assert r.tel["predict_calls"] == 0
    assert not (work / "data" / "s2_report.json").exists()


# --------------------------------------------------------------------------- #
# 3. modality contract: a hard failure that never reaches the report
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("modalities,why", [
    ({"video": "absent"}, "the extractor was deleted before the batch was built"),
    ({"video": "zero"}, "every timestep is the exact zero-fill"),
    ({"video": "one_dead_timestep"}, "a single timestep is the exact zero-fill"),
    ({"video": "real", "audio": "real"}, "a modality declared absent turned up"),
])
def test_modality_violation_is_a_hard_stop_before_the_brain_model(work, modalities,
                                                                  why):
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local, modalities=modalities),
             tag="s2-mod")
    assert r.rc != 0, r
    assert r.tel["exc"] == "ModalityContractViolation", (why, r)
    # the brain model's forward pass was never reached
    assert r.tel["predict_calls"] == 0
    assert r.tel["persist_calls"] == []
    assert r.tel["analyse_calls"] == 0
    assert not (work / "data" / "s2_report.json").exists()
    # ... and the batch WAS actually inspected, so this is a measurement
    assert r.tel["loader_calls"] == 1


def test_a_real_video_modality_passes_the_same_check(work):
    """The contract is not vacuous: the identical wiring accepts a real batch."""
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"],
             scen=_consume(work, local, modalities={"video": "real"}), tag="s2-ok")
    assert r.rc == 0 and r.tel.get("exc") is None, r
    assert r.tel["loader_calls"] == 1
    assert r.tel["predict_calls"] == 1


def test_a_batch_with_no_data_mapping_refuses_rather_than_running_blind(work):
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local, batch_without_data=True),
             tag="s2-blind")
    assert r.rc != 0
    assert r.tel["exc"] == "SystemExit", r
    assert r.tel["predict_calls"] == 0
    assert not (work / "data" / "s2_report.json").exists()


# --------------------------------------------------------------------------- #
# 4. predictions are persisted BEFORE analyse()
# --------------------------------------------------------------------------- #

def test_predictions_are_on_disk_before_analyse_is_entered(work):
    """B6.  Proven by looking at the filesystem from inside analyse(), not by
    reading the source order."""
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2-persist")
    assert r.rc == 0, r
    assert r.tel["analyse_calls"] == 1
    found = r.tel["preds_npz_at_analyse"]
    assert found and len(found) == 1, found
    path, size = found[0]
    assert size > 0
    assert r.tel["persist_calls"] == [path]
    # and it is a real, reloadable array of the right height
    with np.load(path) as z:
        assert z["preds"].shape[0] == z["segment_starts"].shape[0]
        assert z["preds"].shape[0] == int(round(S2.stimulus_duration_s))


def test_nothing_is_persisted_when_the_run_dies_before_predict(work):
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"],
             scen=_consume(work, local, modalities={"video": "zero"}), tag="s2-nopersist")
    assert r.rc != 0
    assert not list((work / "features").glob("*/preds.npz"))


# --------------------------------------------------------------------------- #
# 5. analyse() consumes FROZEN parcels and never imports mne
# --------------------------------------------------------------------------- #

def test_analyse_uses_the_frozen_parcels_and_never_imports_mne(work):
    """B5.  Three separate facts, all measured in the process that ran analyse():
    mne was absent at entry AND still absent afterwards, the live-mne entry point
    was never called, and the vertex indices analyse() received hash to the frozen
    artifact's recorded per-parcel sha256."""
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2-atlas")
    assert r.rc == 0, r
    assert r.tel["mne_at_start"] is False
    assert r.tel["mne_at_analyse_entry"] is False
    assert r.tel["mne_after_analyse"] is False
    assert r.tel["get_vertices_calls"] == 0
    assert r.tel["analyse_parcels"] == {"FFA": [FFA_N, FFA_SHA]}


def test_the_documented_cpu_command_walks_the_frozen_parcel_path(work):
    """``python3 scripts/s2_run.py --infer --stub`` is the command the module
    docstring gives for a box with no GPU, and the one a pre-GPU gate would run.

    It must reach ``analyse()`` through ``assert_atlas_ready(parcels=...)`` +
    ``load_frozen_parcels`` -- the same two calls the GPU path uses -- with no brain
    model constructed, nothing encoded, and mne absent for the whole scientific
    computation.  It also writes to the STUB report path, so a stub can never be
    mistaken for a result.
    """
    r = _run(work, ["--infer", "--stub"], tag="s2-stub")
    assert r.rc == 0 and r.tel.get("exc") is None, r
    assert r.tel["load_model_calls"] == []
    assert r.tel["vjepa_encoded_items"] == []
    assert r.tel["mne_at_start"] is False
    assert r.tel["mne_at_analyse_entry"] is False
    assert r.tel["mne_after_analyse"] is False
    assert r.tel["get_vertices_calls"] == 0
    assert r.tel["analyse_parcels"] == {"FFA": [FFA_N, FFA_SHA]}
    assert (work / "data" / "s2_report_stub.json").is_file()
    assert not (work / "data" / "s2_report.json").exists()
    assert json.loads((work / "data" / "s2_report_stub.json").read_text())["stub"] is True


def test_the_stub_command_also_refuses_a_tampered_frozen_parcel(work):
    """The stub is not a bypass: the same atlas guard fires on the CPU path."""
    from tribe_tools.atlas_preflight import atlas_manifest, load_frozen_parcels
    cache = work / "data" / "s2_parcels.npz"
    man = atlas_manifest(cache)
    verts = np.array(load_frozen_parcels(cache)["FFA"])
    verts[0] = (int(verts[0]) + 7) % N_VERTICES
    _atomic_write(cache, _pack(man, {"FFA": verts}))

    r = _run(work, ["--infer", "--stub"], tag="s2-stub-bad")
    assert r.rc != 0
    assert r.tel["exc"] == "AtlasCacheCorrupt", r
    assert r.tel["analyse_calls"] == 0
    assert r.tel["get_vertices_calls"] == 0
    assert not (work / "data" / "s2_report_stub.json").exists()


def test_the_only_mne_import_left_is_the_report_s_version_probe(work):
    """``mne`` DOES end up in ``sys.modules`` -- strictly after analyse returned,
    and only because ``environment_provenance`` probes ``mne.__version__``
    (neurocheck/s2_design.py:477) while ``_write_report`` builds
    ``provenance.run_environment``.

    That is not B5: it costs ~1.4 s once, after every scientific value is already
    computed, and it cannot influence a vertex index.  It is asserted here rather
    than left implicit, because "mne is not in sys.modules at the end of the run"
    is FALSE on a box that has mne installed and would otherwise read as a
    regression.
    """
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2-mne")
    assert r.rc == 0, r
    assert r.tel["mne_after_analyse"] is False
    report = json.loads((work / "data" / "s2_report.json").read_text())
    probed = report["provenance"]["run_environment"]["mne"]
    # the ONLY thing that pulled mne in is that probe: present iff it found a version
    assert r.tel["mne_at_end"] is (probed is not None)
    assert r.tel["get_vertices_calls"] == 0


def test_a_tampered_frozen_parcel_stops_stage2_before_the_model(work):
    """The frozen-parcel path is load-bearing, not decorative: altering one
    vertex index inside the cache refuses the run."""
    from tribe_tools.atlas_preflight import atlas_manifest, load_frozen_parcels
    local, _ = _artifact_dirs(work)
    cache = work / "data" / "s2_parcels.npz"
    man = atlas_manifest(cache)                       # verifies before we break it
    verts = np.array(load_frozen_parcels(cache)["FFA"])
    verts[0] = (int(verts[0]) + 7) % N_VERTICES
    _atomic_write(cache, _pack(man, {"FFA": verts}))  # same manifest, moved vertex

    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2-atlas-bad")
    assert r.rc != 0
    assert r.tel["exc"] == "AtlasCacheCorrupt", r
    assert r.tel["predict_calls"] == 0
    assert r.tel["get_vertices_calls"] == 0          # no silent live re-resolution
    assert not (work / "data" / "s2_report.json").exists()


# --------------------------------------------------------------------------- #
# 6. an inactive encode counter is refused
# --------------------------------------------------------------------------- #

def test_an_inactive_encode_counter_is_refused(work):
    """Zero is the success value AND the reading of an unplugged instrument.

    The scenario makes exca genuinely unimportable in the Stage-2 process (the
    exact condition ``EncodeCounter.active`` reports on) at the moment the batch
    is built -- i.e. before the counter is entered -- and leaves everything else
    identical to the passing run.
    """
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local, break_exca_at_loader=True),
             tag="s2-inert")
    assert r.rc != 0
    assert r.tel["exc"] == "ConsumeStageRecomputed", r
    assert "INACTIVE" in r.tel["msg"]
    # the run is stopped at the counter, so nothing downstream happened
    assert r.tel["persist_calls"] == []
    assert r.tel["analyse_calls"] == 0
    assert not (work / "data" / "s2_report.json").exists()


def test_the_counter_catches_an_encode_the_firewall_would_have_stopped(work):
    """Defence in depth, measured rather than argued.

    ``infra_override`` puts the extractor back into exca's default ``mode='cached'``
    (exca/map.py:157) -- exactly the state ``model_config_update('consume')`` exists
    to prevent -- and the dataloader asks for a chunk the verified artifact does not
    hold.  exca now happily encodes it, and ``EncodeCounter`` is the layer that stops
    the run.

    Paired with :func:`test_read_only_refuses_an_uncached_chunk_without_encoding_it`
    this is one config key apart, and ``vjepa_encoded_items`` -- appended to inside
    the tensor-producing body -- is 0 with the firewall on and 1 with it off.  That
    is what makes the other test's zero a measurement instead of an assumption.
    """
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"],
             scen=_consume(work, local, extra_uncached_item=True,
                           infra_override={"mode": "cached"}), tag="s2-counter")
    assert r.rc != 0
    assert r.tel["exc"] == "ConsumeStageRecomputed", r
    assert r.tel["vjepa_encoded_items"] == ["s2_stimulus.mp4_99999.00_60.00"]
    assert "encoded 1 item" in r.tel["msg"], r.tel["msg"]
    assert r.tel["persist_calls"] == []
    assert r.tel["analyse_calls"] == 0
    assert not (work / "data" / "s2_report.json").exists()


# --------------------------------------------------------------------------- #
# 7. the sidecar probe is consulted
# --------------------------------------------------------------------------- #

def test_stage1_records_excas_sidecar_digests_in_the_artifact(stage1_session):
    local, _ = _artifact_dirs(stage1_session.work)
    man = json.loads((local / ART_MANIFEST).read_text())
    side = man["exca_sidecars"]
    assert set(side) == {"uid.yaml", "full-uid.yaml", "config.yaml"}
    assert all(v is not None for v in side.values()), side


def test_stage2_consults_the_sidecar_probe(work):
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2-side")
    assert r.rc == 0, r
    assert r.tel["sidecar_digest_calls"], r
    got = r.tel["sidecar_digest_calls"][-1]["digests"]
    man = json.loads((local / ART_MANIFEST).read_text())
    assert got == man["exca_sidecars"]


def test_a_tampered_exca_sidecar_is_refused_before_the_report(work):
    """The probe is not merely called: its answer decides the run.  Altering
    config.yaml -- which exca serves data over without complaint, and which a
    read-only stage will happily RE-CREATE from its own config -- stops Stage 2."""
    local, _ = _artifact_dirs(work)
    cfg = _uid_folder(local) / "config.yaml"
    cfg.write_bytes(cfg.read_bytes() + b"\n# provenance laundering\n")

    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2-side-bad")
    assert r.rc != 0
    # Now that Stage 2 resolves through require_artifact_location with the sidecar
    # probe attached, the tamper is caught one step EARLIER -- the candidate is
    # rejected during resolution rather than during verification. The diagnosis is
    # not lost: ArtifactNotFound embeds Resolution.why(), which carries the
    # underlying ArtifactStale and names the file.
    assert r.tel["exc"] in ("ArtifactStale", "ArtifactCorrupt", "ArtifactNotFound"), r
    assert "config.yaml" in r.tel["msg"], r.tel["msg"]
    assert "Stale" in r.tel["msg"] or "Corrupt" in r.tel["msg"], r.tel["msg"]
    assert r.tel["predict_calls"] == 0
    assert not (work / "data" / "s2_report.json").exists()


# --------------------------------------------------------------------------- #
# the artifact itself must still be digest-verified on the real path
# --------------------------------------------------------------------------- #

def test_a_corrupted_payload_stops_stage2_before_the_brain_model(work):
    """D2's tarpit on the real call graph: exca's index still says COMPLETE and
    `missing == 0`, and the payload is unreadable."""
    local, _ = _artifact_dirs(work)
    blobs = sorted((_uid_folder(local) / "data").glob("*.data"))
    assert blobs
    blobs[0].write_bytes(b"")

    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2-tarpit")
    assert r.rc != 0
    assert r.tel["exc"] in ("ArtifactCorrupt", "ArtifactNotFound"), r
    assert r.tel["predict_calls"] == 0
    assert r.tel["vjepa_encoded_items"] == []
    assert not (work / "data" / "s2_report.json").exists()


def test_stage2_refuses_an_artifact_missing_its_complete_marker(work):
    local, _ = _artifact_dirs(work)
    (local / COMPLETE).unlink()
    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2-incomplete")
    assert r.rc != 0
    assert r.tel["exc"] in ("ArtifactIncomplete", "ArtifactNotFound"), r
    assert not (work / "data" / "s2_report.json").exists()


# --------------------------------------------------------------------------- #
# the two documented workarounds, each proved to be a real defect
# --------------------------------------------------------------------------- #

def test_reader_for_reads_every_item_from_a_real_two_level_exca_cache(work):
    """W1, inverted.

    OLD BUG: ``s2_run._reader_for`` descended ONE level below ``<artifact>/cache``
    and handed that directory to ``CacheDict``. exca nests TWO --
    ``{method,version}/{uid}`` per ``exca/base.py:143`` -- so the reader landed on
    the method directory, every lookup raised KeyError, and Stage 2 reported
    ArtifactNotFound for an artifact that was present and correct.

    FIXED: ``_exca_uid_folder`` descends both levels and refuses to guess if there
    is not exactly one leaf.

    Driven directly against a REAL artifact written by a real Stage 1, so this
    exercises the shipped function rather than a Stage-2 side effect: every expected
    uid must come back, and the bytes must match the digests the artifact recorded.
    """
    import hashlib
    import importlib.util
    import json as _json

    local, _ = _artifact_dirs(work)
    spec = importlib.util.spec_from_file_location(
        "s2run_probe", Path(__file__).resolve().parent.parent / "scripts" / "s2_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    read = mod._reader_for(local)                      # the shipped function
    manifest = _json.loads((local / ART_MANIFEST).read_text())

    for uid, want in manifest["items"].items():
        arr = read(uid)
        h = hashlib.sha256()
        h.update(str(arr.dtype).encode()); h.update(b"|")
        h.update(repr(tuple(arr.shape)).encode()); h.update(b"|")
        h.update(arr.tobytes())
        assert h.hexdigest() == want, f"{uid} read back with the wrong bytes"
    assert len(manifest["items"]) == N_ITEMS


def test_stage2_points_the_extractor_at_the_artifact_it_resolved(work):
    """W2, inverted.

    OLD BUG: ``infer()`` loaded the model with ``cache_folder=None`` -- literally the
    2026-08-25 configuration -- resolved the artifact AFTERWARDS, and never carried
    the location into ``data.<mod>_feature.infra.folder``. Both of Stage 2's payload
    seams hang off that extractor, so ``sidecar_digests(None)`` raised TypeError on
    the first line of ``stage2_infer`` and ``cache_dict`` raised per item.

    FIXED: the artifact is resolved BEFORE the consuming model is built, the model is
    constructed with ``cache_folder=<artifact>/cache``, and ``infer()`` refuses
    outright if ``extractor.infra.folder`` is None.

    The harness no longer supplies the folder, so this run succeeding at all is the
    proof: with the old ordering it could not reach predict.
    """
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2-w2")
    assert r.rc == 0 and r.tel.get("exc") is None, r
    assert r.tel["require_artifact_location_calls"], r
    assert Path(r.tel["artifact_dir"]) == local
    assert r.tel["predict_calls"] == 1
    assert r.tel["vjepa_encoded_items"] == []          # consumed, never encoded
    assert (work / "data" / "s2_report.json").is_file()


def test_publish_calls_the_sidecar_probe_with_a_directory(tmp_path):
    """W3, inverted.

    OLD BUG: ``durable_store.verify_location`` calls ``sidecar_probe(root)`` with one
    argument; ``extract_features`` handed ``publish`` a ZERO-argument lambda. Every
    Stage 1 with ``S2_DURABLE_ROOT`` set raised TypeError *after* the local artifact
    was finalized -- the encode survived the process and nothing survived the session,
    which is precisely the durability the two-stage split exists to provide.

    FIXED: ``_sidecars_for(artifact_dir)`` takes the root it is given, so the probe
    describes the COPY being verified rather than the source -- the same requirement
    B3 imposed on ``reader_factory``.
    """
    work = _build_template(tmp_path / "work")
    r = _run(work, ["--preflight", "--extract-features"],
             scen={}, durable=work / "durable", tag="s1-w3")
    assert r.rc == 0 and r.tel.get("exc") is None, r
    assert r.tel["publish_calls"], r
    assert r.tel["publish_calls"][0]["probe_params"] == 1, r.tel["publish_calls"]
    # BOTH copies exist and both are finalized
    local = sorted(p for p in (work / "features").iterdir() if p.is_dir())
    assert len(local) == 1 and (local[0] / COMPLETE).is_file()
    durable = sorted((work / "durable").glob("s2v1-*"))
    assert len(durable) == 1, durable
    assert (durable[0] / COMPLETE).is_file()
    assert (durable[0] / ART_MANIFEST).is_file()


def test_persist_predictions_writes_the_file_it_then_renames(work):
    """W4, inverted.

    OLD BUG: ``_persist_predictions`` wrote with ``np.savez(dest.with_suffix(
    ".npz.tmp"))``. ``np.savez`` APPENDS ``.npz`` to any name not already ending in
    it, so the bytes landed in ``preds.npz.tmp.npz`` while the next line renamed
    ``preds.npz.tmp``. FileNotFoundError fired immediately after the GPU work and
    before analyse -- the step whose entire purpose is not losing the predictions was
    the step that lost them.

    FIXED: the temp name ends in ``.npz`` so numpy leaves it alone, and the write is
    asserted to exist before the rename.
    """
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local), tag="s2-w4")
    assert r.rc == 0 and r.tel.get("exc") is None, r
    assert r.tel["predict_calls"] == 1
    assert r.tel["persist_calls"], r
    assert (local / "preds.npz").is_file()
    assert r.tel["analyse_calls"] == 1
    assert (work / "data" / "s2_report.json").is_file()
    # and no temp file survived
    assert not list(local.glob("*.tmp*")), list(local.glob("*.tmp*"))


def test_read_only_refuses_an_uncached_chunk_without_encoding_it(work):
    """The firewall is exca's ``mode='read-only'``, and it has to be the thing that
    fires -- not the encode counter noticing afterwards.

    The scenario makes the dataloader ask for one chunk the verified artifact does
    not hold.  With ``data.<mod>_feature.infra.mode == 'read-only'`` actually on the
    extractor, exca raises before the generator is entered and NOTHING is computed.
    Drop the read-only key from ``model_config_update('consume')`` and this same run
    encodes that chunk on the GPU and then fails one step later at
    ``ConsumeStageRecomputed`` -- so ``vjepa_encoded_items`` is what separates the
    two, and it is measured at the tensor-producing body.
    """
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"], scen=_consume(work, local, extra_uncached_item=True),
             tag="s2-readonly")
    assert r.rc != 0
    assert r.tel["exc"] == "RuntimeError", r
    assert "read-only" in r.tel["msg"], r.tel["msg"]
    assert r.tel["vjepa_encoded_items"] == []
    assert r.tel["persist_calls"] == []
    assert r.tel["analyse_calls"] == 0
    assert not (work / "data" / "s2_report.json").exists()


def test_a_design_the_frozen_atlas_does_not_cover_is_refused_before_the_gpu(work):
    """``assert_atlas_ready`` is called WITH ``parcels=`` (s2_run.py:757).

    Here the design asks for two parcels and the frozen cache holds one.  With the
    argument passed, the run stops in the cheap preflight-style check before the
    model is touched.  Without it, the cache verifies fine and the mismatch surfaces
    only inside ``analyse()`` -- i.e. after the GPU has already run -- so
    ``predict_calls == 0`` is the assertion that separates the two.
    """
    local, _ = _artifact_dirs(work)
    r = _run(work, ["--infer"],
             scen=_consume(work, local, parcels=["FFA", "EBA"]), tag="s2-atlas-cover")
    assert r.rc != 0
    assert r.tel["exc"] == "AtlasCacheCorrupt", r
    assert "EBA" in r.tel["msg"]
    assert r.tel["predict_calls"] == 0
    assert r.tel["loader_calls"] == 0
    assert not (work / "data" / "s2_report.json").exists()
