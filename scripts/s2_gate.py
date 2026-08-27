#!/usr/bin/env python3
"""The pre-GPU gate. It EXECUTES the real path and reports what happened.

The gate this replaces asked whether names existed --
``hasattr(ds, "resolve_artifact_location")`` was green while nothing called it.
That is the same presence-based error class as the 57/57 that went green four
hours before a total loss, reintroduced in the tool built to prevent it. It also
had a boolean short-circuit (``not skipped or os.environ.get(...)``) that
reported green WHEN TESTS SKIPPED, an item (``is_owner() is True``) that is true
for every unarmed process, and it built an env dict it never passed to ``run()``.

Every item below is the OUTCOME of running something. There is no item that a
missing implementation could satisfy, and every guard is checked twice: once for
the pass and once with the guard's precondition broken, so an item that cannot
fail is visible as an item that never failed.

    python3 scripts/s2_gate.py               # everything
    python3 scripts/s2_gate.py --preflight   # what must hold before the GPU
    python3 scripts/s2_gate.py --stub        # the CLI's own stub Stage 2
    python3 scripts/s2_gate.py --infer       # real Stage 1 + Stage 2 orchestration
    python3 scripts/s2_gate.py --no-tests    # skip the pytest phase

Exit 0 = GO. Anything else = do not touch Kaggle.

WHAT IS AND IS NOT STUBBED
--------------------------
``tribev2``/``neuralset``/``torch`` are not installable on this box, so the
``--infer`` phase replaces exactly four leaves:

  * ``tribe_tools.model.load_model``     -> a stand-in model whose extractor is
    backed by a REAL ``exca`` 0.5.20 ``CacheDict``. It reproduces
    ``tribev2/demo_utils.py:206-207`` verbatim (``cache_folder`` is written into
    ``data.<mod>_feature.infra.folder`` unconditionally, ``None`` included) and
    ``exca/cachedict/core.py:127-128`` (``folder=None`` + ``keep_in_ram=False``
    -> ``ValueError``). Both facts are re-measured against the real libraries by
    this gate before they are relied on.
  * ``tribe_tools.model.predict_single`` -> deterministic predictions.
  * ``huggingface_hub.hf_hub_download``  -> a local processor config.
  * the V-JEPA HF cache tree             -> a small fixture whose blob names are
    the true sha256 of their contents, so ``force_hash=True`` really hashes.

Everything between those leaves -- ``s2_run.extract_features``, ``s2_run.infer``,
``s2_pipeline.stage1_extract``/``stage2_infer``, ``feature_artifact``,
``durable_store``, ``ledger``, ``provenance``, ``atlas_preflight``, ``analyse``
-- is the real code, called by the real caller with the real arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
S2_RUN = REPO / "scripts" / "s2_run.py"

# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #

@dataclass
class Item:
    phase: str
    name: str
    ok: bool
    evidence: str


ITEMS: list[Item] = []
NOTES: list[str] = []
EXCA_NEST_DEPTH: int | None = None


def check(phase: str, name: str, ok: bool, evidence) -> bool:
    """Record one executed assertion.

    An item with no evidence string is recorded as a FAILURE, not a pass: the
    evidence is what a reader uses to tell an executed check from a vacuous one,
    and an item that cannot say what it observed has not observed anything.
    """
    ev = " ".join(str(evidence).split())
    ok = bool(ok) and bool(ev)
    ITEMS.append(Item(phase, name, ok, ev or "NO EVIDENCE RECORDED"))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if ev:
        print(f"         {ev[:400]}")
    else:
        print("         NO EVIDENCE RECORDED -> counted as a failure")
    return ok


DEFECTS: list[Item] = []


def defect(phase: str, name: str, reproduced: bool, evidence) -> bool:
    """Record a DEMONSTRATED defect in the real execution path.

    A defect item is not an assertion that passes. Reproducing it means the path
    is broken, so it blocks GO no matter how many other items are green -- the
    one thing this file exists to make impossible is a GO printed over a broken
    path. If it stops reproducing it is reported as cleared and the operator is
    told to re-read the item rather than trust it silently.
    """
    ev = " ".join(str(evidence).split())
    DEFECTS.append(Item(phase, name, bool(reproduced), ev or "NO EVIDENCE RECORDED"))
    print(f"  [{'DEFECT' if reproduced else 'cleared'}] {name}")
    print(f"         {ev[:400]}")
    if not reproduced:
        NOTES.append(f"defect demonstration no longer reproduces: {name}. "
                     f"Re-read it -- either it was fixed, or the demonstration "
                     f"rotted. Observed: {ev[:200]}")
    return bool(reproduced)


def note(text: str) -> None:
    NOTES.append(text)
    print(f"  [note] {text}")


# --------------------------------------------------------------------------- #
# process helpers
# --------------------------------------------------------------------------- #

@dataclass
class Run:
    rc: int
    out: str
    err: str
    secs: float

    @property
    def all(self) -> str:
        return self.out + "\n" + self.err

    def tail(self, n: int = 1) -> str:
        lines = [l for l in self.all.strip().splitlines() if l.strip()]
        body = " | ".join(lines[-n:]) if lines else "(no output)"
        if self.rc < 0:
            body = f"KILLED BY SIGNAL {-self.rc} :: {body}"
        return body


def run(cmd, *, cwd, env, timeout=3600, attempts=3) -> Run:
    """Run a subprocess; a SIGNAL death is retried, and the retry is stated.

    A negative return code is a signal, i.e. something OUTSIDE this gate ended
    the process. On a loaded shared box that is a supervisor, not a result, and
    treating it as one would let machine load masquerade as a wiring failure.
    Retries are always reported in the evidence, never silent.
    """
    killed: list[int] = []
    for i in range(max(1, attempts)):
        t0 = time.time()
        p = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env,
                           capture_output=True, text=True, timeout=timeout)
        if p.returncode >= 0:
            err = p.stderr
            if killed:
                err = (f"[gate] attempt(s) killed by signal {killed}; this is "
                       f"attempt {i + 1}\n" + err)
            return Run(p.returncode, p.stdout, err, time.time() - t0)
        killed.append(-p.returncode)
    return Run(p.returncode, p.stdout,
               f"[gate] every attempt was killed by a signal {killed} -- the box "
               f"terminated this process from outside; this is a machine-load "
               f"result, not a wiring result\n" + p.stderr, time.time() - t0)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

PROBE_SITECUSTOMIZE = r'''
# Generated by scripts/s2_gate.py. Counts calls to the LIVE atlas routes so the
# gate can state "Stage 2 resolved zero parcels live" as a measurement rather
# than a reading of the source.
import json as _json
import os as _os
import sys as _sys

_OUT = _os.environ.get("S2_GATE_PROBE")

# chain to whatever sitecustomize we are shadowing, so this stays a probe and
# not a change of behaviour
_here = _os.path.dirname(_os.path.abspath(__file__))
for _p in _sys.path:
    try:
        _cand = _os.path.join(_p, "sitecustomize.py")
    except Exception:
        continue
    if _os.path.abspath(_p) == _here or not _os.path.isfile(_cand):
        continue
    try:
        import importlib.util as _u
        _spec = _u.spec_from_file_location("_s2_chained_sitecustomize", _cand)
        _mod = _u.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
    except Exception:
        pass
    break

_state = {"patched": [], "calls": {}, "imported": []}

TARGETS = {
    "mne": [("read_labels_from_annot", "mne.read_labels_from_annot")],
    "tribe_tools.atlas": [("get_vertices", "tribe_tools.atlas.get_vertices"),
                          ("_get_hcp_labels", "tribe_tools.atlas._get_hcp_labels")],
}


def _flush():
    if not _OUT:
        return
    _state["imported"] = sorted(m for m in _sys.modules
                                if m == "mne" or m.startswith("mne.")
                                or m == "tribe_tools.atlas")
    try:
        with open(_OUT, "w") as f:
            _json.dump(_state, f)
    except Exception:
        pass


def _wrap(mod, attr, label):
    try:
        orig = getattr(mod, attr)
    except Exception:
        return
    if orig is None or getattr(orig, "_s2_probe", False):
        return

    def w(*a, **k):
        _state["calls"][label] = _state["calls"].get(label, 0) + 1
        _flush()
        return orig(*a, **k)

    w._s2_probe = True
    try:
        setattr(mod, attr, w)
    except Exception:
        return
    _state["patched"].append(label)
    _flush()


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        for f in list(_sys.meta_path):
            if f is self:
                continue
            try:
                spec = f.find_spec(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None:
                break
        else:
            return None
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        _orig_exec = loader.exec_module

        def exec_module(module, _o=_orig_exec, _n=fullname):
            _o(module)
            for attr, label in TARGETS[_n]:
                _wrap(module, attr, label)

        try:
            loader.exec_module = exec_module
        except Exception:
            return spec
        return spec


_sys.meta_path.insert(0, _Finder())
import atexit as _atexit
_atexit.register(_flush)
_flush()
'''


def write_probe(root: Path) -> Path:
    d = root / "probe"
    d.mkdir(parents=True, exist_ok=True)
    (d / "sitecustomize.py").write_text(PROBE_SITECUSTOMIZE)
    return d


def read_probe(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {"patched": [], "calls": {}, "imported": []}


def build_hf_cache(root: Path, *, commit: str, repo_id: str,
                   filenames, weights_name: str) -> Path:
    """The cache layout huggingface_hub writes: blobs named by their digest,
    snapshot symlinks, refs/main. Small files, real sha256 names -- so
    ``force_hash=True`` hashes bytes that genuinely match the blob name."""
    root = Path(root)
    repo = root / ("models--" + repo_id.replace("/", "--"))
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    snap = repo / "snapshots" / commit
    snap.mkdir(parents=True, exist_ok=True)
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(commit)
    for name in filenames:
        if name == weights_name:
            content = b"S2-GATE-VJEPA-WEIGHT-FIXTURE\n" * 64
        elif name.endswith("preprocessor_config.json"):
            content = json.dumps(PROCESSOR_CONFIG, sort_keys=True).encode()
        else:
            content = json.dumps({"model_type": "vjepa2", "fixture": True},
                                 sort_keys=True).encode()
        digest = hashlib.sha256(content).hexdigest()
        blob = repo / "blobs" / digest
        blob.write_bytes(content)
        link = snap / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(blob)
    return root


PROCESSOR_CONFIG = {
    "do_resize": True,
    "size": {"shortest_edge": 256},
    "resample": 2,
    "do_center_crop": True,
    "crop_size": {"height": 256, "width": 256},
    "do_rescale": True,
    "rescale_factor": 0.00392156862745098,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "video_processor_type": "VJEPA2VideoProcessor",
}

FAKE_DISTS = {
    "tribev2": "0.1.0", "neuralset": "0.0.2", "torch": "2.4.0",
    "torchvision": "0.19.0", "transformers": "4.44.0", "moviepy": "1.0.3",
}


def build_metadir(root: Path) -> Path:
    """dist-info trees so ``provenance.library_versions`` returns real strings.

    ``_versions_or_die`` refuses the literal "absent", by design: two machines
    that cannot read their own versions would otherwise collapse to one identity.
    Making the versions readable is therefore a precondition of running the path
    at all, not a way around a check.
    """
    d = root / "meta"
    d.mkdir(parents=True, exist_ok=True)
    for name, ver in FAKE_DISTS.items():
        di = d / f"{name}-{ver}.dist-info"
        di.mkdir(exist_ok=True)
        (di / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n")
        (di / "INSTALLER").write_text("s2-gate-fixture\n")
    return d


def make_workdir(root: Path, name: str, *, tamper_video: bool = False) -> Path:
    """A scenario cwd whose ``data/`` holds the real, verified inputs.

    Symlinked, not copied: the run must see the bytes ``--prepare`` recorded.
    Everything the run WRITES (atlas cache, ledger, report, artifacts) lands
    inside the scenario directory, so the repo's own data/ is never touched.
    """
    wd = root / "runs" / name
    (wd / "data").mkdir(parents=True, exist_ok=True)
    src = REPO / "data"
    for f in ("s2_manifest.json", "s2_stimulus_probe.json", "floc"):
        dst = wd / "data" / f
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src / f)
    vid = wd / "data" / "s2_stimulus.mp4"
    if vid.exists() or vid.is_symlink():
        vid.unlink()
    if tamper_video:
        shutil.copy2(src / "s2_stimulus.mp4", vid)
        with open(vid, "r+b") as f:
            f.seek(vid.stat().st_size // 2)
            b = f.read(1)
            f.seek(vid.stat().st_size // 2)
            f.write(bytes([b[0] ^ 0xFF]))
    else:
        vid.symlink_to(src / "s2_stimulus.mp4")
    return wd


def base_env(root: Path, wd: Path, *, hf: Path | None, probe_out: Path | None,
             extra: dict | None = None, stim_root: Path | None = None) -> dict:
    env = dict(os.environ)
    env.pop("S2_ALLOW_NETWORK", None)
    env.pop("S2_ALLOW_DOWNLOAD", None)
    env["S2_STIMULUS_ROOT"] = str((stim_root or (wd / "data")).resolve())
    env["HF_HUB_CACHE"] = str(hf) if hf else str(root / "hf_absent")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # unbuffered: on a loaded box a SIGTERM otherwise discards every
    # line the run had already produced, and the evidence with it
    env["PYTHONUNBUFFERED"] = "1"
    pp = [str(root / "probe")]
    for p in (os.environ.get("S2_DEV_SITE_PACKAGES", "") or "").split(os.pathsep):
        if p:
            pp.append(p)
    if os.environ.get("PYTHONPATH"):
        pp.append(os.environ["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    if probe_out:
        env["S2_GATE_PROBE"] = str(probe_out)
    else:
        env.pop("S2_GATE_PROBE", None)
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def ledger_events(wd: Path) -> list[dict]:
    p = wd / "data" / "s2_ledger.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def ledger_kinds(wd: Path) -> list[str]:
    return [r.get("event") for r in ledger_events(wd)]


# --------------------------------------------------------------------------- #
# phase 0 -- the third-party facts this gate's stand-ins reproduce
# --------------------------------------------------------------------------- #

SQ_SRC = """
import typing as tp
import pydantic


class Sq(pydantic.BaseModel):
    infra: exca.MapInfra = exca.MapInfra()

    @infra.apply(item_uid=str)
    def process(self, items: tp.Sequence[int]) -> tp.Iterator[np.ndarray]:
        for i in items:
            yield np.full((2,), float(i), dtype=np.float32)
"""


def phase_ground_truth(root: Path) -> None:
    print("\n0  ground truth: the library behaviours the --infer stand-in reproduces")
    site = [p for p in (os.environ.get("S2_DEV_SITE_PACKAGES", "") or "")
            .split(os.pathsep) if p]
    for p in site:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import exca  # noqa: F401
        from exca.cachedict import CacheDict
    except Exception as e:
        check("0", "real exca 0.5.20 is importable",
              False,
              f"{type(e).__name__}: {e}. Export S2_DEV_SITE_PACKAGES to a tree "
              f"holding exca 0.5.20; without it the artifact round-trip and the "
              f"encode counter are unmeasured, and an unmeasured counter reads "
              f"zero, which is also the success value.")
        return
    check("0", "real exca is importable and is the pinned version",
          getattr(exca, "__version__", "?") == "0.5.20",
          f"exca {getattr(exca, '__version__', '?')} from {exca.__file__}")

    err = None
    try:
        CacheDict(folder=None, keep_in_ram=False)
    except Exception as e:
        err = e
    check("0", "exca: folder=None + keep_in_ram=False raises (cachedict/core.py:127)",
          isinstance(err, ValueError),
          f"{type(err).__name__ if err else 'no error'}: {err}")

    import numpy as np
    # cold every time: a warm cache would make the round-trip and the
    # cold/warm encode counts prove something other than what they claim
    shutil.rmtree(root / "gt_cache", ignore_errors=True)
    shutil.rmtree(root / "gt_map", ignore_errors=True)
    d = root / "gt_cache"
    d.mkdir(parents=True, exist_ok=True)
    a = np.arange(24, dtype=np.float32).reshape(3, 8)
    cd = CacheDict(folder=d, keep_in_ram=False)
    with cd.write():
        cd["probe_0.00_60.00"] = a
    back = np.asarray(CacheDict(folder=d, keep_in_ram=False)["probe_0.00_60.00"])
    check("0", "exca CacheDict round-trips an array through a fresh handle",
          bool(np.array_equal(a, back)),
          f"wrote {a.shape} {a.dtype}, read back equal={np.array_equal(a, back)}")

    # the encode counter, measured against exca's real funnel.
    # Built through exec because this module uses `from __future__ import
    # annotations` and exca reads the raw annotation objects off the method.
    sys.path.insert(0, str(REPO))
    from tribe_tools.ledger import EncodeCounter
    ns: dict = {"exca": exca, "np": np}
    # dont_inherit: `exec` would otherwise inherit this module's
    # `from __future__ import annotations` and hand exca strings
    exec(compile(SQ_SRC, "<s2gate-sq>", "exec", dont_inherit=True), ns)
    m = ns["Sq"](infra={"folder": str(root / "gt_map"), "keep_in_ram": False})
    with EncodeCounter() as c1:
        list(m.process([1, 2, 3]))
    with EncodeCounter() as c2:
        list(m.process([1, 2, 3]))
    check("0", "EncodeCounter counts a real exca recompute and reports zero on a hit",
          c1.active and c1.items == 3 and c2.active and c2.items == 0,
          f"cold run active={c1.active} items={c1.items}; warm run "
          f"active={c2.active} items={c2.items}")

    # how deep exca actually nests, measured -- the --infer stand-in copies it
    uf = m.infra.uid_folder()
    depth = len(uf.relative_to(root / "gt_map").parts) if uf else None
    global EXCA_NEST_DEPTH
    EXCA_NEST_DEPTH = depth
    check("0", "exca nests its cache TWO levels under infra.folder "
               "('{method},{version}/{uid}', base.py:143)",
          depth == 2,
          f"infra.folder={root / 'gt_map'} -> uid_folder()={uf} "
          f"({depth} level(s) below it)")


# --------------------------------------------------------------------------- #
# phase H -- THIS host, untouched
# --------------------------------------------------------------------------- #

def phase_host(root: Path) -> None:
    """The only phase with no fixtures at all.

    Every other phase supplies a V-JEPA cache fixture and an exca on PYTHONPATH,
    because otherwise the wiring cannot be exercised here. That is a harness,
    and a harness must never be mistaken for the machine. This phase runs the
    real CLI against the real environment, so "the gate went green" can never
    mean "the gate went green against its own fixtures".
    """
    print("\nH  host readiness: the real environment, no fixtures, no PYTHONPATH")
    wd = make_workdir(root, "host_preflight")
    env = dict(os.environ)
    env.pop("S2_ALLOW_NETWORK", None)
    env.pop("S2_ALLOW_DOWNLOAD", None)
    env.pop("PYTHONPATH", None)
    env["S2_STIMULUS_ROOT"] = str((wd / "data").resolve())
    r = run([sys.executable, S2_RUN, "--preflight"], cwd=wd, env=env, timeout=900)
    check("H", "s2_run.py --preflight passes on THIS machine, unassisted",
          r.rc == 0 and "preflight PASSED" in r.all,
          f"rc={r.rc} :: {r.tail(1)}")

    r2 = run([sys.executable, "-c",
              "import exca, sys; print(exca.__version__, exca.__file__)"],
             cwd=wd, env=env, timeout=120)
    check("H", "exca is importable by a bare interpreter here -- otherwise the "
               "encode counter is inert and reads zero, the success value",
          r2.rc == 0 and "0.5.20" in r2.out,
          f"rc={r2.rc} :: {r2.tail(1)}")


# --------------------------------------------------------------------------- #
# phase P -- s2_run.py --preflight, executed
# --------------------------------------------------------------------------- #

def phase_preflight(root: Path) -> dict:
    print("\nP  preflight: python3 scripts/s2_run.py --preflight, run for real")
    sys.path.insert(0, str(REPO))
    from tribe_tools import provenance as P

    hf_ok = build_hf_cache(root / "hf_ok", commit=P.VJEPA2_COMMIT,
                           repo_id=P.VJEPA2_REPO, filenames=P.VJEPA2_FILENAMES,
                           weights_name=P.WEIGHTS_FILENAME)
    wrong_commit = "0" * 39 + "1"
    hf_bad = build_hf_cache(root / "hf_wrong", commit=wrong_commit,
                            repo_id=P.VJEPA2_REPO, filenames=P.VJEPA2_FILENAMES,
                            weights_name=P.WEIGHTS_FILENAME)

    # ---- P1: the whole thing, green
    wd = make_workdir(root, "preflight_ok")
    probe = root / "probe_preflight.json"
    env = base_env(root, wd, hf=hf_ok, probe_out=probe)
    r = run([sys.executable, S2_RUN, "--preflight"], cwd=wd, env=env, timeout=900)
    ok1 = check("P", "s2_run.py --preflight exits 0 and says it passed",
                r.rc == 0 and "preflight PASSED" in r.all,
                f"rc={r.rc} in {r.secs:.1f}s :: {r.tail(2)}")
    kinds = ledger_kinds(wd)
    check("P", "the run appended preflight_passed to the ledger it writes",
          "preflight_passed" in kinds,
          f"ledger events = {kinds}")

    # ---- P3: the atlas cache the run WROTE, read back
    cache_path = wd / "data" / "s2_parcels.npz"
    try:
        from neurocheck.s2_design import ALL_PARCELS
        from tribe_tools.atlas_preflight import load_frozen_parcels
        parcels = load_frozen_parcels(cache_path)
        want = {p.name for p in ALL_PARCELS}
        sizes = {k: int(len(v)) for k, v in parcels.items()}
        ok = set(parcels) == want and all(v > 0 for v in sizes.values())
        check("P", "the frozen atlas the run produced covers every design parcel, "
                   "non-empty", ok,
              f"{len(sizes)} parcels, vertex counts {sizes}")
    except Exception as e:
        parcels = {}
        check("P", "the frozen atlas the run produced covers every design parcel, "
                   "non-empty", False, f"{type(e).__name__}: {e}")

    pr = read_probe(probe)
    annot_reads = int(pr.get("calls", {}).get("mne.read_labels_from_annot", 0))
    check("P", "the live-atlas probe is ARMED (it saw preflight read the annots)",
          annot_reads >= 1 and "mne.read_labels_from_annot" in pr.get("patched", []),
          f"patched={pr.get('patched')} read_labels_from_annot={annot_reads} calls")

    # ---- P4: negative -- one flipped byte in the stimulus
    wd2 = make_workdir(root, "preflight_bad_video", tamper_video=True)
    env2 = base_env(root, wd2, hf=hf_ok, probe_out=None)
    r2 = run([sys.executable, S2_RUN, "--preflight"], cwd=wd2, env=env2, timeout=900)
    k2 = ledger_kinds(wd2)
    check("P", "one flipped stimulus byte stops preflight before the atlas",
          r2.rc != 0 and "hash mismatch" in r2.all
          and "preflight_passed" not in k2
          and not (wd2 / "data" / "s2_parcels.npz").exists(),
          f"rc={r2.rc} ledger={k2} atlas_cache_written="
          f"{(wd2 / 'data' / 's2_parcels.npz').exists()} :: {r2.tail(1)}")

    # ---- P5: negative -- the pinned V-JEPA commit
    wd3 = make_workdir(root, "preflight_wrong_commit")
    env3 = base_env(root, wd3, hf=hf_bad, probe_out=None)
    r3 = run([sys.executable, S2_RUN, "--preflight"], cwd=wd3, env=env3, timeout=900)
    pinned_line = next((l.strip() for l in r3.all.splitlines()
                        if wrong_commit in l or "pinned" in l), r3.tail(1))
    check("P", "a V-JEPA cache at the wrong commit is refused "
               "(expected_commit is really passed)",
          r3.rc != 0 and P.VJEPA2_COMMIT in r3.all and wrong_commit in r3.all,
          f"rc={r3.rc} :: {pinned_line}")

    # ---- P6: negative -- no weight identity available at all
    wd4 = make_workdir(root, "preflight_no_weights")
    env4 = base_env(root, wd4, hf=root / "hf_absent", probe_out=None)
    r4 = run([sys.executable, S2_RUN, "--preflight"], cwd=wd4, env=env4, timeout=900)
    check("P", "an unresolvable weight identity stops preflight (never guessed)",
          r4.rc != 0 and "Refusing to guess an identity" in r4.all,
          f"rc={r4.rc} :: {r4.tail(1)}")

    # ---- P7: negative -- the atlas assets are gone
    wd5 = make_workdir(root, "preflight_no_atlas")
    empty = root / "empty_mne"
    empty.mkdir(parents=True, exist_ok=True)
    env5 = base_env(root, wd5, hf=hf_ok, probe_out=None,
                    extra={"MNE_DATASETS_SAMPLE_PATH": str(empty)})
    r5 = run([sys.executable, S2_RUN, "--preflight"], cwd=wd5, env=env5, timeout=900)
    check("P", "missing atlas assets stop preflight before the weight step",
          r5.rc != 0 and "atlas" in r5.all.lower()
          and "V-JEPA weights identified" not in r5.all,
          f"rc={r5.rc} :: {r5.tail(1)}")

    return {"hf_ok": hf_ok, "atlas_cache": cache_path, "parcels": parcels,
            "workdir": wd, "ok": ok1}


# --------------------------------------------------------------------------- #
# phase S -- s2_run.py --infer --stub, executed
# --------------------------------------------------------------------------- #

def phase_stub(root: Path, pre: dict) -> None:
    print("\nS  stub Stage 2: python3 scripts/s2_run.py --infer --stub, run for real")
    if not pre.get("atlas_cache") or not Path(pre["atlas_cache"]).is_file():
        check("S", "a frozen atlas exists to run the stub against", False,
              "the preflight phase produced no atlas cache, so --infer --stub "
              "cannot be exercised")
        return

    sys.path.insert(0, str(REPO))
    from neurocheck.s2_design import ALL_PARCELS

    wd = pre["workdir"]
    probe = root / "probe_stub.json"
    env = base_env(root, wd, hf=pre["hf_ok"], probe_out=probe)
    r = run([sys.executable, S2_RUN, "--infer", "--stub"], cwd=wd, env=env,
            timeout=3600)
    check("S", "s2_run.py --infer --stub exits 0",
          r.rc == 0, f"rc={r.rc} in {r.secs:.1f}s :: {r.tail(1)}")

    rep_path = wd / "data" / "s2_report_stub.json"
    rep = {}
    try:
        rep = json.loads(rep_path.read_text())
    except Exception as e:
        check("S", "the stub run wrote a parseable report", False,
              f"{type(e).__name__}: {e} at {rep_path}")
    else:
        names = sorted((rep.get("results") or {}))
        want = sorted(p.name for p in ALL_PARCELS)
        check("S", "the stub run wrote a v2 report scoring every design parcel",
              rep.get("schema_version") == 2 and rep.get("stub") is True
              and names == want and bool(rep.get("verdict")),
              f"schema={rep.get('schema_version')} stub={rep.get('stub')} "
              f"parcels={len(names)} verdict.stop={(rep.get('verdict') or {}).get('stop')}")

    # ---- B5, measured
    pr = read_probe(probe)
    calls = pr.get("calls", {})
    imported = pr.get("imported", [])
    live = int(calls.get("mne.read_labels_from_annot", 0)) \
        + int(calls.get("tribe_tools.atlas.get_vertices", 0)) \
        + int(calls.get("tribe_tools.atlas._get_hcp_labels", 0))
    armed = [t for t in pr.get("patched", [])]
    check("S", "B5: Stage 2 resolves ZERO parcels live, and the counter that says "
               "so was installed in THIS process",
          live == 0 and "mne.read_labels_from_annot" in armed,
          f"probe installed on {armed}; call counts {calls or '{}'} "
          f"(the identical probe recorded 4 read_labels_from_annot calls in the "
          f"preflight process, so a live read here would have been counted)")
    if any(m == "mne" for m in imported):
        note("mne is still IMPORTED by the --infer --stub process -- traced to "
             "neurocheck/s2_design.py:468 environment_provenance, which does "
             "__import__(mod).__version__ while WRITING THE REPORT. Zero "
             "parcellation reads, so B5 stays closed, but ~1.4 s / ~180 MB is "
             "paid after inference for a version string")

    # ---- the report's vertices ARE the frozen vertices
    frozen = {k: len(v) for k, v in (pre.get("parcels") or {}).items()}
    diag = {k: (v or {}).get("n_vertices")
            for k, v in (rep.get("diagnostics") or {}).items()}
    check("S", "the scored vertices are the frozen, digest-verified ones",
          bool(frozen) and bool(diag) and frozen == diag,
          f"frozen={frozen} report={diag}")

    # ---- negative: the frozen atlas is load-bearing
    wd_del = make_workdir(root, "stub_no_atlas")
    shutil.copy2(wd / "data" / "s2_ledger.jsonl", wd_del / "data" / "s2_ledger.jsonl") \
        if (wd / "data" / "s2_ledger.jsonl").is_file() else None
    probe2 = root / "probe_stub_noatlas.json"
    env2 = base_env(root, wd_del, hf=pre["hf_ok"], probe_out=probe2)
    r2 = run([sys.executable, S2_RUN, "--infer", "--stub"], cwd=wd_del, env=env2,
             timeout=900)
    pr2 = read_probe(probe2)
    live2 = sum(int(v) for v in pr2.get("calls", {}).values())
    check("S", "with no frozen atlas the stub refuses instead of resolving live",
          r2.rc != 0 and live2 == 0,
          f"rc={r2.rc} live atlas calls={live2} :: {r2.tail(1)}")

    # ---- negative: an atlas frozen for a DIFFERENT parcel set is rejected
    wd_st = make_workdir(root, "stub_stale_atlas")
    try:
        import numpy as np
        src = np.load(pre["atlas_cache"], allow_pickle=False)
        keys = [k for k in src.files]
        drop = next((k for k in keys if k.endswith("FFA")), None) or keys[-1]
        np.savez(wd_st / "data" / "s2_parcels.npz",
                 **{k: src[k] for k in keys if k != drop})
        env3 = base_env(root, wd_st, hf=pre["hf_ok"], probe_out=None)
        r3 = run([sys.executable, S2_RUN, "--infer", "--stub"], cwd=wd_st, env=env3,
                 timeout=900)
        check("S", "an atlas frozen for a different parcel set is rejected",
              r3.rc != 0,
              f"dropped array {drop!r} from the frozen cache -> rc={r3.rc} :: "
              f"{r3.tail(1)}")
    except Exception as e:
        check("S", "an atlas frozen for a different parcel set is rejected", False,
              f"could not build the stale-atlas fixture: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# phase I -- the real Stage 1 / Stage 2 orchestration
# --------------------------------------------------------------------------- #

DRIVER = r'''
# Generated by scripts/s2_gate.py. Runs the REAL s2_run.extract_features /
# s2_run.infer with only the tribev2/torch/hub leaves replaced. Never edits
# s2_run; every argument the orchestration passes is recorded and reported.
import hashlib
import json
import os
import sys
import types
from pathlib import Path

REPO = os.environ["S2_GATE_REPO"]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.environ["S2_GATE_METADIR"])

RESULT = Path(os.environ["S2_GATE_RESULT"])
STAGE = sys.argv[1]

R = {"stage": STAGE, "status": "started", "error": None, "error_type": None,
     "load_model_calls": [], "events": [], "predict_calls": 0,
     "sidecar_probe_arity_error": False}


def save():
    RESULT.write_text(json.dumps(R, indent=2, default=str))


def flag(name):
    return os.environ.get(name, "") == "1"


import numpy as np

PROC = os.environ["S2_GATE_PROC_JSON"]
_hh = types.ModuleType("huggingface_hub")


def _hf_hub_download(repo_id, filename, **kw):
    R["events"].append("hf_hub_download:" + str(filename))
    if not str(filename).endswith("preprocessor_config.json"):
        raise RuntimeError("the gate stand-in only serves the processor config, "
                           "not " + str(filename))
    return PROC


_hh.hf_hub_download = _hf_hub_download
sys.modules["huggingface_hub"] = _hh

import tribe_tools.model as tmodel
import tribe_tools.inference as tinference
import tribe_tools.provenance as prov

ITEM_ROWS = 8
ITEM_DIM = 1408


def det_array(key, rows=ITEM_ROWS, dim=ITEM_DIM):
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(0.0, 1.0, (rows, dim)).astype(np.float32)


class Event:
    def __init__(self, path, offset, duration):
        self._path = str(path)
        self.offset = float(offset)
        self.duration = float(duration)

    def study_relative_path(self):
        return self._path


def chunk_events(video_path, total, max_duration=60.0, min_duration=30.0):
    # neuralset ChunkEvents("Video", 60, 30), the literals at demo_utils.py:78
    out = []
    off = 0.0
    while off < total:
        dur = min(max_duration, total - off)
        if dur >= min_duration:
            out.append(Event(video_path, off, dur))
        off += max_duration
    return out


class Helper:
    def __init__(self, events):
        self._events = events

    def extract(self, events_df):
        return list(self._events)


class Image:
    model_name = "facebook/vjepa2-vitg-fpc64-256"
    pretrained = True
    imsize = 256
    token_aggregation = "mean"
    cache_all_layers = False
    cache_n_layers = None
    layers = None
    layer_aggregation = "mean"


class Infra:
    # tribev2/demo_utils.py:206-207 writes cache_folder into infra.folder for all
    # three modalities UNCONDITIONALLY, None included. Reproduced verbatim.
    def __init__(self, folder, keep_in_ram, mode, version, forbid_single):
        self.folder = None if folder is None else Path(folder)
        self.keep_in_ram = bool(keep_in_ram)
        self.mode = mode
        self.version = version
        self.forbid_single_item_computation = bool(forbid_single)
        self.cluster = None
        # exca/base.py:143 _uid_string = "{method},{version}/{uid}", so the cache
        # sits TWO levels under infra.folder. The gate measures this against real
        # exca in phase 0 before relying on it.
        self._uid = ("_get_data," + str(version) + "/vjepa-"
                     + hashlib.sha256(str(version).encode()).hexdigest()[:16])

    def uid_folder(self):
        if self.folder is None:
            return None
        uf = self.folder / self._uid
        uf.mkdir(parents=True, exist_ok=True)
        for name, body in (("uid.yaml", "uid: " + self._uid),
                           ("full-uid.yaml", "full_uid: " + str(self.version)),
                           ("config.yaml", "version: " + str(self.version))):
            f = uf / name
            if not f.is_file():
                f.write_text(body + "\n")
        return uf

    @property
    def cache_dict(self):
        # exca/cachedict/core.py:127-128, re-measured by the gate's phase 0
        if self.folder is None and not self.keep_in_ram:
            raise ValueError("At least folder or keep_in_ram should be activated")
        from exca.cachedict import CacheDict
        return CacheDict(folder=self.uid_folder(), keep_in_ram=self.keep_in_ram)


class Extractor:
    def __init__(self, infra, events):
        self.infra = infra
        self.image = Image()
        self.frequency = 2.0
        self.clip_duration = 2.0
        self.num_frames = None
        self.max_imsize = 256
        self.layer_type = "encoder"
        self.use_audio = False
        self._event_types_helper = Helper(events)

    def _get_data(self, events):
        if self.infra.mode == "read-only":
            raise RuntimeError("self.mode='read-only' but found "
                               + str(len(list(events))) + " missing items")
        cd = self.infra.cache_dict
        with cd.write():
            for e in events:
                uid = (e.study_relative_path() + "_" + format(e.offset, ".2f")
                       + "_" + format(e.duration, ".2f"))
                arr = det_array(uid)
                cd[uid] = arr
                yield arr


class Batch:
    def __init__(self, data):
        self.data = data


class Loader:
    def __init__(self, batch):
        self._batch = batch

    def __iter__(self):
        yield self._batch


class Data:
    def __init__(self, extractor):
        self.video_feature = extractor
        self.audio_feature = None
        self.text_feature = None
        self.num_workers = 0

    def get_loaders(self, events=None, split_to_build="all"):
        video = det_array("modality-video", rows=16)
        if flag("S2_GATE_ZERO_TIMESTEP"):
            video[3, :] = 0.0
        data = {"video": video}
        if flag("S2_GATE_EXTRA_AUDIO"):
            data["audio"] = det_array("modality-audio", rows=16)
        return {"all": Loader(Batch(data))}


class Model:
    def __init__(self, extractor):
        self.data = Data(extractor)

    def get_events_dataframe(self, video_path=None):
        return {"video_path": video_path}


import s2_run
from neurocheck.s2_design import S2

TOTAL = float(S2.stimulus_duration_s)
EVENTS = chunk_events(s2_run.video_path(), TOTAL)
R["n_expected_items"] = len(EVENTS)

FORCED_CACHE = os.environ.get("S2_GATE_FORCED_CACHE") or None


def fake_load_model(device="cuda", cache_folder=None, config_update=None,
                    revision=None, checkpoint_dir=None):
    cu = dict(config_update or {})
    R["load_model_calls"].append({"device": device,
                                  "cache_folder": None if cache_folder is None
                                  else str(cache_folder),
                                  "revision": revision,
                                  "config_update": cu})
    save()
    folder = cache_folder
    if folder is None and FORCED_CACHE:
        # the one-line correction under test; recorded, never hidden
        folder = FORCED_CACHE
    infra = Infra(folder=folder,
                  keep_in_ram=cu.get("data.video_feature.infra.keep_in_ram", True),
                  mode=cu.get("data.video_feature.infra.mode"),
                  # the checkpoint's own value when the caller supplies none:
                  # tribev2/grids/defaults.py:94 sets infra.version = "release"
                  version=cu.get("data.video_feature.infra.version") or "release",
                  forbid_single=cu.get(
                      "data.video_feature.infra.forbid_single_item_computation", False))
    return Model(Extractor(infra, EVENTS))


def fake_predict_single(model, video_path, **kw):
    R["predict_calls"] += 1
    save()
    if flag("S2_GATE_RECOMPUTE"):
        # a Stage 2 that really re-enters exca's compute funnel
        import typing as tp
        import pydantic
        import exca

        class Sq(pydantic.BaseModel):
            infra: exca.MapInfra = exca.MapInfra()

            @infra.apply(item_uid=str)
            def process(self, items: tp.Sequence[int]) -> tp.Iterator[np.ndarray]:
                for i in items:
                    yield np.full((2,), float(i), dtype=np.float32)

        Sq(infra={"folder": os.environ["S2_GATE_RECOMPUTE_DIR"],
                  "keep_in_ram": False}).process([1, 2, 3])
        list(Sq(infra={"folder": os.environ["S2_GATE_RECOMPUTE_DIR"],
                       "keep_in_ram": False}).process([4, 5, 6]))
    n_rows = int(round(TOTAL))
    preds = np.random.default_rng(7).normal(0, 1, (n_rows, 20484)).astype(np.float32)
    R["preds_checksum"] = float(preds[:, 0].sum())
    R["preds_shape"] = list(preds.shape)
    save()

    class Seg:
        def __init__(self, t):
            self.start = float(t)

    return preds, [Seg(t) for t in range(n_rows)]


tmodel.load_model = fake_load_model
tmodel.predict_single = fake_predict_single
tinference.predict_single = fake_predict_single

if flag("S2_GATE_FIX_B1"):
    # scripts/s2_run.py:553 hands verify_local_weights the HUB ROOT; _locate wants
    # the file, its snapshot dir, or the models--<org>--<name> repo dir. This
    # rewrites ONLY that one argument, so whatever happens next is attributable
    # to it and to nothing else. The function itself is untouched.
    _real_vlw = prov.verify_local_weights

    def _vlw(path_or_cache, expected, **kw):
        p = Path(path_or_cache)
        cand = p / ("models--" + prov.VJEPA2_REPO.replace("/", "--"))
        if cand.is_dir():
            path_or_cache = cand
        return _real_vlw(path_or_cache, expected, **kw)

    prov.verify_local_weights = _vlw

if flag("S2_GATE_FIX_B6"):
    # scripts/s2_run.py:739-741 builds the temp name with
    # dest.with_suffix(".npz.tmp") and hands it to np.savez, which APPENDS .npz
    # when the name does not already end in it -- so the bytes land at
    # preds.npz.tmp.npz and os.replace then raises FileNotFoundError. Corrected
    # here so the rest of Stage 2 is reachable; the defect itself is asserted.
    def _persist(preds, segments, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        starts = np.asarray([float(getattr(sg, "start", i))
                             for i, sg in enumerate(segments)])
        tmp = dest.with_name(dest.name + ".tmp.npz")
        np.savez(tmp, preds=np.asarray(preds), segment_starts=starts)
        os.replace(tmp, dest)
        return dest

    s2_run._persist_predictions = _persist

try:
    if STAGE == "stage1":
        rc = s2_run.extract_features(S2, stub=False)
        R["rc"] = rc
    elif STAGE == "stage2":
        rc = s2_run.infer(S2, stub=False)
        R["rc"] = rc
    else:
        raise SystemExit("unknown stage " + STAGE)
    R["status"] = "ok"
except SystemExit as e:
    R["status"] = "systemexit"
    R["error_type"] = "SystemExit"
    R["error"] = str(e)
    R["rc"] = e.code if isinstance(e.code, int) else 2
except BaseException as e:
    R["status"] = "raised"
    R["error_type"] = type(e).__name__
    R["error"] = str(e)
    if isinstance(e, TypeError) and "positional argument" in str(e):
        R["sidecar_probe_arity_error"] = True
save()
'''


def write_driver(root: Path) -> Path:
    p = root / "gate_driver.py"
    p.write_text(DRIVER)
    return p


def drive(root: Path, driver: Path, wd: Path, stage: str, *, hf: Path,
          extra: dict, timeout=3600, stim_root: Path | None = None) -> tuple[Run, dict]:
    res = wd / f"driver_{stage}_{abs(hash(json.dumps(extra, sort_keys=True))) % 99999}.json"
    env = base_env(root, wd, hf=hf, probe_out=None, stim_root=stim_root, extra={
        "S2_GATE_REPO": str(REPO),
        "S2_GATE_METADIR": str(root / "meta"),
        "S2_GATE_PROC_JSON": str(root / "processor_config.json"),
        "S2_GATE_RESULT": str(res),
        **extra,
    })
    r = run([sys.executable, driver, stage], cwd=wd, env=env, timeout=timeout)
    try:
        data = json.loads(res.read_text())
    except Exception:
        data = {"status": "no-result-file", "error": r.tail(3)}
    return r, data


def phase_infer(root: Path, pre: dict) -> None:
    print("\nI  real Stage 1 + Stage 2 orchestration (tribev2 leaves replaced)")
    if not pre.get("hf_ok"):
        check("I", "a weight-identity fixture exists", False,
              "the preflight phase did not build one")
        return
    (root / "processor_config.json").write_text(
        json.dumps(PROCESSOR_CONFIG, sort_keys=True))
    build_metadir(root)
    driver = write_driver(root)
    hf = pre["hf_ok"]
    atlas_src = pre.get("atlas_cache")

    # One stimulus root shared by every scenario. exca keys each item on the
    # literal filepath string, so two roots = two disjoint key sets; that is
    # its own finding below, not something to smear across every other item.
    stim = make_workdir(root, "_shared_stimulus") / "data"

    def scenario(name, *, tamper=False):
        wd = make_workdir(root, name, tamper_video=tamper)
        if atlas_src and Path(atlas_src).is_file():
            shutil.copy2(atlas_src, wd / "data" / "s2_parcels.npz")
        return wd

    def go(wd, stage, extra, *, timeout=1800, stim_root=stim):
        return drive(root, driver, wd, stage, hf=hf, extra=extra,
                     timeout=timeout, stim_root=stim_root)

    # ---------------------------------------------------------------- I1 / I2
    wd = scenario("infer_stage1_asis")
    r, d = go(wd, "stage1", {"S2_ARTIFACT_ROOT": str(wd / "art")})
    defect("I", "B1 STILL OPEN: Stage 1 aborts before extracting -- "
               "verify_local_weights is given the hub root, not the repo cache dir",
          d.get("status") == "raised" and d.get("error_type") == "WeightFileMissing",
          f"scripts/s2_run.py:553 -> {d.get('error_type')}: "
          f"{str(d.get('error'))[:220]}")

    wd = scenario("infer_stage1_fixed")
    art = wd / "art"
    r, d1 = go(wd, "stage1", {"S2_ARTIFACT_ROOT": str(art), "S2_GATE_FIX_B1": "1"})
    uid_dirs = sorted(p for p in art.glob("*") if p.is_dir()) if art.is_dir() else []
    art_dir = uid_dirs[0] if uid_dirs else None
    man = {}
    if art_dir:
        try:
            man = json.loads((art_dir / "S2_FEATURES.json").read_text())
        except Exception:
            man = {}
    kinds = ledger_kinds(wd)
    check("I", "with that one argument corrected Stage 1 completes and finalizes "
               "a digest-verified artifact",
          d1.get("status") == "ok" and bool(man)
          and (art_dir / "S2_FEATURES.COMPLETE").is_file()
          and man.get("n_items") == d1.get("n_expected_items")
          and "artifact_finalized" in kinds,
          f"status={d1.get('status')} n_items={man.get('n_items')} "
          f"expected={d1.get('n_expected_items')} COMPLETE="
          f"{bool(art_dir and (art_dir / 'COMPLETE').is_file())} ledger={kinds}")

    calls = d1.get("load_model_calls") or []
    extract_call = next((c for c in calls if c["cache_folder"]), None)
    cu = (extract_call or {}).get("config_update", {})
    check("I", "the config Stage 1 actually handed load_model carries "
               "data.num_workers=0, keep_in_ram off and the weight-bound uid",
          bool(extract_call) and cu.get("data.num_workers") == 0
          and all(cu.get(f"data.{m}_feature.infra.keep_in_ram") is False
                  for m in ("video", "audio", "text"))
          and all(cu.get(f"data.{m}_feature.infra.version")
                  for m in ("video", "audio", "text"))
          and not any("mode" in k for k in cu),
          f"cache_folder={(extract_call or {}).get('cache_folder')} "
          f"num_workers={cu.get('data.num_workers')} "
          f"version={str(cu.get('data.video_feature.infra.version'))[:24]} "
          f"read-only keys present={[k for k in cu if k.endswith('.mode')]}")

    check("I", "the artifact digests were computed from bytes READ BACK out of "
               "a real exca cache",
          bool(man.get("items")) and len(man.get("items") or {}) == man.get("n_items")
          and all(len(v) == 64 for v in (man.get("items") or {}).values()),
          f"{len(man.get('items') or {})} sha256 digests recorded; sidecars="
          f"{sorted((man.get('exca_sidecars') or {}))}")

    # ---------------------------------------------------------------- I4 publish
    wd_pub = scenario("infer_stage1_publish")
    art_pub = wd_pub / "art"
    dur = wd_pub / "durable"
    r, dpub = go(wd_pub, "stage1", {"S2_ARTIFACT_ROOT": str(art_pub),
                                    "S2_GATE_FIX_B1": "1",
                                    "S2_DURABLE_ROOT": str(dur)})
    published = list(dur.glob("*")) if dur.is_dir() else []
    defect("I", "NEW DEFECT: Stage 1's publish call crashes -- sidecar_probe is a "
               "zero-argument lambda and durable_store calls it with the directory",
          dpub.get("status") == "raised" and dpub.get("sidecar_probe_arity_error"),
          f"scripts/s2_run.py:596 -> {dpub.get('error_type')}: "
          f"{str(dpub.get('error'))[:180]}; durable root now holds "
          f"{[p.name for p in published]}")

    if not art_dir:
        check("I", "Stage 2 can be exercised", False,
              "Stage 1 produced no artifact, so Stage 2 has nothing to consume")
        return

    # ---------------------------------------------------------------- I5 as-wired
    wd2 = scenario("infer_stage2_asis")
    art2 = wd2 / "art"
    art2.mkdir(parents=True, exist_ok=True)
    shutil.copytree(art_dir, art2 / art_dir.name)
    r, d2 = go(wd2, "stage2", {"S2_ARTIFACT_ROOT": str(art2), "S2_GATE_FIX_B1": "1"})
    c2 = (d2.get("load_model_calls") or [])
    consume = next((c for c in c2
                    if any(k.endswith(".mode") for k in c["config_update"])), None)
    defect("I", "NEW DEFECT: Stage 2 loads the model with cache_folder=None, so the "
               "extractor has no feature cache at all and Stage 2 cannot complete",
          d2.get("status") == "raised"
          and d2.get("error_type") in ("TypeError", "ArtifactCorrupt", "ValueError")
          and bool(consume) and consume["cache_folder"] is None,
          f"scripts/s2_run.py:768 cache_folder={consume and consume['cache_folder']} "
          f"-> {d2.get('error_type')}: {str(d2.get('error'))[:220]}. "
          f"exca/base.py:338 returns uid_folder()=None for folder=None, so "
          f"scripts/s2_run.py:795 sidecar_digests(None) raises before anything is "
          f"verified; past that, cache_dict raises ValueError (phase 0 measured it)")
    defect("I", "and the failure carries NO remedy text -- it is an untyped "
               "TypeError, not one of the four typed artifact errors",
          d2.get("error_type") == "TypeError"
          and "--extract-features" not in str(d2.get("error", "")),
          f"the operator sees: {str(d2.get('error'))[:200]}")

    # ------------------------------------------------------------ B6, isolated
    wd_b6 = scenario("infer_stage2_persist")
    art_b6 = wd_b6 / "art"
    art_b6.mkdir(parents=True, exist_ok=True)
    shutil.copytree(art_dir, art_b6 / art_dir.name)
    r, db6 = go(wd_b6, "stage2",
                {"S2_ARTIFACT_ROOT": str(art_b6), "S2_GATE_FIX_B1": "1",
                 "S2_GATE_FORCED_CACHE": str(art_b6 / art_dir.name / "cache")},
                timeout=3600)
    stray = sorted(pp.name for pp in (art_b6 / art_dir.name).glob("preds*"))
    kb6 = ledger_kinds(wd_b6)
    defect("I", "B6 NOT CLOSED: the persistence step itself raises AFTER inference "
               "-- np.savez appends .npz to a name that already ends .npz.tmp",
          db6.get("error_type") == "FileNotFoundError"
          and db6.get("predict_calls") == 1
          and stray == ["preds.npz.tmp.npz"],
          f"scripts/s2_run.py:739-741 -> {db6.get('error_type')}: "
          f"{str(db6.get('error'))[:160]}; the artifact dir now holds {stray} and "
          f"no preds.npz; ledger stops at {kb6}")

    note("every Stage 2 item below runs with the B1 and B6 corrections applied "
         "in-process (recorded in the driver, never hidden). They prove the "
         "downstream guards fire ONCE those two lines are fixed; they are not "
         "evidence that the shipped path works.")

    # ---------------------------------------------------------------- I6 corrected
    wd3 = scenario("infer_stage2_fixed")
    art3 = wd3 / "art"
    art3.mkdir(parents=True, exist_ok=True)
    shutil.copytree(art_dir, art3 / art_dir.name)
    forced3 = str(art3 / art_dir.name / "cache")
    r, d3 = go(wd3, "stage2", {"S2_ARTIFACT_ROOT": str(art3), "S2_GATE_FIX_B1": "1",
                               "S2_GATE_FIX_B6": "1",
                               "S2_GATE_FORCED_CACHE": forced3}, timeout=3600)
    k3 = ledger_kinds(wd3)
    rep3 = wd3 / "data" / "s2_report.json"
    check("I", "with the cache folder supplied, the REAL Stage 2 runs end to end: "
               "verify -> contract -> predict -> persist -> analyse -> report",
          d3.get("status") == "ok" and "artifact_verified" in k3
          and "infer_completed" in k3 and rep3.is_file(),
          f"status={d3.get('status')} ledger={k3} report={rep3.is_file()} "
          f"in {r.secs:.0f}s :: {r.tail(1)}")

    c3 = (d3.get("load_model_calls") or [])
    consume3 = next((c for c in c3
                     if any(k.endswith(".mode") for k in c["config_update"])), None)
    cu3 = (consume3 or consume or {}).get("config_update", {})
    check("I", "the consume-stage config really carries read-only on all three "
               "extractors",
          all(cu3.get(f"data.{m}_feature.infra.mode") == "read-only"
              for m in ("video", "audio", "text"))
          and all(cu3.get(f"data.{m}_feature.infra.forbid_single_item_computation")
                  is True for m in ("video", "audio", "text"))
          and cu3.get("data.num_workers") == 0,
          f"read from the call the corrected Stage 2 actually made -- modes="
          f"{[cu3.get(f'data.{m}_feature.infra.mode') for m in ('video','audio','text')]}"
          f" forbid_single="
          f"{[cu3.get(f'data.{m}_feature.infra.forbid_single_item_computation') for m in ('video','audio','text')]}")

    # ---- B6: predictions persisted
    npz = list((art3 / art_dir.name).glob("preds.npz"))
    ok_npz = False
    detail = "no preds.npz written"
    if npz:
        try:
            import numpy as np
            z = np.load(npz[0])
            got = float(z["preds"][:, 0].sum())
            ok_npz = abs(got - float(d3.get("preds_checksum", 1e30))) < 1e-3
            detail = (f"{npz[0].name} {z['preds'].shape} checksum {got:.4f} vs "
                      f"predicted {d3.get('preds_checksum')}")
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
    check("I", "B6: the predictions are on disk before analyse() can raise",
          ok_npz, detail)

    # ---------------------------------------------------------------- I9 / I10
    wd4 = scenario("infer_stage2_zerofill")
    art4 = wd4 / "art"
    art4.mkdir(parents=True, exist_ok=True)
    shutil.copytree(art_dir, art4 / art_dir.name)
    r, d4 = go(wd4, "stage2", {"S2_ARTIFACT_ROOT": str(art4), "S2_GATE_FIX_B1": "1",
                               "S2_GATE_FORCED_CACHE": str(art4 / art_dir.name / "cache"),
                               "S2_GATE_ZERO_TIMESTEP": "1"})
    check("I", "B4 forward: one zero-filled video timestep stops Stage 2 BEFORE "
               "predict",
          d4.get("error_type") == "ModalityContractViolation"
          and d4.get("predict_calls") == 0,
          f"{d4.get('error_type')}: {str(d4.get('error'))[:180]} "
          f"(predict called {d4.get('predict_calls')} times)")

    wd5 = scenario("infer_stage2_audio")
    art5 = wd5 / "art"
    art5.mkdir(parents=True, exist_ok=True)
    shutil.copytree(art_dir, art5 / art_dir.name)
    r, d5 = go(wd5, "stage2", {"S2_ARTIFACT_ROOT": str(art5), "S2_GATE_FIX_B1": "1",
                               "S2_GATE_FORCED_CACHE": str(art5 / art_dir.name / "cache"),
                               "S2_GATE_EXTRA_AUDIO": "1"})
    check("I", "B4 backward: a modality the frozen design declared absent stops "
               "Stage 2",
          d5.get("error_type") == "ModalityContractViolation"
          and d5.get("predict_calls") == 0,
          f"{d5.get('error_type')}: {str(d5.get('error'))[:180]}")

    # ---------------------------------------------------------------- I11 counter
    wd6 = scenario("infer_stage2_recompute")
    art6 = wd6 / "art"
    art6.mkdir(parents=True, exist_ok=True)
    shutil.copytree(art_dir, art6 / art_dir.name)
    r, d6 = go(wd6, "stage2", {"S2_ARTIFACT_ROOT": str(art6), "S2_GATE_FIX_B1": "1",
                               "S2_GATE_FORCED_CACHE": str(art6 / art_dir.name / "cache"),
                               "S2_GATE_RECOMPUTE": "1",
                               "S2_GATE_RECOMPUTE_DIR": str(wd6 / "recompute")})
    check("I", "a Stage 2 that really re-enters exca's compute funnel is caught "
               "and refused",
          d6.get("error_type") == "ConsumeStageRecomputed",
          f"{d6.get('error_type')}: {str(d6.get('error'))[:200]}")

    # ---------------------------------------------------------------- I12 / I13
    wd7 = scenario("infer_stage2_durable_only")
    art7 = wd7 / "art"
    art7.mkdir(parents=True, exist_ok=True)      # deliberately empty
    dur7 = wd7 / "durable"
    dur7.mkdir(parents=True, exist_ok=True)
    shutil.copytree(art_dir, dur7 / art_dir.name)
    r, d7 = go(wd7, "stage2", {"S2_ARTIFACT_ROOT": str(art7), "S2_GATE_FIX_B1": "1",
                               "S2_GATE_FIX_B6": "1",
                               "S2_DURABLE_ROOT": str(dur7),
                               "S2_GATE_FORCED_CACHE": str(dur7 / art_dir.name / "cache")},
                   timeout=3600)
    k7 = ledger_kinds(wd7)
    check("I", "B2: an artifact that exists ONLY in the durable store is found and "
               "verified by the real search",
          d7.get("status") == "ok" and "artifact_verified" in k7,
          f"local artifact root was empty; S2_DURABLE_ROOT={dur7.name} -> "
          f"status={d7.get('status')} ledger={k7} :: {r.tail(1)}")

    wd8 = scenario("infer_stage2_no_artifact")
    art8 = wd8 / "art"
    art8.mkdir(parents=True, exist_ok=True)
    r, d8 = go(wd8, "stage2", {"S2_ARTIFACT_ROOT": str(art8), "S2_GATE_FIX_B1": "1",
                               "S2_GATE_FORCED_CACHE": str(wd8 / "nowhere" / "cache")})
    check("I", "no artifact anywhere -> a typed stop, and nothing is encoded",
          d8.get("error_type") == "ArtifactNotFound"
          and d8.get("predict_calls") == 0
          and "will NOT encode" in str(d8.get("error", "")),
          f"{d8.get('error_type')}: {str(d8.get('error'))[:200]} "
          f"(predict called {d8.get('predict_calls')} times)")

    # ---------------------------------------------------------------- corrupt payload
    wd9 = scenario("infer_stage2_corrupt")
    art9 = wd9 / "art"
    art9.mkdir(parents=True, exist_ok=True)
    shutil.copytree(art_dir, art9 / art_dir.name)
    victims = sorted((art9 / art_dir.name / "cache").rglob("*.npy"))
    if not victims:
        victims = [p for p in (art9 / art_dir.name / "cache").rglob("*")
                   if p.is_file() and p.suffix not in (".yaml", ".jsonl", ".json")]
    if victims:
        v = victims[0]
        raw = bytearray(v.read_bytes())
        raw[-1] ^= 0xFF
        v.write_bytes(bytes(raw))
    r, d9 = go(wd9, "stage2", {"S2_ARTIFACT_ROOT": str(art9), "S2_GATE_FIX_B1": "1",
                               "S2_GATE_FORCED_CACHE": str(art9 / art_dir.name / "cache")})
    check("I", "a single flipped byte inside the feature payload is caught before "
               "the model is used",
          bool(victims) and d9.get("error_type") in ("ArtifactCorrupt",
                                                     "ArtifactNotFound")
          and d9.get("predict_calls") == 0,
          f"flipped 1 byte in {victims[0].name if victims else 'nothing'} -> "
          f"{d9.get('error_type')} (predict called {d9.get('predict_calls')} times)")

    # ------------------------------------------------------------ B7, isolated
    # Same bytes, same digest, a different mount point. exca keys each item on
    # the LITERAL filepath (neuralset/extractors/video.py:247), so the key set
    # moves with the path. STIMULUS_ROOT.resolve() makes it absolute; absolute
    # is not portable. durable_store.stage_stimulus exists for exactly this and
    # s2_run never calls it.
    alt_stim = make_workdir(root, "_alt_stimulus") / "data"
    wd10 = scenario("infer_stage2_moved_stimulus")
    art10 = wd10 / "art"
    art10.mkdir(parents=True, exist_ok=True)
    shutil.copytree(art_dir, art10 / art_dir.name)
    r, d10 = go(wd10, "stage2",
                {"S2_ARTIFACT_ROOT": str(art10), "S2_GATE_FIX_B1": "1",
                 "S2_GATE_FORCED_CACHE": str(art10 / art_dir.name / "cache")},
                stim_root=alt_stim)
    moved_msg = str(d10.get("error", ""))
    defect("I", "B7 NOT CLOSED: the same stimulus reached through a second path "
               "invalidates the whole artifact and the remedy is a re-encode",
          d10.get("error_type") in ("ArtifactNotFound", "ArtifactIncomplete")
          and "--extract-features" in moved_msg,
          f"byte-identical stimulus under {alt_stim.parent.name}/data instead of "
          f"{stim.parent.name}/data -> {d10.get('error_type')}: "
          f"{moved_msg[:200]}")
    note("s2_run.py never calls durable_store.stage_stimulus / "
         "staged_stimulus_path, so exca's item keys stay path-addressed. "
         "STIMULUS_ROOT.resolve() (scripts/s2_run.py:65) makes them absolute, "
         "which does not survive a Kaggle Stage 1 in /kaggle/working and a "
         "Stage 2 reading /kaggle/input.")


# --------------------------------------------------------------------------- #
# phase T -- the suite
# --------------------------------------------------------------------------- #

_COUNT = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed)")


def phase_tests() -> None:
    print("\nT  regression suite")
    env = dict(os.environ)
    # A skipped cache-contract test protects nothing, so the suite is run with
    # the flag that turns "exca is missing" into a session failure instead of a
    # green run with holes in it. The previous gate instead OR'd the skip away.
    env["S2_REQUIRE_EXCA"] = "1"
    r = run([sys.executable, "-m", "pytest", "tests/", "-q", "-rs"],
            cwd=REPO, env=env, timeout=5400)
    summary = ""
    for line in reversed(r.out.strip().splitlines()):
        if _COUNT.search(line):
            summary = line.strip()
            break
    counts = {k if k != "errors" else "error": int(n)
              for n, k in _COUNT.findall(summary)}
    check("T", "pytest tests/ -q exits 0 with S2_REQUIRE_EXCA=1",
          r.rc == 0, f"rc={r.rc} in {r.secs:.0f}s :: {summary or r.tail(1)}")
    check("T", "no test failed and none errored",
          counts.get("failed", 0) == 0 and counts.get("error", 0) == 0
          and counts.get("passed", 0) > 0,
          f"passed={counts.get('passed', 0)} failed={counts.get('failed', 0)} "
          f"error={counts.get('error', 0)} skipped={counts.get('skipped', 0)}")
    skip_lines = [l for l in r.out.splitlines() if l.startswith("SKIPPED")]
    check("T", "no test skipped for want of exca",
          not any("exca" in l.lower() for l in skip_lines),
          f"{len(skip_lines)} skip reason(s) reported; exca-related: "
          f"{[l for l in skip_lines if 'exca' in l.lower()] or 'none'}")


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--workdir", default=None,
                    help="where the gate builds its sandbox (default: a temp dir)")
    args = ap.parse_args()
    modes = {"preflight": args.preflight, "stub": args.stub, "infer": args.infer}
    if not any(modes.values()):
        modes = {k: True for k in modes}
        run_tests = not args.no_tests
    else:
        run_tests = not args.no_tests and all(modes.values())

    root = Path(args.workdir) if args.workdir else Path(
        os.environ.get("S2_GATE_WORKDIR")
        or (Path(os.environ.get("TMPDIR", "/tmp")) / f"s2gate-{os.getpid()}"))
    root.mkdir(parents=True, exist_ok=True)
    write_probe(root)

    print("=== S2 pre-GPU gate ===")
    print("Every item below is the outcome of running something. The gate it")
    print("replaces asked whether names existed; that gate would have said GO.")
    print(f"sandbox: {root}\n")

    t0 = time.time()
    phase_host(root)
    phase_ground_truth(root)
    pre = {}
    if modes["preflight"] or modes["stub"] or modes["infer"]:
        pre = phase_preflight(root)
    if modes["stub"]:
        phase_stub(root, pre)
    if modes["infer"]:
        phase_infer(root, pre)
    if run_tests:
        phase_tests()

    bad = [i for i in ITEMS if not i.ok]
    repro = [d for d in DEFECTS if d.ok]
    print(f"\n{len(ITEMS) - len(bad)}/{len(ITEMS)} executed assertions hold; "
          f"{len(repro)}/{len(DEFECTS)} defect demonstrations reproduce "
          f"({time.time() - t0:.0f}s, sandbox {root})")
    for n in NOTES:
        print(f"  note: {n}")
    if bad or repro:
        print("\nNO-GO.")
        if repro:
            print("\nDEFECTS REPRODUCED IN THE REAL PATH -- each of these was "
                  "demonstrated by running it:")
            for i in repro:
                print(f"  [{i.phase}] {i.name}")
                print(f"        {i.evidence[:400]}")
        if bad:
            print("\nASSERTIONS THAT DID NOT HOLD:")
            for i in bad:
                print(f"  [{i.phase}] {i.name}")
                print(f"        {i.evidence[:400]}")
        print("\nDo not upload to Kaggle and do not start S2.")
        return 1
    full = all(modes.values()) and run_tests
    if not full:
        # A partial run can never be a verdict: the whole point of this file is
        # that a green subset is what preceded the incident.
        ran = [k for k, v in modes.items() if v] + (["tests"] if run_tests else [])
        print(f"\nPARTIAL RUN ({', '.join(ran)}) -- no verdict. "
              f"Re-run with no mode flags for a GO/NO-GO.")
        return 3
    print("\nGO for the two-stage S2 run.")
    print("  stage 1: python3 scripts/s2_run.py --extract-features")
    print("  stage 2: python3 scripts/s2_run.py --infer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
