"""Run every failure injection and report: failure -> expected -> observed.

The 2026-08-25 incident passed a 57-item go/no-go and then wasted 4h45m. The gate
checked that things were CONFIGURED; it never checked that the pipeline could carry
an expensive computation through to a scientific result without discarding or
repeating it. This harness asks that question by breaking things on purpose.

    python3 scripts/s2_failure_injection.py

Exit 0 only if every injection produced its expected behaviour.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tribe_tools.feature_artifact import (  # noqa: E402
    ArtifactCorrupt, ArtifactIncomplete, ArtifactMissing, ArtifactStale, COMPLETE,
    MANIFEST, begin_stage1, verify_artifact, write_artifact,
)
from tribe_tools.ledger import Event, Ledger, resume_state  # noqa: E402
from tribe_tools.s2_pipeline import (  # noqa: E402
    ConsumeStageRecomputed, ExtractionIncomplete, ModalityContractViolation,
    Stage1Deps, Stage2Deps, assert_modality_contract, stage1_extract, stage2_infer,
)

UIDS = ["stim_0.00_60.00", "stim_60.00_60.00", "stim_120.00_30.00"]
IDENTITY = {"stimulus_sha256": "a" * 64, "weights_sha256": "b" * 64,
            "preprocessing_sha256": "c" * 64, "model_revision": "f894e783"}

RESULTS: list[tuple[str, str, str, bool]] = []


class Counter:
    def __init__(self, items=0):
        self.items, self.calls, self.active = items, 0, True
    def __enter__(self): return self
    def __exit__(self, *a): return None


class World:
    def __init__(self, uids=UIDS):
        self.store = {u: np.full((4, 3), i + 1.0, dtype=np.float32)
                      for i, u in enumerate(uids)}
        self.encodes = 0
        self.model_loads = 0
    def load_model(self, **kw):
        self.model_loads += 1; return object()
    def build_events(self, m): return list(self.store)
    def extract(self, m, e):
        self.encodes += len(self.store); return list(self.store)
    def read_item(self, uid):
        v = self.store[uid]
        if v is None:
            raise ValueError("cannot mmap an empty file")
        return v
    def predict(self, m, e):
        return np.zeros((10, 20484), dtype=np.float32), list(range(10))
    def analyse(self, p, s): return {"results": {"FFA": {}}}


def s1(w): return Stage1Deps(w.load_model, w.build_events, w.extract, w.read_item, lambda: {})
def s2(w): return Stage2Deps(w.load_model, w.build_events, w.predict, w.read_item, w.analyse)


def _finalized(tmp, w=None, identity=None, uids=UIDS):
    w = w or World(uids)
    led = Ledger(tmp / "l.jsonl")
    stage1_extract(None, identity or IDENTITY, tmp / "art", uids, s1(w), led,
                   counter_factory=lambda: Counter(len(uids)))
    return w, led


def check(name, expected, fn):
    """fn() -> observed string. Passes when observed matches expected."""
    tmp = Path(tempfile.mkdtemp())
    try:
        observed = fn(tmp)
    except Exception as exc:                    # an unexpected crash is a failure
        observed = f"UNEXPECTED {type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = observed == expected
    RESULTS.append((name, expected, observed, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         expected: {expected}")
    print(f"         observed: {observed}")
    return ok


def _expect(exc_type, fn, *a, **kw):
    try:
        fn(*a, **kw)
        return "no error raised"
    except exc_type as e:
        return type(e).__name__
    except Exception as e:
        return f"{type(e).__name__} (wrong type): {str(e)[:60]}"


# ------------------------------------------------------------------ injections

def fi_corrupt_tensor(tmp):
    w, led = _finalized(tmp)
    w.store[UIDS[0]] = np.full((4, 3), 999.0, dtype=np.float32)
    return _expect(ArtifactCorrupt, stage2_infer, None, IDENTITY, tmp / "art", UIDS,
                   s2(w), led, counter_factory=lambda: Counter(0))


def fi_truncate_artifact(tmp):
    w, led = _finalized(tmp)
    w.store[UIDS[1]] = None                      # unreadable payload
    return _expect(ArtifactCorrupt, stage2_infer, None, IDENTITY, tmp / "art", UIDS,
                   s2(w), led, counter_factory=lambda: Counter(0))


def fi_alter_metadata(tmp):
    import json
    w, led = _finalized(tmp)
    man = json.loads((tmp / "art" / MANIFEST).read_text())
    man["items"][UIDS[0]] = "0" * 64
    (tmp / "art" / MANIFEST).write_text(json.dumps(man))
    return _expect(ArtifactCorrupt, stage2_infer, None, IDENTITY, tmp / "art", UIDS,
                   s2(w), led, counter_factory=lambda: Counter(0))


def fi_change_stimulus_sha(tmp):
    w, led = _finalized(tmp)
    ident = dict(IDENTITY, stimulus_sha256="9" * 64)
    return _expect(ArtifactStale, stage2_infer, None, ident, tmp / "art", UIDS,
                   s2(w), led, counter_factory=lambda: Counter(0))


def fi_change_model_revision(tmp):
    w, led = _finalized(tmp)
    ident = dict(IDENTITY, model_revision="deadbeef")
    return _expect(ArtifactStale, stage2_infer, None, ident, tmp / "art", UIDS,
                   s2(w), led, counter_factory=lambda: Counter(0))


def fi_change_weight_sha(tmp):
    w, led = _finalized(tmp)
    ident = dict(IDENTITY, weights_sha256="e" * 64)
    return _expect(ArtifactStale, stage2_infer, None, ident, tmp / "art", UIDS,
                   s2(w), led, counter_factory=lambda: Counter(0))


def fi_change_preprocessing(tmp):
    w, led = _finalized(tmp)
    ident = dict(IDENTITY, preprocessing_sha256="d" * 64)
    return _expect(ArtifactStale, stage2_infer, None, ident, tmp / "art", UIDS,
                   s2(w), led, counter_factory=lambda: Counter(0))


def fi_remove_completion_marker(tmp):
    w, led = _finalized(tmp)
    (tmp / "art" / COMPLETE).unlink()
    return _expect(ArtifactIncomplete, stage2_infer, None, IDENTITY, tmp / "art",
                   UIDS, s2(w), led, counter_factory=lambda: Counter(0))


def fi_missing_artifact(tmp):
    w, led = _finalized(tmp)
    return _expect(ArtifactMissing, stage2_infer, None, IDENTITY, tmp / "gone",
                   UIDS, s2(w), led, counter_factory=lambda: Counter(0))


def fi_partial_extraction(tmp):
    w = World(UIDS[:2])
    led = Ledger(tmp / "l.jsonl")
    out = _expect(ExtractionIncomplete, stage1_extract, None, IDENTITY, tmp / "art",
                  UIDS, s1(w), led, counter_factory=lambda: Counter(2))
    if (tmp / "art" / COMPLETE).exists():
        return "ExtractionIncomplete BUT the artifact was certified COMPLETE"
    return out


def fi_force_extraction_during_consume(tmp):
    w, led = _finalized(tmp)
    return _expect(ConsumeStageRecomputed, stage2_infer, None, IDENTITY, tmp / "art",
                   UIDS, s2(w), led, counter_factory=lambda: Counter(5))


def fi_worker_attempts_cuda(tmp):
    """A real child process, through the real guard."""
    code = (
        f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r});\n"
        "import os, multiprocessing as mp\n"
        "from tribe_tools import cuda_guard as g\n"
        "def child(q):\n"
        "    from tribe_tools import cuda_guard as gg\n"
        "    try:\n"
        "        gg.check('feature extraction'); q.put('ALLOWED')\n"
        "    except gg.ChildGPUViolation: q.put('ChildGPUViolation')\n"
        "if __name__ == '__main__':\n"
        "    g.arm()\n"
        "    ctx = mp.get_context('fork'); q = ctx.Queue()\n"
        "    p = ctx.Process(target=child, args=(q,)); p.start(); p.join(30)\n"
        "    print(q.get(timeout=10))\n"
    )
    f = tmp / "probe.py"; f.write_text(code)
    r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, timeout=90)
    return (r.stdout.strip().splitlines() or ["no output"])[-1]


def fi_extractor_returns_failure(tmp):
    w = World()
    led = Ledger(tmp / "l.jsonl")

    def boom(model, events):
        raise RuntimeError("extractor failed")

    deps = Stage1Deps(w.load_model, w.build_events, boom, w.read_item, lambda: {})
    out = _expect(RuntimeError, stage1_extract, None, IDENTITY, tmp / "art", UIDS,
                  deps, led, counter_factory=lambda: Counter(0))
    if (tmp / "art" / COMPLETE).exists():
        return "RuntimeError BUT the artifact was certified COMPLETE"
    if [r["event"] for r in led.records()][-1] != "aborted":
        return "RuntimeError BUT the ledger did not record an abort"
    return out


def fi_zero_filled_modality(tmp):
    return _expect(ModalityContractViolation, assert_modality_contract,
                   {"video": np.zeros((5, 1408), dtype=np.float32)}, ["video"])


def fi_missing_extractor(tmp):
    return _expect(ModalityContractViolation, assert_modality_contract,
                   {"audio": np.ones((5, 4))}, ["video"])


def fi_kill_between_stages(tmp):
    """SIGKILL after ARTIFACT_FINALIZED, then ask the ledger what to do."""
    w, led = _finalized(tmp)
    code = (
        f"import sys, os, signal; sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    f = tmp / "die.py"; f.write_text(code)
    subprocess.run([sys.executable, str(f)], capture_output=True)
    st = resume_state(led.path, IDENTITY)
    before = w.encodes
    stage2_infer(None, IDENTITY, tmp / "art", UIDS, s2(w), led,
                 counter_factory=lambda: Counter(0))
    if w.encodes != before:
        return f"{st.action} BUT the restart re-encoded"
    return f"{st.action}, 0 re-encodes"


def main() -> int:
    print("=== S2 failure injection ===\n")
    ok = True
    ok &= check("1  corrupt tensor", "ArtifactCorrupt", fi_corrupt_tensor)
    ok &= check("2  truncate artifact payload", "ArtifactCorrupt", fi_truncate_artifact)
    ok &= check("3  alter manifest metadata", "ArtifactCorrupt", fi_alter_metadata)
    ok &= check("4  change stimulus sha256", "ArtifactStale", fi_change_stimulus_sha)
    ok &= check("5  change model revision", "ArtifactStale", fi_change_model_revision)
    ok &= check("6  change V-JEPA weight sha256", "ArtifactStale", fi_change_weight_sha)
    ok &= check("7  change preprocessing", "ArtifactStale", fi_change_preprocessing)
    ok &= check("8  remove completion marker", "ArtifactIncomplete", fi_remove_completion_marker)
    ok &= check("9  artifact absent entirely", "ArtifactMissing", fi_missing_artifact)
    ok &= check("10 partial extraction", "ExtractionIncomplete", fi_partial_extraction)
    ok &= check("11 force extraction during consume", "ConsumeStageRecomputed",
                fi_force_extraction_during_consume)
    ok &= check("12 worker attempts CUDA", "ChildGPUViolation", fi_worker_attempts_cuda)
    ok &= check("13 extractor returns failure", "RuntimeError", fi_extractor_returns_failure)
    ok &= check("14 required modality zero-filled", "ModalityContractViolation",
                fi_zero_filled_modality)
    ok &= check("15 required extractor missing", "ModalityContractViolation",
                fi_missing_extractor)
    ok &= check("16 kill between extraction and inference",
                "verify_then_infer, 0 re-encodes", fi_kill_between_stages)

    n = len(RESULTS); good = sum(1 for *_, o in RESULTS if o)
    print(f"\n{good}/{n} injections behaved as specified")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
