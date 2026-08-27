"""The two-stage boundary, tested as control flow.

Every test here answers a question the 2026-08-25 incident could not:
does the consume stage ever reach the extractor, and is an unusable artifact
refused BEFORE anything expensive happens?

No GPU, no tribev2, no network. The expensive operations are injected, and the
fakes count their own invocations -- so "the extractor was never called" is a
measured integer, not an inference from a config value.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from tribe_tools.feature_artifact import (
    ArtifactCorrupt, ArtifactIncomplete, ArtifactMissing, ArtifactStale, COMPLETE,
)
from tribe_tools.ledger import Event, Ledger, resume_state
from tribe_tools.s2_pipeline import (
    ConsumeStageRecomputed, ExtractionIncomplete, ModalityContractViolation,
    Stage1Deps, Stage2Deps, assert_modality_contract, stage1_extract, stage2_infer,
)

UIDS = ["stim_0.00_60.00", "stim_60.00_60.00", "stim_120.00_30.00"]
IDENTITY = {"stimulus_sha256": "a" * 64, "weights_sha256": "b" * 64,
            "preprocessing_sha256": "c" * 64}


class FakeCounter:
    """Stands in for EncodeCounter. `items` and `active` are set by the test."""
    def __init__(self, items=0, active=True):
        self.items, self.calls, self.active = items, 0, active
    def __enter__(self): return self
    def __exit__(self, *a): return None


class World:
    """A fake extractor + cache. Counts every expensive call it is asked to make."""
    def __init__(self, tmp_path, uids=UIDS):
        self.store = {u: np.full((4, 3), i + 1.0, dtype=np.float32)
                      for i, u in enumerate(uids)}
        self.encode_calls = 0
        self.model_loads = 0
        self.tmp = tmp_path
        self.persisted = None
        rng = np.random.default_rng(0)
        self.modalities = rng.normal(size=(6, 1408)).astype(np.float32)

    def load_model(self, **kw):
        self.model_loads += 1
        return object()

    def build_events(self, model):
        return list(self.store)

    def extract(self, model, events):
        self.encode_calls += len(self.store)
        return list(self.store)

    def read_item(self, uid):
        if uid not in self.store:
            raise KeyError(uid)
        v = self.store[uid]
        if v is None:
            raise ValueError("cannot mmap an empty file")
        return v

    def predict(self, model, events):
        return np.zeros((10, 20484), dtype=np.float32), list(range(10))

    def analyse(self, preds, segments):
        return {"results": {"FFA": {}}}

    def probe_modalities(self, model, events):
        """What ACTUALLY reaches the brain model. Tests override this to simulate
        tribev2 deleting the video extractor and zero-filling the gap."""
        return {"video": self.modalities}

    def persist(self, preds, segments):
        self.persisted = (preds, segments)
        return self.tmp / "preds.npz"


def _s1(w):
    return Stage1Deps(load_model=w.load_model, build_events=w.build_events,
                      extract=w.extract, read_item=w.read_item, sidecars=lambda: {})


def _s2(w, **over):
    kw = dict(load_model=w.load_model, build_events=w.build_events,
              predict=w.predict, read_item=w.read_item, analyse=w.analyse,
              probe_modalities=w.probe_modalities, persist=w.persist,
              sidecar_probe=lambda: {})
    kw.update(over)
    return Stage2Deps(**kw)


REQ = ("video",)
ABS = ("audio", "text")


def _infer(w, identity, art, uids, led, items=0, active=True):
    return stage2_infer(None, identity, art, uids, _s2(w), led,
                        required_modalities=REQ, expected_absent=ABS,
                        counter_factory=lambda: FakeCounter(items, active))


# ------------------------------------------------------------------- stage 1

def test_stage1_extracts_finalizes_and_records(tmp_path):
    w = World(tmp_path)
    led = Ledger(tmp_path / "l.jsonl")
    man = stage1_extract(None, IDENTITY, tmp_path / "art", UIDS, _s1(w), led,
                         counter_factory=lambda: FakeCounter(3))
    assert man["n_items"] == 3
    assert (tmp_path / "art" / COMPLETE).is_file()
    events = [r["event"] for r in led.records()]
    assert events == ["extract_started", "extract_completed", "artifact_finalized"]


def test_stage1_refuses_to_finalize_an_incomplete_extraction(tmp_path):
    """A partial artifact must never be certified. This is the state the incident
    died in, and the state a naive resume would trust."""
    w = World(tmp_path, uids=UIDS[:2])
    led = Ledger(tmp_path / "l.jsonl")
    with pytest.raises(ExtractionIncomplete):
        stage1_extract(None, IDENTITY, tmp_path / "art", UIDS, _s1(w), led,
                       counter_factory=lambda: FakeCounter(2))
    assert not (tmp_path / "art" / COMPLETE).exists()
    assert [r["event"] for r in led.records()][-1] == "aborted"


def test_stage1_clears_a_stale_completion_marker_first(tmp_path):
    """An earlier session's certificate must not survive into a new extraction."""
    art = tmp_path / "art"
    art.mkdir()
    (art / COMPLETE).write_bytes(b"")
    w = World(tmp_path, uids=UIDS[:2])
    led = Ledger(tmp_path / "l.jsonl")
    with pytest.raises(ExtractionIncomplete):
        stage1_extract(None, IDENTITY, art, UIDS, _s1(w), led,
                       counter_factory=lambda: FakeCounter(2))
    assert not (art / COMPLETE).exists(), \
        "a failed extraction left the previous run's certificate in place"


def test_stage1_reads_the_bytes_back_rather_than_trusting_its_own_ram(tmp_path):
    """neuralset/extractors/base.py:201 discards the generator exca returns, so a
    Stage 1 that does not read back never touches what it wrote. Here the store is
    poisoned AFTER extraction: the read-back must surface it."""
    w = World(tmp_path)
    led = Ledger(tmp_path / "l.jsonl")
    real_extract = w.extract

    def extract_then_poison(model, events):
        out = real_extract(model, events)
        w.store[UIDS[1]] = None          # unreadable on read-back
        return out

    deps = Stage1Deps(load_model=w.load_model, build_events=w.build_events,
                      extract=extract_then_poison, read_item=w.read_item,
                      sidecars=lambda: {})
    with pytest.raises(ValueError):
        stage1_extract(None, IDENTITY, tmp_path / "art", UIDS, deps, led,
                       counter_factory=lambda: FakeCounter(3))
    assert not (tmp_path / "art" / COMPLETE).exists()


# ------------------------------------------------------------------- stage 2

def _finalize(tmp_path, w=None, identity=None, uids=UIDS):
    w = w or World(tmp_path, uids=uids)
    led = Ledger(tmp_path / "l.jsonl")
    stage1_extract(None, identity or IDENTITY, tmp_path / "art", uids, _s1(w), led,
                   counter_factory=lambda: FakeCounter(len(uids)))
    return w, led


def test_stage2_consumes_without_calling_the_extractor(tmp_path):
    """THE test. The extractor's own call counter must not move."""
    w, led = _finalize(tmp_path)
    before = w.encode_calls
    out = _infer(w, IDENTITY, tmp_path / "art", UIDS, led, items=0)
    assert out["results"]
    assert w.encode_calls == before, "Stage 2 invoked the extractor"
    evs = [r["event"] for r in led.records()]
    assert "artifact_verified" in evs and "infer_completed" in evs
    assert "extract_started" not in evs[3:], "Stage 2 started an extraction"


def test_stage2_raises_if_anything_was_encoded(tmp_path):
    """Last line of defence if exca's read-only mode is ever misconfigured."""
    w, led = _finalize(tmp_path)
    with pytest.raises(ConsumeStageRecomputed, match="read-only"):
        _infer(w, IDENTITY, tmp_path / "art", UIDS, led, items=7)


def test_stage2_verifies_before_loading_the_model(tmp_path):
    """An unusable artifact must cost seconds, not a 709 MB checkpoint download."""
    w, led = _finalize(tmp_path)
    (tmp_path / "art" / COMPLETE).unlink()
    loads = w.model_loads
    with pytest.raises(ArtifactIncomplete):
        _infer(w, IDENTITY, tmp_path / "art", UIDS, led, items=0)
    assert w.model_loads == loads, "the model was loaded before the artifact was checked"


@pytest.mark.parametrize("damage,exc", [
    ("missing", ArtifactMissing),
    ("no_complete", ArtifactIncomplete),
    ("corrupt", ArtifactCorrupt),
    ("stale", ArtifactStale),
])
def test_stage2_refuses_every_kind_of_bad_artifact(tmp_path, damage, exc):
    w, led = _finalize(tmp_path)
    art, ident = tmp_path / "art", IDENTITY
    if damage == "missing":
        art = tmp_path / "nowhere"
    elif damage == "no_complete":
        (art / COMPLETE).unlink()
    elif damage == "corrupt":
        w.store[UIDS[0]] = np.full((4, 3), 999.0, dtype=np.float32)
    elif damage == "stale":
        ident = dict(IDENTITY, stimulus_sha256="f" * 64)
    with pytest.raises(exc):
        _infer(w, ident, art, UIDS, led, items=0)
    assert [r["event"] for r in led.records()][-1] in ("artifact_rejected", "aborted")


def test_a_rejected_artifact_never_reaches_inference(tmp_path):
    w, led = _finalize(tmp_path)
    w.store[UIDS[0]] = np.full((4, 3), 999.0, dtype=np.float32)
    with pytest.raises(ArtifactCorrupt):
        _infer(w, IDENTITY, tmp_path / "art", UIDS, led, items=0)
    assert "infer_started" not in [r["event"] for r in led.records()]


# --------------------------------------------------- crash / restart behaviour

def test_a_crash_after_extraction_does_not_cause_another_extraction(tmp_path):
    """Restart proof, end to end through the ledger."""
    w, led = _finalize(tmp_path)
    assert resume_state(led.path, IDENTITY).action == "verify_then_infer"
    encodes_before = w.encode_calls
    _infer(w, IDENTITY, tmp_path / "art", UIDS, led, items=0)
    assert w.encode_calls == encodes_before, "the restart re-encoded"


def test_a_new_identity_forces_exactly_one_extraction(tmp_path):
    w, led = _finalize(tmp_path)
    other = dict(IDENTITY, stimulus_sha256="9" * 64)
    assert resume_state(led.path, other).action == "extract"
    before = w.encode_calls
    stage1_extract(None, other, tmp_path / "art2", UIDS, _s1(w), led,
                   counter_factory=lambda: FakeCounter(3))
    assert w.encode_calls == before + len(UIDS), "not exactly one extraction"


# ------------------------------------------------------------ modality contract

def test_a_zero_filled_required_modality_is_refused():
    """The zero-fill signature. Exactly 0.0 across every dimension of a timestep --
    which `_missing_default` is and a real V-JEPA activation is not."""
    feats = {"video": np.zeros((5, 1408), dtype=np.float32)}
    with pytest.raises(ModalityContractViolation, match="exactly zero"):
        assert_modality_contract(feats, required=["video"], expected_absent=[])


def test_a_partially_zero_filled_modality_is_refused():
    """Catches per-segment zero-fill, not just whole-extractor deletion."""
    a = np.random.default_rng(0).normal(size=(5, 1408)).astype(np.float32)
    a[3] = 0.0
    with pytest.raises(ModalityContractViolation, match="timestep"):
        assert_modality_contract({"video": a}, required=["video"], expected_absent=[])


def test_an_absent_required_modality_is_refused():
    with pytest.raises(ModalityContractViolation, match="absent"):
        assert_modality_contract({"audio": np.ones((2, 4))}, required=["video"],
                                 expected_absent=[])


def test_a_real_activation_passes():
    a = np.random.default_rng(0).normal(size=(5, 1408)).astype(np.float32)
    assert_modality_contract({"video": a}, required=["video"],
                             expected_absent=["audio", "text"])


def test_an_unexpectedly_present_modality_is_refused():
    """Backward direction: the frozen design says the video is silent. If audio
    appears, what ran is not what was designed."""
    a = np.random.default_rng(0).normal(size=(5, 1408)).astype(np.float32)
    with pytest.raises(ModalityContractViolation, match="declared absent"):
        assert_modality_contract({"video": a, "audio": a}, required=["video"],
                                 expected_absent=["audio"])


# ------------------------------- the guards that were DEAD before 2026-08-26
# stage2_infer previously DECLARED required_modalities and expected_absent and never
# referenced either. The guard existed; the caller could not switch it on. These tests
# exercise it through the real stage2_infer, not through assert_modality_contract
# directly -- that distinction is the entire finding.

def test_stage2_refuses_a_zero_filled_video_modality(tmp_path):
    """tribev2 deletes an extractor with no matching events and zero-fills the gap.
    With time_pos_embedding on, the output is finite, non-zero, time-varying and
    within 2% of a real run -- and the encode count is 0, the success value."""
    w, led = _finalize(tmp_path)
    w.modalities = np.zeros((6, 1408), dtype=np.float32)
    with pytest.raises(ModalityContractViolation, match="exactly zero"):
        _infer(w, IDENTITY, tmp_path / "art", UIDS, led)


def test_stage2_refuses_an_absent_video_modality(tmp_path):
    w, led = _finalize(tmp_path)
    w.probe_modalities = lambda m, e: {"audio": np.ones((6, 4))}
    with pytest.raises(ModalityContractViolation, match="absent"):
        _infer(w, IDENTITY, tmp_path / "art", UIDS, led)


def test_stage2_refuses_a_modality_the_design_says_is_absent(tmp_path):
    """The frozen design uses a SILENT video. If audio appears, what ran is not what
    was designed."""
    w, led = _finalize(tmp_path)
    real = w.modalities
    w.probe_modalities = lambda m, e: {"video": real, "audio": real}
    with pytest.raises(ModalityContractViolation, match="declared absent"):
        _infer(w, IDENTITY, tmp_path / "art", UIDS, led)


def test_the_modality_check_happens_before_prediction(tmp_path):
    """A doomed run must cost nothing. Predict must not have been called."""
    w, led = _finalize(tmp_path)
    w.modalities = np.zeros((6, 1408), dtype=np.float32)
    called = {"n": 0}
    w.predict = lambda m, e: (called.__setitem__("n", called["n"] + 1),
                              (np.zeros((10, 4)), []))[1]
    with pytest.raises(ModalityContractViolation):
        _infer(w, IDENTITY, tmp_path / "art", UIDS, led)
    assert called["n"] == 0, "the brain model ran despite a broken modality contract"


def test_an_inactive_encode_counter_is_refused(tmp_path):
    """Zero encodes from an unplugged instrument is not evidence of zero encodes.
    EncodeCounter returns early with active=False when exca is absent, and the old
    check read `if counter.items:` -- so an inert counter certified success."""
    w, led = _finalize(tmp_path)
    with pytest.raises(ConsumeStageRecomputed, match="INACTIVE"):
        _infer(w, IDENTITY, tmp_path / "art", UIDS, led, active=False)


def test_predictions_are_persisted_before_analysis(tmp_path):
    """analyse() has ~58 reachable raise sites and is the first thing to touch the
    atlas. The only copy of ~86 MB of predictions must not be a local variable."""
    w, led = _finalize(tmp_path)
    seen = {}
    real_analyse = w.analyse

    def analyse_checking_persistence(preds, segments):
        seen["persisted_first"] = w.persisted is not None
        return real_analyse(preds, segments)

    stage2_infer(None, IDENTITY, tmp_path / "art", UIDS,
                 _s2(w, analyse=analyse_checking_persistence), led,
                 required_modalities=REQ, expected_absent=ABS,
                 counter_factory=lambda: FakeCounter(0))
    assert seen["persisted_first"], "analyse() ran before predictions were persisted"
    rec = [r for r in led.records() if r["event"] == "infer_completed"][-1]
    assert rec["preds_path"], "the ledger does not record where predictions went"


def test_stage2_cannot_be_built_without_its_guards():
    """Every Stage2Deps field is required. A guard that is optional at the call site
    is the exact failure this whole phase exists to remove."""
    with pytest.raises(TypeError):
        Stage2Deps(load_model=lambda: None, build_events=lambda m: None,
                   predict=lambda m, e: (None, None), read_item=lambda u: None,
                   analyse=lambda p, s: {})


def test_the_sidecar_probe_is_consulted(tmp_path):
    """Previously verify_artifact was called with sidecars=None from the real path,
    silently skipping exca's provenance-laundering check."""
    w, led = _finalize(tmp_path)
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return {}

    stage2_infer(None, IDENTITY, tmp_path / "art", UIDS, _s2(w, sidecar_probe=probe),
                 led, required_modalities=REQ, expected_absent=ABS,
                 counter_factory=lambda: FakeCounter(0))
    assert calls["n"] >= 1, "the sidecar probe was never consulted"
