"""Two-stage S2 orchestration: extract once, then consume without being able to extract.

Before this, extraction and consumption happened inside a single `predict()` call.
That is why, after the 2026-08-25 incident, "did the second stage recompute?" could
not be answered -- there was no boundary at which to ask.

Everything expensive or environment-touching is injected, so the control flow is
testable on a box with no GPU, no tribev2 and no network. The tests that matter --
"can Stage 2 reach the extractor?", "is a poisoned artifact refused before the model
is loaded?" -- are about control flow, not about CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from tribe_tools.feature_artifact import (
    begin_stage1, sidecar_digests, verify_artifact, write_artifact,
)
from tribe_tools.ledger import EncodeCounter, Event, Ledger


class ConsumeStageRecomputed(RuntimeError):
    """Stage 2 encoded something. The read-only firewall failed."""


class ModalityContractViolation(RuntimeError):
    """A required modality was absent or zero-filled.

    tribev2/main.py:206-212 deletes an extractor that has no matching events, and
    tribev2/model.py:188-192 substitutes torch.zeros for any modality missing from
    the batch. Because time_pos_embedding defaults True (tribev2/model.py:56), an
    all-zero video input still receives a learned per-timestep embedding through
    eight transformer layers: the output is finite, non-zero, time-varying,
    autocorrelated, and statistically within 2% of a real run. Reviewer C measured
    it. modality_dropout 0.3 during training puts video-absent squarely
    in-distribution, so the result is plausible rather than obviously broken.

    An object-presence check does NOT catch this: `model.data.video_feature is not
    None` passes after the deletion, because the deletion is from a local dict.
    """


class ExtractionIncomplete(RuntimeError):
    """Stage 1 did not produce every expected item."""


@dataclass(frozen=True)
class Stage1Deps:
    load_model: Callable[..., Any]
    build_events: Callable[[Any], Any]
    extract: Callable[[Any, Any], Sequence[str]]
    read_item: Callable[[str], "np.ndarray"]
    sidecars: Callable[[], dict] = lambda: {}


@dataclass(frozen=True)
class Stage2Deps:
    """Every field is REQUIRED. None of these may default to a no-op.

    The 2026-08-26 review found the previous version declared `required_modalities`
    and `expected_absent` and never referenced either: the guard existed and the
    caller could not switch it on. Making the probe a mandatory dependency means a
    Stage 2 that cannot inspect its own modalities cannot be constructed at all.
    """
    load_model: Callable[..., Any]
    build_events: Callable[[Any], Any]
    predict: Callable[[Any, Any], tuple]
    read_item: Callable[[str], "np.ndarray"]
    analyse: Callable[..., dict]
    # (model, events) -> {modality: tensor} ACTUALLY reaching the brain model
    probe_modalities: Callable[[Any, Any], dict]
    # (preds, segments) -> path. Called immediately after predict, before analyse.
    persist: Callable[[Any, Any], Any]
    # () -> {filename: digest} for exca's own provenance files
    sidecar_probe: Callable[[], dict]


# --------------------------------------------------------------------- modality

ZERO_TOL = 0.0


def assert_modality_contract(features: dict, required: Sequence[str],
                             expected_absent: Sequence[str]) -> None:
    """Two-sided check. Both directions are load-bearing.

    Forward: a required modality must be present and must not be exactly zero across
    a whole timestep. `_missing_default` is exactly 0.0 across all 1408 dims; a real
    V-JEPA activation never is. Exact comparison is deliberate -- a tolerance would
    let a genuinely tiny activation trip the check.

    Backward: a modality we declared absent must actually be absent. If audio
    suddenly appears, the frozen design (a silent video) is not what ran.
    """
    for name in required:
        if name not in features or features[name] is None:
            raise ModalityContractViolation(
                f"required modality {name!r} is absent from the batch. tribev2 "
                f"deletes extractors with no matching events and zero-fills the gap; "
                f"the resulting report would look normal and mean nothing. "
                f"Present: {sorted(features)}")
        arr = np.asarray(features[name])
        if arr.size == 0:
            raise ModalityContractViolation(f"required modality {name!r} is empty")
        # collapse everything but the time axis; a zero-filled modality is exactly
        # zero across the entire feature dimension for every timestep it covers
        flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(1, -1)
        dead = np.flatnonzero((flat == ZERO_TOL).all(axis=1))
        if dead.size:
            raise ModalityContractViolation(
                f"required modality {name!r} is exactly zero across all "
                f"{flat.shape[1]} dimensions at {dead.size} timestep(s) "
                f"(first: {int(dead[0])}). That is the zero-fill signature, not a "
                f"V-JEPA activation.")
    for name in expected_absent:
        if name in features and features[name] is not None:
            raise ModalityContractViolation(
                f"modality {name!r} was declared absent by the frozen design but is "
                f"present. The stimulus is not the one the design describes.")


# ----------------------------------------------------------------------- stage 1

def stage1_extract(cfg: Any, identity: dict, artifact_dir: Path,
                   expected_uids: Sequence[str], deps: Stage1Deps, ledger: Ledger,
                   *, counter_factory=EncodeCounter) -> dict:
    """Extract, read back, digest, finalize. Raises rather than leaving a partial."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # FIRST act: an artifact under construction must not carry an earlier session's
    # certificate. Found by execution -- COMPLETE survived a failed read-back.
    begin_stage1(artifact_dir)
    ledger.record(Event.EXTRACT_STARTED, identity=identity,
                  n_expected=len(expected_uids), artifact_dir=str(artifact_dir))
    try:
        model = deps.load_model()
        events = deps.build_events(model)
        with counter_factory() as counter:
            produced = list(deps.extract(model, events))
        ledger.record(Event.EXTRACT_COMPLETED, n_produced=len(produced),
                      encoded_items=counter.items, counter_active=counter.active)

        missing = [u for u in expected_uids if u not in set(produced)]
        if missing:
            raise ExtractionIncomplete(
                f"{len(missing)} expected item(s) were never produced: {missing[:3]}. "
                f"The artifact will NOT be finalized.")

        # Read the bytes back out of the cache. neuralset/extractors/base.py:201
        # discards the generator exca returns, so without this Stage 1 never touches
        # what it wrote and cannot attest to it.
        materialised = {uid: deps.read_item(uid) for uid in expected_uids}
        man = write_artifact(artifact_dir, identity, materialised, deps.sidecars())
        ledger.record(Event.ARTIFACT_FINALIZED, identity=identity,
                      n_items=man["n_items"], artifact_dir=str(artifact_dir),
                      encoded_items=counter.items)
        return man
    except Exception as exc:
        ledger.record(Event.ABORTED, stage="extract", error=f"{type(exc).__name__}: {exc}")
        raise


# ----------------------------------------------------------------------- stage 2

def stage2_infer(cfg: Any, identity: dict, artifact_dir: Path,
                 expected_uids: Sequence[str], deps: Stage2Deps, ledger: Ledger,
                 *, required_modalities: Sequence[str],
                 expected_absent: Sequence[str],
                 counter_factory=EncodeCounter) -> dict:
    """Verify, then consume. Structurally unable to extract.

    `required_modalities` and `expected_absent` have NO defaults. The previous
    version defaulted them and then never used them, which is how a guard comes to
    exist without protecting anything.
    """
    artifact_dir = Path(artifact_dir)
    # BEFORE load_model: an unusable artifact must cost seconds, not a 709 MB download.
    # sidecar_probe is a dependency, not an optional kwarg -- omitting it previously
    # skipped exca's provenance-laundering check in silence.
    try:
        man = verify_artifact(artifact_dir, identity, expected_uids, deps.read_item,
                              deps.sidecar_probe())
    except Exception as exc:
        ledger.record(Event.ARTIFACT_REJECTED, artifact_dir=str(artifact_dir),
                      error=f"{type(exc).__name__}: {exc}")
        raise
    ledger.record(Event.ARTIFACT_VERIFIED, n_items=man["n_items"],
                  artifact_dir=str(artifact_dir))

    ledger.record(Event.INFER_STARTED, identity=identity)
    try:
        model = deps.load_model()
        events = deps.build_events(model)

        # THE scientific hard stop, on the real path. tribev2/main.py:200-212 deletes
        # an extractor with no matching events and model.py:188-192 zero-fills the gap;
        # with time_pos_embedding on, the output is finite, non-zero, time-varying and
        # within 2% of a real run. It reports as success with an encode count of 0.
        # Checked BEFORE predict so a doomed run costs nothing.
        assert_modality_contract(deps.probe_modalities(model, events),
                                 required_modalities, expected_absent)

        with counter_factory() as counter:
            preds, segments = deps.predict(model, events)

        # An inert counter reads zero. Zero is also the success value. Refuse to treat
        # an unplugged instrument as evidence.
        if not counter.active:
            raise ConsumeStageRecomputed(
                "the encode counter is INACTIVE, so 'no features were computed' is "
                "unmeasured rather than proven. exca must be importable in the "
                "process that runs Stage 2.")
        if counter.items:
            raise ConsumeStageRecomputed(
                f"the consume stage encoded {counter.items} item(s). Stage 2 must "
                f"never compute features. Check that "
                f"data.<mod>_feature.infra.mode == 'read-only' actually reached the "
                f"extractor, and that the artifact covers every requested uid.")

        # B6: the only copy of ~86 MB of predictions must not live solely in RAM
        # across analyse(), which has ~58 reachable raise sites.
        preds_path = deps.persist(preds, segments)
        ledger.record(Event.INFER_COMPLETED, n_rows=int(np.shape(preds)[0]),
                      encoded_items=counter.items, counter_active=counter.active,
                      preds_path=str(preds_path))

        out = deps.analyse(preds, segments)
        ledger.record(Event.REPORT_WRITTEN, n_results=len(out.get("results", {}) or {}))
        return out
    except Exception as exc:
        ledger.record(Event.ABORTED, stage="infer", error=f"{type(exc).__name__}: {exc}")
        raise
