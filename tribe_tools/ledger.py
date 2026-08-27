"""Append-only execution record for an S2 GPU session.

The 2026-08-25 incident spent 4h45m of V-JEPA and left NOTHING on disk: afterwards
there was no way to tell whether the encoding had been reused, repeated, or thrown
away. The answer had to be reconstructed from tqdm bars in a notebook.

This module exists so that question is never asked again, and so that a process
which dies between stages leaves a resume state that is unambiguous rather than
merely suggestive.

Two rules the design encodes:

* Every record is flushed AND fsynced before `record()` returns. A ledger that
  loses its last line to a SIGKILL is worse than no ledger, because it certifies
  a state that never happened.
* The ledger never authorises skipping verification. The most it can say is
  "an artifact was finalized for this identity, go and verify it".
"""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Event(str, Enum):
    """The fixed vocabulary. A stage inventing an event name is a bug."""

    PREFLIGHT_STARTED = "preflight_started"
    PREFLIGHT_PASSED = "preflight_passed"
    EXTRACT_STARTED = "extract_started"
    EXTRACT_COMPLETED = "extract_completed"
    ARTIFACT_FINALIZED = "artifact_finalized"
    ARTIFACT_VERIFIED = "artifact_verified"
    ARTIFACT_REJECTED = "artifact_rejected"
    INFER_STARTED = "infer_started"
    INFER_COMPLETED = "infer_completed"
    REPORT_WRITTEN = "report_written"
    ABORTED = "aborted"


class Ledger:
    """Append-only JSONL, fsynced per record."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: Event, **payload) -> dict:
        rec = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "event": Event(event).value,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            **payload,
        }
        line = json.dumps(rec, sort_keys=True, default=str) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        return rec

    def records(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line is expected after SIGKILL. Ignoring it is safe
                # BECAUSE a truncated record can only ever remove information, and
                # every action below is derived from positive evidence.
                continue
        return out


@dataclass(frozen=True)
class ResumeState:
    action: str
    reason: str
    identity: dict | None
    last_event: str | None


def resume_state(path: Path, identity: dict) -> ResumeState:
    """Derive the resume action from the ledger alone.

    Deliberately conservative. The only way to reach "verify_then_infer" is a
    ARTIFACT_FINALIZED whose recorded identity matches EXACTLY -- and even that
    only authorises *verification*, never consumption.
    """
    recs = Ledger(path).records()
    last = recs[-1]["event"] if recs else None

    finalized = [r for r in recs if r["event"] == Event.ARTIFACT_FINALIZED.value]
    if finalized:
        rec_id = finalized[-1].get("identity")
        if rec_id == identity:
            return ResumeState("verify_then_infer",
                               "an artifact was finalized for this exact identity; "
                               "it must still be digest-verified before use",
                               rec_id, last)
        return ResumeState("extract",
                           "the ledger describes a DIFFERENT stimulus/model identity; "
                           "a previous artifact cannot authorise this run",
                           rec_id, last)

    if any(r["event"] == Event.EXTRACT_STARTED.value for r in recs):
        return ResumeState("extract",
                           "extraction started but no artifact was finalized; a "
                           "half-written cache is not a checkpoint",
                           None, last)

    if any(r["event"] == Event.PREFLIGHT_PASSED.value for r in recs):
        return ResumeState("extract", "preflight passed, nothing extracted yet",
                           None, last)

    return ResumeState("run_preflight", "no usable history", None, last)


class EncodeCounter:
    """Count V-JEPA items actually encoded, by patching exca's single funnel.

    `MapInfra._call_and_store` is the one place every exca recomputation passes
    through -- both the in-process branch and the process-pool branch. Counting
    there catches a recompute no matter which extractor or which stage caused it.

    NEVER assert `.items == 0` on its own. Zero is the shared signature of success,
    of a poisoned cache that exca believes is complete, and of an extractor that was
    silently deleted before it ever ran. Pair it with a digest verification.
    """

    def __init__(self):
        self.items = 0
        self.calls = 0
        self._orig = None
        self._target = None

    def __enter__(self) -> "EncodeCounter":
        try:
            from exca import map as emap
        except Exception:
            return self          # exca absent: counter stays at zero, .active False
        self._target = emap.MapInfra
        self._orig = emap.MapInfra._call_and_store
        counter = self

        def counting(self_infra, items, use_cache_dict=True):
            counter.calls += 1
            items = list(items)
            counter.items += len(items)
            return counter._orig(self_infra, items, use_cache_dict=use_cache_dict)

        emap.MapInfra._call_and_store = counting
        return self

    def __exit__(self, *exc) -> None:
        if self._orig is not None and self._target is not None:
            self._target._call_and_store = self._orig
            self._orig = None

    @property
    def active(self) -> bool:
        """False when exca is not importable, i.e. the count means nothing."""
        return self._target is not None
