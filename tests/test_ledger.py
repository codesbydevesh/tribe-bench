"""The ledger must make the resume state unambiguous after a crash.

The 2026-08-25 incident left nothing on disk. Afterwards the only way to answer
"was the 4h45m of encoding reused or repeated?" was to read tqdm bars out of a
notebook. These tests pin the properties that make that question answerable.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tribe_tools.ledger import EncodeCounter, Event, Ledger, ResumeState, resume_state

ID_A = {"stimulus_sha256": "a" * 64, "weights_sha256": "b" * 64}
ID_B = {"stimulus_sha256": "z" * 64, "weights_sha256": "b" * 64}


@pytest.fixture
def led(tmp_path):
    return Ledger(tmp_path / "ledger.jsonl")


# ------------------------------------------------------------- basic record

def test_a_record_is_durable_before_it_returns(led):
    """fsync, not just flush. A ledger that loses its last line to SIGKILL is worse
    than no ledger: it certifies a state that never happened."""
    led.record(Event.EXTRACT_STARTED, n_items=18)
    raw = led.path.read_text().strip()
    rec = json.loads(raw)
    assert rec["event"] == "extract_started"
    assert rec["n_items"] == 18
    assert rec["pid"] == os.getpid()
    assert "ts" in rec and "iso" in rec and "host" in rec


def test_every_record_is_fsynced(led, monkeypatch):
    """Durability, asserted as a call rather than as an outcome.

    HONEST LIMITATION: no in-process test can observe a MISSING fsync. SIGKILL does
    not lose the page cache -- only a machine crash or power loss does, which is
    exactly the Kaggle 12-hour-wall / preemption case we care about. Demonstrated:
    the mutation that deletes the fsync leaves the SIGKILL test green. So this test
    asserts the syscall is made, on the file we just wrote, which is the strongest
    check available without fault-injecting the filesystem.
    """
    import tribe_tools.ledger as mod
    seen = []
    real = mod.os.fsync
    monkeypatch.setattr(mod.os, "fsync", lambda fd: (seen.append(fd), real(fd))[1])
    led.record(Event.EXTRACT_STARTED)
    assert seen, "the record was not fsynced; a SIGKILL could lose it silently"


def test_records_append_and_preserve_order(led):
    for e in (Event.PREFLIGHT_STARTED, Event.PREFLIGHT_PASSED, Event.EXTRACT_STARTED):
        led.record(e)
        time.sleep(0.002)
    got = [r["event"] for r in led.records()]
    assert got == ["preflight_started", "preflight_passed", "extract_started"]


def test_an_unknown_event_name_is_rejected(led):
    """The vocabulary is fixed; a stage inventing an event is a bug, not a feature."""
    with pytest.raises(ValueError):
        led.record("i_made_this_up")


def test_a_torn_final_line_is_survivable(led):
    """Expected after SIGKILL. Ignoring it is only safe because every resume action
    is derived from POSITIVE evidence, so losing a record can never authorise more."""
    led.record(Event.PREFLIGHT_PASSED)
    with open(led.path, "a") as f:
        f.write('{"event": "artifact_finali')      # torn
    assert [r["event"] for r in led.records()] == ["preflight_passed"]
    assert resume_state(led.path, ID_A).action == "extract"


# ----------------------------------------------------------- resume states

def test_an_empty_ledger_asks_for_preflight(tmp_path):
    st = resume_state(tmp_path / "absent.jsonl", ID_A)
    assert st.action == "run_preflight"


def test_extraction_started_but_not_finalized_must_re_extract(led):
    """A half-written cache is not a checkpoint. This is the exact state the
    2026-08-25 run died in."""
    led.record(Event.PREFLIGHT_PASSED)
    led.record(Event.EXTRACT_STARTED, identity=ID_A)
    st = resume_state(led.path, ID_A)
    assert st.action == "extract"
    assert "not a checkpoint" in st.reason


def test_a_finalized_artifact_authorises_verification_not_consumption(led):
    """The strongest thing the ledger may ever say."""
    led.record(Event.ARTIFACT_FINALIZED, identity=ID_A, n_items=18)
    st = resume_state(led.path, ID_A)
    assert st.action == "verify_then_infer"
    assert "verified" in st.reason


def test_a_ledger_for_a_different_identity_never_authorises_reuse(led):
    led.record(Event.ARTIFACT_FINALIZED, identity=ID_A, n_items=18)
    st = resume_state(led.path, ID_B)
    assert st.action == "extract", "a different stimulus was allowed to reuse features"


def test_no_ledger_state_ever_skips_verification(led):
    """Exhaustive over the vocabulary: no sequence of events produces an action that
    consumes features without verifying them first."""
    actions = set()
    for e in Event:
        led.record(e, identity=ID_A)
        actions.add(resume_state(led.path, ID_A).action)
    assert actions <= {"run_preflight", "extract", "verify_then_infer"}, actions
    assert "infer" not in actions, "a ledger state authorised inference directly"


# ------------------------------------------------------------ crash / resume

def test_a_killed_process_leaves_a_readable_unambiguous_ledger(tmp_path):
    """Kill -9 between EXTRACT_STARTED and ARTIFACT_FINALIZED, then ask the ledger."""
    p = tmp_path / "ledger.jsonl"
    code = (
        "import sys, os, signal;"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r});"
        "from tribe_tools.ledger import Ledger, Event;"
        f"lg = Ledger({str(p)!r});"
        "lg.record(Event.PREFLIGHT_PASSED);"
        f"lg.record(Event.EXTRACT_STARTED, identity={ID_A!r});"
        "os.kill(os.getpid(), signal.SIGKILL)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert r.returncode == -9, r.stderr[:300]
    st = resume_state(p, ID_A)
    assert st.action == "extract"
    assert st.last_event == "extract_started"


# ----------------------------------------------------------- encode counter

def test_the_encode_counter_reports_whether_it_is_active():
    """A counter that silently reads zero because exca is absent would certify the
    invariant it is supposed to test."""
    with EncodeCounter() as c:
        pass
    assert isinstance(c.active, bool)


def test_the_encode_counter_counts_real_encodes_and_reads_zero_on_a_warm_cache():
    """Drives the REAL exca funnel with a real cached model.

    Cold cache -> the counter sees every item. Warm cache -> zero, because
    _call_and_store is never entered. That second number is the operator's
    expensive-operation invariant, measured rather than asserted about.
    """
    pytest.importorskip("exca", reason="set S2_DEV_SITE_PACKAGES")
    import tempfile, typing as tp
    import pydantic, exca

    calls = {"n": 0}

    class Enc(pydantic.BaseModel):
        tag: str = "v1"
        infra: exca.MapInfra = exca.MapInfra(version="v5")

        @infra.apply(item_uid=lambda x: str(x))
        def _get_data(self, items: tp.Sequence[int]) -> tp.Iterator[int]:
            for i in items:
                calls["n"] += 1
                yield i * 10

    folder = tempfile.mkdtemp()
    cfg = dict(folder=folder, keep_in_ram=False, version="v5")

    with EncodeCounter() as cold:
        out = list(Enc(infra=cfg)._get_data([1, 2, 3]))
    assert out == [10, 20, 30]
    assert cold.active
    assert cold.items == 3, f"cold cache should encode 3 items, counted {cold.items}"
    assert calls["n"] == 3

    with EncodeCounter() as warm:
        out2 = list(Enc(infra=cfg)._get_data([1, 2, 3]))
    assert out2 == [10, 20, 30]
    assert warm.items == 0, f"warm cache must encode nothing, counted {warm.items}"
    assert calls["n"] == 3, "the expensive function ran again on a warm cache"


def test_the_counter_sees_the_incident_configuration_recompute_every_time():
    """folder=None + keep_in_ram=False is what shipped on 2026-08-25. exca caches
    NOTHING and recomputes on every access, silently. The counter makes that
    visible as a number instead of a 4h45m tqdm bar."""
    pytest.importorskip("exca", reason="set S2_DEV_SITE_PACKAGES")
    import typing as tp
    import pydantic, exca

    class Enc(pydantic.BaseModel):
        tag: str = "v1"
        infra: exca.MapInfra = exca.MapInfra(version="v5")

        @infra.apply(item_uid=lambda x: str(x))
        def _get_data(self, items: tp.Sequence[int]) -> tp.Iterator[int]:
            for i in items:
                yield i * 10

    cfg = dict(folder=None, keep_in_ram=False, version="v5")
    with EncodeCounter() as a:
        list(Enc(infra=cfg)._get_data([1, 2, 3]))
    with EncodeCounter() as b:
        list(Enc(infra=cfg)._get_data([1, 2, 3]))
    assert a.items == 3 and b.items == 3, (
        "the incident configuration is supposed to recompute every single time; "
        f"got {a.items} then {b.items}")


def test_the_counter_restores_the_original_on_exception():
    exca_map = pytest.importorskip("exca.map", reason="pip install exca==0.5.20")
    orig = exca_map.MapInfra._call_and_store
    with pytest.raises(RuntimeError):
        with EncodeCounter():
            raise RuntimeError("boom")
    assert exca_map.MapInfra._call_and_store is orig
