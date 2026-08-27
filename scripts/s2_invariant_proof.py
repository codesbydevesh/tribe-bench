"""Prove the expensive-operation invariant with the REAL exca, across process boundaries.

    Within one invocation, new identity : encodes == exactly 1 per item
    Restart, identity already valid     : encodes == 0
    Restart, new identity               : encodes == exactly 1 per item
    Anything else                       : failure

Stage 1 and Stage 2 run in SEPARATE INTERPRETERS. That is not ceremony: a
same-process test can pass features through exca's in-RAM cache while the disk
cache is empty, and would report zero encodes for the wrong reason.

The "V-JEPA" here is a stand-in that increments a counter on disk, so the encode
count is observed across processes rather than inferred. Everything else -- exca's
MapInfra, its cache uid, its read-only mode, its index -- is the real library at the
pinned 0.5.20 that ran on Kaggle.

    python3 scripts/s2_invariant_proof.py

Exit 0 only if every row matches.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

WORKER = r'''
import json, os, sys, typing as tp
from pathlib import Path
sys.path.insert(0, os.environ["S2_REPO"])
for p in os.environ.get("S2_DEV_SITE_PACKAGES", "").split(os.pathsep):
    if p: sys.path.insert(0, p)

import numpy as np, pydantic, exca
from tribe_tools.feature_artifact import verify_artifact, write_artifact, begin_stage1
from tribe_tools.ledger import Ledger, Event, EncodeCounter, resume_state

STAGE   = sys.argv[1]
CACHE   = os.environ["S2_CACHE"]
ART     = Path(os.environ["S2_ART"])
TALLY   = Path(os.environ["S2_TALLY"])
LEDGER  = Path(os.environ["S2_LEDGER"])
IDENT   = json.loads(os.environ["S2_IDENT"])
UIDS    = ["a", "b", "c"]

def bump():
    n = int(TALLY.read_text()) if TALLY.exists() else 0
    TALLY.write_text(str(n + 1))

class VJEPA(pydantic.BaseModel):
    """Stand-in encoder behind a REAL exca MapInfra."""
    tag: str = "s2"
    infra: exca.MapInfra = exca.MapInfra(version="v5")

    @infra.apply(item_uid=lambda x: str(x))
    def _get_data(self, items: tp.Sequence[str]) -> tp.Iterator[np.ndarray]:
        for i in items:
            bump()                                   # stands in for ~15 min of GPU
            yield np.full((4, 3), float(ord(i)), dtype=np.float32)

def infra_for(stage):
    cfg = dict(folder=CACHE, keep_in_ram=False, version="v5")
    if stage == "consume":
        cfg["mode"] = "read-only"
        cfg["forbid_single_item_computation"] = True
    return cfg

led = Ledger(LEDGER)

if STAGE == "extract":
    begin_stage1(ART)
    led.record(Event.EXTRACT_STARTED, identity=IDENT)
    enc = VJEPA(infra=infra_for("extract"))
    with EncodeCounter() as c:
        produced = list(enc._get_data(UIDS))          # MATERIALISED, not discarded
    led.record(Event.EXTRACT_COMPLETED, encoded_items=c.items)
    cd = VJEPA(infra=infra_for("extract")).infra.cache_dict
    materialised = {u: np.asarray(cd[u]) for u in UIDS}   # READ BACK from cache
    man = write_artifact(ART, IDENT, materialised)
    led.record(Event.ARTIFACT_FINALIZED, identity=IDENT, n_items=man["n_items"])
    print(json.dumps({"stage": "extract", "encodes": c.items}))

elif STAGE == "consume":
    cd = VJEPA(infra=infra_for("consume")).infra.cache_dict
    verify_artifact(ART, IDENT, UIDS, lambda u: np.asarray(cd[u]))
    led.record(Event.ARTIFACT_VERIFIED)
    enc = VJEPA(infra=infra_for("consume"))
    with EncodeCounter() as c:
        out = list(enc._get_data(UIDS))
    led.record(Event.INFER_COMPLETED, encoded_items=c.items)
    print(json.dumps({"stage": "consume", "encodes": c.items,
                      "first": float(np.asarray(out[0]).ravel()[0])}))

elif STAGE == "resume":
    print(json.dumps({"action": resume_state(LEDGER, IDENT).action}))
'''

ROWS: list[tuple[str, str, str, bool]] = []


def run(stage, env):
    w = Path(env["S2_TMP"]) / "worker.py"
    w.write_text(WORKER)
    r = subprocess.run([sys.executable, str(w), stage], capture_output=True,
                       text=True, env={**os.environ, **env}, cwd=REPO, timeout=300)
    if r.returncode != 0:
        return {"error": (r.stderr.strip().splitlines() or ["?"])[-1][:120]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def row(name, expected, observed):
    ok = str(observed) == str(expected)
    ROWS.append((name, str(expected), str(observed), ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:52s} expected={expected!s:22s} observed={observed}")
    return ok


def main() -> int:
    print("=== S2 expensive-operation invariant, real exca, separate processes ===\n")
    tmp = Path(tempfile.mkdtemp())
    ident_a = {"stimulus_sha256": "a" * 64, "weights_sha256": "b" * 64}
    ident_b = {"stimulus_sha256": "z" * 64, "weights_sha256": "b" * 64}
    base = {"S2_REPO": str(REPO), "S2_TMP": str(tmp),
            "S2_DEV_SITE_PACKAGES": os.environ.get("S2_DEV_SITE_PACKAGES", "")}

    def env_for(ident, tag):
        d = tmp / tag
        (d).mkdir(parents=True, exist_ok=True)
        return {**base, "S2_CACHE": str(d / "cache"), "S2_ART": str(d / "art"),
                "S2_TALLY": str(d / "tally"), "S2_LEDGER": str(d / "ledger.jsonl"),
                "S2_IDENT": json.dumps(ident, sort_keys=True)}

    ok = True
    eA = env_for(ident_a, "A")

    r1 = run("extract", eA)
    ok &= row("1 first run, new identity -> extract", 3, r1.get("encodes", r1))
    ok &= row("1 tally on disk after extract", 3, Path(eA["S2_TALLY"]).read_text())

    r2 = run("consume", eA)
    ok &= row("2 consume in a FRESH process -> zero encodes", 0, r2.get("encodes", r2))
    ok &= row("2 tally unchanged by consume", 3, Path(eA["S2_TALLY"]).read_text())
    ok &= row("2 consumed value is the extracted value", 97.0, r2.get("first"))

    r3 = run("resume", eA)
    ok &= row("3 ledger after a crash -> verify, do not extract",
              "verify_then_infer", r3.get("action", r3))

    r4 = run("consume", eA)
    ok &= row("4 restart with a valid identity -> zero encodes", 0, r4.get("encodes", r4))
    ok &= row("4 tally still unchanged", 3, Path(eA["S2_TALLY"]).read_text())

    eB = env_for(ident_b, "B")
    r5 = run("extract", eB)
    ok &= row("5 new identity -> exactly one extraction", 3, r5.get("encodes", r5))

    # a consume against an EMPTY cache must refuse, not encode
    eC = env_for(ident_a, "C")
    r6 = run("consume", eC)
    ok &= row("6 consume with no artifact -> refuses, encodes nothing",
              "0", str(Path(eC["S2_TALLY"]).read_text() if Path(eC["S2_TALLY"]).exists() else "0"))
    ok &= row("6 and it raised rather than returning", True, "error" in r6)

    n = len(ROWS); good = sum(1 for *_, o in ROWS if o)
    print(f"\n{good}/{n} invariant checks hold")
    if not ok:
        print("\nINVARIANT VIOLATED -- do not run S2.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
