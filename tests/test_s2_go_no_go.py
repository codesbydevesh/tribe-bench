"""Regression tests for the four go/no-go CHECKER defects found on Kaggle.

All four printed a red item while the design was fine. They surfaced only because
the gate ran somewhere other than the machine that wrote it, which is exactly the
situation the final GPU session will be in — so they are pinned here.

The rule these enforce: a check must verify its invariant *wherever it runs*, and
must not fail because of where a file happens to live or because a neighbouring
check failed.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "s2_go_no_go.py"


def _load():
    spec = importlib.util.spec_from_file_location("s2gng", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(*args, cwd=REPO, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=cwd, env=e)


# ---------------------------------------------------------------- defect 1

def test_image_hash_check_honours_the_stimulus_root():
    """DEFECT 1. The image-hash gate hardcoded `data/floc`, so on Kaggle's
    read-only mount it raised FileNotFoundError and reported a red item for a
    perfectly good upload. It must resolve images the same way s2_run.py does."""
    src = SCRIPT.read_text()
    assert 'resolve_stimulus_images("data/floc"' not in src, \
        "the image root is hardcoded again; it will fail on any other machine"
    assert "--stimulus-root" in src, "the gate offers no stimulus-root knob"
    assert "S2_STIMULUS_ROOT" in src, "the gate ignores the env var s2_run.py uses"


def test_image_hash_check_actually_reads_from_the_given_root(tmp_path):
    """And it must USE the knob, not merely accept it. Point it at an empty dir:
    the image gate must fail there while the design gates still pass."""
    out = _run("--review-clean", "--neuralset-timestamp",
               "--stimulus-root", str(tmp_path)).stdout
    assert "manifest image hashes match the files on disk" in out
    line = [l for l in out.split("\n")
            if "manifest image hashes match" in l][0]
    assert line.strip().startswith("[ ]"), \
        f"an empty stimulus root should fail the image gate, got: {line}"
    # ...but the design itself must still be reported as sound
    assert "SOA = 8 s and is frozen" in out
    soa = [l for l in out.split("\n") if "SOA = 8 s and is frozen" in l][0]
    assert soa.strip().startswith("[x]"), "a bad image root broke an unrelated design gate"


# ---------------------------------------------------------------- defect 2

def test_checkpoint_check_is_independent_of_the_image_check(tmp_path):
    """DEFECT 2. Both lived in one try block, so a missing image directory also
    failed the checkpoint gate — which reads the manifest and has nothing to do
    with images. The manifest had the revision and hash the whole time."""
    out = _run("--review-clean", "--neuralset-timestamp",
               "--stimulus-root", str(tmp_path)).stdout
    line = [l for l in out.split("\n")
            if "checkpoint revision + hash recorded" in l][0]
    assert line.strip().startswith("[x]"), \
        f"the checkpoint gate inherited the image gate's failure: {line}"


def test_the_manifest_really_does_carry_the_checkpoint_identity():
    """The gate above is only meaningful if the thing it checks is real."""
    man = json.loads((REPO / "data" / "s2_manifest.json").read_text())
    prov = man["provenance"]
    assert prov["model_revision"] == "f894e783020944dcd96e5568550afe2aa9743f9f"
    ck = prov["checkpoint"]
    assert ck["sha256"] == \
        "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"
    assert ck["bytes"] == 708_856_138


# ---------------------------------------------------------------- defects 3 & 4

def test_local_only_dry_run_scratch_is_not_a_kaggle_prerequisite():
    """DEFECTS 3 and 4. Two gates required data/s2_dry_run/, which is gitignored
    scratch on the PREPARING machine and never reaches the GPU box. They failed on
    Kaggle for a reason that has nothing to do with readiness."""
    src = SCRIPT.read_text()
    assert 's2_manifest_tiny.json' not in src or 'stub.exists()' in src, \
        "the gate still hard-requires gitignored dry-run scratch"
    # with the scratch directory absent, the validation gates must still pass on
    # the committed stub report
    out = _run("--review-clean", "--neuralset-timestamp",
               "--dry-run-dir", "/nonexistent/scratch").stdout
    for name in ("CPU end-to-end validation on record",
                 "validation outputs are machine-readable"):
        line = [l for l in out.split("\n") if name in l][0]
        assert line.strip().startswith("[x]"), \
            f"absent local-only scratch failed a gate: {line}"


def test_the_committed_stub_report_is_real_evidence():
    """The replacement evidence must actually prove the pipeline ran end to end,
    or the gate has been weakened rather than corrected."""
    rep = json.loads((REPO / "data" / "s2_report_stub.json").read_text())
    assert rep["stub"] is True, "must be labelled a stub, never mistaken for a result"
    assert rep["results"], "no per-parcel results — the pipeline did not reach the end"
    assert rep["verdict"]["stop_eligible"] == ["FFA", "EBA"]
    # both lags were actually scored
    for name, r in rep["results"].items():
        assert set(r["by_lag"]) == {"5", "0"} or set(map(int, r["by_lag"])) == {5, 0}, name


# ------------------------------------------------------ the classification itself

def test_every_check_declares_what_kind_of_failure_it_would_be():
    """The point of the fix: '53/57' must not read as 'the experiment is broken'.
    Each item declares design / environment / checker / local-only."""
    mod = _load()
    assert hasattr(mod, "check")
    src = SCRIPT.read_text()
    for kind in ("design", "environment", "checker", "local-only"):
        assert f'"{kind}"' in src, f"failure kind '{kind}' is not represented"
    assert "DESIGN FAILURE" in src and "ENVIRONMENT/INPUT FAILURE" in src


def test_a_failing_environment_item_is_not_reported_as_a_design_failure(tmp_path):
    out = _run("--review-clean", "--neuralset-timestamp",
               "--stimulus-root", str(tmp_path)).stdout
    assert "NO-GO" in out
    assert "ENVIRONMENT/INPUT FAILURE" in out, "an input problem was not classified"
    design_block = out.split("DESIGN FAILURE")[1] if "DESIGN FAILURE" in out else ""
    assert "manifest image hashes" not in design_block, \
        "a missing image directory was reported as a DESIGN failure"


# ------------------------------------------------------------------ end to end

def test_the_full_gate_runs_all_items_and_no_design_item_fails():
    """All 57 from scratch — but assert the right invariant.

    An earlier version asserted a flat 57/57, which coupled the test to git state:
    "working tree is clean at freeze time" is legitimately environment-dependent,
    so the test failed on every uncommitted edit and would have cried wolf
    constantly. What must hold at all times is that NO DESIGN item fails; a dirty
    tree is an environment condition to fix before the real run, not a defect."""
    r = _run("--review-clean", "--neuralset-timestamp")
    total = [l for l in r.stdout.split("\n") if "checklist items satisfied" in l][0]
    n, d = total.strip().split()[0].split("/")
    assert int(d) >= 57, f"expected at least 57 items, got {d}"

    if "GPU GO." in r.stdout:
        assert n == d and r.returncode == 0, total
        return
    # not clean: every outstanding item must be environment-kind, never design
    assert "DESIGN FAILURE" not in r.stdout, (
        "a DESIGN item failed — the frozen experiment is wrong:\n"
        + r.stdout.split("NO-GO")[-1])
    assert "CHECKER DEFECT" not in r.stdout, (
        "a CHECKER item failed:\n" + r.stdout.split("NO-GO")[-1])


def test_a_clean_tree_yields_a_flat_gpu_go():
    """And when the tree IS clean, it must be a flat pass — no partial credit."""
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, cwd=REPO).stdout.strip()
    if dirty:
        pytest.skip(f"working tree has {len(dirty.splitlines())} uncommitted change(s)")
    r = _run("--review-clean", "--neuralset-timestamp")
    assert "GPU GO." in r.stdout, r.stdout.strip().split("\n")[-12:]
    assert r.returncode == 0
