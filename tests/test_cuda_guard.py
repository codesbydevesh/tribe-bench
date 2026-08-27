"""Executable proof of the S2 GPU firewall, on a box with no GPU and no torch.

The incident of 2026-08-25 died on `Cannot re-initialize CUDA in forked
subprocess`. `data.num_workers=0` closes exactly one of the 22 routes by which a
child process can reach CUDA, so the guard must hold regardless of configuration.

Reviewer C established, by execution, that `spawn` is NOT the fix: a spawned
worker re-initialises CUDA cleanly and then honestly re-encodes V-JEPA in every
worker in parallel -- the crash disappears and the catastrophe remains. Measured
`os.register_at_fork` hit counts: subprocess 0, fork 1, ProcessPoolExecutor +2,
spawn +0, forkserver +0. So the at-fork layer can never be the only layer; the
PID sentinel travels through the ENVIRONMENT, which fork, spawn, forkserver and
exec all inherit.
"""
import multiprocessing as mp
import os
import sys
import types

import pytest

from tribe_tools import cuda_guard


@pytest.fixture
def fake_torch(monkeypatch):
    """Just the surface the guard poisons."""
    state = {"init_calls": 0}
    t, c = types.ModuleType("torch"), types.ModuleType("torch.cuda")

    def _lazy_init():
        state["init_calls"] += 1
        return "CUDA CONTEXT CREATED"

    c._lazy_init = _lazy_init
    c.init = _lazy_init
    c.set_device = lambda i: None
    c.is_available = lambda: True
    t.cuda = c
    monkeypatch.setitem(sys.modules, "torch", t)
    monkeypatch.setitem(sys.modules, "torch.cuda", c)
    return state


@pytest.fixture
def armed(fake_torch, monkeypatch):
    monkeypatch.delenv(cuda_guard.ENV_OWNER, raising=False)
    monkeypatch.delenv(cuda_guard.ENV_ARMED, raising=False)
    cuda_guard.arm()
    yield
    monkeypatch.delenv(cuda_guard.ENV_OWNER, raising=False)


def _child_attempt(q):
    """Runs in the child. Reports whether the guard stopped it."""
    from tribe_tools import cuda_guard as g
    try:
        g.check("V-JEPA model construction")
        q.put(("ALLOWED", os.getpid()))
    except g.ChildGPUViolation as e:
        q.put(("BLOCKED", str(e).splitlines()[0]))


def _run_in(method):
    ctx = mp.get_context(method)
    q = ctx.Queue()
    p = ctx.Process(target=_child_attempt, args=(q,))
    p.start()
    p.join(30)
    return q.get(timeout=10)


def test_the_owning_process_may_build_the_model(armed):
    cuda_guard.check("V-JEPA model construction")   # must not raise


def test_an_unarmed_process_is_not_restricted(monkeypatch, fake_torch):
    """The guard must not change behaviour when it was never armed -- otherwise it
    breaks every unrelated import."""
    monkeypatch.delenv(cuda_guard.ENV_OWNER, raising=False)
    assert cuda_guard.is_owner() is True
    cuda_guard.check("anything")


@pytest.mark.parametrize("method", ["fork", "spawn", "forkserver"])
def test_no_child_process_may_build_a_model(armed, method):
    """The route that actually killed the run, plus the two 'fixes' that do not fix
    it. All three must be blocked by the PID sentinel, which rides the environment."""
    if method not in mp.get_all_start_methods():
        pytest.skip(f"{method} unavailable on this platform")
    verdict, detail = _run_in(method)
    assert verdict == "BLOCKED", f"{method} child was allowed to build a model: {detail}"
    assert str(os.getpid()) in detail, "the message does not name the owning process"


def test_a_process_pool_worker_may_not_build_a_model(armed):
    """exca dispatches to ProcessPoolExecutor when infra.cluster='processpool',
    which three places in tribev2/neuralset set as a hardcoded default."""
    import concurrent.futures as cf
    with cf.ProcessPoolExecutor(max_workers=1) as ex:
        verdict = ex.submit(_pool_probe).result(timeout=30)
    assert verdict.startswith("BLOCKED"), verdict


def _pool_probe():
    from tribe_tools import cuda_guard as g
    try:
        g.check("feature extraction")
        return "ALLOWED"
    except g.ChildGPUViolation as e:
        return "BLOCKED: " + str(e)[:60]


def test_a_forked_child_calling_cuda_directly_is_poisoned(armed, fake_torch):
    """Layer 2. Even a child that never consults the sentinel finds the CUDA entry
    points replaced with raisers."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_direct_cuda_probe, args=(q,))
    p.start(); p.join(30)
    verdict, detail = q.get(timeout=10)
    assert verdict == "BLOCKED", detail


def _direct_cuda_probe(q):
    import sys
    try:
        sys.modules["torch.cuda"]._lazy_init()
        q.put(("ALLOWED", "cuda initialised in a forked child"))
    except Exception as e:
        q.put(("BLOCKED", f"{type(e).__name__}: {str(e)[:60]}"))


def test_a_forked_child_sees_no_cuda_devices(armed):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=lambda qq: qq.put(os.environ.get("CUDA_VISIBLE_DEVICES")), args=(q,))
    p.start(); p.join(30)
    assert q.get(timeout=10) == "", "a forked child can still see the GPU"


def test_the_violation_message_names_the_likely_cause(armed):
    """An operator hitting this at hour five needs the diagnosis in the message."""
    import subprocess
    env = dict(os.environ)
    env[cuda_guard.ENV_OWNER] = str(os.getpid() + 1)   # pretend we are not the owner
    r = subprocess.run(
        [sys.executable, "-c",
         "from tribe_tools import cuda_guard as g\n"
         "try: g.check('feature extraction')\n"
         "except g.ChildGPUViolation as e: print(e)"],
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = r.stdout
    assert "num_workers" in out and "processpool" in out, out
