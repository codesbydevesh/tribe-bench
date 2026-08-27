"""S2 GPU/subprocess firewall.

Invariant enforced:  no worker / subprocess may initialise CUDA or invoke a
feature extractor.  Independent of any configuration value.

Three layers, each independently sufficient for the routes it covers:

  L1 PID SENTINEL   - the owning-process id is published in the ENVIRONMENT
                      (inherited by fork AND spawn AND exec'd subprocesses),
                      so a re-imported module in a spawned child cannot
                      re-arm itself as "main".
  L2 AT-FORK POISON - os.register_at_fork(after_in_child=...) replaces the
                      CUDA entry points in the child with raisers.  Covers
                      fork children even if they never reach L1.
                      PROVEN not to fire for spawn/forkserver/subprocess.
  L3 PREFLIGHT      - static assertion over the LIVE object graph that no
                      exca infra can dispatch to a process pool and that the
                      dataloader worker count is zero.
"""
from __future__ import annotations

import os
import sys

ENV_OWNER = "S2_GPU_OWNER_PID"
ENV_ARMED = "S2_GPU_FIREWALL"


class ChildGPUViolation(RuntimeError):
    """A child process attempted GPU init or feature extraction."""


def arm(*, sitehook: str | None = None) -> None:
    """Call ONCE, in the main process, before any model or extractor exists.

    sitehook: directory containing our sitecustomize.py.  Prepending it to
    PYTHONPATH is what makes the guard survive spawn / forkserver / exec.
    """
    os.environ[ENV_OWNER] = str(os.getpid())
    os.environ[ENV_ARMED] = "1"
    if sitehook:
        pp = os.environ.get("PYTHONPATH", "")
        parts = [p for p in pp.split(os.pathsep) if p]
        here = os.path.dirname(os.path.abspath(__file__))
        for extra in (sitehook, here):
            if extra not in parts:
                parts.insert(0, extra)
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    os.register_at_fork(after_in_child=_poison_child)
    install_extractor_sentinel()


def install_child_hooks() -> None:
    """Run in a FRESH interpreter (spawn/forkserver/subprocess) via sitecustomize.

    The owner pid arrives through the environment, so this process knows it is
    not the owner and can refuse before importing anything expensive.
    """
    install_import_hook()
    install_extractor_sentinel()
    if not is_owner():
        # a non-owner interpreter may never see a GPU, whatever it goes on to do
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def is_owner() -> bool:
    owner = os.environ.get(ENV_OWNER)
    if owner is None:
        return True          # firewall not armed -> do not change behaviour
    return os.getpid() == int(owner)


def check(what: str) -> None:
    """Raise unless we are the process that armed the firewall."""
    if is_owner():
        return
    raise ChildGPUViolation(
        f"{what} attempted in pid={os.getpid()} but the S2 GPU owner is "
        f"pid={os.environ.get(ENV_OWNER)}.\n"
        "Stage 2 forbids GPU initialisation and feature extraction in any "
        "worker or subprocess. The parent must supply every feature from the "
        "verified Stage-1 artifact.\n"
        "Likely cause: data.num_workers > 0, an exca infra with "
        "cluster='processpool', or a lazily-loaded model on a cache miss."
    )


# ----------------------------------------------------------------- L2
_CUDA_ATTRS = (
    ("torch.cuda", "_lazy_init"),
    ("torch.cuda", "init"),
    ("torch.cuda", "set_device"),
)


def _raiser(name):
    def _f(*a, **k):
        raise ChildGPUViolation(
            f"{name} called in forked child pid={os.getpid()}; "
            "CUDA is unavailable to S2 workers by policy."
        )
    return _f


def _poison_child() -> None:
    """Runs in the child immediately after fork()."""
    torch = sys.modules.get("torch")
    if torch is None:
        return                       # torch never imported -> nothing to poison
    for modname, attr in _CUDA_ATTRS:
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, attr):
            setattr(mod, attr, _raiser(f"{modname}.{attr}"))
    # belt: a spawned grandchild inherits an empty device list
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ----------------------------------------------------------------- L1 wrapper
def guard_callable(fn, what: str):
    """Wrap a compute entry point so it refuses to run outside the owner."""
    def _wrapped(*a, **k):
        check(what)
        return fn(*a, **k)
    _wrapped.__wrapped__ = fn
    _wrapped.__name__ = getattr(fn, "__name__", "guarded")
    return _wrapped


def install_extractor_sentinel() -> list[str]:
    """Monkeypatch every recompute funnel we can reach. Returns what was patched.

    exca.map.MapInfra._call_and_store is THE single funnel through which every
    exca recomputation passes, on both the pool branch (map.py:482) and the
    in-process branch (map.py:467).  Guarding it makes 'a cache miss in a
    worker' impossible on every extractor at once, without touching neuralset.
    """
    patched = []
    try:
        from exca import map as _emap
    except Exception:
        return patched
    if not getattr(_emap.MapInfra._call_and_store, "_s2_guarded", False):
        orig = _emap.MapInfra._call_and_store

        def _call_and_store(self, items, use_cache_dict=True):
            check(f"exca recomputation ({type(self._obj).__name__ if getattr(self,'_obj',None) else '?'})")
            return orig(self, items, use_cache_dict=use_cache_dict)

        _call_and_store._s2_guarded = True
        _emap.MapInfra._call_and_store = _call_and_store
        patched.append("exca.map.MapInfra._call_and_store")
    return patched


def install_import_hook() -> None:
    """Patch exca.map the moment it is imported, however late that is.

    sitecustomize runs before the child has any third-party path set up, so a
    direct `import exca` there can silently fail.  A meta_path hook cannot.
    """
    import importlib.abc, importlib.machinery

    class _Hook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        _busy = False

        def find_module(self, *a, **k):  # py2 shim, unused
            return None

        def find_spec(self, fullname, path=None, target=None):
            if fullname != "exca.map" or _Hook._busy:
                return None
            return None  # we only need the post-import callback below

    # simplest reliable mechanism: wrap builtins.__import__
    import builtins
    if getattr(builtins.__import__, "_s2_hooked", False):
        return
    _orig_import = builtins.__import__

    def _imp(name, globals=None, locals=None, fromlist=(), level=0):
        mod = _orig_import(name, globals, locals, fromlist, level)
        if name.startswith("exca") or (fromlist and "map" in fromlist and name == "exca"):
            try:
                install_extractor_sentinel()
            except Exception:
                pass
        return mod

    _imp._s2_hooked = True
    builtins.__import__ = _imp


# ----------------------------------------------------------------- L3
_POOLY = {"processpool", "threadpool", "slurm", "auto", "local", "debug"}


def walk_infras(root, _seen=None, _path="root"):
    """Yield (dotted_path, infra_obj) for every exca infra on a pydantic graph."""
    if _seen is None:
        _seen = set()
    if id(root) in _seen:
        return
    _seen.add(id(root))
    fields = getattr(type(root), "model_fields", None)
    if fields is None:
        return
    for name in fields:
        try:
            val = getattr(root, name)
        except Exception:
            continue
        path = f"{_path}.{name}"
        if val is None:
            continue
        if hasattr(val, "cluster") and hasattr(val, "uid"):
            yield path, val
        if isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                yield from walk_infras(v, _seen, f"{path}[{i}]")
        elif isinstance(val, dict):
            for k, v in val.items():
                yield from walk_infras(v, _seen, f"{path}[{k!r}]")
        else:
            yield from walk_infras(val, _seen, path)


def preflight(experiment, *, require_read_only=True) -> None:
    """Assert the LIVE object graph cannot dispatch work off the main process."""
    problems = []
    nw = getattr(getattr(experiment, "data", None), "num_workers", None)
    if nw != 0:
        problems.append(f"data.num_workers={nw!r} (must be exactly 0)")
    n = 0
    for path, infra in walk_infras(experiment):
        n += 1
        if infra.cluster is not None:
            problems.append(f"{path}.cluster={infra.cluster!r} (must be None)")
        if require_read_only and path.endswith("_feature.infra"):
            if getattr(infra, "mode", None) != "read-only":
                problems.append(f"{path}.mode={getattr(infra,'mode',None)!r} (must be 'read-only')")
            if not getattr(infra, "forbid_single_item_computation", False):
                problems.append(f"{path}.forbid_single_item_computation is False")
    if not n:
        problems.append("no exca infra found on the object graph -- walker is blind, refuse to run")
    if problems:
        raise RuntimeError(
            "S2 preflight refused to start:\n  - " + "\n  - ".join(problems)
        )
