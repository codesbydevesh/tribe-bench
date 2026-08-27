"""Auto-arms the S2 firewall in EVERY interpreter that inherits our PYTHONPATH.

This is the only hook that survives spawn / forkserver / exec'd subprocesses,
because those start a fresh interpreter which re-runs `site`.
Chain-loads any pre-existing sitecustomize so we do not shadow the host's.
"""
import os, sys

# 1. chain: let a pre-existing sitecustomize still run
_me = os.path.dirname(os.path.abspath(__file__))
_rest = [p for p in sys.path if os.path.abspath(p) != _me]
try:
    import importlib.util as _ilu
    _spec = _ilu.find_spec("sitecustomize", None)
except Exception:
    _spec = None
for _p in _rest:
    _cand = os.path.join(_p, "sitecustomize.py")
    if os.path.isfile(_cand):
        try:
            import importlib.util as _u
            _s = _u.spec_from_file_location("_host_sitecustomize", _cand)
            _m = _u.module_from_spec(_s); _s.loader.exec_module(_m)
        except Exception:
            pass
        break

# 2. arm
if os.environ.get("S2_GPU_FIREWALL") == "1":
    try:
        import s2guard
        s2guard.install_child_hooks()
    except Exception as _e:  # never break the interpreter
        sys.stderr.write(f"[s2guard] sitecustomize failed: {_e!r}\n")
