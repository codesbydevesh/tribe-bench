"""Is TRIBE v2's learned absolute position embedding actually trained, or still random noise?

This is a ten-minute, zero-GPU, no-torch kill test. It decides whether one proposed flagship
("sweep a temporal-aperture knob built on the position embedding") rests on a live parameter or a
dead one. Answer, found 2026-08-10: it is dead. The model ignores absolute position and relies on
rotary (relative) position instead.

WHY THIS IS THE RIGHT TEST. `model.time_pos_embed` is created as `torch.randn(1, max_seq_len,
hidden)` (tribev2/model.py:151-152), i.e. std exactly 1.0 at init. So "std ~ 1.0 after training"
is ambiguous -- it could mean used-and-kept-at-unit-scale, or never-moved-from-noise. MAGNITUDE
cannot separate those. STRUCTURE can:

  * A learned position code is SMOOTH: adjacent slots become correlated (cos > 0), correlation
    decays with slot distance, and the used slots (a 100-TR window uses ~100 of 1024) differentiate
    from the untouched tail.
  * Random init has NONE of that: adjacent-slot cosine ~ 0 (reference sd = 1/sqrt(dim)), flat norm
    across all slots, no decay with distance.

RESULT (facebook/tribev2 best.ckpt, 708,856,138 bytes):
  * per-slot L2 norm flat at ~33.9 = sqrt(1152) across ALL 1024 slots, used region included
  * adjacent-slot cosine in [-0.004, +0.001] everywhere vs a random reference sd of 0.0295
  * cosine-vs-distance flat at ~0 for distances 1..128
  -> statistically indistinguishable from its own random initialization on three independent
     structural measures. The additive absolute position embedding is untrained.

WHAT SURVIVES. This does NOT kill the underlying research question. The load-bearing fact for
"the model's prediction at within-window position p uses future stimulus" is that the encoder is
MASKLESS/bidirectional (config.yaml: causal=false -> x_transformers Encoder, not Decoder), which
is independent of this embedding. The lookahead experiment is therefore done by truncating/masking
FUTURE INPUTS, not by twiddling this dead knob. See position_embed_dead.md.

Usage:  python3 probe_position_embed.py --ckpt /path/to/best.ckpt
No torch, no GPU. Parses the checkpoint zip directly.
"""

import argparse
import collections
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np

STORAGE_DTYPE = {"FloatStorage": "<f4", "HalfStorage": "<f2", "DoubleStorage": "<f8"}


class _Tracer(pickle.Unpickler):
    """Reads a torch checkpoint's pickle for tensor -> (storage key, dtype, shape), no torch."""

    def persistent_load(self, pid):
        return {
            "key": pid[2],
            "dtype": pid[1].__name__ if hasattr(pid[1], "__name__") else str(pid[1]),
        }

    def find_class(self, module, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            def rebuild(storage, offset, size, stride, *a):
                return {"key": storage["key"], "dtype": storage["dtype"], "size": tuple(size)}
            return rebuild
        if name == "OrderedDict":
            return collections.OrderedDict
        if name.endswith("Storage"):
            return type(name, (), {})  # keep __name__ for the dtype lookup
        return type("G", (), {"__init__": lambda s, *a, **k: None,
                              "__setstate__": lambda s, st: None})


def _find(d, needle, prefix=""):
    for k, v in (d.items() if isinstance(d, dict) else []):
        key = prefix + str(k)
        if isinstance(v, dict) and "key" in v and needle in key:
            return key, v
        if isinstance(v, dict):
            hit = _find(v, needle, key + ".")
            if hit:
                return hit
    return None


def load_tensor(ckpt_path: Path, needle: str = "time_pos_embed") -> tuple[str, np.ndarray]:
    """Return (name, ndarray) for the first tensor whose key contains `needle`."""
    zf = zipfile.ZipFile(ckpt_path)
    root = zf.namelist()[0].split("/")[0] + "/"
    meta_obj = _Tracer(io.BytesIO(zf.read(root + "data.pkl"))).load()
    state = meta_obj.get("state_dict", meta_obj) if isinstance(meta_obj, dict) else meta_obj
    hit = _find(state, needle)
    if not hit:
        raise KeyError(f"no tensor matching {needle!r} in {ckpt_path}")
    name, meta = hit
    dtype = STORAGE_DTYPE.get(meta["dtype"])
    raw = zf.read(root + "data/" + str(meta["key"]))
    if meta["dtype"] == "BFloat16Storage":
        u16 = np.frombuffer(raw, "<u2")
        arr = ((u16.astype(np.uint32)) << 16).view(np.float32)
    elif dtype is None:
        raise ValueError(f"unhandled storage dtype {meta['dtype']}")
    else:
        arr = np.frombuffer(raw, dtype).astype(np.float32)
    n = int(np.prod(meta["size"]))
    return name, arr[:n].reshape(meta["size"])


def structural_verdict(embed: np.ndarray, used_window: int = 100) -> dict:
    """Three structure tests that separate a trained position code from random init."""
    E = embed[0] if embed.ndim == 3 else embed  # (slots, dim)
    slots, dim = E.shape
    unit = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    adj_cos = np.sum(unit[:-1] * unit[1:], axis=1)
    norm = np.linalg.norm(E, axis=1)
    rand_ref_sd = 1.0 / np.sqrt(dim)  # sd of cos between two iid gaussian vectors in `dim` dims

    used = slice(0, min(used_window, slots - 1))
    tail = slice(min(used_window, slots - 1), slots - 1)
    dist_cos = {}
    reg = unit[: used_window + 60]
    for d in (1, 2, 4, 8, 16, 32, 64, 128):
        if d < reg.shape[0]:
            dist_cos[d] = float(np.mean(np.sum(reg[:-d] * reg[d:], axis=1)))

    used_adj = float(np.mean(np.abs(adj_cos[used])))
    trained = used_adj > 5 * rand_ref_sd  # 5-sigma above the random-init floor
    return {
        "slots": slots, "dim": dim,
        "norm_mean_all": float(norm.mean()),
        "norm_sqrt_dim": float(np.sqrt(dim)),
        "adj_cos_used_absmean": used_adj,
        "adj_cos_tail_absmean": float(np.mean(np.abs(adj_cos[tail]))),
        "random_ref_sd": float(rand_ref_sd),
        "dist_cos": dist_cos,
        "verdict": "TRAINED (used position code)" if trained
                   else "UNTRAINED (indistinguishable from random init) -- the model ignores it",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True, help="path to facebook/tribev2 best.ckpt")
    ap.add_argument("--window", type=int, default=100, help="TRs a real window uses (duration_trs)")
    args = ap.parse_args()

    name, embed = load_tensor(args.ckpt)
    print(f"tensor {name}  shape {embed.shape}")
    v = structural_verdict(embed, used_window=args.window)
    print(f"\nnorm: mean {v['norm_mean_all']:.2f} vs sqrt(dim) {v['norm_sqrt_dim']:.2f} "
          f"({'MATCHES random-init scale' if abs(v['norm_mean_all']-v['norm_sqrt_dim'])<1 else 'departs from init'})")
    print(f"adjacent-slot |cos|, used region : {v['adj_cos_used_absmean']:.4f}")
    print(f"adjacent-slot |cos|, tail        : {v['adj_cos_tail_absmean']:.4f}")
    print(f"random-init reference sd         : {v['random_ref_sd']:.4f}")
    print("cosine vs slot-distance:")
    for d, c in v["dist_cos"].items():
        print(f"  {d:3d}: {c:+.4f}")
    print(f"\nVERDICT: {v['verdict']}")


if __name__ == "__main__":
    main()
