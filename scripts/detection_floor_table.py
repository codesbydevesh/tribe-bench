"""Detection floors for the Gate 0 v3b design, under each candidate statistic (S1, D-3).

"No verdict without a floor." The 2026-07-31 run reported NO-GO without ever stating
the smallest FFCr effect it could have detected. This computes that number — for the
retracted statistic and for each of its replacements — so every future verdict can be
read as "no effect larger than X", not "no effect".

Method. Reuse the synthetic brain from scripts/compositional_demo.py (auditory mass
above visual, per-clip global gain, condition-independent noise) at the auditory drive
difference selected by scripts/sensitivity_surface.py -- an AVERAGED objective over many
seeded draws per grid point, on a grid widened well past the old boundary, using seeds
DISJOINT from those used to report. The previous value (0.24) was an argmin over 25
single noisy draws and was the last point of its own grid.
Inject a GENUINE additive FFCr effect of known size on FACE clips only, run the design's
own one-sided permutation test at alpha=0.025, and repeat to estimate power. The floor
is the effect at 80% power, linearly interpolated between grid points.

Units are RAW predicted-response units throughout, so the four statistics are directly
comparable: the question asked of each is "how big must the true FFCr effect be before
THIS statistic detects it at this n?"

CPU only, no model, no GPU. Deterministic (seeded). ~2-4 minutes.

Writes: data/floor_table_v3b.md
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.compositional_demo import ROIS, build_brain, clip_map  # noqa: E402
from tribe_tools.roi_stats import (  # noqa: E402
    glm_contrast_z,
    perm_p,
    raw_roi_mean,
    roi_minus_reference,
    spatial_z,
)

# D_AUD and RHO are inputs, not constants of nature. Both are read from the
# selection artifact so no boundary-selected literal can re-enter the pipeline.
_SEL = Path(__file__).resolve().parent.parent / "data" / "sensitivity_selection.json"


def _selected_d_aud():
    """Read the selected D_AUD, or fail loudly. Never fall back to a literal.

    Deliberately lazy: raising at import time would make the module
    un-importable (breaking test collection and introspection), but silently
    defaulting to the old boundary-selected 0.24 would reintroduce the bug.
    """
    if env := os.environ.get("FLOOR_D_AUD"):
        return float(env)
    if not _SEL.exists():
        raise FileNotFoundError(
            f"{_SEL} not found -- run `python3 scripts/sensitivity_surface.py` first. "
            "Refusing to fall back to the old boundary-selected D_AUD=0.24."
        )
    return float(json.loads(_SEL.read_text())["selection"]["d_best"])


D_AUD = None                                     # resolved in main(); see _selected_d_aud
RHO = float(os.environ.get("FLOOR_RHO", 0.0))    # within-parcel noise correlation
N_PER_GROUP = 15    # the v3b design as run
ALPHA = 0.025       # the pre-registered one-sided level
POWER = 0.80
N_SIM = int(os.environ.get("FLOOR_N_SIM", 2000))  # ~+-1% per point at 2000
N_PERM = 500
EFFECTS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)


def one_experiment(base, idx, ref, face_effect, rng):
    """Simulate one v3b-shaped experiment; return per-statistic one-sided p-values."""
    ffc = idx["FFCr"]
    face, nonface = [], []
    for _ in range(N_PER_GROUP):
        a_f = 0.30 + D_AUD + rng.normal(0, 0.10)
        a_n = 0.30 + rng.normal(0, 0.10)
        pf = clip_map(base, idx, a_f, seed=int(rng.integers(1 << 31)), rho=RHO)
        if face_effect:
            pf[:, ffc] += face_effect
        face.append(pf)
        nonface.append(clip_map(base, idx, a_n, seed=int(rng.integers(1 << 31)), rho=RHO))

    seed = int(rng.integers(1 << 31))
    out = {}
    for name, fn in (
        ("spatial_z", lambda p: spatial_z(p, ffc)),
        ("raw_roi_mean", lambda p: raw_roi_mean(p, ffc)),
        ("roi_minus_reference", lambda p: roi_minus_reference(p, ffc, ref)),
    ):
        out[name] = perm_p([fn(p) for p in face], [fn(p) for p in nonface],
                           n_perm=N_PERM, seed=seed)

    # glm_contrast_z is a single across-clip number, not a per-clip value, so its
    # permutation null shuffles the clip labels and recomputes it each time.
    fa = np.array([p.mean(axis=0) for p in face])
    fb = np.array([p.mean(axis=0) for p in nonface])
    obs = glm_contrast_z(fa, fb, ffc)
    # Slice the ROI ONCE. glm_contrast_z only ever reads `verts` columns, so
    # permuting the pre-sliced (n_clips, |ROI|) array is exactly equivalent and
    # avoids copying a (15, 20484) array N_PERM times per simulated experiment.
    pool_roi = np.concatenate([fa[:, ffc], fb[:, ffc]], axis=0)
    roi_all = np.arange(pool_roi.shape[1])
    prng = np.random.default_rng(seed)
    ge = 0
    for _ in range(N_PERM):
        perm = prng.permutation(pool_roi.shape[0])
        if glm_contrast_z(pool_roi[perm[:N_PER_GROUP]],
                          pool_roi[perm[N_PER_GROUP:]], roi_all) >= obs - 1e-12:
            ge += 1
    out["glm_contrast_z"] = (ge + 1) / (N_PERM + 1)
    return out


def interpolate_floor(effects, powers, target=POWER):
    """First crossing of `target`, linearly interpolated. None if never reached."""
    for i in range(1, len(effects)):
        if powers[i] >= target > powers[i - 1]:
            span = powers[i] - powers[i - 1]
            if span <= 0:
                return effects[i]
            frac = (target - powers[i - 1]) / span
            return effects[i - 1] + frac * (effects[i] - effects[i - 1])
    return None


def main():
    global D_AUD
    D_AUD = _selected_d_aud()
    base, idx = build_brain()
    ref = idx["REST"][:2000]
    stats = ("spatial_z", "raw_roi_mean", "roi_minus_reference", "glm_contrast_z")
    power = {s: [] for s in stats}

    print(f"v3b design: n={N_PER_GROUP} per group, alpha={ALPHA}, "
          f"auditory drive delta={D_AUD}, rho={RHO}, "
          f"{N_SIM} sims x {N_PERM} perms per point")
    print()
    header = f"{'true FFCr effect':>16} | " + " ".join(f"{s:>20}" for s in stats)
    print(header)
    print("-" * len(header))

    for eff in EFFECTS:
        rng = np.random.default_rng(20260820)
        hits = {s: 0 for s in stats}
        for _ in range(N_SIM):
            ps = one_experiment(base, idx, ref, eff, rng)
            for s in stats:
                if ps[s] <= ALPHA:
                    hits[s] += 1
        for s in stats:
            power[s].append(hits[s] / N_SIM)
        print(f"{eff:>16.3f} | " + " ".join(f"{power[s][-1]:>20.2f}" for s in stats))

    floors = {s: interpolate_floor(EFFECTS, power[s]) for s in stats}
    print()
    print("DETECTION FLOOR (minimum detectable effect at 80% power, raw units)")
    for s in stats:
        f = floors[s]
        print(f"  {s:>22}: " + (f"{f:.4f}" if f is not None else f"> {EFFECTS[-1]:.2f} (never reached)"))

    # Never overwrite the canonical artifact from a parameter sweep. Only a run at the
    # SELECTED D_AUD with rho=0 may write floor_table_v3b.md; anything else is tagged.
    tag = "" if (D_AUD == _selected_d_aud() and RHO == 0.0) else f"_daud{D_AUD:g}_rho{RHO:g}"
    out = Path(__file__).resolve().parent.parent / "data" / f"floor_table_v3b{tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Detection floors — Gate 0 v3b design",
        "",
        f"Generated by `scripts/detection_floor_table.py` (seeded, CPU, no model). "
        f"n={N_PER_GROUP} per group, alpha={ALPHA} one-sided, auditory drive difference "
        f"{D_AUD} (selected, not boundary-picked), within-parcel noise correlation "
        f"rho={RHO}, {N_SIM} simulations x {N_PERM} permutations per point.",
        "",
        "The question asked of each statistic: **how large must a genuine FFCr effect be "
        "before this statistic detects it at 80% power, in this design?** Units are raw "
        "predicted-response units, so the columns are directly comparable.",
        "",
        "## Power curves",
        "",
        "| true FFCr effect | " + " | ".join(f"`{s}`" for s in stats) + " |",
        "|---|" + "---|" * len(stats),
    ]
    for i, eff in enumerate(EFFECTS):
        lines.append(f"| {eff:.3f} | " + " | ".join(f"{power[s][i]:.2f}" for s in stats) + " |")
    lines += [
        "",
        "## Floors (MDE at 80% power)",
        "",
        "| statistic | detection floor |",
        "|---|---|",
    ]
    for s in stats:
        f = floors[s]
        lines.append(f"| `{s}` | " + (f"**{f:.4f}**" if f is not None
                                      else f"**> {EFFECTS[-1]:.2f}** (not reached)") + " |")
    lines += [
        "",
        "## How to read this",
        "",
        "A NO-GO from a statistic only means \"no effect larger than its floor\". "
        "`spatial_z` is the retracted primary (G020, D027) — its floor is the number that "
        "makes the 2026-07-31 verdict uninterpretable. Any verdict reported without the "
        "corresponding floor beside it violates doctrine **D-3**.",
        "",
        "Caveat, stated plainly: these floors are computed on a **synthetic** brain whose "
        "mass structure is modelled on the printed top-k ROIs, not on the real prediction "
        "cache. They are correct for the *design*; the constant relating them to real "
        "TRIBE units is unverified until S2 produces real predictions.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
