"""Phase A: replace the single-draw, boundary-selected point estimate with a
reproducible sensitivity analysis over (D_AUD, rho).

WHAT WAS WRONG. `compositional_demo.main()` chose D_AUD by
`argmin over np.arange(0.0, 0.25, 0.01)` of the squared error against the observed
2026-07-31 pattern, evaluating ONE draw per grid point from a module-level mutable
RNG. At a FIXED d_aud the objective's input varies by sd 0.033 in FFCr z_d
(range -0.279..-0.167) -- far larger than the effect of one 0.01 grid step. The
argmin therefore selected noise. Its winner, 0.24, is also the LAST point of that
grid: a boundary solution, the signature of a search that wanted to go further.
That selected D_AUD was then reused to report performance, so every downstream
number was conditioned on a parameter fitted to the same draws.

WHAT THIS DOES INSTEAD.
  1. Averages N draws per grid point BEFORE any argmin, and reports the standard
     error, so "is this minimum real?" is answerable rather than assumed.
  2. Widens the grid well past the old boundary and states explicitly whether the
     new optimum is interior. A boundary optimum is surfaced, never hidden.
  3. Uses DISJOINT seed sets for selection and reporting: no parameter is chosen
     using a draw later used to quote performance.
  4. Sweeps rho, the within-parcel correlation of the prediction noise. rho=0 is
     not a neutral default -- TRIBE's head is nn.Linear(hidden, low_rank_head=2048)
     (tribev2/model.py:139-141, grids/defaults.py:198), so its 20,484-vertex output
     has rank <= 2048 and spatially independent noise is structurally impossible.

CPU only, deterministic, no GPU. Writes data/sensitivity_surface.md + .json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.compositional_demo import OBSERVED, run  # noqa: E402

# Disjoint by construction: SELECT fits D_AUD, REPORT quotes results.
SELECT_SEEDS = tuple(range(0, 225))
REPORT_SEEDS = tuple(range(100_000, 100_225))
assert not set(SELECT_SEEDS) & set(REPORT_SEEDS), "seed sets must be disjoint"

D_AUD_GRID = tuple(round(float(x), 2) for x in np.arange(0.0, 0.66, 0.05))
RHO_GRID = (0.0, 0.3, 0.6, 0.9)
OLD_D_AUD, OLD_GRID_MAX = 0.24, 0.24
ROIS_REPORTED = ("A1", "V1", "FFCr", "EBA")


def _mean_sem(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size))


def objective(d_aud, seeds, rho=0.0):
    """Mean squared error vs the observed pattern, averaged over `seeds`."""
    errs = []
    for s in seeds:
        # NOTE: call run() ONCE per seed. Putting it inside the `for k in OBSERVED`
        # generator evaluates it four times (same seed, same answer, 4x the work).
        r = run(float(d_aud), seed=int(s), rho=rho, compute_p=False)
        errs.append(sum((r[k]["z_d"] - OBSERVED[k]) ** 2 for k in OBSERVED))
    m, sem = _mean_sem(errs)
    return {"d_aud": float(d_aud), "mean": m, "sem": sem, "n": len(errs)}


def select_d_aud(seeds, rho=0.0, grid=D_AUD_GRID):
    """Averaged-objective selection, with boundary and separation checks."""
    curve = [objective(d, seeds, rho) for d in grid]
    best = min(curve, key=lambda r: r["mean"])
    rest = [r for r in curve if r["d_aud"] != best["d_aud"]]
    runner = min(rest, key=lambda r: r["mean"])
    # Two-sample z between the winner and its nearest rival, on independent means.
    # The threshold MUST account for the number of looks: picking the minimum of k+1
    # noisy means is k implicit comparisons, so an uncorrected 2-sigma rule would
    # call a noise minimum "resolved". Bonferroni at family-wise 0.05.
    sep = (runner["mean"] - best["mean"]) / float(np.hypot(best["sem"], runner["sem"]))
    k = len(grid) - 1
    z_crit = float(NormalDist().inv_cdf(1 - 0.05 / (2 * k)))
    # every grid point not separated from the winner at z_crit is indistinguishable
    band = sorted(r["d_aud"] for r in curve
                  if (r["mean"] - best["mean"]) / float(np.hypot(best["sem"], r["sem"])) < z_crit)
    return {
        "curve": curve,
        "d_best": best["d_aud"],
        "on_boundary": best["d_aud"] in (grid[0], grid[-1]),
        "runner_up": runner["d_aud"],
        "separation_sigma": sep,
        "n_comparisons": k,
        "z_critical_bonferroni": z_crit,
        "distinguishable": sep >= z_crit,
        "indistinguishable_band": band,
    }


def report_pattern(d_aud, seeds, rho=0.0):
    """Per-ROI z_d, mean +/- sd over `seeds`. Reporting only -- never used to fit."""
    per = {k: [] for k in ROIS_REPORTED}
    for s in seeds:
        r = run(float(d_aud), seed=int(s), rho=rho, compute_p=False)
        for k in ROIS_REPORTED:
            per[k].append(r[k]["z_d"])
    out = {}
    for k, vals in per.items():
        a = np.asarray(vals)
        out[k] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                  "sem": float(a.std(ddof=1) / np.sqrt(a.size)),
                  "observed": OBSERVED[k], "n": a.size}
    return out


def main():
    n_sel, n_rep = len(SELECT_SEEDS), len(REPORT_SEEDS)
    print(f"selection seeds: {n_sel} (disjoint from {n_rep} reporting seeds)")
    print(f"D_AUD grid: {D_AUD_GRID[0]}..{D_AUD_GRID[-1]} step 0.05  "
          f"(old grid stopped at {OLD_GRID_MAX})\n")

    sel = select_d_aud(SELECT_SEEDS, rho=0.0)
    print(f"{'d_aud':>6} {'objective':>11} {'sem':>8}")
    for r in sel["curve"]:
        mark = "  <-- min" if r["d_aud"] == sel["d_best"] else ""
        old = "   (old boundary)" if r["d_aud"] == OLD_GRID_MAX else ""
        print(f"{r['d_aud']:>6.2f} {r['mean']:>11.5f} {r['sem']:>8.5f}{mark}{old}")
    print(f"\nselected D_AUD = {sel['d_best']}   (old, boundary-selected: {OLD_D_AUD})")
    print(f"on boundary of the widened grid: {sel['on_boundary']}")
    print(f"nearest rival {sel['runner_up']}, separated by "
          f"{sel['separation_sigma']:.2f} sigma -> "
          f"{'distinguishable' if sel['distinguishable'] else 'NOT distinguishable from noise'}")

    print("\nreported pattern at the selected D_AUD, on the DISJOINT reporting seeds:")
    rep = report_pattern(sel["d_best"], REPORT_SEEDS, rho=0.0)
    print(f"{'ROI':>6} {'sim mean':>10} {'sd':>8} {'sem':>8} {'observed':>10}")
    for k in ROIS_REPORTED:
        v = rep[k]
        print(f"{k:>6} {v['mean']:>+10.4f} {v['sd']:>8.4f} {v['sem']:>8.4f} {v['observed']:>+10.3f}")

    out = {"select_seeds": [SELECT_SEEDS[0], SELECT_SEEDS[-1]],
           "report_seeds": [REPORT_SEEDS[0], REPORT_SEEDS[-1]],
           "selection": sel, "reported_pattern": rep}
    Path("data").mkdir(exist_ok=True)
    Path("data/sensitivity_selection.json").write_text(json.dumps(out, indent=2))
    print("\nwritten: data/sensitivity_selection.json")


if __name__ == "__main__":
    main()
