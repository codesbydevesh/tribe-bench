"""Can spatial_z manufacture Gate 0 v3b's observed pattern with ZERO face information?

Observed 2026-07-31 (FACE vs NONFACE, 15v15, within-film):
    A1   d = +0.280   (auditory)
    V1   d = -0.046   (523 vtx, sits near the brain average)
    FFCr d = -0.244   ( 58 vtx)
    EBA  d = -0.382   (116 vtx)

spatial_z(clip) = (roi_mean - brain_mean) / brain_sd, so every clip's z-map has
mean exactly 0 and sd exactly 1. Two nuisance channels therefore exist:

  mu-shift:  brain_mean rises  -> every unchanged ROI's z falls by the SAME amount.
  sd-shift:  brain_sd rises    -> every z SHRINKS TOWARD ZERO in proportion to |z|,
                                  so ROIs far ABOVE the average fall the most.

The observed pattern is ordered by distance from the brain average (EBA < FFC < V1 < 0
in delta, A1 alone positive). That is the sd-shift signature, not the mu-shift signature.
FACE clips carry 18% more speech (23.3 vs 19.8 words, p=0.157), and the printed top-k
says TRIBE's output mass on this material is auditory/STS-dominated -- so more speech
means more auditory mass means larger spatial sd.

This script injects ZERO face-category information anywhere, varies only the auditory
drive between conditions, and asks whether spatial_z reproduces the observed pattern.

CPU only, no model, no GPU. Uses the project's own statistics module unmodified.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tribe_tools.roi_stats import perm_p, u_statistic  # noqa: E402

N_VERTS = 20484  # fsaverage5, both hemispheres
DEFAULT_SEED = 0

# Permutation settings. perm_p returns the MC estimator (ge+1)/(n_perm+1), whose
# FLOOR is 1/(n_perm+1). A printed "0.0005" at n_perm=2000 is that floor -- i.e.
# "p < 5e-4" -- and is NOT a measured value. fmt_p() below enforces that wording.
PERM_N = 2000
P_FLOOR = 1.0 / (PERM_N + 1)

# Parcels that PARTITION the cortex here (A1 lives inside AUD, so it is not listed).
# Used to give the noise within-parcel correlation; see _parcel_noise.
_PARCELS = ("AUD", "V1", "FFCr", "EBA", "REST")

# ROI sizes as used by Gate 0 (from the record: FFC 58, EBA proxy 116, V1 523).
ROIS = {"A1": 70, "V1": 523, "FFCr": 58, "EBA": 116}

OBSERVED = {"A1": +0.280, "V1": -0.046, "FFCr": -0.244, "EBA": -0.382}


def fmt_p(p, n_perm=PERM_N):
    """Render a Monte-Carlo permutation p-value with correct floor semantics.

    At the estimator floor the only honest statement is an upper bound: the
    experiment cannot distinguish p = floor from p = 0. Reporting the floor as if
    it were a point estimate ("p=0.0005") overstates precision.
    """
    floor = 1.0 / (n_perm + 1)
    return f"<{floor:.1e}" if p <= floor * (1 + 1e-9) else f"{p:.4f}"


def run_many(delta_aud, seeds, *, rho=0.0, face_effect=0.0, n=15, compute_p=False):
    """Aggregate `run` across independent seeds. THE reporting entry point.

    Returns {roi: {stat: {"mean","sd","sem","n"}}}. Every published figure from
    this module must come from here, not from a single `run()` call: at a fixed
    setting the single-draw sd of FFCr z_d is 0.033, so one draw is not a result.
    """
    acc = {}
    for s in seeds:
        r = run(float(delta_aud), n=n, face_effect=face_effect,
                seed=int(s), rho=rho, compute_p=compute_p)
        for roi, d in r.items():
            for stat, val in d.items():
                acc.setdefault(roi, {}).setdefault(stat, []).append(float(val))
    out = {}
    for roi, stats in acc.items():
        out[roi] = {}
        for stat, vals in stats.items():
            a = np.asarray(vals, dtype=float)
            entry = {"mean": float(a.mean()),
                     "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                     "sem": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0,
                     "n": int(a.size)}
            if stat.endswith("_p"):
                # A MEAN P-VALUE IS NOT A P-VALUE. The interpretable summary of a
                # p-value across independent replications is the rejection RATE
                # (empirical power at the stated alpha). `mean` is retained only
                # for diagnostics; never publish it as "the p-value".
                entry["reject_rate_025"] = float((a <= 0.025).mean())
                entry["reject_rate_05"] = float((a <= 0.05).mean())
            out[roi][stat] = entry
    return out


def build_brain(rng=None, *, seed=DEFAULT_SEED):
    """A baseline predicted-response map with realistic mass structure.

    Deterministic: pass an ``rng`` (to share a stream) or a ``seed``. Previously
    this consumed a module-level mutable RNG, so two calls returned different
    brains (max |diff| 0.17) and every published number from this module was a
    single unreproducible draw.

    Auditory/STS carries the most output mass (matches the printed top-k ROIs:
    A5, STSdp, TPOJ1, A4, PBelt, LBelt, STSvp, STSda, 55b, A1). Visual sits above
    the whole-brain average but below auditory. Everything else is low.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    idx, cursor = {}, 0
    # auditory/STS mass (the normaliser's owner) -- ~2000 vertices
    idx["AUD"] = np.arange(cursor, cursor + 2000)
    cursor += 2000
    for name, size in ROIS.items():
        if name == "A1":
            idx[name] = idx["AUD"][:size]  # A1 lives inside the auditory mass
            continue
        idx[name] = np.arange(cursor, cursor + size)
        cursor += size
    idx["REST"] = np.arange(cursor, N_VERTS)

    base = np.full(N_VERTS, 0.20)          # low-drive default
    base[idx["AUD"]] = 1.00                # auditory/STS: the output mass
    base[idx["V1"]] = 0.42                 # visual, modestly above the brain average
    base[idx["FFCr"]] = 0.62               # higher-order visual, further above
    base[idx["EBA"]] = 0.72                # further still
    base += rng.normal(0, 0.03, N_VERTS)   # spatial texture
    return base, idx


def _parcel_noise(rng, idx, sigma, rho):
    """Noise with within-parcel correlation `rho`, MARGINAL VARIANCE HELD FIXED.

    ``x_i = sqrt(1-rho)*eps_i + sqrt(rho)*c_P`` for vertex i in parcel P, with
    ``eps_i, c_P ~ N(0, sigma)`` independent. Then ``Var(x_i) = sigma**2`` exactly
    and ``Corr(x_i, x_j) = rho`` within a parcel, 0 across parcels.

    Why rho=0 is a stipulation, not a neutral default. TRIBE v2's head is
    ``nn.Linear(hidden, low_rank_head=2048)`` (``tribev2/model.py:139-141``,
    ``grids/defaults.py:198``), so the 20,484-vertex output is a linear image of a
    2048-dim latent: **rank <= 2048**. Spatially independent noise across 20,484
    vertices is therefore structurally impossible for this model. That much is
    verified from source. The *magnitude* of rho's effect on each statistic's
    detection floor is measured, not assumed -- see ``data/sensitivity_surface.md``;
    do not quote a ratio here that this module has not itself produced.
    """
    eps = rng.normal(0.0, sigma, N_VERTS)
    if rho <= 0.0:
        return eps
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    out = np.sqrt(1.0 - rho) * eps
    for name in _PARCELS:
        out[idx[name]] += np.sqrt(rho) * rng.normal(0.0, sigma)
    return out


def clip_map(base, idx, aud_drive, seed, rho=0.0):
    """One clip's (n_segments, n_vertices) prediction. NO face information anywhere.

    aud_drive is the only thing that differs between conditions. FFCr, V1 and EBA
    receive identical distributions in both conditions by construction.
    """
    rng = np.random.default_rng(seed)
    g = base.copy()
    g[idx["AUD"]] *= 1.0 + aud_drive           # speech -> auditory mass
    g += _parcel_noise(rng, idx, 0.05, rho)    # per-clip noise, condition-independent
    g *= 1.0 + rng.normal(0, 0.04)             # per-clip global gain, condition-independent
    return np.repeat(g[None, :], 11, axis=0)   # ~11 kept TRs for a 10 s clip


def spatial_z(preds, verts):
    """The project's statistic, inlined so the mu/sd channels stay visible."""
    g = preds.mean(axis=0)
    return float((g[verts].mean() - g.mean()) / g.std())


def raw_roi_mean(preds, verts):
    """Non-compositional alternative: the ROI's predicted response, unnormalised."""
    return float(preds.mean(axis=0)[verts].mean())


def roi_minus_reference(preds, verts, ref):
    """Non-compositional alternative: ROI minus a pre-registered off-target reference."""
    g = preds.mean(axis=0)
    return float(g[verts].mean() - g[ref].mean())


def run(delta_aud, n=15, face_effect=0.0, *, seed=DEFAULT_SEED, rho=0.0,
        compute_p=True, verbose=False):
    """delta_aud = the FACE-minus-NONFACE difference in auditory drive.

    face_effect = a GENUINE additive face response injected into FFCr on FACE clips
    only (raw units). This is the case that decides whether the NO-GO is safe to act
    on: can spatial_z report a negative FFCr delta while a real positive effect is
    present in the data?
    """
    rng = np.random.default_rng(seed)
    base, idx = build_brain(rng)
    ref = idx["REST"][:2000]  # low-drive off-target reference region

    face, nonface = [], []
    for _ in range(n):
        # condition-independent scatter in speech drive, plus the condition offset
        a_f = 0.30 + delta_aud + rng.normal(0, 0.10)
        a_n = 0.30 + rng.normal(0, 0.10)
        # clip seeds derive from the run stream, so distinct `seed` values give
        # genuinely independent realizations (the old fixed 1000+i / 2000+i reused
        # identical clip noise across every draw).
        pf = clip_map(base, idx, a_f, seed=int(rng.integers(1 << 31)), rho=rho)
        if face_effect:
            pf[:, idx["FFCr"]] += face_effect
        face.append(pf)
        nonface.append(clip_map(base, idx, a_n, seed=int(rng.integers(1 << 31)), rho=rho))

    out = {}
    for name in ROIS:
        v = idx[name]
        zf = [spatial_z(p, v) for p in face]
        zn = [spatial_z(p, v) for p in nonface]
        rf = [raw_roi_mean(p, v) for p in face]
        rn = [raw_roi_mean(p, v) for p in nonface]
        df = [roi_minus_reference(p, v, ref) for p in face]
        dn = [roi_minus_reference(p, v, ref) for p in nonface]
        out[name] = {
            "z_d": np.mean(zf) - np.mean(zn),
            "z_U": u_statistic(zf, zn),
            "z_base": float(np.mean(zf + zn)),
            "raw_d": np.mean(rf) - np.mean(rn),
            "ref_d": np.mean(df) - np.mean(dn),
        }
        if compute_p:
            # NOTE: perm_p returns the MC estimator (ge+1)/(n_perm+1); with
            # n_perm=2000 its FLOOR is 1/2001 = 4.998e-4. A printed "0.0005" is
            # that floor, i.e. "p < 5e-4", not a measured value.
            out[name].update({
                "z_p": perm_p(zf, zn, 2000, 0),
                "raw_p": perm_p(rf, rn, 2000, 0),
                "ref_p": perm_p(df, dn, 2000, 0),
            })

    # the normaliser itself -- diagnostic #3 in the parked Kaggle code
    mu_f = [p.mean(axis=0).mean() for p in face]
    mu_n = [p.mean(axis=0).mean() for p in nonface]
    sd_f = [p.mean(axis=0).std() for p in face]
    sd_n = [p.mean(axis=0).std() for p in nonface]
    out["_norm"] = {
        "mu_f": np.mean(mu_f), "mu_n": np.mean(mu_n),
        "sd_f": np.mean(sd_f), "sd_n": np.mean(sd_n),
        "sd_ratio": np.mean(sd_f) / np.mean(sd_n),
    }
    if compute_p:
        out["_norm"]["mu_p"] = perm_p(mu_f, mu_n, 2000, 0)
        out["_norm"]["sd_p"] = perm_p(sd_f, sd_n, 2000, 0)
    return out


SELECTION_JSON = Path(__file__).resolve().parent.parent / "data" / "sensitivity_selection.json"


def load_selected_d_aud():
    """The D_AUD chosen by the averaged, disjoint-seed procedure.

    Deliberately raises rather than falling back to a literal. The old code chose
    D_AUD by an argmin over 25 SINGLE noisy draws on a grid whose last point was
    the winner (0.24) -- a boundary solution selected on noise, then reused to
    report performance. Hard-coding any replacement constant here would recreate
    exactly that: a number with no visible provenance.
    """
    if not SELECTION_JSON.exists():
        raise FileNotFoundError(
            f"{SELECTION_JSON} not found. Run `python3 scripts/sensitivity_surface.py` "
            "first; it selects D_AUD by averaging many draws per grid point over a "
            "widened grid, using seeds disjoint from those used for reporting."
        )
    sel = json.loads(SELECTION_JSON.read_text())["selection"]
    return float(sel["d_best"]), sel


def main(n_report_seeds=40):
    sel_d_aud, sel = load_selected_d_aud()
    report_seeds = range(100_000, 100_000 + n_report_seeds)

    print("=" * 78)
    print("SWEEP: FACE-minus-NONFACE auditory drive difference -> spatial_z deltas")
    print(f"(zero face information injected at any delta; mean +/- sd over "
          f"{n_report_seeds} seeds)")
    print("=" * 78)
    print(f"{'d_aud':>7} | {'A1':>16} {'V1':>16} {'FFCr':>16} {'EBA':>16}")
    print("-" * 78)
    for d_aud in (0.0, 0.05, 0.15, 0.30, 0.45, 0.60):
        agg = run_many(d_aud, report_seeds)
        cells = " ".join(f"{agg[k]['z_d']['mean']:>+8.3f}+/-{agg[k]['z_d']['sd']:<6.3f}"
                         for k in ("A1", "V1", "FFCr", "EBA"))
        print(f"{d_aud:>7.2f} | {cells}")

    print()
    print("OBSERVED 2026-07-31 (real run, not simulated):  "
          + "  ".join(f"{k} {OBSERVED[k]:+.3f}" for k in ("A1", "V1", "FFCr", "EBA")))
    print()
    print("=" * 78)
    print(f"SELECTED D_AUD = {sel_d_aud}   (averaged objective, {sel['curve'][0]['n']} draws/point,")
    print(f"  grid {sel['curve'][0]['d_aud']}..{sel['curve'][-1]['d_aud']}, "
          f"on boundary: {sel['on_boundary']}, "
          f"separation from nearest rival {sel['separation_sigma']:.2f} sigma)")
    if sel["on_boundary"]:
        print("  !! WARNING: the optimum sits on the grid boundary. Widen the grid.")
    if not sel.get("distinguishable", True):
        band = sel.get("indistinguishable_band", [])
        print(f"  !! The optimum is NOT resolved once the {sel.get('n_comparisons','?')} implicit")
        print(f"     comparisons are accounted for (needs {sel.get('z_critical_bonferroni',0):.2f} "
              f"sigma, has {sel['separation_sigma']:.2f}).")
        print(f"     Statistically indistinguishable: D_AUD in {band}.")
        print("     Treat D_AUD as a STIPULATED nuisance level, not a fitted optimum.")
    print("=" * 78)

    agg = run_many(sel_d_aud, report_seeds)
    print(f"{'ROI':>6} | {'sim mean':>10} {'sd':>8} {'observed':>10}   (n={n_report_seeds} seeds)")
    print("-" * 78)
    for name in ("A1", "V1", "FFCr", "EBA"):
        s = agg[name]["z_d"]
        print(f"{name:>6} | {s['mean']:>+10.4f} {s['sd']:>8.4f} {OBSERVED[name]:>+10.3f}")
    print()
    print("Read the sd column before quoting any single number: at a fixed setting the")
    print("per-draw spread is ~0.03, so one draw is not a result. Every figure above is")
    print(f"a mean over {n_report_seeds} independent seeds, disjoint from the "
          f"{sel['curve'][0]['n']} seeds used to select D_AUD.")

    print()
    print("=" * 78)
    print("THE DECISION-RELEVANT CASE: a GENUINE face effect is present in FFCr.")
    print(f"(p from perm_p at n_perm={PERM_N}; its floor is {P_FLOOR:.1e}, so '<' means")
    print(" the test cannot resolve further, NOT that p was measured that small)")
    print("=" * 78)
    print("Reported as DETECTION RATE: the fraction of seeds where p <= 0.025.")
    print("(A mean p-value is not a p-value; the rate is the interpretable summary.)")
    print(f"{'true FFCr':>10} | {'spatial_z d':>18} {'det':>6} | "
          f"{'raw d':>18} {'det':>6} | {'ref d':>18} {'det':>6}")
    print("-" * 78)
    for eff in (0.0, 0.02, 0.05, 0.10, 0.20):
        a = run_many(sel_d_aud, report_seeds, face_effect=eff, compute_p=True)["FFCr"]
        print(f"{eff:>10.2f} | {a['z_d']['mean']:>+9.3f}+/-{a['z_d']['sd']:<7.3f} "
              f"{a['z_p']['reject_rate_025']:>6.2f} | "
              f"{a['raw_d']['mean']:>+9.4f}+/-{a['raw_d']['sd']:<7.4f} "
              f"{a['raw_p']['reject_rate_025']:>6.2f} | "
              f"{a['ref_d']['mean']:>+9.4f}+/-{a['ref_d']['sd']:<7.4f} "
              f"{a['ref_p']['reject_rate_025']:>6.2f}")
    print()
    print("The raw and reference statistics track the injected effect; spatial_z lags,")
    print("because it is fighting the normaliser shift the auditory difference creates.")


if __name__ == "__main__":
    main()
