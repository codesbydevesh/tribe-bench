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

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tribe_tools.roi_stats import perm_p, u_statistic  # noqa: E402

N_VERTS = 20484  # fsaverage5, both hemispheres
RNG = np.random.default_rng(0)

# ROI sizes as used by Gate 0 (from the record: FFC 58, EBA proxy 116, V1 523).
ROIS = {"A1": 70, "V1": 523, "FFCr": 58, "EBA": 116}

OBSERVED = {"A1": +0.280, "V1": -0.046, "FFCr": -0.244, "EBA": -0.382}


def build_brain():
    """A baseline predicted-response map with realistic mass structure.

    Auditory/STS carries the most output mass (matches the printed top-k ROIs:
    A5, STSdp, TPOJ1, A4, PBelt, LBelt, STSvp, STSda, 55b, A1). Visual sits above
    the whole-brain average but below auditory. Everything else is low.
    """
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
    base += RNG.normal(0, 0.03, N_VERTS)   # spatial texture
    return base, idx


def clip_map(base, idx, aud_drive, seed):
    """One clip's (n_segments, n_vertices) prediction. NO face information anywhere.

    aud_drive is the only thing that differs between conditions. FFCr, V1 and EBA
    receive identical distributions in both conditions by construction.
    """
    rng = np.random.default_rng(seed)
    g = base.copy()
    g[idx["AUD"]] *= 1.0 + aud_drive           # speech -> auditory mass
    g += rng.normal(0, 0.05, N_VERTS)          # per-clip noise, condition-independent
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


def run(delta_aud, n=15, face_effect=0.0, verbose=False):
    """delta_aud = the FACE-minus-NONFACE difference in auditory drive.

    face_effect = a GENUINE additive face response injected into FFCr on FACE clips
    only (raw units). This is the case that decides whether the NO-GO is safe to act
    on: can spatial_z report a negative FFCr delta while a real positive effect is
    present in the data?
    """
    base, idx = build_brain()
    ref = idx["REST"][:2000]  # low-drive off-target reference region

    face, nonface = [], []
    for i in range(n):
        # condition-independent scatter in speech drive, plus the condition offset
        a_f = 0.30 + delta_aud + RNG.normal(0, 0.10)
        a_n = 0.30 + RNG.normal(0, 0.10)
        pf = clip_map(base, idx, a_f, seed=1000 + i)
        if face_effect:
            pf[:, idx["FFCr"]] += face_effect
        face.append(pf)
        nonface.append(clip_map(base, idx, a_n, seed=2000 + i))

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
            "z_p": perm_p(zf, zn, 2000, 0),
            "z_base": np.mean(zf + zn),
            "raw_d": np.mean(rf) - np.mean(rn),
            "raw_p": perm_p(rf, rn, 2000, 0),
            "ref_d": np.mean(df) - np.mean(dn),
            "ref_p": perm_p(df, dn, 2000, 0),
        }

    # the normaliser itself -- diagnostic #3 in the parked Kaggle code
    mu_f = [p.mean(axis=0).mean() for p in face]
    mu_n = [p.mean(axis=0).mean() for p in nonface]
    sd_f = [p.mean(axis=0).std() for p in face]
    sd_n = [p.mean(axis=0).std() for p in nonface]
    out["_norm"] = {
        "mu_f": np.mean(mu_f), "mu_n": np.mean(mu_n),
        "mu_p": perm_p(mu_f, mu_n, 2000, 0),
        "sd_f": np.mean(sd_f), "sd_n": np.mean(sd_n),
        "sd_p": perm_p(sd_f, sd_n, 2000, 0),
        "sd_ratio": np.mean(sd_f) / np.mean(sd_n),
    }
    return out


def main():
    print(__doc__.split("CPU only")[0].strip()[:0] or "", end="")
    print("=" * 78)
    print("SWEEP: FACE-minus-NONFACE auditory drive difference -> spatial_z deltas")
    print("(zero face information injected at any delta)")
    print("=" * 78)
    print(f"{'d_aud':>7} | {'A1':>8} {'V1':>8} {'FFCr':>8} {'EBA':>8} | "
          f"{'sd ratio':>8} | {'FFCr raw':>9} {'FFCr-ref':>9}")
    print("-" * 78)
    for d_aud in (0.0, 0.02, 0.05, 0.08, 0.12, 0.18):
        r = run(d_aud)
        print(f"{d_aud:>7.2f} | "
              f"{r['A1']['z_d']:>+8.3f} {r['V1']['z_d']:>+8.3f} "
              f"{r['FFCr']['z_d']:>+8.3f} {r['EBA']['z_d']:>+8.3f} | "
              f"{r['_norm']['sd_ratio']:>8.4f} | "
              f"{r['FFCr']['raw_d']:>+9.4f} {r['FFCr']['ref_d']:>+9.4f}")

    print()
    print("OBSERVED 2026-07-31:  "
          f"A1 {OBSERVED['A1']:+.3f}  V1 {OBSERVED['V1']:+.3f}  "
          f"FFCr {OBSERVED['FFCr']:+.3f}  EBA {OBSERVED['EBA']:+.3f}")
    print()

    # detail at the setting that best matches the observed FFCr delta
    best, best_err = None, 1e9
    for d_aud in np.arange(0.0, 0.25, 0.01):
        r = run(float(d_aud))
        err = sum((r[k]["z_d"] - OBSERVED[k]) ** 2 for k in OBSERVED)
        if err < best_err:
            best, best_err = (float(d_aud), r), err
    d_aud, r = best
    print("=" * 78)
    print(f"CLOSEST MATCH TO THE OBSERVED PATTERN: auditory drive difference {d_aud:.2f}")
    print(f"(sum of squared error across all four ROIs = {best_err:.4f})")
    print("=" * 78)
    print(f"{'ROI':>6} | {'z base':>7} | {'sim d':>8} {'obs d':>8} | "
          f"{'sim U':>7} {'sim p':>7} | {'raw d':>9} {'ref d':>9}")
    print("-" * 78)
    for name in ("A1", "V1", "FFCr", "EBA"):
        s = r[name]
        print(f"{name:>6} | {s['z_base']:>7.3f} | {s['z_d']:>+8.3f} "
              f"{OBSERVED[name]:>+8.3f} | {s['z_U']:>7.1f} {s['z_p']:>7.4f} | "
              f"{s['raw_d']:>+9.4f} {s['ref_d']:>+9.4f}")
    n = r["_norm"]
    print()
    print("THE NORMALISER (the falsifiable prediction for the real cache):")
    print(f"  brain_mean  FACE {n['mu_f']:.5f}  NONFACE {n['mu_n']:.5f}  p={n['mu_p']:.4f}")
    print(f"  brain_sd    FACE {n['sd_f']:.5f}  NONFACE {n['sd_n']:.5f}  p={n['sd_p']:.4f}"
          f"   ratio {n['sd_ratio']:.4f}")

    print()
    print("=" * 78)
    print("THE DECISION-RELEVANT CASE: a GENUINE face effect is present in FFCr,")
    print("alongside the same speech/auditory difference. What does each statistic say?")
    print("=" * 78)
    print(f"{'true FFCr':>10} | {'spatial_z':>10} {'z p':>7} {'z U':>7} | "
          f"{'raw d':>9} {'raw p':>7} | {'ref d':>9} {'ref p':>7} | verdict")
    print(f"{'effect':>10} | {'delta':>10} {'':>7} {'/225':>7} | "
          f"{'':>9} {'':>7} | {'':>9} {'':>7} | on spatial_z")
    print("-" * 78)
    for eff in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40):
        s = run(d_aud, face_effect=eff)["FFCr"]
        verdict = "NO-GO" if s["z_d"] <= 0 else ("detected" if s["z_p"] <= 0.025 else "ambiguous")
        print(f"{eff:>10.2f} | {s['z_d']:>+10.3f} {s['z_p']:>7.4f} {s['z_U']:>7.1f} | "
              f"{s['raw_d']:>+9.4f} {s['raw_p']:>7.4f} | "
              f"{s['ref_d']:>+9.4f} {s['ref_p']:>7.4f} | {verdict}")
    print()
    print("Read the left column against the last: the raw and reference statistics track")
    print("the injected effect monotonically; spatial_z keeps printing NO-GO until the")
    print("true effect is large enough to overcome the normaliser shift it is fighting.")


if __name__ == "__main__":
    main()
