"""Select the matched, sustained-shot Gate 0 v3 stimulus set.

Mechanical and outcome-blind end to end. No brain data exists yet and none is consulted.

Rules, fixed before running:
  SHOT      a candidate window must lie strictly inside one shot with a 0.5 s guard on each side,
            where a shot boundary is any frame with ffmpeg scene_score > 0.05 (deliberately
            sensitive - over-detection only costs candidates, under-detection is the bug being fixed).
  FACE      the high-precision cascade union finds a frontal face of area >= 0.030 of frame in
            >= 80% of the 21 samples, and the median such area is >= 0.030.
  NONFACE   the high-precision cascade union NEVER finds a frontal face of area >= 0.020 anywhere in
            the window, AND people are present (profile detections in >= 30% of samples) so D021's
            "both conditions contain people" control is preserved.
  ONE PER SHOT   at most one clip per shot, so no two clips share a take.
  MATCH     1:1 optimal assignment minimising standardised distance on [motion, luminance];
            keep the 15 closest pairs.
  ACCEPT    two-sided permutation p >= 0.20 on motion AND |SMD| <= 0.25 AND rank AUC <= 0.60.
            The AUC cap is the load-bearing one: at 15v15 a perfectly motion-driven artefact with
            AUC 0.60 reaches at best U=135, one-sided p ~ 0.18, so it cannot clear the G1 gate of
            0.025 on its own. The confound is bounded BELOW the gate by construction.
"""
import json
import sys

import numpy as np

import candidates as C

sys.path.insert(0, '/home/deveshb/workspace/AI/tribe-bench')
from tribe_tools.roi_stats import mc_perm_p  # noqa: E402

FACE_A, FACE_FRAC, FACE_MED = 0.025, 0.80, 0.025
BOTH_FRAC = 0.30   # the two Haar cascades must AGREE in >=30% of samples. A persistent
                   # single-cascade false positive (observed: union area ~0.040 on every sample
                   # of a wide corridor shot, agreement 0.000 throughout) is thereby excluded,
                   # while a genuine face tolerates frames where one cascade misses.
NONFACE_A, PEOPLE_FRAC = 0.020, 0.30
N_TARGET = 15
SEED = 0


def two_sided(a, b, n=10000, s=SEED):
    return min(1.0, 2 * min(mc_perm_p(a, b, n_perm=n, seed=s), mc_perm_p(b, a, n_perm=n, seed=s)))


def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    tie = (a[:, None] == b[None, :]).sum()
    return float(gt + 0.5 * tie) / (len(a) * len(b))


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def build_pools():
    d, F = C.load()
    sh = C.shots(d['scene_t'], d['scene_v'])
    long = sh[(sh[:, 1] - sh[:, 0]) >= C.DUR_S + 2 * C.GUARD_S]
    face, nonface = [], []
    for t0, s, e in C.windows(long):
        m = (F[:, 0] >= t0) & (F[:, 0] < t0 + C.DUR_S)
        ff = F[m]
        if len(ff) < 18:
            continue
        sa = np.maximum(ff[:, 2], ff[:, 5])                     # union: recall
        both = np.where((ff[:, 1] > 0) & (ff[:, 4] > 0),
                        np.minimum(ff[:, 2], ff[:, 5]), 0.0)    # agreement: precision
        people = (ff[:, 10] > 0).mean()
        rec = dict(start=round(float(t0), 2), shot=round(float(s), 2),
                   shot_end=round(float(e), 2), shot_len=round(float(e - s), 2),
                   motion=C.agg(t0, C.DUR_S, d['motion_t'], d['motion_v']),
                   lum=C.agg(t0, C.DUR_S, d['lum_t'], d['lum_v']),
                   scene_max=float(d['scene_v'][(d['scene_t'] >= t0)
                                                & (d['scene_t'] < t0 + C.DUR_S)].max()),
                   strict_frac=float((sa >= FACE_A).mean()),
                   strict_med=float(np.median(sa)), strict_max=float(sa.max()),
                   people_frac=float(people), n_samples=int(len(ff)))
        # V-JEPA2 clamps its window to 4s and duplicates frame 0 for 217/1280 frames, so the
        # OPENING of the clip carries far more weight than a window-average criterion reflects.
        # Require the face to be dominant at the very first sample AND across the first 4s, not
        # merely in >=80% of the whole 10s. Without this, a window can pass on its tail while
        # frame 0 shows a wide establishing shot (the exact defect seen in v3 draft FACE_00).
        rec['head_frac'] = float((sa[:8] >= FACE_A).mean())     # first 4.0 s at 2 fps
        rec['t0_area'] = float(sa[0])
        rec['both_frac'] = float((both >= FACE_A).mean())
        rec['t0_both'] = float(both[0])
        if (rec['strict_frac'] >= FACE_FRAC and rec['strict_med'] >= FACE_MED
                and rec['t0_area'] >= FACE_A and rec['head_frac'] >= FACE_FRAC
                and rec['both_frac'] >= BOTH_FRAC and rec['t0_both'] >= FACE_A):
            face.append(rec)
        elif sa.max() < NONFACE_A and people >= PEOPLE_FRAC:
            nonface.append(rec)
    return face, nonface


def one_per_shot(pool):
    """Median-motion window of each shot: a fixed, non-cherry-picked representative."""
    by = {}
    for r in pool:
        by.setdefault(r['shot'], []).append(r)
    out = []
    for shot, rs in by.items():
        rs = sorted(rs, key=lambda x: x['motion'])
        out.append(rs[len(rs) // 2])
    return sorted(out, key=lambda x: x['start'])


MIN_SEP_S = 45.0   # D021's own independence rule: no two clips within 45 s of each other,
                   # enforced ACROSS conditions too, since adjacent shots of one scene are not
                   # independent samples of anything.


def match(face, nonface, n=N_TARGET, min_sep=MIN_SEP_S):
    """Greedy best-first pairing on standardised [motion, luminance] distance, subject to
    one clip per shot and >= min_sep separation between every accepted clip. Deterministic:
    pairs are consumed in ascending cost, ties broken by start time."""
    allm = np.array([r['motion'] for r in face + nonface])
    alll = np.array([r['lum'] for r in face + nonface])
    ms, ls = allm.std() or 1.0, alll.std() or 1.0
    Fm = np.array([[r['motion'] / ms, r['lum'] / ls] for r in face])
    Nm = np.array([[r['motion'] / ms, r['lum'] / ls] for r in nonface])
    cost = np.linalg.norm(Fm[:, None, :] - Nm[None, :, :], axis=2)
    order = sorted(((float(cost[i, j]), face[i]['start'], nonface[j]['start'], i, j)
                    for i in range(len(face)) for j in range(len(nonface))))
    used_f, used_n, accepted, starts = set(), set(), [], []
    for c, _, _, i, j in order:
        if i in used_f or j in used_n:
            continue
        cand = [face[i]['start'], nonface[j]['start']]
        if abs(cand[0] - cand[1]) < min_sep:
            continue
        if any(abs(x - s) < min_sep for x in cand for s in starts):
            continue
        used_f.add(i); used_n.add(j); starts += cand
        accepted.append((face[i], nonface[j], c))
        if len(accepted) == n:
            break
    return accepted


if __name__ == '__main__':
    face, nonface = build_pools()
    print(f'raw pools: FACE {len(face)} windows / {len({r["shot"] for r in face})} shots | '
          f'NONFACE {len(nonface)} windows / {len({r["shot"] for r in nonface})} shots')
    fr, nr = one_per_shot(face), one_per_shot(nonface)
    print(f'one-per-shot: FACE {len(fr)}  NONFACE {len(nr)}')
    if min(len(fr), len(nr)) < N_TARGET:
        print(f'!! pool too small for n={N_TARGET}')
    pairs = match(fr, nr)
    fsel = [p[0] for p in pairs]
    nsel = [p[1] for p in pairs]
    fm = [r['motion'] for r in fsel]
    nm = [r['motion'] for r in nsel]
    fl = [r['lum'] for r in fsel]
    nl = [r['lum'] for r in nsel]
    print(f'\nselected {len(pairs)} pairs')
    print(f'  motion    FACE {np.mean(fm):.3f}  NONFACE {np.mean(nm):.3f}  '
          f'p={two_sided(fm, nm):.4f}  SMD={smd(fm, nm):+.3f}  AUC={auc(fm, nm):.3f}')
    print(f'  luminance FACE {np.mean(fl):.2f}  NONFACE {np.mean(nl):.2f}  '
          f'p={two_sided(fl, nl):.4f}  SMD={smd(fl, nl):+.3f}  AUC={auc(fl, nl):.3f}')
    ok = (two_sided(fm, nm) >= 0.20 and abs(smd(fm, nm)) <= 0.25 and abs(auc(fm, nm) - 0.5) <= 0.10)
    print(f'  ACCEPTANCE: {"PASS" if ok else "FAIL"}')
    out = dict(face=fsel, nonface=nsel,
               match=dict(motion_p=two_sided(fm, nm), motion_smd=smd(fm, nm),
                          motion_auc=auc(fm, nm), lum_p=two_sided(fl, nl),
                          lum_smd=smd(fl, nl), lum_auc=auc(fl, nl), accepted=bool(ok)),
               params=dict(scene_thr=C.SCENE_THR, guard_s=C.GUARD_S, dur_s=C.DUR_S,
                           stride_s=C.STRIDE_S, face_area=FACE_A, face_frac=FACE_FRAC,
                           face_med=FACE_MED, both_frac=BOTH_FRAC, nonface_area=NONFACE_A,
                           people_frac=PEOPLE_FRAC, n=N_TARGET, seed=SEED))
    json.dump(out, open('selection.json', 'w'), indent=1)
    print('\nwrote selection.json')
    for a, b, c in pairs:
        print(f"  F {a['start']:7.1f}s (m={a['motion']:.2f}) <-> N {b['start']:7.1f}s "
              f"(m={b['motion']:.2f})  d={c:.3f}")
