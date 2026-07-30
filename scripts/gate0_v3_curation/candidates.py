"""Enumerate and label candidate 10s windows from Charade, from the pre-computed scans.

All thresholds are parameters at the top so the spec can set them without touching logic.
Nothing here looks at brain data - it is a pure stimulus-side, outcome-blind procedure.
"""
import json

import numpy as np

# ---------------------------------------------------------------- parameters
SCENE_THR = 0.05     # a frame with scene_score above this is treated as a shot boundary.
                     # Deliberately SENSITIVE: over-detecting boundaries only costs candidates,
                     # while under-detecting is exactly the defect being fixed (the 0.4 threshold
                     # reported 0.13 cuts/clip against 2.20 actual). At 0.05 the film still has
                     # ~140 shots >= 11s, so strictness is affordable.
GUARD_S = 0.5        # required cut-free margin on BOTH sides of the window. The V-JEPA2 window is
                     # clamped at 4s and duplicates frame 0 for 217/1280 frames, so frame 0 must be
                     # unambiguously inside the shot.
DUR_S = 10.0
STRIDE_S = 0.5
EXCLUDE_HEAD_S = 120.0   # titles
EXCLUDE_TAIL_S = 120.0   # credits

# FACE: a dominant frontal face, persistently, by the HIGH-PRECISION detector
FACE_MIN_AREA = 0.030      # max frontal bbox area as a fraction of frame
FACE_MIN_FRAC = 0.85       # fraction of the 21 samples that must satisfy it
FACE_MEDIAN_AREA = 0.045   # median area across the window ("fills the frame")

# NONFACE: certified absent by the HIGH-RECALL detector. False negatives here are the dangerous
# error (they put a face into the face-absent baseline), so the bar is deliberately strict.
NONFACE_MAX_AREA = 0.012   # no permissive frontal detection larger than this...
NONFACE_MAX_FRAC = 0.05    # ...in more than this fraction of samples
NONFACE_REQUIRE_PEOPLE = True   # preserve D021's "both contain people" control
PEOPLE_MIN_FRAC = 0.40     # profile detections as a cheap people proxy


def load():
    d = np.load('scans_full.npz')
    F = np.load('face_timeline.npy')
    return d, F


def shots(scene_t, scene_v, thr=SCENE_THR):
    b = scene_t[scene_v > thr]
    edges = np.concatenate(([0.0], b, [scene_t.max()]))
    return np.stack([edges[:-1], edges[1:]], axis=1)   # (n,2) start,end


def windows(shot_list, dur=DUR_S, guard=GUARD_S, stride=STRIDE_S):
    out = []
    for s, e in shot_list:
        lo, hi = s + guard, e - guard - dur
        if hi < lo:
            continue
        t = lo
        while t <= hi + 1e-9:
            if t >= EXCLUDE_HEAD_S and t + dur <= shot_list[-1][1] - EXCLUDE_TAIL_S:
                out.append((round(float(t), 2), float(s), float(e)))
            t += stride
    return out


def agg(t0, dur, xt, xv):
    m = (xt >= t0) & (xt < t0 + dur)
    return float(xv[m].mean()) if m.any() else float('nan')


def face_stats(t0, dur, F):
    m = (F[:, 0] >= t0) & (F[:, 0] < t0 + dur)
    if not m.any():
        return None
    f = F[m]
    strict_area = np.maximum(f[:, 2], f[:, 5])     # union of the two strict frontal cascades
    perm_area = np.maximum(f[:, 7], f[:, 9])       # union of the two permissive frontal cascades
    prof_n = f[:, 10]
    return dict(
        n=int(m.sum()),
        strict_frac=float((strict_area >= FACE_MIN_AREA).mean()),
        strict_med_area=float(np.median(strict_area)),
        strict_max_area=float(strict_area.max()),
        perm_over_frac=float((perm_area >= NONFACE_MAX_AREA).mean()),
        perm_max_area=float(perm_area.max()),
        people_frac=float((prof_n > 0).mean()),
    )


def label(fs):
    if fs is None:
        return 'DISCARD'
    if (fs['strict_frac'] >= FACE_MIN_FRAC and fs['strict_med_area'] >= FACE_MEDIAN_AREA):
        return 'FACE'
    if (fs['perm_over_frac'] <= NONFACE_MAX_FRAC
            and (not NONFACE_REQUIRE_PEOPLE or fs['people_frac'] >= PEOPLE_MIN_FRAC)):
        return 'NONFACE'
    return 'DISCARD'


def build():
    d, F = load()
    sh = shots(d['scene_t'], d['scene_v'])
    long_shots = sh[(sh[:, 1] - sh[:, 0]) >= DUR_S + 2 * GUARD_S]
    win = windows(long_shots)
    print(f'shots total {len(sh)}, long enough {len(long_shots)}, candidate windows {len(win)}')
    rows = []
    for t0, s, e in win:
        fs = face_stats(t0, DUR_S, F)
        lab = label(fs)
        if lab == 'DISCARD':
            continue
        rows.append(dict(start=t0, shot_start=round(s, 2), shot_end=round(e, 2),
                         shot_len=round(e - s, 2), label=lab,
                         motion=agg(t0, DUR_S, d['motion_t'], d['motion_v']),
                         lum=agg(t0, DUR_S, d['lum_t'], d['lum_v']),
                         scene_max=float(d['scene_v'][(d['scene_t'] >= t0)
                                                      & (d['scene_t'] < t0 + DUR_S)].max()),
                         **{k: v for k, v in fs.items()}))
    return rows


if __name__ == '__main__':
    rows = build()
    for lab in ('FACE', 'NONFACE'):
        sub = [r for r in rows if r['label'] == lab]
        shots_used = len({r['shot_start'] for r in sub})
        print(f'{lab}: {len(sub)} windows across {shots_used} distinct shots')
        if sub:
            mo = np.array([r['motion'] for r in sub])
            print(f'   motion: min {mo.min():.3f}  p25 {np.percentile(mo,25):.3f}  '
                  f'median {np.median(mo):.3f}  p75 {np.percentile(mo,75):.3f}  max {mo.max():.3f}')
    json.dump(rows, open('candidates.json', 'w'), indent=1)
    print(f'wrote candidates.json ({len(rows)} rows)')
