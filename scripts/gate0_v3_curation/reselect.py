"""v3b: repair the v3 SELECTION so it optimises the quantity the acceptance test actually measures.

The v3 bug: candidates were matched on motion estimated from a 160 px full-film scan, but the
acceptance criterion is evaluated on the delivered 480 p clips. The two differ enough that the
shipped set passed at selection (AUC 0.404) and failed on delivery (AUC 0.631 vs a 0.10 cap on
|AUC - 0.5|). Fix: cut every shot representative at 480 p first, measure it with the SAME estimator
used for verification, and match on that.

Also corrected here:
  - balance is certified in RANKS (AUC), because the primary gate is a rank test (Mann-Whitney U).
    Raw SMD on a positive, ~14x-range energy measure is outlier-dominated and uninformative; the
    supplementary SMD is therefore taken on LOG motion.
Nothing about the pool, the shot rule, the detector protocol or the thresholds changes.
"""
import json
import re
import subprocess
import sys

import numpy as np

import importlib.util as _ilu

import candidates as C

# NB: the local module is called select.py, which shadows Python's stdlib `select`.
# Load it by path so the stdlib is not involved.
_spec = _ilu.spec_from_file_location('gate0_select', 'select.py')
S = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(S)

sys.path.insert(0, '/home/deveshb/workspace/AI/tribe-bench')
from tribe_tools.roi_stats import mc_perm_p  # noqa: E402

FF = './ffmpeg'
NUM = r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?'
TMP = 'reps'
MIN_SEP = 45.0
N = 15


def two_sided(a, b):
    return min(1.0, 2 * min(mc_perm_p(a, b, n_perm=10000, seed=0),
                            mc_perm_p(b, a, n_perm=10000, seed=0)))


def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(((a[:, None] > b[None, :]).sum()
                  + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sp) if sp else 0.0


def cut(start, path):
    subprocess.run([FF, '-y', '-v', 'error', '-ss', str(start), '-t', '10', '-i', 'charade.mp4',
                    '-vf', 'scale=-2:480', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                    '-c:a', 'aac', path], check=False, capture_output=True)


def meta_mean(path, vf, key):
    r = subprocess.run([FF, '-v', 'info', '-i', path, '-vf', vf, '-an', '-f', 'null', '-'],
                       capture_output=True, text=True)
    v = [float(x) for x in re.findall(rf'{re.escape(key)}=\s*({NUM})', r.stderr)]
    return sum(v) / len(v) if v else None


def measure(reps, tag):
    import os
    os.makedirs(TMP, exist_ok=True)
    out = []
    for i, r in enumerate(reps):
        p = f'{TMP}/{tag}_{i:02d}.mp4'
        cut(r['start'], p)
        r = dict(r)
        r['motion_480'] = meta_mean(p, 'tblend=all_mode=difference,signalstats,'
                                       'metadata=print:key=lavfi.signalstats.YAVG',
                                    'lavfi.signalstats.YAVG')
        r['lum_480'] = meta_mean(p, 'signalstats,metadata=print:key=lavfi.signalstats.YAVG',
                                 'lavfi.signalstats.YAVG')
        out.append(r)
        print(f'  {tag}_{i:02d} t={r["start"]:.1f}s motion480={r["motion_480"]:.3f}', flush=True)
    return out


def pair(face, non, n=N, min_sep=MIN_SEP):
    """Greedy best-first on standardised [log motion, luminance] measured at 480p."""
    lm = np.log([r['motion_480'] for r in face + non])
    ll = np.array([r['lum_480'] for r in face + non])
    ms, ls = lm.std() or 1.0, ll.std() or 1.0
    Fv = np.array([[np.log(r['motion_480']) / ms, r['lum_480'] / ls] for r in face])
    Nv = np.array([[np.log(r['motion_480']) / ms, r['lum_480'] / ls] for r in non])
    cost = np.linalg.norm(Fv[:, None, :] - Nv[None, :, :], axis=2)
    order = sorted((float(cost[i, j]), face[i]['start'], non[j]['start'], i, j)
                   for i in range(len(face)) for j in range(len(non)))
    uf, un, acc, starts = set(), set(), [], []
    for c, _, _, i, j in order:
        if i in uf or j in un:
            continue
        cand = [face[i]['start'], non[j]['start']]
        if abs(cand[0] - cand[1]) < min_sep or any(abs(x - s) < min_sep for x in cand for s in starts):
            continue
        uf.add(i); un.add(j); starts += cand
        acc.append((face[i], non[j], c))
        if len(acc) == n:
            break
    return acc


if __name__ == '__main__':
    f, nf = S.build_pools()
    fr, nr = S.one_per_shot(f), S.one_per_shot(nf)
    print(f'shot representatives: FACE {len(fr)}  NONFACE {len(nr)} -> cutting at 480p to measure')
    fr = measure(fr, 'F')
    nr = measure(nr, 'N')
    pairs = pair(fr, nr)
    F = [p[0] for p in pairs]; Nn = [p[1] for p in pairs]
    fm = [r['motion_480'] for r in F]; nm = [r['motion_480'] for r in Nn]
    fl = [r['lum_480'] for r in F]; nl = [r['lum_480'] for r in Nn]
    A = auc(nm, fm)
    res = dict(n=len(pairs),
               motion_p=two_sided(fm, nm), motion_auc=A,
               motion_smd_log=smd(np.log(fm), np.log(nm)),
               motion_smd_raw=smd(fm, nm),
               lum_p=two_sided(fl, nl), lum_auc=auc(nl, fl), lum_smd=smd(fl, nl),
               min_sep=float(np.diff(sorted([r['start'] for r in F + Nn])).min()))
    res['accepted'] = bool(res['motion_p'] >= 0.20 and abs(A - 0.5) <= 0.10
                           and abs(res['motion_smd_log']) <= 0.25)
    print('\n=== v3b, measured on the delivered 480p clips ===')
    for k, v in res.items():
        print(f'  {k:16} {v if isinstance(v,(bool,int)) else round(v,4)}')
    json.dump(dict(face=F, nonface=Nn, acceptance=res), open('selection_v3b.json', 'w'), indent=1)
    print('wrote selection_v3b.json')
