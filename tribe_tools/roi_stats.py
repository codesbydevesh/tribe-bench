"""ROI contrast statistics for in-silico functional localization.

Small-n exact statistics used by Gate 0 (and, later, the NeuroCheck harness):
a per-clip spatial z-score, the Mann-Whitney U direction statistic, and its
EXACT permutation null. These are decision-critical — a wrong permutation null
turned Gate 0's original "7/9 pairs" rule into a p=0.20 coin flip — so they live
here, unit-tested, not in a notebook cell that only ever runs on a GPU.

All functions are pure NumPy/stdlib and run on CPU.
"""

from itertools import combinations
from math import comb

import numpy as np

# Guard: exact enumeration is C(n+m, n) labelings. Gate 0 is 4v4 = 70. Refuse to
# silently churn on a combinatorially huge input instead of returning a wrong number.
_MAX_LABELINGS = 200_000


def spatial_z(preds: np.ndarray, verts: np.ndarray) -> float:
    """Spatial z-score of an ROI for a single clip.

    Collapses the clip to a per-vertex mean map, then measures how many
    spatial standard deviations the ROI's mean sits above the whole-cortex
    mean. This cancels per-clip global gain (a clip that is merely "louder
    everywhere") before any across-condition comparison.

    Args:
        preds: (n_segments, n_vertices) or (n_vertices,).
        verts: 1D array of vertex indices for the ROI.

    Returns:
        (mean over ROI - mean over cortex) / std over cortex, on the clip-mean map.
    """
    g = preds.mean(axis=0) if preds.ndim == 2 else np.asarray(preds)
    if len(verts) == 0:
        raise ValueError("empty ROI vertex set")
    sd = g.std()
    if sd == 0:
        return 0.0
    return float((g[verts].mean() - g.mean()) / sd)


def u_statistic(face_vals, scene_vals) -> float:
    """Mann-Whitney U: count of (face_i > scene_j) pairs; ties count 0.5.

    Ranges 0 .. len(face)*len(scene). Max = perfect face>scene separation.
    """
    u = 0.0
    for f in face_vals:
        for s in scene_vals:
            if f > s:
                u += 1.0
            elif f == s:
                u += 0.5
    return u


def _labelings(n_total: int, n_face: int):
    k = comb(n_total, n_face)
    if k > _MAX_LABELINGS:
        raise ValueError(
            f"exact permutation would enumerate {k} labelings (> {_MAX_LABELINGS}); "
            "Gate 0 is small (4v4=70) — a huge input here is a bug"
        )
    return combinations(range(n_total), n_face)


def exact_perm_p(face_vals, scene_vals) -> float:
    """One-sided exact permutation p-value for U under label exchangeability.

    H0: the condition labels are exchangeable across all clips. Returns
    P(U_permuted >= U_observed) over every way to assign the labels. With
    4 vs 4 clips there are C(8,4)=70 labelings, so perfect separation (U=16)
    gives p=1/70≈0.014 and U>=15 gives p=2/70≈0.029 — the numbers Gate 0's
    GO rule is pre-registered against.
    """
    vals = list(face_vals) + list(scene_vals)
    n_total, n_face = len(vals), len(face_vals)
    u_obs = u_statistic(face_vals, scene_vals)
    ge = total = 0
    for combo in _labelings(n_total, n_face):
        sel = set(combo)
        f = [vals[i] for i in range(n_total) if i in sel]
        s = [vals[i] for i in range(n_total) if i not in sel]
        total += 1
        if u_statistic(f, s) >= u_obs - 1e-9:
            ge += 1
    return ge / total


def perm_null_deltas(face_vals, scene_vals) -> np.ndarray:
    """All permuted mean-differences (face_mean - scene_mean) under label swaps.

    The magnitude null for Gate 0's G2 criterion: the observed Delta-z is
    "large" only if it exceeds the 95th percentile of THIS distribution — a
    per-ROI null tied to the same exchangeability assumption as the direction
    test, so no arbitrary shared magnitude constant is needed.
    """
    vals = np.array(list(face_vals) + list(scene_vals), dtype=float)
    n_total, n_face = len(vals), len(face_vals)
    out = []
    for combo in _labelings(n_total, n_face):
        sel = list(combo)
        rest = [i for i in range(n_total) if i not in set(combo)]
        out.append(vals[sel].mean() - vals[rest].mean())
    return np.array(out)
