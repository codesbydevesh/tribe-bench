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
    """LEGACY — DO NOT USE AS A PRIMARY STATISTIC. Kept as the comparison only.

    .. warning::
       **This statistic inverts real effects (G020, D027).** It is compositional:
       every clip's z-map has mean exactly 0 and sd exactly 1, so brain-wide
       condition deltas sum to exactly zero. If one region's share rises, others
       MUST fall — with no change in their actual predicted response.

       Demonstrated in ``scripts/compositional_demo.py`` and pinned by
       ``tests/test_roi_stats.py::test_spatial_z_inverts_a_real_effect``:

       * inject ZERO face information, vary only auditory drive -> reproduces the
         SIGN and ORDERING of the 2026-07-31 pattern. Quantitatively, over 225
         seeded draws at the selected auditory drive (D_AUD=0.30): FFCr
         -0.262 +/- 0.035 (sd) vs observed -0.244, and EBA -0.325 +/- 0.042 vs
         -0.382. Two of four ROIs are NOT reproduced: V1 -0.136 vs -0.046, and
         A1 +0.035 vs +0.280 (~8x short). The ordering EBA < FFCr < V1 < A1 is
         robust across draws but is a STIPULATED INPUT (build_brain sets the
         baseline z of each ROI by hand), not a measured agreement.
       * inject a GENUINE +0.05 FFCr effect -> raw/reference statistics detect it
         while spatial_z does not.

       .. note:: CORRECTED 2026-08-23 (D030). This docstring previously read
          "FFCr sim -0.239 vs obs -0.244, ordering exact" and "p=0.0005 ...
          p=0.9985". Those came from a single draw of a simulator with a
          module-level mutable RNG (two build_brain() calls differed by 0.17), at
          a D_AUD chosen by argmin over 25 single noisy draws on a grid whose last
          point was the winner. "p=0.0005" was the permutation estimator FLOOR
          1/(n_perm+1) = 1/2001, i.e. "p < 5e-4", not a measurement. Every figure
          above now comes from `run_many` over independent seeds; see
          data/sensitivity_selection.json and scripts/sensitivity_surface.py.

       This is not a novel finding: it is global signal regression (Murphy et al.
       2009, doi:10.1016/j.neuroimage.2008.09.036) and, for interpretability,
       arXiv 2512.18792 "The Dead Salmons of AI Interpretability". Cite, fix, move on.

       Use :func:`event_locked_contrast` (the model authors' own statistic, D027),
       :func:`roi_minus_reference`, or :func:`glm_contrast_z` instead. Every verdict
       needs a :func:`detection_floor` beside it (D-3).

    Spatial z-score of an ROI for a single clip.

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
    g = _as_vertex_map(preds, "spatial_z preds")
    verts = _as_vertex_indices(verts, "spatial_z ROI", g.shape[-1])
    if len(verts) == 0:
        raise ValueError("empty ROI vertex set")
    # Guard the WHOLE map, not just the ROI: unlike raw_roi_mean and glm_contrast_z,
    # which read only their selected region, this statistic divides by the brain-wide
    # mean and sd. A non-finite value at any vertex silently returned nan (F4).
    g = _require_finite(g, "spatial_z map")
    sd = g.std()
    if sd == 0:
        return 0.0
    return float((g[verts].mean() - g.mean()) / sd)


def u_statistic(face_vals, scene_vals) -> float:
    """Mann-Whitney U: count of (face_i > scene_j) pairs; ties count 0.5.

    Ranges 0 .. len(face)*len(scene). Max = perfect face>scene separation.
    """
    # A NaN compares False against everything, so it silently scored as a loss
    # with no tie credit -- a FINITE WRONG U, which is worse than an error.
    face_list, scene_list = list(face_vals), list(scene_vals)
    n_face = len(face_list)
    vals = _require_finite(face_list + scene_list, "u_statistic input")
    # Iterate the VALIDATED array, never the original arguments: a single-pass
    # iterable was exhausted by the guard above, both loops then saw nothing, and
    # the result was U=0.0 -- finite, wrong, and maximally anti-selective (F6).
    face_list, scene_list = vals[:n_face], vals[n_face:]
    u = 0.0
    for f in face_list:
        for s in scene_list:
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
    # I3: materialise ONCE at the boundary. This previously consumed its arguments
    # three times -- guard, re-materialise, then len() and u_statistic() -- so a
    # single-pass iterable silently became an empty sample (F6, same class).
    face_list, scene_list = list(face_vals), list(scene_vals)
    n_face = len(face_list)
    vals = _require_finite(face_list + scene_list, "exact_perm_p input").tolist()
    n_total = len(vals)
    u_obs = u_statistic(vals[:n_face], vals[n_face:])
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
    # I3: n_face came from len(face_vals) AFTER the argument had been consumed
    # above, which raises on a generator. Measure the materialised copy.
    face_list = list(face_vals)
    n_face = len(face_list)
    vals = _require_finite(np.array(face_list + list(scene_vals), dtype=float),
                           "perm_null_deltas input")
    n_total = len(vals)
    out = []
    for combo in _labelings(n_total, n_face):
        sel = list(combo)
        rest = [i for i in range(n_total) if i not in set(combo)]
        out.append(vals[sel].mean() - vals[rest].mean())
    return np.array(out)


def _u_fast(a: np.ndarray, b: np.ndarray) -> float:
    """Vectorised Mann-Whitney U (count a_i > b_j, ties 0.5) for the MC inner loop."""
    gt = (a[:, None] > b[None, :]).sum()
    tie = (a[:, None] == b[None, :]).sum()
    return float(gt) + 0.5 * float(tie)


def mc_perm_p(face_vals, other_vals, n_perm: int = 10000, seed: int = 0) -> float:
    """Monte-Carlo one-sided permutation p-value for U (face > other).

    For Gate 0 v2 the conditions have ~15 clips each, so exact enumeration
    (C(30,15) ≈ 1.55e8) is infeasible; this shuffles the pooled labels n_perm
    times. Uses the (perm >= observed) + 1 over (n_perm + 1) estimator so the
    p-value is never zero and stays valid. Seeded for reproducibility.
    """
    # I3: `n` came from len(face_vals), re-reading an argument already consumed
    # above -- which raises on a generator instead of working.
    face_list = list(face_vals)
    n = len(face_list)
    vals = _require_finite(np.array(face_list + list(other_vals), dtype=float),
                           "mc_perm_p input")
    N = len(vals)
    u_obs = _u_fast(vals[:n], vals[n:])
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(N)
        if _u_fast(vals[p[:n]], vals[p[n:]]) >= u_obs - 1e-9:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def perm_p(face_vals, other_vals, n_perm: int = 10000, seed: int = 0) -> float:
    """One-sided permutation p for U: exact when small enough to enumerate, else Monte-Carlo."""
    # I3: materialise before measuring, so a single-pass iterable is not consumed
    # by the size check and then handed on empty to the estimator.
    face_list, other_list = list(face_vals), list(other_vals)
    N, n = len(face_list) + len(other_list), len(face_list)
    if comb(N, n) <= 20000:
        return exact_perm_p(face_list, other_list)
    return mc_perm_p(face_list, other_list, n_perm=n_perm, seed=seed)


def iut_pass(p_a: float, p_b: float, alpha: float = 0.025) -> bool:
    """Intersection-union test: BOTH one-sided contrasts must clear alpha.

    Gate 0 v2's face-selectivity rule is 'faces>objects AND faces>bodies'. An IUT
    controls the family-wise error at alpha without further correction, because the
    null (selectivity fails) is rejected only if every sub-null is rejected.
    """
    return (p_a <= alpha) and (p_b <= alpha)


# ---------------------------------------------------------------------------
# NON-COMPOSITIONAL STATISTICS (D027)
#
# spatial_z above is zero-sum by construction and inverts real effects (G020).
# Everything below measures the ROI without dividing by a brain-wide normaliser,
# so a rise in one region does not force a fall in another.
# ---------------------------------------------------------------------------


def _require_finite(a, what: str) -> np.ndarray:
    """Reject NaN and +/-inf. ONE policy, used at every entry point (M3).

    Non-finite values do not merely propagate here — they can be ranked as
    *maximally selective* (``np.argsort`` sorts NaN last ascending, and
    ``[::-1]`` then promotes it to first), or be silently absorbed into a
    plausible finite number. Both produce a confident wrong answer with no
    warning, which is the failure class this module exists to prevent.
    """
    arr = np.asarray(a, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        idx = np.flatnonzero(bad.ravel())[:5].tolist()
        raise ValueError(
            f"{what} contains {int(bad.sum())} non-finite value(s) (NaN or +/-inf); "
            f"first flat indices {idx}. Refusing to continue: a non-finite value here "
            "can be ranked as maximally selective or absorbed into a finite-looking "
            "result. Clean the input or exclude those vertices explicitly."
        )
    return arr


def _as_vertex_map(preds, what: str) -> np.ndarray:
    """Collapse ``(n_trs, n_vertices)`` or ``(n_vertices,)`` to one per-vertex map.

    The accepted RANK is a precondition, checked before ``n_vertices`` is derived
    from the shape. Without it a 3-D array was silently accepted: ``n_vertices``
    came from ``shape[-1]`` while ``g[verts]`` indexed axis 0, so the range and
    ambiguity checks validated against an axis the selector never touched and the
    result was a finite, plausible, wrong number.
    """
    arr = np.asarray(preds, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError(
            f"{what} must be (n_trs, n_vertices) or (n_vertices,), got shape {arr.shape}. "
            "A higher-rank array indexes the wrong axis and returns a plausible wrong number."
        )
    return arr.mean(axis=0) if arr.ndim == 2 else arr


def _as_vertex_indices(v, what: str = "vertex selector", n_vertices: int | None = None) -> np.ndarray:
    """Normalise and VALIDATE a vertex selector. The single entry point for every
    function in this module that accepts one (C7, A0, A8, F4).

    A boolean mask and an integer index array select the same vertices but do not
    compare correctly against each other: ``np.intersect1d`` compares the boolean
    *values* (0/1) against the integers, so a total overlap goes undetected. That
    was the reported instance. The MECHANISM is wider, and all of it is handled
    here rather than at the call sites (M008):

    * **boolean mask** -> converted to indices.
    * **0/1 integer or float mask** -> **REJECTED as ambiguous**, not guessed. A
      length-N array of 0s and 1s is indistinguishable from the index list
      ``[0, 1]``, and guessing "indices" made ``raw_roi_mean(g, mask.astype(int8))``
      return 0.15 where the bool mask gives 11.0, AND made the overlap guard fire
      falsely against a genuinely disjoint reference. Both were silent.
    * **negative indices** -> **REJECTED**. ``ROI[-1]`` IS vertex ``n-1``, but
      ``np.intersect1d([-1], [n-1])`` is empty, so negatives defeated the overlap
      guard entirely.
    * **duplicate indices** -> **REJECTED**. A repeated index double-weights that
      vertex in every ROI mean, and a "subset" implies uniqueness.
    * **out-of-range indices** -> **REJECTED** when ``n_vertices`` is known.
    * **non-integer floats** -> rejected.
    * **multi-dimensional selectors** -> **REJECTED**, checked on the raw input so
      the rule binds masks and index arrays alike. A ``(30, 2)`` mask over 60
      vertices was read in flat C order, selected every other vertex, and let the
      overlap guard pass on an ROI compared against itself.
    * **object/string dtypes** -> **REJECTED** with a ValueError naming the dtype,
      rather than a TypeError raised from inside ``np.isfinite``.

    Returns indices in **ascending order** in every case. Uniqueness is enforced
    below, so sorting is lossless, and it makes the mask and index representations
    of one vertex set genuinely interchangeable.

    Args:
        v: boolean mask or integer index array.
        what: name used in error messages.
        n_vertices: total vertex count, when the caller knows it. Enables the
            range check and the ambiguity check.
    """
    if isinstance(v, np.ma.MaskedArray):
        raise ValueError(
            f"{what} is a masked array. np.asarray drops the mask, so the masked entries would "
            "silently rejoin the selection and two encodings of one vertex set would disagree. "
            "Pass the compressed values explicitly."
        )
    arr = np.asarray(v)
    if arr.ndim == 0:
        raise ValueError(
            f"{what} is a scalar ({arr.tolist()!r}). A single vertex is still a SET of one: "
            f"pass [{arr.tolist()!r}]. Accepting a bare scalar would make `verts=5` ambiguous "
            "between 'vertex 5' and 'the first 5 vertices'."
        )
    if arr.ndim != 1:
        raise ValueError(
            f"{what} must be 1-D, got shape {arr.shape}. A multi-dimensional selector was "
            "previously read in flat C order, which silently selected a DIFFERENT vertex set "
            "(np.column_stack([lh_mask, rh_mask]) selected every other vertex) and let the "
            "overlap guard pass on an ROI compared against itself. This rule is checked on the "
            "raw input so it applies to masks and index arrays alike."
        )
    if not (arr.dtype == bool or np.issubdtype(arr.dtype, np.integer)
            or np.issubdtype(arr.dtype, np.floating)):
        raise ValueError(
            f"{what} has dtype {arr.dtype}; expected a boolean mask or integer vertex indices. "
            "Object and string arrays previously reached np.isfinite, which raises TypeError "
            "rather than reporting the real problem."
        )
    if arr.dtype == bool:
        if n_vertices is not None and arr.size != n_vertices:
            raise ValueError(
                f"{what}: boolean mask has length {arr.size} but there are "
                f"{n_vertices} vertices"
            )
        return np.flatnonzero(arr)
    if arr.size == 0:
        return arr.astype(int, copy=False)
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{what} contains non-finite values; expected indices or a mask")
        if not np.all(np.equal(np.mod(arr, 1), 0)):
            raise ValueError(f"{what} must be integer vertex indices or a boolean mask")
    idx = arr.astype(int)
    # Ambiguity: a full-length 0/1 array could be a mask OR the indices [0, 1].
    # Refuse to guess -- guessing is what produced a silent wrong ROI.
    if (n_vertices is not None and idx.size == n_vertices and n_vertices > 2
            and np.all((idx == 0) | (idx == 1))):
        raise ValueError(
            f"{what} is ambiguous: a length-{n_vertices} array of only 0s and 1s could be a "
            "boolean mask or the index list [0, 1]. Pass `selector.astype(bool)` for a mask, "
            "or `np.flatnonzero(selector)` for indices. Guessing here previously returned a "
            "silently wrong ROI."
        )
    if (idx < 0).any():
        raise ValueError(
            f"{what} contains negative indices {idx[idx < 0][:5].tolist()}. Negative indices "
            "alias positive vertices (ROI[-1] IS vertex n-1) but compare as distinct, which "
            "defeats the overlap guard. Pass non-negative indices."
        )
    if n_vertices is not None and (idx >= n_vertices).any():
        raise ValueError(
            f"{what} contains indices >= n_vertices={n_vertices}: "
            f"{idx[idx >= n_vertices][:5].tolist()}"
        )
    if np.unique(idx).size != idx.size:
        # np.unique returns the values that repeat. The previous expression built
        # the repeat mask from np.sort(idx) and then applied it to the UNSORTED
        # idx, so it named whichever value happened to sit at that position.
        vals, counts = np.unique(idx, return_counts=True)
        dup = [int(x) for x in vals[counts > 1]][:5]
        raise ValueError(
            f"{what} contains duplicate indices (e.g. {dup}). A repeated index double-weights "
            "that vertex in every ROI mean; pass a unique set."
        )
    # Ordering contract: ALWAYS sorted. A bool mask yields sorted indices via
    # np.flatnonzero while an index array preserved caller order, so the two
    # representations of one vertex set could produce different fROIs under ties.
    return np.sort(idx)

def raw_roi_mean(preds: np.ndarray, verts: np.ndarray) -> float:
    """The ROI's predicted response, unnormalised. The simplest honest readout.

    No brain-wide reference at all, so it carries any per-clip global gain
    straight through — that is the trade. Pair it with :func:`roi_minus_reference`
    (which cancels an additive gain) and report both, per D027.

    Args:
        preds: (n_trs, n_vertices) or (n_vertices,).
        verts: 1D array of vertex indices for the ROI.
    """
    g = _as_vertex_map(preds, "raw_roi_mean preds")
    verts = _as_vertex_indices(verts, "raw_roi_mean ROI", g.shape[-1])
    if len(verts) == 0:
        raise ValueError("empty ROI vertex set")
    _require_finite(g[verts], "raw_roi_mean ROI values")
    return float(g[verts].mean())


def roi_minus_reference(preds: np.ndarray, verts: np.ndarray, ref_verts: np.ndarray) -> float:
    """ROI mean minus a PRE-REGISTERED off-target reference region.

    Cancels an additive per-clip global gain without the zero-sum trap: the
    reference is a fixed, named, low-drive region chosen in advance, not the
    whole-brain average. So a condition difference confined to some third region
    (e.g. auditory) leaves this statistic alone, which is exactly the failure
    mode spatial_z has (G020).

    The reference must be declared before seeing data and must not overlap the
    ROI — otherwise this becomes a different, undocumented normaliser.

    Args:
        preds: (n_trs, n_vertices) or (n_vertices,).
        verts: ROI vertex indices.
        ref_verts: reference-region vertex indices. Must be disjoint from verts.
    """
    # Normalise BOTH selectors to integer indices before comparing them: a
    # boolean mask vs an int array defeats np.intersect1d entirely (C7).
    _n = preds.shape[-1] if np.ndim(preds) else None
    verts = _as_vertex_indices(verts, "roi_minus_reference ROI", _n)
    ref_verts = _as_vertex_indices(ref_verts, "roi_minus_reference reference", _n)
    if len(verts) == 0 or len(ref_verts) == 0:
        raise ValueError("empty ROI or reference vertex set")
    if np.intersect1d(verts, ref_verts).size:
        raise ValueError(
            "ROI and reference overlap — the reference must be off-target and "
            "pre-registered, or this is an undeclared normaliser"
        )
    g = _as_vertex_map(preds, "roi_minus_reference preds")
    _require_finite(g[verts], "roi_minus_reference ROI values")
    _require_finite(g[ref_verts], "roi_minus_reference reference values")
    return float(g[verts].mean() - g[ref_verts].mean())


def row_times_from_segments(segments) -> np.ndarray:
    """Absolute start time (seconds) of each prediction row, from ``predict()``'s all_segments.

    **Why this exists.** ``predict()`` returns ``(preds, all_segments)`` where
    ``preds`` has shape ``(n_kept_segments, n_vertices)`` — only the KEPT rows,
    concatenated across 100 s windows and across timelines. Segments with no
    events are dropped (``remove_empty_segments=True``, demo_utils.py:148,
    370-376). So **row index is not TR index** and never was; Meta's own demo
    prints ``Predicted 53 / 100 segments`` for a 52.21 s clip.

    Meta guarantees the 1:1 mapping we rely on here — demo_utils.py:382-384
    raises unless ``len(all_segments) == preds.shape[0]``.

    This is the ONLY function that touches segment internals. It duck-types
    ``.start`` and fails loudly if the object is not what we expect, rather than
    returning a plausible wrong array.

    Args:
        segments: the ``all_segments`` list returned beside ``preds``.

    Returns:
        (n_rows,) float array of absolute segment start times, in seconds.
    """
    # I3: materialise ONCE before measuring. len() on a single-pass iterable
    # raised TypeError, so this boundary did not honour the module's stated
    # "any iterable, consumed exactly once" contract.
    if segments is None:
        raise ValueError("no segments given — pass predict()'s all_segments")
    segments = list(segments)
    if len(segments) == 0:
        raise ValueError("no segments given — pass predict()'s all_segments")
    try:
        times = np.array([float(s.start) for s in segments], dtype=float)
    except AttributeError as exc:
        attrs = [a for a in dir(segments[0]) if not a.startswith("_")][:20]
        raise AttributeError(
            f"segment objects have no usable .start ({exc}); available attributes: {attrs}. "
            "Do not work around this by indexing rows arithmetically — that is the bug this "
            "function exists to prevent."
        ) from exc
    # NaN also defeats `diff <= 0`, so a NaN start passed the monotonicity guard.
    _require_finite(times, "row_times_from_segments segment start times")
    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "segment start times are not strictly increasing — this list spans multiple "
            "timelines or is unsorted. Pass the segments for ONE timeline at a time; "
            "row->time resolution is ambiguous otherwise."
        )
    return times


def _resolve_rows(row_times_s: np.ndarray, want_s, tol: float = 0.5) -> np.ndarray:
    """Map absolute times to prediction rows, or raise. Never silently approximate."""
    # NaN defeats `err > tol` (NaN > x is False), so a NaN onset silently
    # resolved to a row instead of raising -- the exact drift this guards (A1).
    # I3: np.asarray(generator) yields a 0-d object array, not the values.
    rt = _require_finite(np.array(list(row_times_s), dtype=float), "_resolve_rows row times")
    want = _require_finite(np.asarray(want_s, dtype=float), "_resolve_rows requested times")
    idx = np.searchsorted(rt, want)
    idx = np.clip(idx, 1, len(rt) - 1)
    left, right = rt[idx - 1], rt[np.minimum(idx, len(rt) - 1)]
    pick = np.where(np.abs(want - left) <= np.abs(right - want), idx - 1, idx)
    err = np.abs(rt[pick] - want)
    bad = err > tol
    if np.any(bad):
        i = int(np.argmax(err))
        raise IndexError(
            f"{int(bad.sum())} of {len(want)} requested times have no prediction row within "
            f"{tol} s — worst is t={want[i]:.3f} s (nearest row at {rt[pick[i]]:.3f} s, "
            f"off by {err[i]:.3f} s). Rows span {rt[0]:.3f}..{rt[-1]:.3f} s. This means the "
            "stimulus timing and the returned segments disagree; do NOT proceed."
        )
    return pick


def peri_event_timecourse(
    preds: np.ndarray,
    verts: np.ndarray,
    onset_times_s,
    row_times_s,
    pre_trs: int = 2,
    post_trs: int = 9,
) -> np.ndarray:
    """ROI response around each event onset, as a time course. **The primary readout.**

    Returns the whole evoked response rather than one hand-picked lag, because the
    correct lag is a thing to MEASURE, not assume — see :func:`peak_lag_trs` and the
    warning in :func:`event_locked_response`.

    Rows are resolved by absolute TIME against ``row_times_s``, never by arithmetic
    on a row index. Use :func:`row_times_from_segments` to build ``row_times_s``.

    Args:
        preds: (n_rows, n_vertices) as returned by ``predict()``.
        verts: ROI vertex indices.
        onset_times_s: stimulus onset times in seconds, absolute on the same clock
            as the segments.
        row_times_s: (n_rows,) absolute time of each prediction row.
        pre_trs: TRs before onset to include (baseline).
        post_trs: TRs after onset to include.

    Returns:
        (n_events, pre_trs + post_trs + 1) array of ROI-mean responses. Column
        ``pre_trs`` is the onset itself; column ``pre_trs + k`` is onset + k TRs.

    Raises:
        IndexError: if any requested time has no matching prediction row. Loud on
            purpose — silent misalignment is the failure this function prevents.
    """
    p = np.asarray(preds, dtype=float)
    if p.ndim != 2:
        raise ValueError("peri_event_timecourse needs (n_rows, n_vertices)")
    verts = _as_vertex_indices(verts, "peri_event_timecourse ROI", p.shape[1])
    if len(verts) == 0:
        raise ValueError("empty ROI vertex set (selector selected no vertices)")
    # I3: materialise once -- np.asarray(generator) does not iterate it.
    rt = _require_finite(np.array(list(row_times_s), dtype=float), "peri_event_timecourse row_times_s")
    if len(rt) != p.shape[0]:
        raise ValueError(
            f"row_times_s has {len(rt)} entries but preds has {p.shape[0]} rows — "
            "these must correspond 1:1 (Meta asserts this at demo_utils.py:382)"
        )
    onsets = _require_finite(np.asarray(list(onset_times_s), dtype=float),
                             "peri_event_timecourse onset_times_s")
    if onsets.size == 0:
        raise ValueError("no event onsets given")
    if pre_trs < 0 or post_trs < 0:
        raise ValueError("pre_trs and post_trs must be >= 0")

    lags = np.arange(-pre_trs, post_trs + 1, dtype=float)
    _require_finite(p[:, verts], "peri_event_timecourse ROI values")
    roi = p[:, verts].mean(axis=1)
    out = np.empty((onsets.size, lags.size), dtype=float)
    for j, lag in enumerate(lags):
        out[:, j] = roi[_resolve_rows(rt, onsets + lag)]
    return out


def peak_lag_trs(category_timecourses, pre_trs: int = 2) -> int:
    """Measured peak lag (TRs relative to onset) of the POOLED evoked response.

    Report this instead of assuming a lag. On a movie-trained encoder whose outputs
    are already hemodynamically aligned, the expected answer is **0**, not 5 — but
    measuring costs nothing and settles it (D-3: no verdict without evidence).

    .. important:: **The peak MUST be selected from the pooled (grand-average)
       response across ALL categories, never from the target category alone.**
       Selecting the lag on the same category you then test at is selection on the
       test statistic — double dipping along the time axis. Measured type-I error
       with a true effect of zero at a nominal one-sided alpha of 0.025:
       fixed lag 0.0050 · peak from the **pooled** course 0.0100 ·
       peak from the **target's** course **0.0417** (C5).

       This function therefore takes the per-category time courses and pools them
       itself, requires **two or more**, and rejects the degenerate configurations
       that satisfy that rule while still pooling to the target alone (identical
       courses, flat courses, empty courses).

       .. warning:: **These guards are heuristics, not a proof.** The API cannot
          verify provenance: ``[tc[:k], tc[k:]]`` splits ONE category into two and
          is undetectable, and because ``pooled`` is the unweighted mean of category
          means, a target that responds more strongly than the others -- the normal
          case in a selectivity study -- makes the pooled peak simply BE the
          target's peak. **The only sound protection is to re-select the lag inside
          every permutation, or to use a fixed lag.** Treat this function's output
          as a diagnostic to REPORT, not as a lag to test at.
          The degeneracy check is value-based, so it can also false-positive on
          synthetic data where genuinely different categories happen to produce
          proportional means. It is expressed on the **pooled course** -- the only
          quantity this function derives -- so one rule covers the same course
          passed twice, with duplicated rows, rescaled or offset; a second category
          whose *mean course* is flat even though its event matrix is not; and
          three or more categories that cancel to leave one. Inputs with fewer than
          three lags are refused outright rather than given a weaker guard.

          **What it deliberately does NOT reject:** a category whose mean course is
          weak but non-zero. "Flat" means zero to floating point at the data's own
          scale, not statistically small -- a condition that genuinely shows no
          time-locked response is a real result, not a malformed input. Such a
          category still dilutes the pool, which is the dominance limit above.

    Args:
        category_timecourses: sequence of >= 2 arrays, each (n_events_k, n_lags),
            one per stimulus category — as returned by
            :func:`peri_event_timecourse`. All must share n_lags.
        pre_trs: how many TRs before onset the courses begin (column ``pre_trs``
            is the onset itself).

    Returns:
        The lag, in TRs relative to onset, at which the pooled grand average peaks.
    """
    # A bare 2-D array iterates as its ROWS, which would otherwise produce a
    # confusing per-row shape error instead of naming the real mistake.
    if isinstance(category_timecourses, np.ndarray) and category_timecourses.ndim == 2:
        raise ValueError(
            "peak_lag_trs received a SINGLE (n_events, n_lags) array. It requires the "
            "time courses of every category, as a sequence, so it can pool them. "
            "Selecting the peak lag from one category and then testing at that lag "
            "inflates type-I error to ~0.042 against a nominal 0.025 (C5). "
            "Pass e.g. [faces_tc, objects_tc, places_tc]."
        )
    courses = [np.asarray(c, dtype=float) for c in category_timecourses]
    if len(courses) < 2:
        raise ValueError(
            f"peak_lag_trs needs the time courses of >= 2 categories so it can pool "
            f"them, got {len(courses)}. Selecting the peak from one category and then "
            "testing at that lag inflates type-I error to ~0.042 against a nominal "
            "0.025 (C5). Pass every category's course."
        )
    for i, c in enumerate(courses):
        if c.ndim != 2 or c.shape[1] == 0:
            raise ValueError(f"category_timecourses[{i}] must be (n_events, n_lags), got {c.shape}")
        if c.shape[1] != courses[0].shape[1]:
            raise ValueError(
                f"category_timecourses[{i}] has {c.shape[1]} lags but [0] has "
                f"{courses[0].shape[1]}; all categories must share the lag grid"
            )
        _require_finite(c, f"peak_lag_trs category_timecourses[{i}]")
    # ">= 2 categories" alone is a FORMALITY: [tc, tc], [tc, tc.copy()],
    # [tc[:k], tc[k:]] and [tc, zeros_like(tc)] all satisfy it while restoring
    # target-only selection (measured type-I 0.2032 vs a nominal 0.025). Reject
    # the degenerate configurations explicitly (F6/M008).
    for i, c in enumerate(courses):
        if c.shape[0] == 0:
            raise ValueError(
                f"category_timecourses[{i}] has 0 events. Its mean is all-NaN and argmax then "
                "returns 0, fabricating a lag. Drop the empty category deliberately."
            )
    n_lags = courses[0].shape[1]
    if n_lags < 3:
        raise ValueError(
            f"peak_lag_trs needs at least 3 lags, got {n_lags}. With two lags every pair of "
            "mean-centred courses is [a, -a] and [b, -b], so every degeneracy test below is "
            "either vacuous or rejects all legitimate data. Refusing is honest; a weaker guard "
            "below three lags was a silent hole."
        )

    # I4. The degeneracy is a property of the POOLED course -- the only thing this
    # function derives -- not of pairs of raw event matrices. The previous version
    # tested pairwise collinearity of category means, which missed every
    # configuration where no PAIR is collinear but the pool still collapses:
    # a category whose mean course is flat (its raw matrix is not constant, so the
    # old flat check keyed on `c.std(axis=0)` never fired, and the pairwise check
    # `continue`d past it on zero norm), and mutually cancelling categories.
    means = [c.mean(axis=0) for c in courses]
    centred = [m - m.mean() for m in means]
    norms = [float(np.linalg.norm(c)) for c in centred]
    # "Zero" means zero to floating point at the data's own scale -- NOT a
    # statistical smallness threshold. A category with a weak but real mean
    # response is accepted: refusing to compute because one condition showed no
    # time-locked response would reject a legitimate empirical result. That case
    # falls under the dominance limit in the warning above, which no input check
    # can fix.
    def _is_zero(nrm: float, m: np.ndarray) -> bool:
        return nrm <= 1e-12 * max(1.0, float(np.abs(m).max()))

    for i, (nrm, m) in enumerate(zip(norms, means)):
        if _is_zero(nrm, m):
            raise ValueError(
                f"category_timecourses[{i}] has a flat mean course (no structure across lags). "
                "It contributes nothing to the pooled peak and its only effect is to dilute the "
                "others -- which restores target-only selection. Note this is a property of the "
                "MEAN course: events that disagree in sign average to flat while the raw event "
                "matrix looks perfectly structured."
            )

    # Grand average = mean of the per-category means, so a category with more
    # events does not dominate the pooled course.
    pooled = np.mean(means, axis=0)
    pc = pooled - pooled.mean()
    pn = float(np.linalg.norm(pc))
    if _is_zero(pn, pooled):
        raise ValueError(
            "the pooled course is flat: the categories cancel exactly, so argmax returns 0 and "
            "fabricates a lag. Pass categories that do not sum to a constant."
        )
    for i, (c, nrm) in enumerate(zip(centred, norms)):
        if np.isclose(float(pc @ c) / (pn * nrm), 1.0):
            raise ValueError(
                f"the pooled course is argmax-identical to category_timecourses[{i}] alone (it "
                "is that category's mean course up to a positive scale and a constant offset), "
                "so pooling carries no information that one category did not already carry. "
                "This is the '>= 2 categories' formality -- measured type-I 0.2032 against a "
                "nominal 0.025. It covers the same course passed twice, with duplicated rows, "
                "rescaled or offset; a second category whose mean course is flat; and three or "
                "more categories that cancel to leave one. Pass genuinely different categories."
            )
    return int(np.argmax(pooled)) - int(pre_trs)


def event_locked_response(
    preds: np.ndarray,
    verts: np.ndarray,
    onset_times_s,
    row_times_s,
    lag_trs: int = 0,
) -> np.ndarray:
    """ROI response at a single lag, one value per event. Thin wrapper on the time course.

    .. warning::
       **``lag_trs`` defaults to 0, and that is deliberate. Do not set it to 5.**

       TRIBE v2's predictions are ALREADY hemodynamically aligned. Its own README
       states: *"They are offset by 5 seconds in the past, in order to compensate
       for the hemodynamic lag"* (``tribev2-source/README.md``), implemented as
       ``FmriExtractor(offset=5)`` in ``tribev2/grids/defaults.py:67``. The model
       learns ``stimulus(t) -> BOLD(t+5)``, so output row *t* is already the peak
       response to the stimulus at *t*.

       Reading at ``lag_trs=5`` therefore reads predicted BOLD at **onset + 10 s** —
       past the peak, at ~18% of its amplitude (canonical HRF ``h(10)/h(5) = 0.183``).
       Simulated on the S2 design: a true +0.425 effect is recovered as **+0.4266 at
       lag 0** and **+0.0769 at lag 5**, a 5.5x attenuation that would fail the
       detection floor and fire the S2 stop rule on a real effect.

       This repo had already recorded the fact twice before the bug was written —
       ``MASTER-PLAN`` §3.8 *"do not double-apply"* and
       ``scripts/algonauts/prepare_submission.py`` A3. See M006.

       Meta's Fig 4A says predicted activity *"peaks 5 seconds after stimulus onset"*,
       which is consistent with plotting the same array on the BOLD clock rather than
       the row index. That ambiguity is settled by **measurement**, not assumption:
       use :func:`peri_event_timecourse` + :func:`peak_lag_trs` and report the peak
       you actually observe.

    Args:
        preds: (n_rows, n_vertices).
        verts: ROI vertex indices.
        onset_times_s: onset times in seconds, absolute.
        row_times_s: (n_rows,) absolute time of each prediction row; see
            :func:`row_times_from_segments`.
        lag_trs: TRs after onset to read. **Default 0.** Any non-zero value must be
            justified by a measured peak, not by HRF reasoning.

    Returns:
        (n_events,) array of ROI-mean responses.
    """
    if lag_trs < 0:
        raise ValueError("lag_trs must be >= 0")
    tc = peri_event_timecourse(
        preds, verts, onset_times_s, row_times_s, pre_trs=0, post_trs=int(lag_trs)
    )
    return tc[:, int(lag_trs)]


def _as_event_vector(a, what: str) -> np.ndarray:
    """Validate a per-event response vector. Used for EVERY such argument (F3).

    The reported instance was a 2-D ``target_responses``. The mechanism is that
    ``peri_event_timecourse`` returns ``(n_events, n_lags)`` and is always in
    scope beside ``event_locked_response``'s ``(n_events,)``, so the wrong one
    can be passed to ANY of these arguments -- and the original fix guarded only
    the first of two (M008). 0-d input is also handled here, so it raises a
    ValueError about the contract rather than a TypeError from ``list()``.
    """
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 0:
        raise ValueError(
            f"{what} must be a 1-D (n_events,) vector, got a scalar. "
            "A single event is [x], not x."
        )
    if arr.ndim != 1:
        raise ValueError(
            f"{what} must be 1-D (n_events,), got shape {arr.shape}. If this came from "
            "peri_event_timecourse (n_events, n_lags), select a lag column first -- e.g. "
            "via event_locked_response."
        )
    if arr.size == 0:
        raise ValueError(
            f"{what} is empty. An empty category silently changes the number of categories "
            "averaged into the baseline; pass only categories that have events, deliberately."
        )
    return _require_finite(arr, what)


def event_locked_contrast(target_responses, other_responses) -> float:
    """Target category minus the mean of the OTHER categories, at the same lag.

    The second half of the published protocol (arXiv 2605.04326): *"subtract[ing]
    the average responses at t=5 for the other categories"*. Each element of
    ``other_responses`` is one other category's array from
    :func:`event_locked_response`; each category is averaged first, then those
    category means are averaged, so a category with more exemplars does not
    dominate the baseline.

    Note this is mean-of-category-means, not a pool of all other exemplars. The
    two coincide only at equal n. The published wording says "the average
    responses ... for the other categories", which reads as per-category; the
    choice is recorded here so it can be challenged rather than discovered.

    Args:
        target_responses: (n_events,) responses for the category under test.
        other_responses: sequence of (n_events_k,) arrays, one per other category.

    Returns:
        mean(target) - mean over categories of mean(category).
    """
    # A bare 2-D array passed AS other_responses iterates into rows, and each row
    # becomes a "category" -- the same 2-D hazard through the other argument (F3).
    if isinstance(other_responses, np.ndarray) and other_responses.ndim >= 2:
        raise ValueError(
            f"other_responses must be a SEQUENCE of 1-D per-category vectors, but received a "
            f"single {other_responses.ndim}-D array of shape {other_responses.shape}. Iterating "
            "it would turn each row into a separate 'category' and silently average both axes. "
            "Pass e.g. [objects_resp, places_resp]."
        )
    # Every array-valued argument goes through the SAME validator (M008): the
    # original fix guarded target_responses only, and the identical 2-D hazard
    # reaches the arithmetic through other_responses.
    tgt = _as_event_vector(target_responses, "event_locked_contrast target_responses")
    others = [_as_event_vector(o, f"event_locked_contrast other_responses[{i}]")
              for i, o in enumerate(other_responses)]
    if not others:
        raise ValueError("event_locked_contrast needs at least one other category")
    return float(tgt.mean() - np.mean([o.mean() for o in others]))


def glm_contrast_z(preds_a, preds_b, verts: np.ndarray) -> float:
    """Per-vertex Welch two-sample contrast, averaged over the ROI. Non-compositional.

    Each vertex gets a two-sample statistic — difference of condition means over
    the **Welch** (unequal-variance) standard error across observations — and the
    ROI value is the mean of those. Because the scaling is per-vertex noise, not a
    brain-wide sd, a condition difference elsewhere in the brain cannot move this
    number (contrast with :func:`spatial_z`, G020).

    .. note:: **This is NOT the estimator from the TRIBE v2 paper (M4, D030).**
       An earlier version of this docstring claimed it was "what Meta's Fig 4
       actually reports". That is false under both readings of the paper: the
       Fig 4 caption describes a GLM fit on the predicted **time-series**, and
       §5.9 describes the visual contrasts as a plain **t = +5 s subtraction with
       no GLM at all**. What is implemented here is a two-sample contrast across
       **observations**, chosen by us because it is non-compositional. It is an
       explicit, recorded deviation — see `ops/interface-contracts.md` and D027.
       For the paper's own protocol use :func:`event_locked_contrast`.

    .. warning:: **Not on the z scale.** This is a mean of per-vertex t-like
       statistics; its null SD depends on the within-ROI correlation and is not 1.
       Never threshold at 1.96 and never compare it numerically to a per-vertex
       z-map — always permute.

    **Welch, not pooled (M2).** The pooled (equal-variance) SE is not level-alpha
    when the group sizes differ, which is the design S2 plans (faces vs the pooled
    other categories, ~1:4). Measured at 10v40 with a 6:1 sd ratio, pooled gives
    z=10.74 where Welch gives 5.59 — 1.9x anticonservative, i.e. a false-positive
    risk on the headline claim. Welch and pooled are algebraically identical at
    equal n, so results computed at equal n are unchanged.

    Args:
        preds_a: (n_obs_a, n_vertices) — one row per clip/event for condition A.
        preds_b: (n_obs_b, n_vertices) — condition B.
        verts: ROI vertex indices.

    Returns:
        Mean per-vertex z over the ROI. Positive = A > B.
    """
    a = np.asarray(preds_a, dtype=float)
    b = np.asarray(preds_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("glm_contrast_z needs (n_obs, n_vertices) for both conditions")
    if a.shape[1] != b.shape[1]:
        raise ValueError("condition arrays disagree on vertex count")
    na, nb = a.shape[0], b.shape[0]
    if na < 2 or nb < 2:
        raise ValueError("need >= 2 observations per condition for a standard error")
    # Normalise BEFORE the empty check: an all-False boolean mask has
    # len == n_vertices, so checking first let it through and returned nan (A2).
    verts = _as_vertex_indices(verts, "glm_contrast_z ROI", a.shape[1])
    if len(verts) == 0:
        raise ValueError("empty ROI vertex set (selector selected no vertices)")
    av, bv = a[:, verts], b[:, verts]
    # Raise rather than renormalise: a silently shrinking ROI is itself a result
    # the operator must see (M3).
    _require_finite(av, "glm_contrast_z condition A over the ROI")
    _require_finite(bv, "glm_contrast_z condition B over the ROI")
    diff = av.mean(axis=0) - bv.mean(axis=0)
    # Welch (unequal-variance) SE of the difference of means -- see the note above.
    se = np.sqrt(av.var(axis=0, ddof=1) / na + bv.var(axis=0, ddof=1) / nb)
    z = np.zeros_like(diff)
    ok = se > 0
    z[ok] = diff[ok] / se[ok]
    return float(z.mean())


def define_froi(
    loc_a: np.ndarray,
    loc_b: np.ndarray,
    parcel_verts: np.ndarray,
    top_n: int = 100,
) -> np.ndarray:
    """Top-N most A-selective vertices inside a parcel, from an INDEPENDENT localizer.

    Fixes the ROI, not just the statistic. Gate 0 used the raw 58-vertex Glasser
    right-FFC parcel as its face region; a proper functional ROI (fROI) is defined
    by selectivity measured on data that is NOT the data you then test on
    (Bladon & Bent use ~104 vertices).

    .. warning::
       ``loc_a`` / ``loc_b`` MUST come from a held-out localizer run — a separate
       set of exemplars, or one half of a split. Defining the fROI on the same
       responses you then contrast is double dipping: it manufactures selectivity
       out of noise and the resulting p-value means nothing.

    Args:
        loc_a: (n_vertices,) or (n_obs, n_vertices) localizer response, condition A.
        loc_b: same shape, condition B.
        parcel_verts: candidate vertex indices (the anatomical parcel to search in).
        top_n: how many vertices to keep. Must be STRICTLY LESS than the parcel
            size — see the no-op note below.

    Returns:
        Sorted array of the selected vertex indices (a strict subset of parcel_verts).

    Raises:
        ValueError: if ``top_n >= len(parcel_verts)``. Selecting the whole parcel
            is not selection (M1). The previous behaviour silently capped
            ``k = min(top_n, parcel_size)``, so the DEFAULT ``top_n=100`` applied
            to the project's own 58-vertex right-FFC parcel returned the entire
            parcel — the unfixed anatomical ROI — with no warning, while the
            caller believed it had defined a functional ROI.
        ValueError: if the localizer contrast contains non-finite values (M3).
            ``np.argsort`` sorts NaN last ascending and ``[::-1]`` then promotes
            it to FIRST, so a dead vertex would be ranked maximally selective.
    """
    a = _as_vertex_map(loc_a, "define_froi loc_a")
    b = _as_vertex_map(loc_b, "define_froi loc_b")
    if a.shape != b.shape:
        raise ValueError("localizer conditions disagree on shape")
    # define_froi was the ONLY selector entry point that never called the
    # normaliser -- and it is the function M1 was about (A0).
    pv = _as_vertex_indices(parcel_verts, "define_froi parcel", a.shape[-1])
    if pv.size == 0:
        raise ValueError("empty parcel")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    k = int(top_n)
    if k >= pv.size:
        raise ValueError(
            f"top_n={k} >= parcel size {pv.size}: this would return the whole parcel "
            "and perform no functional selection at all. The right-FFC parcel is 58 "
            "vertices, so the old default of 100 was a silent no-op on it. Either "
            "widen the candidate region beyond the parcel, or pass a top_n well "
            "below the parcel size (e.g. 30) and record the choice."
        )
    sel = a[pv] - b[pv]
    _require_finite(sel, "define_froi localizer contrast over the parcel")
    # Tie-break contract (I1/F7), stated rather than inherited from numpy:
    # rank by DESCENDING contrast, and among equal contrasts prefer the LOWEST
    # vertex index. `pv` is ascending by the selector contract and the sort is
    # stable, so argsort(-sel) delivers exactly that. The previous
    # argsort(sel)[::-1] was unstable AND reversed the tie order, so two
    # representations of one parcel could return different fROIs.
    keep = pv[np.argsort(-sel, kind="stable")[:k]]
    return np.sort(keep)


def detection_floor(
    n_per_group: int,
    noise_sd: float,
    alpha: float = 0.025,
    power: float = 0.80,
    n_sim: int = 200,
    n_perm: int = 400,
    seed: int = 0,
    tol: float = 1e-3,
    max_effect: float = 1e6,
) -> float:
    """Smallest effect this design could detect at `power`, by simulation (D-3).

    "No verdict without a floor." A null is only informative if you can say what
    size of effect you would have caught. This simulates the design's own test —
    the same one-sided permutation test the verdict uses — under additive
    Gaussian noise, and bisects on the effect size until the rejection rate hits
    `power`.

    Reports the MDE in the same units as the statistic being tested. It is a
    property of the DESIGN (n, noise, alpha), not of any observed data.

    Args:
        n_per_group: clips/events per condition.
        noise_sd: across-observation sd of the statistic under the null.
        alpha: one-sided significance level the verdict uses.
        power: target detection probability. 0.80 by convention.
        n_sim: simulated experiments per effect size.
        n_perm: permutations per simulated experiment.
        seed: RNG seed. Reproducibility is required — the floor goes in the paper.
        tol: bisection tolerance on the effect size.
        max_effect: bracketing ceiling; exceeding it raises rather than looping.

    Returns:
        The minimum detectable effect (MDE) at `power`.
    """
    if n_per_group < 2:
        raise ValueError("need >= 2 per group")
    if noise_sd <= 0:
        raise ValueError("noise_sd must be > 0")
    if not 0 < power < 1 or not 0 < alpha < 1:
        raise ValueError("alpha and power must be in (0, 1)")

    def achieved_power(effect: float) -> float:
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(n_sim):
            a = rng.normal(effect, noise_sd, n_per_group)
            b = rng.normal(0.0, noise_sd, n_per_group)
            if mc_perm_p(a, b, n_perm=n_perm, seed=int(rng.integers(1 << 31))) <= alpha:
                hits += 1
        return hits / n_sim

    # bracket: double until powered
    lo, hi = 0.0, max(noise_sd, 1e-6)
    while achieved_power(hi) < power:
        lo, hi = hi, hi * 2
        if hi > max_effect:
            raise RuntimeError(
                f"no effect below {max_effect} reaches {power:.0%} power at "
                f"n={n_per_group}, sd={noise_sd}, alpha={alpha} — the design is "
                "underpowered for any plausible effect; report that, do not run it"
            )
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if achieved_power(mid) >= power:
            hi = mid
        else:
            lo = mid
    return float(hi)
