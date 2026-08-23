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
    _require_finite(list(face_vals) + list(scene_vals), "exact_perm_p input")
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
    vals = np.array(list(face_vals) + list(other_vals), dtype=float)
    _require_finite(vals, "mc_perm_p input")
    n, N = len(face_vals), len(vals)
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
    N, n = len(face_vals) + len(other_vals), len(face_vals)
    if comb(N, n) <= 20000:
        return exact_perm_p(face_vals, other_vals)
    return mc_perm_p(face_vals, other_vals, n_perm=n_perm, seed=seed)


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


def _as_vertex_indices(v, what: str = "vertex selector") -> np.ndarray:
    """Normalise a vertex selector to an INTEGER index array (C7).

    A boolean mask and an integer index array select the same vertices but do
    NOT compare correctly against each other: ``np.intersect1d`` compares the
    boolean *values* (0/1) with the integers, so a total overlap goes
    undetected and the overlap guard in :func:`roi_minus_reference` silently
    passes. Normalising both sides first makes the comparison semantic.
    """
    arr = np.asarray(v)
    if arr.dtype == bool:
        return np.flatnonzero(arr)
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.equal(np.mod(arr, 1), 0)):
            raise ValueError(f"{what} must be integer vertex indices or a boolean mask")
        return arr.astype(int)
    return arr.astype(int, copy=False)


def raw_roi_mean(preds: np.ndarray, verts: np.ndarray) -> float:
    """The ROI's predicted response, unnormalised. The simplest honest readout.

    No brain-wide reference at all, so it carries any per-clip global gain
    straight through — that is the trade. Pair it with :func:`roi_minus_reference`
    (which cancels an additive gain) and report both, per D027.

    Args:
        preds: (n_trs, n_vertices) or (n_vertices,).
        verts: 1D array of vertex indices for the ROI.
    """
    g = preds.mean(axis=0) if preds.ndim == 2 else np.asarray(preds, dtype=float)
    verts = _as_vertex_indices(verts, "ROI")
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
    verts = _as_vertex_indices(verts, "ROI")
    ref_verts = _as_vertex_indices(ref_verts, "reference")
    if len(verts) == 0 or len(ref_verts) == 0:
        raise ValueError("empty ROI or reference vertex set")
    if np.intersect1d(verts, ref_verts).size:
        raise ValueError(
            "ROI and reference overlap — the reference must be off-target and "
            "pre-registered, or this is an undeclared normaliser"
        )
    g = preds.mean(axis=0) if preds.ndim == 2 else np.asarray(preds, dtype=float)
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
    if segments is None or len(segments) == 0:
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
    if np.any(np.diff(times) <= 0):
        raise ValueError(
            "segment start times are not strictly increasing — this list spans multiple "
            "timelines or is unsorted. Pass the segments for ONE timeline at a time; "
            "row->time resolution is ambiguous otherwise."
        )
    return times


def _resolve_rows(row_times_s: np.ndarray, want_s, tol: float = 0.5) -> np.ndarray:
    """Map absolute times to prediction rows, or raise. Never silently approximate."""
    rt = np.asarray(row_times_s, dtype=float)
    want = np.asarray(want_s, dtype=float)
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
    if len(verts) == 0:
        raise ValueError("empty ROI vertex set")
    rt = np.asarray(row_times_s, dtype=float)
    if len(rt) != p.shape[0]:
        raise ValueError(
            f"row_times_s has {len(rt)} entries but preds has {p.shape[0]} rows — "
            "these must correspond 1:1 (Meta asserts this at demo_utils.py:382)"
        )
    onsets = np.asarray(list(onset_times_s), dtype=float)
    if onsets.size == 0:
        raise ValueError("no event onsets given")
    if pre_trs < 0 or post_trs < 0:
        raise ValueError("pre_trs and post_trs must be >= 0")

    verts = _as_vertex_indices(verts, "ROI")
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
       itself. It deliberately requires **two or more** categories, so selecting on
       a single category's course is not expressible through this API.

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
    # Grand average = mean of the per-category means, so a category with more
    # events does not dominate the pooled course.
    pooled = np.mean([c.mean(axis=0) for c in courses], axis=0)
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
    tgt = np.asarray(list(target_responses), dtype=float)
    # Reject a 2-D target at the CONTRACT boundary, not by letting a later
    # operation happen to fail (S6). peri_event_timecourse returns
    # (n_events, n_lags) and is always in scope beside event_locked_response's
    # (n_events,), so passing the wrong one is a live hazard: it averages both
    # axes and silently returns an attenuated contrast (measured 0.0404 for a
    # true 0.0941 -- right sign, 2.3x too small).
    if tgt.ndim != 1:
        raise ValueError(
            f"target_responses must be 1-D (n_events,), got shape {tgt.shape}. "
            "If this came from peri_event_timecourse (n_events, n_lags), select a "
            "lag column first — e.g. via event_locked_response."
        )
    if tgt.size == 0:
        raise ValueError("no target responses")
    _require_finite(tgt, "event_locked_contrast target")
    others = []
    for i, o in enumerate(other_responses):
        arr = np.asarray(list(o), dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"other_responses[{i}] must be 1-D (n_events,), got shape {arr.shape}"
            )
        if arr.size == 0:
            # Raise rather than drop: silently discarding a category changes the
            # denominator of the baseline without telling the caller.
            raise ValueError(
                f"other_responses[{i}] is empty. An empty category silently changes "
                "the number of categories averaged into the baseline; pass only "
                "categories that have events, deliberately."
            )
        _require_finite(arr, f"event_locked_contrast other_responses[{i}]")
        others.append(arr)
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
    if len(verts) == 0:
        raise ValueError("empty ROI vertex set")
    na, nb = a.shape[0], b.shape[0]
    if na < 2 or nb < 2:
        raise ValueError("need >= 2 observations per condition for a standard error")
    verts = _as_vertex_indices(verts, "ROI")
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
    a = np.asarray(loc_a, dtype=float)
    b = np.asarray(loc_b, dtype=float)
    a = a.mean(axis=0) if a.ndim == 2 else a
    b = b.mean(axis=0) if b.ndim == 2 else b
    if a.shape != b.shape:
        raise ValueError("localizer conditions disagree on shape")
    pv = np.asarray(parcel_verts, dtype=int)
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
    keep = pv[np.argsort(sel)[::-1][:k]]
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
