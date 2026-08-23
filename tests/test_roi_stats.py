"""Gate 0's decision-critical statistics — the exact permutation null in particular.

Porthos's Flaw 1 was that a "7/9 pairs" rule is really Mann-Whitney U at p=0.20, not
0.05. These tests pin the exact numbers the pre-registered GO rule depends on, so the
threshold can never silently drift away from its true p-value.
"""

import numpy as np
import pytest

from tribe_tools.roi_stats import (
    exact_perm_p,
    iut_pass,
    mc_perm_p,
    perm_p,
    perm_null_deltas,
    spatial_z,
    u_statistic,
)


def test_u_statistic_perfect_and_symmetry():
    face, scene = [10, 9, 8, 7], [4, 3, 2, 1]
    assert u_statistic(face, scene) == 16  # perfect separation, 4x4
    # U(face,scene) + U(scene,face) == n*m when there are no ties
    assert u_statistic(scene, face) == 0
    assert u_statistic(face, scene) + u_statistic(scene, face) == 16


def test_u_ties_count_half():
    assert u_statistic([1, 2], [2, 3]) == 0.5  # only 2>... none; tie 2==2 -> 0.5


def test_perm_p_4v4_perfect_is_one_in_70():
    # C(8,4)=70; perfect separation is the single most extreme labeling.
    p = exact_perm_p([10, 9, 8, 7], [4, 3, 2, 1])
    assert abs(p - 1 / 70) < 1e-9  # U=16 -> p≈0.0143


def test_perm_p_4v4_u15_is_two_in_70():
    # One adjacent swap -> U=15; two labelings are >= it. This is the GO threshold.
    p = exact_perm_p([10, 9, 8, 5], [6, 3, 2, 1])  # U = 15
    assert u_statistic([10, 9, 8, 5], [6, 3, 2, 1]) == 15
    assert abs(p - 2 / 70) < 1e-9  # p≈0.0286, clears the p<=0.029 GO bar


def test_perm_p_3v3_perfect_is_one_in_20():
    # The min-viable-subset rule requires U=9/9 -> p=0.05 exactly.
    p = exact_perm_p([10, 9, 8], [3, 2, 1])
    assert abs(p - 1 / 20) < 1e-9


def test_perm_p_3v3_u7_is_the_coin_flip_porthos_caught():
    # face all>1, 9>8&7, 2.5>1 -> U = 3+2+1... construct an exact U=7 and confirm p≈0.20.
    face, scene = [10, 9, 2.5], [8, 7, 1]
    assert u_statistic(face, scene) == 7
    assert abs(exact_perm_p(face, scene) - 4 / 20) < 1e-9  # p=0.20, NOT 0.05


def test_spatial_z_positive_when_roi_is_hot():
    g = np.zeros(100, dtype="float32")
    roi = np.arange(0, 10)
    g[roi] = 5.0
    z = spatial_z(g, roi)
    assert z > 0


def test_spatial_z_accepts_2d_and_flat_zero_std():
    preds = np.ones((5, 50), dtype="float32")  # constant -> std 0 -> defined as 0.0
    assert spatial_z(preds, np.arange(3)) == 0.0


def test_mc_perm_p_matches_exact_on_small_case():
    # 4v4: exact and Monte-Carlo (many perms) should agree within noise.
    face, scene = [10, 9, 8, 5], [6, 3, 2, 1]
    exact = exact_perm_p(face, scene)
    mc = mc_perm_p(face, scene, n_perm=20000, seed=1)
    assert abs(mc - exact) < 0.02


def test_mc_perm_p_tiny_for_clean_separation():
    face = list(range(20, 35))          # 15 clips, all above
    scene = list(range(0, 15))          # 15 clips, all below
    p = mc_perm_p(face, scene, n_perm=5000, seed=2)
    assert p < 0.001                     # far beyond any labelling could reach by chance


def test_mc_perm_p_seeded_deterministic():
    a, b = [3, 1, 4, 1, 5, 9, 2, 6], [2, 7, 1, 8, 2, 8, 1, 8]
    assert mc_perm_p(a, b, n_perm=3000, seed=7) == mc_perm_p(a, b, n_perm=3000, seed=7)


def test_perm_p_dispatches_exact_then_mc():
    # small -> exact (deterministic, matches exact_perm_p); large -> MC (runs, small for clean sep)
    small_f, small_s = [10, 9, 8], [3, 2, 1]
    assert perm_p(small_f, small_s) == exact_perm_p(small_f, small_s)
    big_f, big_s = list(range(20, 35)), list(range(0, 15))
    assert perm_p(big_f, big_s, n_perm=3000, seed=3) < 0.01


def test_iut_requires_both():
    assert iut_pass(0.01, 0.02, alpha=0.025) is True
    assert iut_pass(0.01, 0.20, alpha=0.025) is False   # one contrast fails -> no pass
    assert iut_pass(0.20, 0.01, alpha=0.025) is False


def test_perm_null_deltas_size_and_centering():
    d = perm_null_deltas([3, 2, 1, 0], [-1, -2, -3, -4])
    assert len(d) == 70                      # C(8,4)
    assert abs(d.mean()) < 1e-9              # permutation null is centered at 0
    # observed (first labeling is the identity face-set) is the max separation
    assert d.max() == d[0]


# ---------------------------------------------------------------------------
# D027 / G020 — the corrected, non-compositional statistics.
#
# The first test below is the important one: it ASSERTS the spatial_z artifact.
# The bug becomes a regression test, so the statistic that produced the retracted
# 2026-07-31 NO-GO can never quietly come back as a primary measure.
# ---------------------------------------------------------------------------

from tribe_tools.roi_stats import (  # noqa: E402
    define_froi,
    detection_floor,
    event_locked_contrast,
    event_locked_response,
    glm_contrast_z,
    peak_lag_trs,
    peri_event_timecourse,
    raw_roi_mean,
    roi_minus_reference,
    row_times_from_segments,
)

N_VERTS_TEST = 20484  # fsaverage5, as the real pipeline uses


def _synthetic_brain(seed=0):
    """Baseline map with auditory mass above visual, mirroring scripts/compositional_demo.py."""
    rng = np.random.default_rng(seed)
    idx, cursor = {}, 0
    idx["AUD"] = np.arange(cursor, cursor + 2000)
    cursor += 2000
    for name, size in (("V1", 523), ("FFCr", 58), ("EBA", 116)):
        idx[name] = np.arange(cursor, cursor + size)
        cursor += size
    idx["REST"] = np.arange(cursor, N_VERTS_TEST)
    base = np.full(N_VERTS_TEST, 0.20)
    base[idx["AUD"]] = 1.00
    base[idx["V1"]] = 0.42
    base[idx["FFCr"]] = 0.62
    base[idx["EBA"]] = 0.72
    base += rng.normal(0, 0.03, N_VERTS_TEST)
    return base, idx


def _clip(base, idx, aud_drive, seed, face_effect=0.0):
    rng = np.random.default_rng(seed)
    g = base.copy()
    g[idx["AUD"]] *= 1.0 + aud_drive
    g += rng.normal(0, 0.05, N_VERTS_TEST)
    g *= 1.0 + rng.normal(0, 0.04)
    if face_effect:
        g[idx["FFCr"]] += face_effect
    return g[None, :]


def test_spatial_z_inverts_a_real_effect():
    """THE REGRESSION TEST FOR G020.

    A GENUINE +0.05 face effect is injected into FFCr on FACE clips only, while
    FACE clips also carry more auditory drive (delta 0.24, a STIPULATED
    demonstration level -- see the correction note below).

    What this test pins is the SIGN INVERSION, which is the finding: the
    non-compositional statistics must find the real effect while the
    compositional one reports the wrong sign. Measured over 40 seeds under the
    current generator: spatial_z delta -0.113 +/- 0.035 (0/40 seeds positive,
    worst margin 0.030 below zero), roi_minus_reference +0.049 +/- 0.006
    (min +0.033). The assertions are therefore on signs and loose thresholds by
    design, not on point values.

    .. note:: CORRECTED 2026-08-23 (D030). This docstring previously claimed
       0.24 was "the setting at which compositional_demo.py best matches the
       observed pattern, sum-sq error 0.081", and quoted "spatial_z delta -0.124,
       p=0.9985 / raw +0.0446, p=0.0005 / ref +0.0463, p=0.0005". All of that came
       from ONE draw of a simulator with a module-level mutable RNG, at a D_AUD
       chosen by argmin over 25 single noisy draws on a grid whose LAST point
       (0.24) was the winner. Averaging 225 draws per point over a grid widened to
       0.65 puts the true interior optimum at 0.30 (2.43 sigma clear of its
       neighbour). "p=0.0005" was the estimator floor 1/(n_perm+1)=1/2001, i.e.
       "p < 5e-4". 0.24 is retained here only as a demonstration level; nothing in
       this test depends on it being optimal.
    """
    base, idx = _synthetic_brain()
    ffc, ref = idx["FFCr"], idx["REST"][:2000]
    rng = np.random.default_rng(7)

    face, nonface = [], []
    for i in range(15):
        face.append(_clip(base, idx, 0.30 + 0.24 + rng.normal(0, 0.10), 1000 + i, 0.05))
        nonface.append(_clip(base, idx, 0.30 + rng.normal(0, 0.10), 2000 + i))

    raw_f = [raw_roi_mean(p, ffc) for p in face]
    raw_n = [raw_roi_mean(p, ffc) for p in nonface]
    ref_f = [roi_minus_reference(p, ffc, ref) for p in face]
    ref_n = [roi_minus_reference(p, ffc, ref) for p in nonface]
    z_f = [spatial_z(p, ffc) for p in face]
    z_n = [spatial_z(p, ffc) for p in nonface]

    # the real effect is present and the honest statistics see it
    assert np.mean(raw_f) - np.mean(raw_n) > 0
    assert np.mean(ref_f) - np.mean(ref_n) > 0
    assert perm_p(ref_f, ref_n, n_perm=2000, seed=0) < 0.01

    # and spatial_z reports the OPPOSITE SIGN on the very same data
    assert np.mean(z_f) - np.mean(z_n) < 0, "G020 no longer reproduces — investigate"
    assert perm_p(z_f, z_n, n_perm=2000, seed=0) > 0.5


def test_spatial_z_manufactures_a_pattern_from_zero_face_information():
    """With NO face effect at all, spatial_z still prints a negative FFCr delta."""
    base, idx = _synthetic_brain()
    ffc = idx["FFCr"]
    rng = np.random.default_rng(11)
    face = [_clip(base, idx, 0.30 + 0.24 + rng.normal(0, 0.10), 3000 + i) for i in range(15)]
    nonface = [_clip(base, idx, 0.30 + rng.normal(0, 0.10), 4000 + i) for i in range(15)]

    z_d = np.mean([spatial_z(p, ffc) for p in face]) - np.mean(
        [spatial_z(p, ffc) for p in nonface]
    )
    raw_d = np.mean([raw_roi_mean(p, ffc) for p in face]) - np.mean(
        [raw_roi_mean(p, ffc) for p in nonface]
    )
    assert z_d < 0, "the artifact should manufacture a negative delta from nothing"
    assert abs(raw_d) < abs(z_d), "the raw statistic should stay near zero"


def test_roi_minus_reference_rejects_overlap():
    preds = np.ones((3, 100))
    with pytest.raises(ValueError, match="overlap"):
        roi_minus_reference(preds, np.arange(0, 10), np.arange(5, 20))


def test_roi_minus_reference_cancels_global_gain():
    """An additive per-clip gain must not move the statistic; raw_roi_mean must."""
    rng = np.random.default_rng(3)
    g = rng.normal(0, 1, 500)
    roi, ref = np.arange(0, 50), np.arange(200, 300)
    a = g[None, :]
    b = (g + 4.0)[None, :]  # same brain, +4 everywhere
    assert abs(roi_minus_reference(a, roi, ref) - roi_minus_reference(b, roi, ref)) < 1e-9
    assert abs(raw_roi_mean(a, roi) - raw_roi_mean(b, roi)) > 3.9


class _Seg:
    """Minimal stand-in for a neuralset Segment (only .start is used)."""

    def __init__(self, start):
        self.start = start


def _hrf(t):
    """Canonical SPM double-gamma HRF, peak ~5 s."""
    from math import gamma
    t = np.asarray(t, dtype=float)
    pos = (t ** 5) * np.exp(-t) / gamma(6)
    neg = (t ** 15) * np.exp(-t) / gamma(16)
    return np.where(t < 0, 0.0, pos - neg / 6.0)


def test_row_times_from_segments_extracts_starts():
    segs = [_Seg(0.0), _Seg(1.0), _Seg(2.0)]
    assert row_times_from_segments(segs).tolist() == [0.0, 1.0, 2.0]


def test_row_times_from_segments_rejects_non_monotonic():
    """Multiple concatenated timelines make row->time ambiguous. Fail, do not guess."""
    segs = [_Seg(0.0), _Seg(1.0), _Seg(0.0), _Seg(1.0)]
    with pytest.raises(ValueError, match="strictly increasing"):
        row_times_from_segments(segs)


def test_row_times_from_segments_is_loud_on_a_wrong_object():
    with pytest.raises(AttributeError, match="no usable .start"):
        row_times_from_segments([object(), object()])


def test_event_locked_response_resolves_by_time_not_row_index():
    """THE ROW-DRIFT REGRESSION TEST.

    predict() returns only KEPT rows, so row index != TR index. Here row 3 is
    missing entirely (a dropped segment), which is exactly what shifts every
    later read by one if you index arithmetically. Resolution is by absolute
    time, so the answer must be unaffected.
    """
    times = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0])  # 3.0 s dropped
    preds = np.zeros((len(times), 4))
    preds[times == 5.0] = 9.0
    got = event_locked_response(preds, np.arange(4), onset_times_s=[5.0],
                                row_times_s=times, lag_trs=0)
    assert got.tolist() == [9.0]
    # and the naive arithmetic read (row 5) would have returned the WRONG row
    assert preds[5].mean() == 0.0


def test_event_locked_response_is_loud_when_a_time_has_no_row():
    times = np.arange(0.0, 10.0)
    preds = np.zeros((len(times), 4))
    with pytest.raises(IndexError, match="no prediction row within"):
        event_locked_response(preds, np.arange(4), onset_times_s=[100.0],
                              row_times_s=times, lag_trs=0)


def test_event_locked_response_default_lag_is_zero():
    """Pins the default. TRIBE output is ALREADY hemodynamically aligned
    (README: 'offset by 5 seconds in the past'; defaults.py:67 offset=5), so the
    correct default is 0. A default of 5 double-applies the HRF. See M006."""
    import inspect
    sig = inspect.signature(event_locked_response)
    assert sig.parameters["lag_trs"].default == 0, (
        "lag_trs must default to 0 — TRIBE predictions are already HRF-aligned; "
        "a default of 5 reads onset+10 s and attenuates real effects ~5.5x"
    )


def test_reading_at_lag_5_attenuates_a_real_effect():
    """THE DOUBLE-LAG REGRESSION TEST — quantitative.

    Build predictions under TRIBE's verified convention (row t = BOLD(t+5), i.e.
    already HRF-aligned), inject a real category effect, and confirm that reading
    at lag 0 recovers it while reading at lag 5 destroys it. This pins the reason
    the default is 0 so it cannot be 'tidied' back to 5.
    """
    soa, n_trials, dur = 9, 40, 400
    rng = np.random.default_rng(0)
    times = np.arange(0.0, dur)
    roi = np.zeros(dur)
    onsets_a = np.arange(0, n_trials, 2) * soa          # amplitude 1.0
    onsets_b = np.arange(1, n_trials, 2) * soa          # amplitude 0.5
    tt = np.arange(0, 25.0)
    for on, amp in [(o, 1.0) for o in onsets_a] + [(o, 0.5) for o in onsets_b]:
        # row t already holds the PEAK response to stimulus at t -> shift by -5
        idx = (on + tt - 5).astype(int)
        ok = (idx >= 0) & (idx < dur)
        roi[idx[ok]] += amp * _hrf(tt)[ok]
    preds = np.repeat(roi[:, None], 8, axis=1) + rng.normal(0, 1e-9, (dur, 8))

    keep_a = onsets_a[onsets_a + 5 < dur - 1]
    keep_b = onsets_b[onsets_b + 5 < dur - 1]
    for lag, expect_big in ((0, True), (5, False)):
        a = event_locked_response(preds, np.arange(8), keep_a, times, lag_trs=lag)
        b = event_locked_response(preds, np.arange(8), keep_b, times, lag_trs=lag)
        d = event_locked_contrast(a, [b])
        if expect_big:
            lag0 = d
            assert d > 0.05, f"lag 0 must recover the effect, got {d}"
        else:
            assert abs(d) < 0.35 * abs(lag0), (
                f"lag 5 must strongly attenuate the effect: lag0={lag0:.4f} lag5={d:.4f}"
            )


def test_peak_lag_is_measured_and_lands_at_zero_for_prealigned_output():
    """peak_lag_trs must recover 0 on already-aligned output — the empirical check
    that replaces assuming a lag."""
    dur = 400
    times = np.arange(0.0, dur)
    roi = np.zeros(dur)
    tt = np.arange(0, 25.0)
    onsets = np.arange(2, 40) * 9
    for on in onsets:
        idx = (on + tt - 5).astype(int)
        ok = (idx >= 0) & (idx < dur)
        roi[idx[ok]] += _hrf(tt)[ok]
    preds = np.repeat(roi[:, None], 8, axis=1)
    keep = onsets[(onsets - 2 >= 0) & (onsets + 9 < dur)]
    tc = peri_event_timecourse(preds, np.arange(8), keep, times, pre_trs=2, post_trs=9)
    assert tc.shape == (len(keep), 12)
    # peak_lag_trs pools across categories by construction (C5), so split the
    # events into two pseudo-categories rather than handing it one course.
    half = len(keep) // 2
    assert peak_lag_trs([tc[:half], tc[half:]], pre_trs=2) == 0


def test_event_locked_contrast_averages_categories_not_exemplars():
    """A category with more exemplars must not dominate the 'other categories' baseline."""
    tgt = [10.0, 10.0]
    many = [0.0] * 100
    few = [4.0, 4.0]
    # category means are 0 and 4 -> baseline 2, contrast 8
    assert abs(event_locked_contrast(tgt, [many, few]) - 8.0) < 1e-9


def test_glm_contrast_z_is_unmoved_by_a_distant_region():
    """Unlike spatial_z, a change somewhere else in the brain must not shift the ROI z."""
    rng = np.random.default_rng(5)
    a = rng.normal(0.5, 0.1, (12, 400))
    b = rng.normal(0.0, 0.1, (12, 400))
    roi = np.arange(0, 50)
    before = glm_contrast_z(a, b, roi)
    a2, b2 = a.copy(), b.copy()
    a2[:, 200:] += 9.0  # enormous change far outside the ROI
    after = glm_contrast_z(a2, b2, roi)
    assert abs(before - after) < 1e-9


def test_glm_contrast_z_needs_two_observations():
    with pytest.raises(ValueError, match="standard error"):
        glm_contrast_z(np.ones((1, 10)), np.ones((5, 10)), np.arange(3))


def test_define_froi_picks_the_selective_vertices():
    parcel = np.arange(100, 200)
    a = np.zeros(500)
    b = np.zeros(500)
    a[np.arange(150, 160)] = 1.0  # ten genuinely selective vertices
    froi = define_froi(a, b, parcel, top_n=10)
    assert froi.tolist() == list(range(150, 160))


def test_define_froi_refuses_to_return_the_whole_parcel():
    """M1 REGRESSION. This test previously asserted the OPPOSITE — that
    define_froi(..., top_n=100) on a 5-vertex parcel returns all 5. That blessed
    a silent no-op: with the project's own 58-vertex right-FFC parcel and the
    old default top_n=100, k = min(100, 58) = 58, so the "fROI" was bit-identical
    to the unfixed anatomical parcel and S2 would have reported that while
    believing it had fixed the ROI.

    Boundary is tested on all three sides.
    """
    parcel = np.arange(100, 158)          # 58 vertices, the real FFC parcel size
    a, b = np.zeros(200), np.zeros(200)
    a[np.arange(120, 140)] = 1.0

    # k < parcel size -> valid, and a STRICT subset
    froi = define_froi(a, b, parcel, top_n=30)
    assert froi.size == 30
    assert set(froi.tolist()) < set(parcel.tolist())

    # k == parcel size -> raises (no selection occurred)
    with pytest.raises(ValueError, match="no functional selection"):
        define_froi(a, b, parcel, top_n=58)

    # k > parcel size -> same contract, including the old default
    with pytest.raises(ValueError, match="no functional selection"):
        define_froi(a, b, parcel, top_n=100)

    # and the largest legal request still works
    assert define_froi(a, b, parcel, top_n=57).size == 57


def test_define_froi_rejects_non_finite_localizer():
    """M3. np.argsort sorts NaN LAST ascending; [::-1] promotes it to FIRST, so a
    dead vertex was ranked maximally selective and displaced a real one."""
    parcel = np.arange(100, 158)
    a, b = np.zeros(200), np.zeros(200)
    a[np.arange(120, 140)] = 1.0
    for bad in (np.nan, np.inf, -np.inf):
        a_bad = a.copy(); a_bad[123] = bad
        with pytest.raises(ValueError, match="non-finite"):
            define_froi(a_bad, b, parcel, top_n=10)


def test_glm_contrast_z_uses_welch_not_pooled_se():
    """M2. Pooled SE is anticonservative at unequal n — the design S2 plans.
    Checked against scipy as an independent oracle, and the equal-n invariant
    (Welch == pooled) is pinned so the floor table cannot silently move."""
    from scipy.stats import ttest_ind
    rng = np.random.default_rng(0)
    v = np.arange(40)

    # unequal n, heterogeneous variance -> must equal Welch, NOT pooled
    A = rng.normal(0.5, 0.30, (10, 40)); B = rng.normal(0.0, 0.05, (40, 40))
    mine = glm_contrast_z(A, B, v)
    welch = float(np.mean([ttest_ind(A[:, i], B[:, i], equal_var=False).statistic for i in v]))
    pooled = float(np.mean([ttest_ind(A[:, i], B[:, i], equal_var=True).statistic for i in v]))
    assert mine == pytest.approx(welch, abs=1e-9), "not Welch"
    assert abs(mine - pooled) > 1.0, "indistinguishable from the pooled SE it replaced"
    assert abs(pooled) > abs(mine), "pooled should be the anticonservative one here"

    # equal n -> algebraically identical to pooled; this is what keeps the
    # published 15v15 floor table valid after the change
    C = rng.normal(0.4, 0.3, (12, 40)); D = rng.normal(0.0, 0.1, (12, 40))
    pooled_eq = float(np.mean([ttest_ind(C[:, i], D[:, i], equal_var=True).statistic for i in v]))
    assert glm_contrast_z(C, D, v) == pytest.approx(pooled_eq, abs=1e-9)


def test_glm_contrast_z_has_direction_semantics():
    """S1. `return 0.0` AND a sign flip both passed the previous suite. Pin the
    sign, the antisymmetry, and a magnitude floor so neither mutant survives."""
    rng = np.random.default_rng(5)
    roi = np.arange(20)
    a = rng.normal(0.5, 0.1, (12, 50)); b = rng.normal(0.0, 0.1, (12, 50))
    fwd, rev = glm_contrast_z(a, b, roi), glm_contrast_z(b, a, roi)
    assert fwd > 0, "A > B must give a positive contrast"          # kills return 0.0
    assert rev < 0, "swapping conditions must flip the sign"        # kills the sign flip
    assert fwd == pytest.approx(-rev, rel=1e-9), "must be antisymmetric"
    assert abs(fwd) > 1.0, "a real separation must produce a substantial statistic"


def test_glm_contrast_z_rejects_non_finite():
    rng = np.random.default_rng(6)
    a = rng.normal(0.5, 0.1, (8, 30)); b = rng.normal(0.0, 0.1, (8, 30))
    for bad in (np.nan, np.inf):
        a_bad = a.copy(); a_bad[3, 5] = bad
        with pytest.raises(ValueError, match="non-finite"):
            glm_contrast_z(a_bad, b, np.arange(10))


def test_event_locked_contrast_rejects_2d_target():
    """S6. peri_event_timecourse's (n_events, n_lags) is always in scope beside
    event_locked_response's (n_events,); passing the wrong one silently averaged
    both axes and returned an attenuated contrast."""
    with pytest.raises(ValueError, match="must be 1-D"):
        event_locked_contrast(np.ones((2, 3)), [np.array([1.0, 2.0])])
    with pytest.raises(ValueError, match="must be 1-D"):
        event_locked_contrast(np.array([1.0, 2.0]), [np.ones((2, 3))])
    # the valid 1-D path is untouched
    assert event_locked_contrast([10.0, 10.0], [[0.0] * 100, [4.0, 4.0]]) == pytest.approx(8.0)


def test_event_locked_contrast_raises_on_an_empty_category():
    """Dropping an empty category silently changes the baseline's denominator."""
    with pytest.raises(ValueError, match="empty"):
        event_locked_contrast([1.0, 2.0], [[3.0, 4.0], []])


def test_roi_minus_reference_overlap_guard_survives_a_dtype_mismatch():
    """C7. np.intersect1d compares a boolean mask's VALUES (0/1) against integer
    indices, so a total overlap went undetected and the guard silently passed —
    the module's only defence against an undeclared normaliser."""
    g = np.arange(20.0)[None, :]
    mask = np.zeros(20, dtype=bool); mask[[10, 11, 12]] = True
    with pytest.raises(ValueError, match="overlap"):
        roi_minus_reference(g, mask, np.array([10, 11]))
    # non-overlapping bool mask still works, and agrees with the integer form
    mask2 = np.zeros(20, dtype=bool); mask2[[1, 2, 3]] = True
    assert roi_minus_reference(g, mask2, np.array([10, 11])) == pytest.approx(
        roi_minus_reference(g, np.array([1, 2, 3]), np.array([10, 11])))


def test_peak_lag_must_pool_across_categories():
    """C5. Selecting the peak lag on the target category and then testing at that
    lag is selection on the test statistic — measured type-I 0.0417 against a
    nominal 0.025. The API now makes single-category selection inexpressible."""
    tc_a = np.zeros((5, 12)); tc_a[:, 6] = 1.0     # target peaks late
    tc_b = np.zeros((5, 12)); tc_b[:, 3] = 1.0
    tc_c = np.zeros((5, 12)); tc_c[:, 3] = 1.0     # two categories agree on 3

    # a bare 2-D array (the old signature) is refused, by name
    with pytest.raises(ValueError, match="SINGLE"):
        peak_lag_trs(tc_a, pre_trs=2)
    # so is a single-element sequence
    with pytest.raises(ValueError, match=">= 2 categories"):
        peak_lag_trs([tc_a], pre_trs=2)

    # pooling gives the majority peak (3 -> lag 1), NOT the target's (6 -> lag 4)
    assert peak_lag_trs([tc_a, tc_b, tc_c], pre_trs=2) == 1

    # mismatched lag grids are caught
    with pytest.raises(ValueError, match="lag grid"):
        peak_lag_trs([tc_a, np.zeros((5, 8))], pre_trs=2)


def test_non_finite_policy_is_consistent_across_entry_points():
    """M3. One policy, not five subtly different ones — NaN and both infinities
    are rejected everywhere, with the same error wording."""
    g = np.ones((3, 50))
    roi, ref = np.arange(10), np.arange(20, 30)
    for bad in (np.nan, np.inf, -np.inf):
        dirty = g.copy(); dirty[0, 5] = bad
        with pytest.raises(ValueError, match="non-finite"):
            raw_roi_mean(dirty, roi)
        with pytest.raises(ValueError, match="non-finite"):
            roi_minus_reference(dirty, roi, ref)
        with pytest.raises(ValueError, match="non-finite"):
            perm_p([1.0, 2.0, bad], [0.0, 0.1, 0.2], n_perm=50)


def test_detection_floor_scales_the_right_way():
    """More clips -> smaller detectable effect. The floor must be a real number, not a vibe."""
    small_n = detection_floor(6, noise_sd=1.0, n_sim=60, n_perm=200, seed=0, tol=0.02)
    large_n = detection_floor(24, noise_sd=1.0, n_sim=60, n_perm=200, seed=0, tol=0.02)
    assert small_n > large_n > 0
    # and it must scale with the noise
    louder = detection_floor(6, noise_sd=2.0, n_sim=60, n_perm=200, seed=0, tol=0.02)
    assert louder > small_n
