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
    tc_a = np.zeros((5, 12)); tc_a[:, 6] = 1.0            # target peaks late
    tc_b = np.zeros((5, 12)); tc_b[:, 3] = 1.0
    tc_c = np.zeros((5, 12)); tc_c[:, 3] = 0.9            # agrees on lag 3, NOT identical
    tc_c[:, 8] = 0.1                                       # (identical courses are rejected)

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


# ==========================================================================
# MECHANISM TESTS (M008)
#
# Phase B's first pass fixed each finding at the example level and four fixes
# still did not close the class. These tests ENUMERATE the entry points by
# introspection rather than naming a fixed list, so a NEW function that accepts
# a selector or array data is covered the moment it is added — and an
# unguarded one fails here rather than in S2.
# ==========================================================================

import inspect as _inspect

from tribe_tools import roi_stats as _R

N_V = 60


_SELECTOR_ARGS = ("verts", "ref_verts", "parcel_verts")


def _selector_entry_points():
    """Every public function taking a vertex selector, found by introspection."""
    out = []
    # Enumerate via _public_functions (module membership, imports declared by
    # name) rather than vars() + a __module__ filter. The filter dropped every
    # public callable that is not a plain `def`, so a module-level partial with a
    # parameter named `verts` was never discovered here either (I5/F2).
    for name, fn in _public_functions().items():
        try:
            params = _inspect.signature(fn).parameters
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        for arg in _SELECTOR_ARGS:
            if arg in params:
                out.append((name, fn, arg))   # one entry PER selector argument
    return out


def _call_with_selector(name, fn, selector, which="verts"):
    """Invoke each selector-taking function with valid data and a given selector."""
    preds = np.tile(np.arange(N_V, dtype=float), (4, 1))
    times = np.arange(4.0)
    if name in ("spatial_z", "raw_roi_mean"):
        return fn(preds, selector)
    if name == "roi_minus_reference":
        # exercise BOTH selector arguments — guarding one of two is the exact
        # mistake this suite exists to catch
        if which == "ref_verts":
            # fixed ROI must be DISJOINT from the selectors under test (10..12),
            # or the overlap guard fires and masks what we are checking
            return fn(preds, np.array([30, 31, 32]), selector)
        return fn(preds, selector, np.array([50, 51]))
    if name == "glm_contrast_z":
        return fn(preds + np.arange(4)[:, None], preds, selector)
    if name == "peri_event_timecourse":
        return fn(preds, selector, [1.0], times, pre_trs=0, post_trs=0)
    if name == "event_locked_response":
        return fn(preds, selector, [1.0], times, lag_trs=0)
    if name == "define_froi":
        # selector IS the parcel here, so top_n must be < its size (M1's guard)
        a = np.zeros(N_V); a[10:13] = [3.0, 2.0, 1.0]
        return fn(a, np.zeros(N_V), selector, top_n=2)
    raise AssertionError(f"unhandled selector entry point {name!r} — extend this helper")


def test_every_selector_entry_point_rejects_the_whole_bad_selector_CLASS():
    """MECHANISM TEST for C7/A0/A8/F4.

    The reported instance was a boolean mask vs integer indices defeating the
    overlap guard in ONE function. The class is: any selector representation that
    aliases or duplicates vertices, at ANY entry point. `define_froi` was the one
    function the original fix never reached — and it is the function M1 was about.
    """
    eps = _selector_entry_points()
    assert len(eps) >= 6, f"expected >=6 selector entry points, introspection found {len(eps)}"
    good = np.array([10, 11, 12])
    mask = np.zeros(N_V, bool); mask[[10, 11, 12]] = True
    # Each case pins the SPECIFIC reason. A generic pytest.raises(ValueError) is
    # not enough: with the ambiguity check removed, an int8 mask trips the
    # *duplicate* check instead, so a generic assertion passes while the fix is
    # gone. Mutation testing caught exactly that in an earlier draft of this test.
    bad_cases = {
        "int8 0/1 mask (ambiguous)":  (mask.astype(np.int8),   "ambiguous"),
        "float 0/1 mask (ambiguous)": (mask.astype(np.float64), "ambiguous"),
        "negative index":             (np.array([10, -1]),      "negative"),
        "duplicate index":            (np.array([10, 10, 11]),  "duplicate"),
        "out of range":               (np.array([10, N_V + 5]), ">= n_vertices"),
        "non-integer float":          (np.array([10.5, 11.0]),  "integer vertex indices"),
        # --- shape and dtype rules the docstring advertised with no coverage (F8).
        # A 2-D BOOLEAN mask whose size equals n_vertices was accepted and read in
        # flat C order, silently selecting a different vertex set (F1).
        "2-D bool mask, size == n":   (np.column_stack([mask[:N_V // 2], mask[N_V // 2:]]),
                                       "must be 1-D"),
        "2-D integer selector":       (np.array([[10, 11], [12, 13]]), "must be 1-D"),
        "3-D bool mask, size == n":   (mask.reshape(2, 3, 10), "must be 1-D"),
        "mask of wrong length":       (np.zeros(N_V - 1, dtype=bool), "boolean mask has length"),
        "python scalar":              (5, "scalar"),
        "numpy scalar":               (np.int64(5), "scalar"),
        "object dtype":               (np.array([10, 11], dtype=object), "dtype"),
        "string dtype":               (np.array(["10", "11"]), "dtype"),
    }
    for name, fn, _arg in eps:
        # the good selector must still work in this argument position
        _call_with_selector(name, fn, good, _arg)
        # and a genuine boolean mask must be accepted and mean the same thing
        _call_with_selector(name, fn, mask, _arg)
        for label, (bad, why) in bad_cases.items():
            with pytest.raises(ValueError, match=why):
                _call_with_selector(name, fn, bad, _arg)


def _equivalent_selector_representations(verts=(10, 11, 12)):
    """Every accepted encoding of ONE vertex set, including a SHUFFLED index array.

    The predecessor of this harness compared a mask against np.flatnonzero(mask),
    which is already ascending — so it could not detect that integer arrays
    preserved caller order while masks did not (I1/F7).
    """
    mask = np.zeros(N_V, bool); mask[list(verts)] = True
    return {
        "bool mask":        mask,
        "ascending idx":    np.array(verts),
        "shuffled idx":     np.array(list(verts)[::-1]),
        "python list":      list(verts),
        "python tuple":     tuple(verts),
        "int8 indices":     np.array(verts, dtype=np.int8),
        "whole-valued float": np.array(verts, dtype=float),
        "non-contiguous":   np.array([v for v in verts for _ in (0, 1)])[::2],
    }


def test_selector_canonicalises_identically_across_every_representation():
    """I1. A selector denotes a SET, so every encoding of one set must canonicalise
    to one array — 1-D, ascending, unique, integer."""
    canon = {k: _R._as_vertex_indices(v, n_vertices=N_V)
             for k, v in _equivalent_selector_representations().items()}
    expect = np.array([10, 11, 12])
    for label, got in canon.items():
        assert np.array_equal(got, expect), f"{label} canonicalised to {got}, expected {expect}"
        assert got.dtype.kind == "i", f"{label} produced dtype {got.dtype}"


def test_every_entry_point_agrees_across_every_selector_representation():
    """I1, applied per (function, selector argument). Representation must never
    change the answer, because order carries no meaning in a vertex set."""
    reps = _equivalent_selector_representations()
    for name, fn, arg in _selector_entry_points():
        results = {label: np.asarray(_call_with_selector(name, fn, sel, arg), dtype=float)
                   for label, sel in reps.items()}
        ref_label, ref = next(iter(results.items()))
        for label, got in results.items():
            assert np.allclose(got, ref), (
                f"{name}({arg}) gave {got} for '{label}' but {ref} for '{ref_label}' — "
                "two encodings of the same vertex set disagree"
            )


def test_froi_is_deterministic_under_ties_across_representations():
    """I1/F7. With a tied localizer contrast an unstable argsort returned different
    fROIs for two encodings of one parcel. The contract: rank by descending
    contrast, break ties by LOWEST vertex index."""
    tied = np.zeros(N_V)
    parcel = (0, 1, 2)
    outs = {label: define_froi(tied, tied, sel, top_n=2)
            for label, sel in _equivalent_selector_representations(parcel).items()}
    first = next(iter(outs.values()))
    for label, got in outs.items():
        assert np.array_equal(got, first), f"{label} gave {got}, another encoding gave {first}"
    assert np.array_equal(first, [0, 1]), (
        f"tie-break must prefer the lowest vertex index, got {first}"
    )


def test_froi_is_returned_in_canonical_ascending_order():
    """define_froi returns a SELECTOR, so it must satisfy the selector contract it
    will be fed back into. Ranking is by contrast; the returned set is ascending."""
    # Contrast RANK must be the reverse of vertex order, or "sorted output" and
    # "selection order" coincide and the assertion proves nothing.
    loc_a = np.zeros(N_V); loc_a[[22, 31, 40]] = [3.0, 5.0, 9.0]
    out = define_froi(loc_a, np.zeros(N_V), np.arange(20, 50), top_n=3)
    assert np.array_equal(out, np.sort(out)), f"fROI not ascending: {out}"
    assert set(out.tolist()) == {22, 31, 40}
    # and it round-trips as a selector without further normalisation
    assert np.array_equal(_R._as_vertex_indices(out, n_vertices=N_V), out)


def test_permutation_p_can_never_be_exactly_zero():
    """The (ge + 1) / (n_perm + 1) estimator keeps the p-value valid. Reporting
    p = 0 from a finite permutation set is a claim the data cannot support."""
    face, other = [10.0] * 8, [0.0] * 8          # perfect separation
    p = mc_perm_p(face, other, n_perm=200, seed=0)
    assert p > 0.0, "Monte-Carlo p reached exactly zero"
    assert p == pytest.approx(1 / 201), f"estimator floor should be 1/(n_perm+1), got {p}"
    assert 0.0 < exact_perm_p(face, other) <= 1.0


def test_contrast_without_a_comparison_category_raises_not_returns_a_bare_mean():
    """An 'contrast' computed against nothing is just the target mean wearing the
    name of a contrast — the most misleading thing this module could return."""
    with pytest.raises(ValueError, match="at least one other category"):
        event_locked_contrast([1.0, 2.0, 3.0], [])


def test_reference_region_is_guarded_as_well_as_the_roi():
    """I2, per REGION. roi_minus_reference reads two disjoint regions and must
    guard both: poisoning only the reference left the ROI-side guard satisfied."""
    preds = np.tile(np.arange(N_V, dtype=float), (4, 1))
    roi, ref = np.array([10, 11, 12]), np.array([50, 51])
    for label, bad_vertex in (("inside the ROI", 10), ("inside the reference", 50)):
        dirty = preds.copy(); dirty[0, bad_vertex] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            roi_minus_reference(dirty, roi, ref)
    # a non-finite value in NEITHER region is genuinely irrelevant to this
    # statistic, so rejecting it would be stricter than the mechanism requires
    elsewhere = preds.copy(); elsewhere[0, 30] = np.nan
    assert np.isfinite(roi_minus_reference(elsewhere, roi, ref))


def test_iterable_representations_of_one_sample_agree():
    """I3. Any iterable accepted at a public boundary is materialised exactly once.
    Validating one copy and then computing on the original argument exhausted
    single-pass iterables and returned U=0.0 — finite, wrong, and maximally
    anti-selective (F6). 'It did not crash' is not the assertion; the VALUE is."""
    f, s = [3.0, 4.0, 7.0, 9.0], [1.0, 2.0, 5.0, 6.0]
    cases = {
        "list":      (lambda: list(f), lambda: list(s)),
        "tuple":     (lambda: tuple(f), lambda: tuple(s)),
        "ndarray":   (lambda: np.array(f), lambda: np.array(s)),
        "generator": (lambda: (x for x in f), lambda: (x for x in s)),
        "iterator":  (lambda: iter(list(f)), lambda: iter(list(s))),
    }
    for fname, fn in (("u_statistic", u_statistic),
                      ("exact_perm_p", exact_perm_p),
                      ("perm_p", perm_p),
                      ("perm_null_deltas", perm_null_deltas)):
        vals = {label: np.asarray(fn(mk_f(), mk_s()), dtype=float)
                for label, (mk_f, mk_s) in cases.items()}
        ref = vals["list"]
        for label, got in vals.items():
            assert np.allclose(got, ref), f"{fname}: {label} gave {got}, list gave {ref}"
    # the specific regression: the generator case must be the real statistic, and
    # emphatically not 0.0, which is what discarding the validated copy produced
    assert u_statistic((x for x in f), (x for x in s)) == u_statistic(f, s) == 12.0
    a = mc_perm_p((x for x in f), (x for x in s), n_perm=200)
    assert a == mc_perm_p(f, s, n_perm=200)


def test_empty_selection_raises_at_every_entry_point_not_just_two():
    """MECHANISM TEST for A2. An all-False mask has len == n_vertices, so an
    empty-check placed BEFORE normalisation let it through and returned nan.
    Two functions checked first and two checked after — the policy must be one."""
    empty_mask = np.zeros(N_V, bool)
    for name, fn, _arg in _selector_entry_points():
        with pytest.raises(ValueError, match="empty"):
            _call_with_selector(name, fn, empty_mask, _arg)


def _nonfinite_entry_points():
    """Every public function that consumes array data, with a NaN-poisoned call."""
    preds = np.tile(np.arange(N_V, dtype=float), (4, 1))
    dirty = preds.copy(); dirty[0, 10] = np.nan
    roi, ref = np.array([10, 11, 12]), np.array([50, 51])
    times = np.arange(4.0)
    tc = np.zeros((3, 5)); tc[:, 2] = 1.0
    tc_bad = tc.copy(); tc_bad[0, 0] = np.nan
    tc_b = np.zeros((3, 5)); tc_b[:, 1] = 1.0
    nanvals = [1.0, 2.0, np.nan, 3.0]
    ok = [0.1, 0.2, 0.3, 0.4]
    parcel = np.arange(20, 50)
    nan_in_parcel = np.where(np.arange(N_V) == 25, np.nan, 0.0)
    # Keys are "function:parameter". Coverage is keyed on the PAIR, not on the
    # function name: keying on the name let one poisoned argument satisfy
    # completeness for a function with several, which is the exact
    # roi_minus_reference-two-selectors precedent recurring inside the mechanism
    # written to close it (I5/F3).
    return {
        "spatial_z:preds":            lambda: _R.spatial_z(dirty, roi),
        "raw_roi_mean:preds":         lambda: _R.raw_roi_mean(dirty, roi),
        "roi_minus_reference:preds":  lambda: _R.roi_minus_reference(dirty, roi, ref),
        "glm_contrast_z:preds_a":     lambda: _R.glm_contrast_z(dirty, preds, roi),
        "glm_contrast_z:preds_b":     lambda: _R.glm_contrast_z(preds, dirty, roi),
        "peri_event_timecourse:preds":         lambda: _R.peri_event_timecourse(dirty, roi, [1.0], times, 0, 0),
        "peri_event_timecourse:onset_times_s": lambda: _R.peri_event_timecourse(preds, roi, [np.nan], times, 0, 0),
        "peri_event_timecourse:row_times_s":   lambda: _R.peri_event_timecourse(preds, roi, [1.0], [0.0, 1.0, np.nan, 3.0], 0, 0),
        "event_locked_response:preds":         lambda: _R.event_locked_response(dirty, roi, [1.0], times, 0),
        "event_locked_response:onset_times_s": lambda: _R.event_locked_response(preds, roi, [np.nan], times, 0),
        "event_locked_response:row_times_s":   lambda: _R.event_locked_response(preds, roi, [1.0], [0.0, 1.0, np.nan, 3.0], 0),
        "event_locked_contrast:target_responses": lambda: _R.event_locked_contrast(nanvals, [ok]),
        "event_locked_contrast:other_responses":  lambda: _R.event_locked_contrast(ok, [nanvals]),
        # NaN INSIDE the parcel: outside it cannot affect the selection, so
        # rejecting that would be stricter than the mechanism requires.
        "define_froi:loc_a":    lambda: _R.define_froi(nan_in_parcel, np.zeros(N_V), parcel, top_n=5),
        "define_froi:loc_b":    lambda: _R.define_froi(np.zeros(N_V), nan_in_parcel, parcel, top_n=5),
        "peak_lag_trs:category_timecourses": lambda: _R.peak_lag_trs([tc_bad, tc_b], pre_trs=2),
        "u_statistic:face_vals":        lambda: _R.u_statistic(nanvals, ok),
        "u_statistic:scene_vals":       lambda: _R.u_statistic(ok, nanvals),
        "exact_perm_p:face_vals":       lambda: _R.exact_perm_p(nanvals, ok),
        "exact_perm_p:scene_vals":      lambda: _R.exact_perm_p(ok, nanvals),
        "mc_perm_p:face_vals":          lambda: _R.mc_perm_p(nanvals + ok, ok + ok, n_perm=20),
        "mc_perm_p:other_vals":         lambda: _R.mc_perm_p(ok + ok, nanvals + ok, n_perm=20),
        "perm_null_deltas:face_vals":   lambda: _R.perm_null_deltas(nanvals, ok),
        "perm_null_deltas:scene_vals":  lambda: _R.perm_null_deltas(ok, nanvals),
        "row_times_from_segments:segments": lambda: _R.row_times_from_segments(
            [_Seg(0.0), _Seg(np.nan), _Seg(2.0)]),
    }


def test_non_finite_is_rejected_at_EVERY_entry_point_the_docstring_claims():
    """MECHANISM TEST for M3.

    The original test checked 3 of ~10 call sites while its docstring claimed
    "rejected everywhere", and its perm_p call never even reached mc_perm_p.
    A test that claims universal coverage and does not prove it is worse than no
    test, because it retires the question. This enumerates every path.
    """
    unguarded = []
    for label, call in _nonfinite_entry_points().items():
        try:
            call()
            unguarded.append(label)
        except ValueError:
            pass                      # correct: rejected
        except Exception as exc:      # wrong error type is also a finding
            unguarded.append(f"{label} (raised {type(exc).__name__}, want ValueError)")
    assert not unguarded, "non-finite input NOT rejected at: " + "; ".join(unguarded)


def test_inf_is_treated_exactly_like_nan_everywhere():
    """+/-inf must not be a second, weaker policy."""
    preds = np.tile(np.arange(N_V, dtype=float), (4, 1))
    roi = np.array([10, 11, 12])
    for bad in (np.inf, -np.inf):
        dirty = preds.copy(); dirty[0, 10] = bad
        for fn in (_R.spatial_z, _R.raw_roi_mean):
            with pytest.raises(ValueError, match="non-finite"):
                fn(dirty, roi)
        with pytest.raises(ValueError, match="non-finite"):
            _R.u_statistic([1.0, bad], [0.0, 0.5])


def test_event_locked_contrast_rejects_bad_shapes_on_BOTH_arguments():
    """MECHANISM TEST for S6/F3. The original fix guarded target_responses only;
    the identical hazard reaches the arithmetic through other_responses."""
    good = np.array([1.0, 2.0])
    tc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="1-D"):                 # target 2-D
        event_locked_contrast(tc, [good])
    with pytest.raises(ValueError, match="1-D"):                 # one other 2-D
        event_locked_contrast(good, [tc])
    with pytest.raises(ValueError, match="SEQUENCE"):             # bare 2-D AS other_responses
        event_locked_contrast(good, tc)
    with pytest.raises(ValueError, match="scalar"):               # 0-d, was a TypeError
        event_locked_contrast(np.array(3.0), [good])
    with pytest.raises(ValueError, match="scalar"):
        event_locked_contrast(good, [np.array(3.0)])
    for empty in ([], np.array([])):                              # empty either side
        with pytest.raises(ValueError, match="empty"):
            event_locked_contrast(good, [empty])
    assert event_locked_contrast([10.0, 10.0], [[0.0] * 100, [4.0, 4.0]]) == pytest.approx(8.0)


def test_peak_lag_rejects_every_degenerate_configuration_that_satisfies_the_count():
    """MECHANISM TEST for C5/F6. ">= 2 categories" is a formality: each of these
    satisfies it while pooling to exactly the target's own course (measured
    type-I 0.2032 vs a nominal 0.025)."""
    # Events DIFFER within each category, so a guard that inspects only the first
    # event rather than the mean course cannot pass this fixture.
    rng = np.random.default_rng(7)
    tc = rng.normal(0.0, 0.05, (6, 12)); tc[:, 6] += 1.0 + rng.normal(0, 0.01, 6)
    other = rng.normal(0.0, 0.05, (6, 12)); other[:, 3] += 1.0 + rng.normal(0, 0.01, 6)
    for label, courses, why in [
        ("same object twice",     [tc, tc],                       "same mean course"),
        ("a copy",                [tc, tc.copy()],                "same mean course"),
        # I4/F5: the semantic degeneracy is "the pooled course carries nothing the
        # target's own course did not". These are all argmax-identical to the
        # target alone, and every one of them passed the old syntactic check.
        ("rows duplicated",       [tc, np.vstack([tc, tc])],      "same mean course"),
        ("rescaled",              [tc, tc * 2.0],                 "same mean course"),
        ("constant offset",       [tc, tc + 1.0],                 "same mean course"),
        ("rescaled and offset",   [tc, tc * 3.0 + 7.0],           "same mean course"),
        ("rows duplicated 3x, rescaled",
                                  [tc, np.vstack([tc, tc, tc]) * 0.5], "same mean course"),
        ("all-zero filler",       [tc, np.zeros_like(tc)],        "flat"),
        ("constant filler",       [tc, np.full_like(tc, 7.0)],    "flat"),
        ("an empty category",     [tc, np.zeros((0, 12))],        "0 events"),
    ]:
        # match= pins WHICH rule fired: a bare pytest.raises(ValueError) let a
        # reverted guard survive because a different guard raised instead.
        with pytest.raises(ValueError, match=why):
            peak_lag_trs(courses, pre_trs=2)
    # A negatively-proportional course is NOT degenerate — it carries real
    # opposing information — so the guard must not reject it.
    peak_lag_trs([tc, -tc], pre_trs=2)
    # Genuinely different categories still work, and the expected value is pinned
    # EXACTLY. This previously asserted `in (1, 4)`, where 4 is the target-only
    # answer -- the C5 defect the function exists to prevent -- so the assertion
    # could not distinguish correct pooling from the bug (F9). A third category
    # agreeing with `other` breaks the two-category argmax tie, so the pooled peak
    # is unambiguous.
    third = rng.normal(0.0, 0.05, (6, 12)); third[:, 3] += 1.0 + rng.normal(0, 0.01, 6)
    assert peak_lag_trs([tc, other, third], pre_trs=2) == 1
    # The pooled course must come from the per-category MEAN, not from a single
    # event standing in for its category. `atypical` has one outlier event peaking
    # at lag 9 and five peaking at 3, so a guard or a pooling step that inspects
    # only the first event gets a different answer than the mean does.
    atypical = rng.normal(0.0, 0.02, (6, 12))
    atypical[0, 9] += 3.0     # 3/6 = 0.50 at lag 9 in the mean ...
    atypical[1:, 3] += 1.0    # ... vs 5/6 = 0.83 at lag 3, an unambiguous margin
    assert int(np.argmax(atypical[0])) == 9 and int(np.argmax(atypical.mean(axis=0))) == 3
    assert peak_lag_trs([tc, atypical, third], pre_trs=2) == 1


def test_resolve_rows_rejects_non_finite_directly_not_only_via_its_callers():
    """MECHANISM TEST for A1, isolated.

    `peri_event_timecourse` guards its onsets before `_resolve_rows` ever sees
    them, so testing only through the caller masks whether `_resolve_rows` itself
    is safe — mutation testing showed a revert of its guard surviving. The
    function's own docstring promises "or raise. Never silently approximate",
    and `err > tol` is False for NaN, so a NaN silently resolved to a row.
    """
    from tribe_tools.roi_stats import _resolve_rows
    rt = np.arange(10.0)
    assert _resolve_rows(rt, [3.0]).tolist() == [3]          # the valid path still works
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="non-finite"):
            _resolve_rows(rt, [bad])
        with pytest.raises(ValueError, match="non-finite"):
            _resolve_rows(np.array([0.0, 1.0, bad, 3.0]), [1.0])


def test_row_times_from_segments_rejects_non_finite_directly():
    """A NaN start also defeats `np.diff(t) <= 0`, so the strictly-increasing
    guard passed and rows then resolved to the wrong times."""
    with pytest.raises(ValueError, match="non-finite"):
        row_times_from_segments([_Seg(0.0), _Seg(np.nan), _Seg(2.0)])


# ==========================================================================
# THE DISCOVERY CONTRACT — what these tests do and do not find automatically
#
# CLAIM (precise): every module-level, public, non-imported function in
# tribe_tools.roi_stats whose signature contains a parameter named in
# _SELECTOR_ARGS is automatically included in the selector tests; and every
# such function consuming array data must appear in the hand-written
# _nonfinite_entry_points map, which is CHECKED FOR COMPLETENESS below.
#
# TWO SEPARATE GUARANTEES, and the first does not imply the second:
#   function coverage       — we found every function in the selector family;
#   representation coverage — for each one we exercised every selector argument
#                             position and every accepted encoding.
# The previous version of this block asserted only the first while claiming both.
#
# Discovered:      every public callable that is a MEMBER of roi_stats, by dir(),
#                  regardless of how it was constructed. Enumeration deliberately
#                  does NOT filter on __module__: that dropped module-level
#                  functools.partial / np.vectorize / callable instances, one of
#                  which passed every safeguard with a parameter named `verts`.
# NOT discovered:  names starting with "_" (private helpers) — tested explicitly
#                  names declared in _IMPORTED_CALLABLES (imported, not defined here)
# Fails loudly:    an unclassified parameter on any public callable — every name
#                  must be declared a selector, array data, or a scalar, so a NEW
#                  parameter is a failure by default rather than a silent miss.
#                  A function whose selector argument goes unvalidated fails here
#                  rather than in S2.
# Genuine limit:   a public callable with NO inspectable signature is reported as
#                  unclassified rather than skipped, but cannot be auto-exercised.
# ==========================================================================

# Public functions that legitimately take NO array data and NO selector.
_NO_ARRAY_INPUT = {"iut_pass", "detection_floor", "perm_p"}
# perm_p is a dispatcher: it forwards to exact_perm_p / mc_perm_p, both of which
# ARE in the map. Recorded here rather than silently omitted.

# Public callables that are imported into roi_stats rather than defined by it.
# Declared BY NAME. Enumeration below walks module membership rather than
# filtering on __module__, because __module__ filtering silently dropped every
# callable that is not a plain `def` -- a module-level functools.partial,
# np.vectorize wrapper, callable class instance, or any decorator that does not
# preserve __module__. One of those with a parameter named `verts` passed all
# three safeguards with a green suite (I5/F2).
_IMPORTED_CALLABLES = {"comb", "combinations"}

# Parameters that DO carry array data and therefore need non-finite coverage at
# their own argument position (selectors are covered by the selector harness).
_ARRAY_DATA_PARAMS = {
    "preds", "preds_a", "preds_b", "loc_a", "loc_b",
    "target_responses", "other_responses", "onset_times_s", "row_times_s",
    "category_timecourses", "face_vals", "scene_vals", "other_vals", "segments",
}

# Parameters that carry no array data. Declared explicitly so that a NEW
# parameter is a failure by default rather than a silent miss.
_NON_ARRAY_PARAMS = {
    "top_n", "lag_trs", "pre_trs", "post_trs", "n_perm", "seed", "alpha",
    "power", "n_sim", "tol", "max_effect", "n_per_group", "noise_sd",
    "p_a", "p_b",
}


def _public_callables():
    """Every public callable that is a MEMBER of roi_stats, however it was made.

    Deliberately does not filter on __module__ — see _IMPORTED_CALLABLES.
    """
    return {n: getattr(_R, n) for n in dir(_R)
            if not n.startswith("_") and callable(getattr(_R, n))}


def _public_functions():
    """Public callables roi_stats itself defines (imports declared and excluded)."""
    return {n: f for n, f in _public_callables().items()
            if n not in _IMPORTED_CALLABLES}


def test_discovery_finds_the_functions_it_claims_to():
    """The discovery mechanism needs its own test, or the coverage claim rests on
    an untested helper."""
    pub = _public_functions()
    assert len(pub) >= 15, f"discovery found only {len(pub)} public functions"
    # no classes hiding methods that discovery would miss
    assert not [n for n, f in pub.items() if _inspect.isclass(f)], "a class appeared; discovery only walks functions"
    # every discovered selector function really does take a selector
    for name, fn, arg in _selector_entry_points():
        assert arg in _inspect.signature(fn).parameters
    # and the reverse: no public function takes a selector arg without being discovered
    found = {n for n, _, _ in _selector_entry_points()}
    for name, fn in pub.items():
        if set(_inspect.signature(fn).parameters) & set(_SELECTOR_ARGS):
            assert name in found, f"{name} takes a selector but discovery missed it"


def test_selector_arg_names_still_cover_every_selector_parameter():
    """Known limit made loud: discovery keys on parameter NAME. If a new function
    introduces a differently-named selector this fails, rather than silently
    skipping it."""
    # Inverted: EVERY parameter must be classified, so an unrecognised NEW name
    # fails by default. The previous substring heuristic ("vert"/"parcel"/"roi")
    # passed for any plausible alternative -- `mask`, `labels`, `nodes`,
    # `region_a`, `selection` -- and `froi_a` only failed by the luck of
    # containing "roi" (I5/F2).
    known = set(_SELECTOR_ARGS) | _NON_ARRAY_PARAMS | _ARRAY_DATA_PARAMS
    unclassified = set()
    for name, fn in _public_functions().items():
        try:
            params = _inspect.signature(fn).parameters
        except (TypeError, ValueError):  # pragma: no cover - defensive
            unclassified.add(f"{name}(<no signature>)")
            continue
        for pname in params:
            if pname not in known:
                unclassified.add(f"{name}({pname})")
    assert not unclassified, (
        "these parameters are in no declared category, so the harness does not know "
        "whether they are selectors, array data, or scalars. Add each to "
        "_SELECTOR_ARGS, _ARRAY_DATA_PARAMS, or _NON_ARRAY_PARAMS: "
        f"{sorted(unclassified)}"
    )


def test_discovery_machinery_catches_a_planted_partial(monkeypatch):
    """The machinery needs a POSITIVE test, or 'it found everything' is unfalsifiable.

    A module-level functools.partial with a parameter named `verts` carries no
    __module__, so the previous __module__ filter dropped it from all three
    safeguards while it sat in the public namespace, unvalidated (I5/F2)."""
    import functools

    def _leaky(preds, verts, scale=1.0):
        return float(np.asarray(preds)[..., np.asarray(verts)].mean() * scale)

    monkeypatch.setattr(_R, "roi_dice_fast", functools.partial(_leaky, scale=2.0),
                        raising=False)
    assert "roi_dice_fast" in _public_callables(), "enumeration missed a partial"
    assert "roi_dice_fast" in _public_functions(), "a partial escaped classification"
    # and it is picked up as a selector entry point, so the selector rules apply
    assert "roi_dice_fast" in {n for n, _, _ in _selector_entry_points()}


def test_discovery_machinery_catches_a_planted_unclassified_parameter(monkeypatch):
    """A new selector under a name nobody predicted — `mask`, `labels`, `nodes`,
    `region_a` — must fail by default. The previous substring heuristic passed
    every one of these."""
    def _leaky(preds, region_a):
        return float(np.asarray(preds)[..., np.asarray(region_a)].mean())

    monkeypatch.setattr(_R, "roi_dice", _leaky, raising=False)
    known = set(_SELECTOR_ARGS) | _NON_ARRAY_PARAMS | _ARRAY_DATA_PARAMS
    unclassified = {f"{n}({p})" for n, f in _public_functions().items()
                    for p in _inspect.signature(f).parameters if p not in known}
    assert "roi_dice(region_a)" in unclassified, (
        "an unrecognised parameter name was silently skipped instead of failing"
    )


def test_the_nonfinite_map_is_complete_not_merely_long():
    """The predecessor of this suite claimed 'rejected everywhere' while covering
    3 of ~10 call sites. _nonfinite_entry_points is a HAND-WRITTEN map, so the
    same overclaim is possible — this cross-checks it against introspection so an
    unlisted public function fails here instead of going untested."""
    covered = set(_nonfinite_entry_points())
    required = set()
    for name, fn in _public_functions().items():
        if name in _NO_ARRAY_INPUT:
            continue
        for pname in _inspect.signature(fn).parameters:
            if pname in _ARRAY_DATA_PARAMS:
                required.add(f"{name}:{pname}")
    missing = sorted(required - covered)
    assert not missing, (
        "these (function, argument) pairs consume array data and are never poisoned. "
        "Keying coverage on the function NAME let one poisoned argument stand in for "
        f"all of them -- the roi_minus_reference precedent: {missing}"
    )
    # every key in the map must correspond to a live (function, parameter) pair,
    # so a rename cannot leave a test silently pointing at nothing
    stale = sorted(k for k in covered if k not in required)
    assert not stale, f"map keys that are not live (function, parameter) pairs: {stale}"
    # and the exemption list must stay honest — every name in it must still exist
    assert not (_NO_ARRAY_INPUT - set(_public_functions())), "stale name in _NO_ARRAY_INPUT"
    assert not (_IMPORTED_CALLABLES - set(_public_callables())), "stale name in _IMPORTED_CALLABLES"
