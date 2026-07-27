"""Gate 0's decision-critical statistics — the exact permutation null in particular.

Porthos's Flaw 1 was that a "7/9 pairs" rule is really Mann-Whitney U at p=0.20, not
0.05. These tests pin the exact numbers the pre-registered GO rule depends on, so the
threshold can never silently drift away from its true p-value.
"""

import numpy as np

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
