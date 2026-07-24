"""Gate 0's decision-critical statistics — the exact permutation null in particular.

Porthos's Flaw 1 was that a "7/9 pairs" rule is really Mann-Whitney U at p=0.20, not
0.05. These tests pin the exact numbers the pre-registered GO rule depends on, so the
threshold can never silently drift away from its true p-value.
"""

import numpy as np

from tribe_tools.roi_stats import (
    exact_perm_p,
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


def test_perm_null_deltas_size_and_centering():
    d = perm_null_deltas([3, 2, 1, 0], [-1, -2, -3, -4])
    assert len(d) == 70                      # C(8,4)
    assert abs(d.mean()) < 1e-9              # permutation null is centered at 0
    # observed (first labeling is the identity face-set) is the max separation
    assert d.max() == d[0]
