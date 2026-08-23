"""Phase A invariants for the simulation that produces the project's published numbers.

Every test here is written to FAIL if the pre-Phase-A behaviour is reintroduced:
a module-level mutable RNG, a single-draw headline, or a boundary-selected D_AUD.
"""

import numpy as np
import pytest

from scripts import compositional_demo as cd
from scripts import sensitivity_surface as ss


# --------------------------------------------------------------------------
# Reproducibility: the failure that made every published number unrepeatable
# --------------------------------------------------------------------------

def test_module_has_no_mutable_shared_rng():
    """The original bug: `RNG = default_rng(0)` at module scope, consumed by both
    build_brain() and run(), so every call advanced global state."""
    assert not hasattr(cd, "RNG"), (
        "a module-level RNG is back; it makes every published number a "
        "single unreproducible draw (two build_brain() calls differed by 0.17)"
    )


def test_build_brain_is_deterministic_and_seed_sensitive():
    assert np.array_equal(cd.build_brain()[0], cd.build_brain()[0])
    assert np.array_equal(cd.build_brain(seed=5)[0], cd.build_brain(seed=5)[0])
    assert not np.array_equal(cd.build_brain(seed=5)[0], cd.build_brain(seed=6)[0])


def test_run_is_reproducible_across_repeated_calls():
    """Would fail under the old global RNG: consecutive identical calls diverged."""
    a = cd.run(0.24, seed=11, compute_p=False)["FFCr"]["z_d"]
    b = cd.run(0.24, seed=11, compute_p=False)["FFCr"]["z_d"]
    assert a == b, "run() is not reproducible at a fixed seed"
    assert cd.run(0.24, seed=12, compute_p=False)["FFCr"]["z_d"] != a


def test_rng_propagates_through_the_whole_path():
    """Interleaving other calls must not perturb a seeded result — i.e. no hidden
    global state anywhere in build_brain -> clip_map -> run."""
    ref = cd.run(0.20, seed=3, compute_p=False)["FFCr"]["z_d"]
    for _ in range(3):
        cd.build_brain(seed=99)
        cd.run(0.55, seed=42, compute_p=False)
    assert cd.run(0.20, seed=3, compute_p=False)["FFCr"]["z_d"] == ref


# --------------------------------------------------------------------------
# The correlation model: rho must be real, and must not inflate variance
# --------------------------------------------------------------------------

def test_parcel_noise_holds_marginal_variance_fixed():
    """If rho changed the marginal variance, every floor comparison across rho
    would be confounded by total noise rather than by its structure."""
    _, idx = cd.build_brain()
    for rho in (0.0, 0.5, 0.9):
        rng = np.random.default_rng(2)
        s = np.array([cd._parcel_noise(rng, idx, 0.05, rho) for _ in range(1500)])
        assert s.var(axis=0).mean() == pytest.approx(0.05 ** 2, rel=0.05)


def test_parcel_noise_correlation_tracks_rho_within_parcel_only():
    _, idx = cd.build_brain()
    rng = np.random.default_rng(3)
    s = np.array([cd._parcel_noise(rng, idx, 0.05, 0.6) for _ in range(2500)])
    f = idx["FFCr"][:30]
    c = np.corrcoef(s[:, f].T)
    within = (c.sum() - np.trace(c)) / (len(f) ** 2 - len(f))
    assert within == pytest.approx(0.6, abs=0.08)
    cross = np.corrcoef(np.hstack([s[:, idx["AUD"][:15]], s[:, idx["V1"][:15]]]).T)[:15, 15:]
    assert abs(cross.mean()) < 0.08


def test_rho_is_actually_plumbed_into_run():
    a = cd.run(0.24, seed=4, rho=0.0, compute_p=False)["FFCr"]["z_d"]
    b = cd.run(0.24, seed=4, rho=0.9, compute_p=False)["FFCr"]["z_d"]
    assert a != b, "rho is not reaching clip_map"


def test_rho_is_validated():
    _, idx = cd.build_brain()
    with pytest.raises(ValueError, match="rho"):
        cd._parcel_noise(np.random.default_rng(0), idx, 0.05, 1.5)


# --------------------------------------------------------------------------
# Permutation p-value floor semantics
# --------------------------------------------------------------------------

def test_fmt_p_reports_the_estimator_floor_as_a_bound():
    """`p=0.0005` at n_perm=2000 is 1/2001, the floor -- not a measurement."""
    assert cd.P_FLOOR == pytest.approx(1 / 2001)
    assert cd.fmt_p(cd.P_FLOOR).startswith("<"), "the floor must render as an upper bound"
    assert cd.fmt_p(0.9186) == "0.9186"
    assert not cd.fmt_p(0.02).startswith("<")


# --------------------------------------------------------------------------
# Multi-draw reporting: a single draw is not a result
# --------------------------------------------------------------------------

def test_run_many_aggregates_and_is_reproducible():
    a = cd.run_many(0.24, range(6))
    b = cd.run_many(0.24, range(6))
    assert a["FFCr"]["z_d"]["mean"] == b["FFCr"]["z_d"]["mean"]
    assert a["FFCr"]["z_d"]["n"] == 6
    assert a["FFCr"]["z_d"]["sd"] > 0


def test_single_draw_is_demonstrably_not_representative():
    """The empirical justification for banning single-draw headlines: at a fixed
    setting the spread across draws dwarfs the grid step used to select D_AUD."""
    vals = [cd.run(0.24, seed=s, compute_p=False)["FFCr"]["z_d"] for s in range(12)]
    assert np.std(vals, ddof=1) > 0.01, (
        "single-draw spread collapsed; if this ever passes trivially, re-examine "
        "whether the simulation still has realistic per-draw variability"
    )


# --------------------------------------------------------------------------
# Selection procedure: widened grid, averaged objective, disjoint seeds
# --------------------------------------------------------------------------

def test_selection_and_reporting_seeds_are_disjoint():
    """No parameter may be chosen on a draw later used to quote performance."""
    assert not set(ss.SELECT_SEEDS) & set(ss.REPORT_SEEDS)


def test_selection_grid_extends_well_past_the_old_boundary():
    """The old grid was np.arange(0.0, 0.25, 0.01) and its argmin landed on the
    last point, 0.24 -- a boundary solution."""
    assert max(ss.D_AUD_GRID) > ss.OLD_GRID_MAX * 2


def test_objective_averages_the_full_seed_set():
    r = ss.objective(0.2, range(7))
    assert r["n"] == 7 and r["sem"] > 0


def test_selection_reports_boundary_and_separation_honestly():
    """A boundary optimum, or one indistinguishable from noise, must be surfaced."""
    r = ss.select_d_aud(range(4), grid=(0.0, 0.3, 0.6))
    assert set(r) >= {"d_best", "on_boundary", "separation_sigma", "distinguishable",
                      "z_critical_bonferroni", "indistinguishable_band", "n_comparisons"}
    assert isinstance(r["on_boundary"], bool)
    # the threshold must be corrected for the number of looks, not a bare 2 sigma:
    # picking the min of k+1 noisy means is k implicit comparisons.
    assert r["z_critical_bonferroni"] > 2.0
    assert r["distinguishable"] == (r["separation_sigma"] >= r["z_critical_bonferroni"])
    assert r["d_best"] in r["indistinguishable_band"]


def test_multiple_comparison_correction_scales_with_grid_size():
    """A wider grid means more looks, so a stricter threshold is required."""
    small = ss.select_d_aud(range(3), grid=(0.0, 0.3, 0.6))
    large = ss.select_d_aud(range(3), grid=tuple(round(x * 0.05, 2) for x in range(14)))
    assert large["z_critical_bonferroni"] > small["z_critical_bonferroni"]


def test_rho_grid_covers_the_intended_space():
    assert 0.0 in ss.RHO_GRID and max(ss.RHO_GRID) >= 0.9 and len(ss.RHO_GRID) >= 3


# --------------------------------------------------------------------------
# The headline must be DERIVED, never hard-coded
# --------------------------------------------------------------------------

def test_demo_refuses_to_run_without_the_selection_artifact(tmp_path, monkeypatch):
    """main() must not fall back to a literal D_AUD. A fallback would silently
    recreate the boundary-selected 0.24."""
    monkeypatch.setattr(cd, "SELECTION_JSON", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="sensitivity_surface"):
        cd.load_selected_d_aud()


def test_selected_d_aud_comes_from_the_artifact_not_a_literal():
    import pathlib
    src = pathlib.Path(cd.__file__).read_text()
    assert "np.arange(0.0, 0.25, 0.01)" not in src, "the boundary grid is back"
    # no bare 0.24 assignment to a D_AUD-like name
    assert "D_AUD = 0.24" not in src


def test_forty_seed_aggregation_is_reproducible():
    a = cd.run_many(0.30, range(100_000, 100_040))
    b = cd.run_many(0.30, range(100_000, 100_040))
    for roi in ("A1", "V1", "FFCr", "EBA"):
        assert a[roi]["z_d"]["mean"] == b[roi]["z_d"]["mean"]
        assert a[roi]["z_d"]["n"] == 40


def test_reported_sd_is_the_across_seed_sd_not_the_sem():
    """'+/- x' in published text must be the SD across draws (per-draw spread),
    not the SEM (which shrinks with n and would overstate agreement)."""
    a = cd.run_many(0.30, range(100_000, 100_040))["FFCr"]["z_d"]
    assert a["sd"] > a["sem"] * 3, "sd and sem are being confused"
    assert a["sem"] == pytest.approx(a["sd"] / np.sqrt(a["n"]), rel=1e-9)


def test_floor_table_refuses_the_old_boundary_literal():
    from scripts import detection_floor_table as dft
    import pathlib
    src = pathlib.Path(dft.__file__).read_text()
    assert "D_AUD = 0.24" not in src
    assert dft.D_AUD is None, "D_AUD must resolve lazily from the selection artifact"


def test_p_values_are_summarised_as_a_rejection_rate_not_a_mean():
    """Averaging p-values across replications produces a number that is not a
    p-value. The publishable summary is the rejection rate."""
    a = cd.run_many(0.30, range(100_000, 100_006), face_effect=0.2, compute_p=True)["FFCr"]
    assert "reject_rate_025" in a["z_p"], "p-values must carry a rejection rate"
    assert 0.0 <= a["z_p"]["reject_rate_025"] <= 1.0
    import pathlib
    src = pathlib.Path(cd.__file__).read_text()
    assert "fmt_p(a['z_p']['mean'])" not in src, "a mean p-value is being printed as a p-value"


def test_parameter_sweeps_cannot_overwrite_the_canonical_floor_table():
    """A sweep at non-default parameters must write a TAGGED file.

    This is a regression test for a real incident: a control run at
    D_AUD=0.24 overwrote data/floor_table_v3b.md and destroyed its
    'SUPERSEDED' annotation, because a filename-tagging patch had silently
    failed to apply and nobody verified it.
    """
    import pathlib
    from scripts import detection_floor_table as dft
    src = pathlib.Path(dft.__file__).read_text()
    assert 'f"floor_table_v3b{tag}.md"' in src, "the output filename is no longer parameterised"
    assert 'tag = "" if (D_AUD == _selected_d_aud() and RHO == 0.0)' in src, (
        "the guard that reserves the canonical filename for the selected parameters is gone"
    )
