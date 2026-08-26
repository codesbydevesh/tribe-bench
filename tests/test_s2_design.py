"""Tests for the frozen S2 design.

Phase B's lesson applied to an experiment: the design must be asserted, not just
written down. Every branch that can stop the study, every timing number that the
GPU cost depends on, and the three-way stimulus/manifest/analysis agreement are
pinned here so that a later edit cannot quietly move them.
"""
from dataclasses import replace

import numpy as np
import pytest

from neurocheck import s2_design as D
from neurocheck.s2_design import (
    ALL_PARCELS, PARCELS_SECONDARY, RULES, S2, S2Config, build_manifest,
    build_schedule, check_three_way_consistency, events_dataframe, gpu_cost_estimate,
    grey_intervals, replication_verdict, stop_eligible_parcels,
)


# ---------------------------------------------------------------------- timing

def test_soa_is_a_cycle_not_an_isi():
    """The paper says "flashed for 1 second every 8 seconds" — an 8 s CYCLE. An
    earlier plan line read "1 s on / 8 s ISI", a 9 s cycle: an 11% cost error and a
    protocol deviation from the very result being replicated."""
    assert S2.soa_s == 8.0
    assert S2.on_duration_s == 1.0
    assert S2.isi_s == 7.0, "ISI must be SOA minus presentation, not SOA itself"
    ev = build_schedule()
    gaps = {round(b.onset_s - a.onset_s, 6) for a, b in zip(ev, ev[1:])}
    assert gaps == {8.0}, f"onset-to-onset spacing must be the SOA, got {gaps}"


def test_lead_in_precedes_every_event():
    ev = build_schedule()
    assert ev[0].onset_s == S2.lead_in_s == 25.0
    assert all(e.onset_s >= S2.lead_in_s for e in ev)


def test_the_last_event_gets_a_full_post_onset_window():
    """A response is read from the frames AFTER the onset. If the video ends at the
    last offset, the last event has no window and its response is truncated."""
    ev = build_schedule()
    assert S2.stimulus_duration_s - ev[-1].offset_s >= S2.tail_out_s


# --------------------------------------------------------------- randomisation

def test_order_is_randomised_but_reproducible():
    a, b = build_schedule(), build_schedule()
    assert [e.stimulus_id for e in a] == [e.stimulus_id for e in b], "not reproducible"
    assert [e.category for e in a] != sorted([e.category for e in a]), "order looks unshuffled"
    other = build_schedule(replace(S2, order_seed=S2.order_seed + 1))
    assert [e.stimulus_id for e in a] != [e.stimulus_id for e in other], "seed does nothing"


def test_every_exemplar_appears_exactly_once():
    ev = build_schedule()
    ids = [e.stimulus_id for e in ev]
    assert len(ids) == len(set(ids)) == S2.n_events
    from collections import Counter
    counts = Counter(e.category for e in ev)
    assert set(counts.values()) == {S2.exemplars_per_category}, counts


# --------------------------------------------------------------------- parcels

def test_only_record_parcels_can_stop_the_study():
    """The most expensive branch in the project. A secondary analysis must be
    structurally incapable of ending GPU spend."""
    for p in ALL_PARCELS:
        if p.stop_eligible:
            assert p.provenance == "record", f"{p.name} is secondary but stop-eligible"
    for p in PARCELS_SECONDARY:
        assert not p.stop_eligible, f"secondary parcel {p.name} can stop the study"
    assert {p.name for p in stop_eligible_parcels()} == {"FFA", "EBA"}


def test_the_eba_union_that_straddles_a_category_boundary_cannot_decide_anything():
    """§5.9 assigns PH to a different functional label than EBA, so the Gate-0 proxy
    pooled two categories — the paper's own boundary fell INSIDE our old ROI. The
    union is kept for continuity and must not gate; the record EBA is V4t alone."""
    union = next(p for p in ALL_PARCELS if p.name == "EBA_gate0_union")
    assert set(union.labels) == {"V4t", "PH"}
    assert not union.stop_eligible and union.provenance == "secondary"
    eba = next(p for p in ALL_PARCELS if p.name == "EBA")
    assert eba.labels == ("V4t",), "the record EBA must be V4t alone, not the union"
    assert eba.stop_eligible and eba.mapping_verified


def test_only_securely_mapped_parcels_are_stop_eligible():
    """§5.9 lists nine functional names against eight parcels, so one visual region
    has no parcel and PH is claimed by both PPA and VWFA. A parcel whose identity is
    ambiguous must never be able to end GPU spend."""
    for p in ALL_PARCELS:
        if p.stop_eligible:
            assert p.mapping_verified, f"{p.name} gates on an unverified mapping"
    ambiguous = [p for p in ALL_PARCELS if p.name in ("PPA", "VWFA")]
    assert len(ambiguous) == 2
    for p in ambiguous:
        assert p.labels == ("PH",), "both readings must carry the SAME contested parcel"
        assert not p.stop_eligible and not p.mapping_verified


def test_the_parcel_misalignment_is_recorded_verbatim_not_paraphrased():
    """The off-by-one is the reason two regions cannot gate. It must stay quoted so a
    later reader can re-derive the conclusion instead of trusting the summary."""
    txt = D.PARCEL_LIST_MISALIGNMENT
    assert "FFC, V4t, PH, A5, 45, STSv, PGi, TE1a" in txt, "the parcel list is not quoted"
    assert "NINE" in txt and "EIGHT" in txt
    assert D.ROI_MAPPING_STATUS is txt


def test_a_secondary_parcel_cannot_be_made_stop_eligible_by_editing_one_field():
    with pytest.raises(ValueError, match="secondary"):
        bad = replace(PARCELS_SECONDARY[0], name="planted_secondary",
                      stop_eligible=True)
        original = D.ALL_PARCELS
        try:
            D.ALL_PARCELS = original + (bad,)
            D._validate_parcels()
        finally:
            D.ALL_PARCELS = original


# ------------------------------------------------------------------ decisions

def _res(p, effect, floor, *, alt=None, peak=None):
    """Build a lag-keyed result. `alt` defaults to the same numbers at the
    alternative lag, so a test that does not care about the lag dimension reads
    as before; pass `alt` explicitly to make the two lags disagree."""
    prim = {"p": p, "effect": effect, "floor": floor}
    other = prim if alt is None else {"p": alt[0], "effect": alt[1], "floor": alt[2]}
    return {"by_lag": {S2.primary_lag_trs: prim, S2.alternative_lag_trs: other},
            "peak_lag_trs": peak, "statistic": S2.primary_statistic}


def test_stop_fires_only_when_every_stop_eligible_parcel_fails():
    fail, win = _res(0.9, 0.01, 0.05), _res(0.001, 0.5, 0.05)
    assert replication_verdict({"FFA": fail, "EBA": fail})["stop"] is True
    assert replication_verdict({"FFA": win, "EBA": fail})["stop"] is False
    assert replication_verdict({"FFA": fail, "EBA": win})["stop"] is False


def test_secondary_failure_can_never_fire_the_stop_rule():
    win, fail = _res(0.001, 0.5, 0.05), _res(0.9, 0.01, 0.05)
    v = replication_verdict({"FFA": win, "EBA": win,
                             "PPA": fail, "VWFA": fail, "PPA_literature": fail,
                             "EBA_gate0_union": fail, "V1_control": fail})
    assert v["stop"] is False, "a secondary null stopped the study"
    assert v["per_parcel"]["EBA_gate0_union"]["status"] == "not_recovered"
    assert v["per_parcel"]["EBA_gate0_union"]["stop_eligible"] is False


def test_recovery_requires_the_detection_floor_not_just_the_p_value():
    """Doctrine D-3: no verdict without a floor. A significant effect smaller than
    the minimum detectable effect is not a recovery."""
    below = _res(0.001, 0.01, 0.05)          # significant but under its own floor
    v = replication_verdict({"FFA": below, "EBA": below})
    prim = v["per_parcel"]["FFA"]["by_lag"][str(S2.primary_lag_trs)]
    assert prim["cleared_alpha"] is True
    assert prim["cleared_floor"] is False
    assert v["per_parcel"]["FFA"]["status"] == "not_recovered"
    assert v["stop"] is True


def test_a_missing_parcel_blocks_the_stop_rule_even_when_the_other_one_failed():
    """The interesting case, which the previous fixture avoided: the present parcel
    FAILS and the other is missing. Stopping there would end GPU spend having
    measured one of the two parcels the rule requires. Incomplete is not the same
    outcome as not-recovered."""
    v = replication_verdict({"FFA": _res(0.9, 0.01, 0.05)})    # EBA never ran
    assert v["per_parcel"]["FFA"]["status"] == "not_recovered"
    assert v["per_parcel"]["EBA"]["status"] == "not_run"
    assert v["incomplete"] == ["EBA"]
    assert v["stop"] is False, "the study stopped on half the required evidence"
    # and with the other one recovering, still no stop
    assert replication_verdict({"FFA": _res(0.001, 0.5, 0.05)})["stop"] is False
    # complete evidence, both failed -> the rule fires as designed
    both = replication_verdict({"FFA": _res(0.9, 0.01, 0.05), "EBA": _res(0.9, 0.01, 0.05)})
    assert both["incomplete"] == [] and both["stop"] is True


def test_an_unusable_number_is_an_error_not_a_null_result():
    """NaN compares False against everything, so a NaN p-value read as 'did not
    clear alpha' — indistinguishable from a genuine null, and able to help fire the
    stop rule on a measurement that never happened."""
    nan = float("nan")
    for label, bad in (("p", _res(nan, 0.5, 0.05)),
                       ("effect", _res(0.001, nan, 0.05)),
                       ("floor", _res(0.001, 0.5, nan))):
        v = replication_verdict({"FFA": bad, "EBA": bad})
        assert v["per_parcel"]["FFA"]["status"] == "invalid", label
        reason = v["per_parcel"]["FFA"]["by_lag"][str(S2.primary_lag_trs)]["reason"]
        assert label in reason, reason
        assert v["incomplete"] == ["FFA", "EBA"], label
        assert v["stop"] is False, f"a non-finite {label} fired the stop rule"


def test_a_zero_or_negative_floor_is_rejected_rather_than_silently_disabling_D3():
    """A floor of 0 makes `effect > floor` true for any positive effect, which
    switches off the detection-floor doctrine while appearing to honour it."""
    for floor in (0.0, -1.0):
        v = replication_verdict({"FFA": _res(0.001, 0.5, floor)})
        assert v["per_parcel"]["FFA"]["status"] == "invalid", floor
        assert "not positive" in \
            v["per_parcel"]["FFA"]["by_lag"][str(S2.primary_lag_trs)]["reason"]
    # a genuine positive floor still works
    assert replication_verdict(
        {"FFA": _res(0.001, 0.5, 0.05)})["per_parcel"]["FFA"]["status"] == "recovered"


def test_the_isi_baseline_is_secondary_by_construction():
    """C2's mitigation answers a different question — response vs rest, not vs the
    other categories. It must be classified before the results, not after."""
    assert S2.isi_baseline_as_category is True
    assert S2.isi_baseline_role == "secondary"
    assert "DIFFERENT question" in RULES.estimand_note


def test_the_lag_conflict_is_pre_registered_with_both_reads_from_one_run():
    """§5.9 reads at t=5 and calls it the peak; source-of-truth says the output is
    already offset by 5 so that read lands on BOLD(onset+10), ~18% of peak. Both
    cannot be true. Primary follows the paper (this is a replication); the
    alternative is pre-specified; both come from the SAME peri-event timecourse, so
    resolving it costs no extra GPU and neither can be chosen after the fact."""
    assert S2.primary_lag_trs == 5, "primary must follow the paper's published protocol"
    assert S2.alternative_lag_trs == 0
    assert 5 in S2.report_lags and 0 in S2.report_lags
    assert S2.peak_lag_policy == "measure_and_report_only"
    assert "never used to select" in RULES.lag_policy
    assert "double-lag" in RULES.lag_adjudication
    assert "NOT as a plain replication" in RULES.lag_adjudication


def test_recovering_only_at_the_alternative_lag_is_not_called_a_replication():
    """The adjudication must forbid relabelling a lag-0 recovery as a successful
    replication of a protocol that specifies t=5."""
    assert "not replicated at the published lag" in RULES.lag_adjudication


def test_the_visual_contrast_uses_no_glm():
    """§5.9 assigns the GLM to the LANGUAGE experiments; the visual contrast is the
    plain peak-minus-mean-of-others subtraction. Figure 4's caption says otherwise
    and contradicts its own Methods."""
    assert S2.primary_statistic == "event_locked_contrast"
    assert "no GLM" in D.LAG_CONFLICT
    assert "glm_contrast_z" in S2.secondary_statistics


# ------------------------------------------------------------------- manifest

def test_manifest_round_trips_to_the_schedule():
    m = build_manifest()
    assert m["design_fingerprint"] == S2.fingerprint()
    assert len(m["events"]) == S2.n_events
    assert m["stop_eligible"] == ["FFA", "EBA"]
    ev = build_schedule()
    assert [e["stimulus_id"] for e in m["events"]] == [e.stimulus_id for e in ev]


def test_manifest_is_json_serialisable_and_reconstructs_the_stimulus():
    import json
    m = build_manifest(code_commit="deadbeef")
    blob = json.loads(json.dumps(m))       # must survive a round trip
    assert blob["provenance"]["code_commit"] == "deadbeef"
    assert blob["provenance"]["model_id"] == "facebook/tribev2"
    for key in ("onset_s", "duration_s", "category", "stimulus_id", "event_id", "isi_s", "soa_s"):
        assert key in blob["events"][0], f"manifest event lacks {key}"
    assert blob["grey_intervals_s"], "grey intervals must be recorded"


def test_grey_intervals_tile_every_non_stimulus_moment():
    cfg, ev = S2, build_schedule()
    greys = grey_intervals(cfg, ev)
    spans = sorted(greys + [(e.onset_s, e.offset_s) for e in ev])
    cursor = 0.0
    for a, b in spans:
        assert abs(a - cursor) < 1e-9, f"gap/overlap at {cursor}"
        cursor = b
    assert abs(cursor - cfg.stimulus_duration_s) < 1e-9


def test_design_fingerprint_changes_when_any_number_changes():
    base = S2.fingerprint()
    assert replace(S2, soa_s=9.0).fingerprint() != base
    assert replace(S2, order_seed=1).fingerprint() != base
    assert replace(S2, exemplars_per_category=50).fingerprint() != base
    assert S2Config().fingerprint() == base, "fingerprint must be stable for one design"


# ---------------------------------------------------------------- consistency

def _ok_args(cfg=S2):
    return dict(cfg=cfg, manifest=build_manifest(cfg),
                rendered_duration_s=cfg.stimulus_duration_s, rendered_fps=cfg.fps,
                events_df=events_dataframe(cfg))


def test_three_way_consistency_passes_on_the_frozen_design():
    assert check_three_way_consistency(**_ok_args()) == []


def test_consistency_catches_a_short_render():
    a = _ok_args(); a["rendered_duration_s"] = S2.stimulus_duration_s - 8.0
    assert any("rendered video" in p for p in check_three_way_consistency(**a))


def test_consistency_catches_analysis_events_that_drift_from_the_manifest():
    a = _ok_args()
    df = a["events_df"].copy()
    df.loc[3, "onset"] = float(df.loc[3, "onset"]) + 0.5     # half-second drift
    a["events_df"] = df
    problems = check_three_way_consistency(**a)
    assert any("onset" in p for p in problems), problems


def test_consistency_catches_a_relabelled_event():
    a = _ok_args()
    df = a["events_df"].copy()
    df.loc[0, "trial_type"] = "not_a_category"
    a["events_df"] = df
    assert any("category" in p for p in check_three_way_consistency(**a))


def test_consistency_catches_a_manifest_from_a_different_design():
    a = _ok_args()
    a["manifest"] = build_manifest(replace(S2, order_seed=999))
    problems = check_three_way_consistency(**a)
    assert any("fingerprint" in p for p in problems), problems


def test_consistency_catches_a_dropped_event():
    a = _ok_args()
    a["events_df"] = a["events_df"].iloc[:-1]
    assert any("analysis events" in p for p in check_three_way_consistency(**a))


# ----------------------------------------------------------------- gpu budget

def test_cost_is_charged_on_the_whole_rendered_timeline():
    """Lead-in, every ISI and the tail-out are real frames the model consumes.
    Costing only the 1 s presentations understates this by roughly 8x."""
    est = gpu_cost_estimate()
    assert est["rendered_stimulus_s"] == pytest.approx(S2.stimulus_duration_s)
    presentations_only = S2.n_events * S2.on_duration_s
    assert est["rendered_stimulus_s"] > 8 * presentations_only
    assert est["estimated_gpu_h"] == pytest.approx(
        S2.stimulus_duration_s * S2.compute_s_per_stimulus_s / 3600.0, abs=0.01)


def test_cost_accounts_for_runs_subjects_and_repetitions_explicitly():
    """The brief: do not quote a cost until trials, subjects/runs, repetitions and
    timing are all accounted for. In silico there is one model and no subject
    dimension — stated, not assumed."""
    est = gpu_cost_estimate()
    for k in ("n_events", "runs", "subjects", "repetitions"):
        assert k in est, f"cost estimate does not state {k}"
    assert est["runs"] == est["subjects"] == est["repetitions"] == 1


def test_the_window_packing_concern_is_quantified_not_asserted():
    """C2 says the 100 s window mixes ~11 exemplars from all categories. That is a
    number the design produces, so compute it rather than repeating it."""
    est = gpu_cost_estimate()
    assert est["n_model_windows"] == int(np.ceil(S2.stimulus_duration_s / S2.model_window_s))
    assert 10 <= est["events_per_window"] <= 13, est["events_per_window"]


# --------------------------------------------------------------------- events

def test_events_dataframe_is_built_from_the_schedule_not_from_audio():
    """get_events_dataframe runs WhisperX to derive word events. Over a silent video
    it transcribes nothing while still costing ~65 s/clip, and any timing it derived
    would come from the audio track rather than from this schedule."""
    import inspect
    src = inspect.getsource(events_dataframe)
    assert "get_events_dataframe" not in src.replace("``get_events_dataframe``", "")
    df = events_dataframe()
    assert len(df) == S2.n_events
    assert list(df["onset"]) == [e.onset_s for e in build_schedule()]
    assert set(df["trial_type"]) == set(S2.categories)


def test_the_run_script_disables_dataloader_workers():
    """REGRESSION: the first Kaggle run died here.

    neuralset's SegmentDataset.__getitem__ runs the video extractor, which moves
    V-JEPA onto the GPU. With num_workers > 0 the torch DataLoader forks, and a
    forked child cannot initialise CUDA:

        RuntimeError: Cannot re-initialize CUDA in forked subprocess.
        To use CUDA with multiprocessing, you must use the 'spawn' start method

    num_workers=0 is therefore required, not a tuning preference. It costs nothing:
    the bottleneck is the ViT-giant forward pass, not data loading.
    """
    import pathlib
    src = pathlib.Path(__file__).resolve().parent.parent.joinpath(
        "scripts", "s2_run.py").read_text()
    assert '"num_workers": 0' in src, \
        "the DataLoader will fork and CUDA will fail in the worker"
    # and it must sit in the real inference call, not a comment or the stub path
    infer = src.split("from tribe_tools.model import load_model")[1]
    assert '"num_workers": 0' in infer.split("elapsed = time.time()")[0]
