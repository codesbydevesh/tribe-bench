"""S2 experimental design — the frozen specification, as code.

**Why this is a module and not a document.** Phase B's lesson was that passing the
example is not enough; you must verify the mechanism. The experimental equivalent
is: a runnable S2 is not enough, the design must be frozen *before* the answer is
seen. A design that lives only in prose drifts silently; a design that lives here
is executable, diffable, hashable, and can be dry-run on CPU.

Everything the GPU run needs is derived from :data:`S2` — one frozen config. The
stimulus schedule, the event table, the analysis events and the cost estimate are
all functions of it, so they cannot disagree with each other by construction, and
:func:`check_three_way_consistency` proves they do not.

Design-of-record: ``.notes/plans/corticall/MASTER-PLAN.md`` §3.1/§3.4/§3.6, the
Phase C brief, and ``ops/source-of-truth.md``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# ROI parcels
# ---------------------------------------------------------------------------

Provenance = Literal["record", "secondary"]


@dataclass(frozen=True)
class Parcel:
    """One ROI, with its provenance and its eligibility to stop the study.

    ``stop_eligible`` is deliberately NOT derived from ``provenance``. The stop
    rule is the most expensive branch in the project -- it ends GPU spend -- so
    each ROI's eligibility is stated explicitly and asserted below, rather than
    inferred from a label that a later edit might change for another reason.
    """

    name: str                 # functional name, e.g. "FFA"
    labels: tuple[str, ...]   # HCP-MMP1 parcel labels
    hemi: str                 # "left" | "right" | "both"
    provenance: Provenance
    stop_eligible: bool
    source: str               # where the mapping comes from, verbatim-ish
    mapping_verified: bool    # True only if traced to the paper's own text


# Replication-of-record: mappings quoted from the paper's own Methods §5.9.
PARCELS_RECORD: tuple[Parcel, ...] = (
    Parcel(
        name="FFA", labels=("FFC",), hemi="right",
        provenance="record", stop_eligible=True,
        source="Methods §5.9, position 1 -> FFC. SECURE (see PARCEL_LIST_MISALIGNMENT). "
               "Right hemisphere: FFA is right-lateralised and Gate 0 used FFCr.",
        mapping_verified=True,
    ),
    Parcel(
        name="EBA", labels=("V4t",), hemi="both",
        provenance="record", stop_eligible=True,
        source="Methods §5.9, position 2 -> V4t. SECURE. NOTE this is V4t ALONE: the Gate-0 "
               "proxy also pooled PH, which §5.9 assigns to a different functional label, so "
               "the paper's category boundary fell INSIDE our old ROI.",
        mapping_verified=True,
    ),
)

# Named in the paper but NOT securely mappable, so barred from gating anything.
PARCELS_UNRESOLVED: tuple[Parcel, ...] = (
    Parcel(
        name="PPA", labels=("PH",), hemi="both",
        provenance="record", stop_eligible=False,
        source="AMBIGUOUS. §5.9 lists NINE functional names against EIGHT parcels, so exactly "
               "one visual region has no parcel. PH is claimed by PPA (if VWFA is the "
               "omission) or by VWFA (if PPA is the omission). Unresolvable from the paper.",
        mapping_verified=False,
    ),
    Parcel(
        name="VWFA", labels=("PH",), hemi="left",
        provenance="record", stop_eligible=False,
        source="AMBIGUOUS -- the same PH as PPA, for the same reason. Both readings are "
               "reported; neither may fire the stop rule.",
        mapping_verified=False,
    ),
)

# Secondary analyses. Reported, never decisive.
PARCELS_SECONDARY: tuple[Parcel, ...] = (
    Parcel(
        name="PPA_literature", labels=("PHA1", "PHA2", "PHA3", "VMV1", "VMV2", "VMV3"),
        hemi="both", provenance="secondary", stop_eligible=False,
        source="The standard literature PPA definition, NOT the paper's. Gate 0 v3b reached "
               "d=+2.529, p=0.0002 with it, so it is kept for continuity -- but §5.9 names a "
               "single parcel, so this union cannot be the replication-of-record.",
        mapping_verified=False,
    ),
    Parcel(
        name="EBA_gate0_union", labels=("V4t", "PH"), hemi="both",
        provenance="secondary", stop_eligible=False,
        source="Gate-0 union (D021). Retained for continuity with the July run ONLY. "
               "It straddles the paper's own category boundary, so a null here means "
               "nothing about replication.",
        mapping_verified=False,
    ),
    Parcel(
        name="V1_control", labels=("V1",), hemi="both",
        provenance="secondary", stop_eligible=False,
        source="Non-specific control: should NOT show category selectivity. A positive "
               "here indicates a pipeline artefact, not a finding.",
        mapping_verified=False,
    ),
)

ALL_PARCELS: tuple[Parcel, ...] = PARCELS_RECORD + PARCELS_UNRESOLVED + PARCELS_SECONDARY

LAG_CONFLICT = """\
The single most decision-critical ambiguity in S2, resolved by measurement rather than by
choosing a side in advance.

  Methods §5.9, verbatim:
    "we obtain contrast maps by simply selecting the predicted response at t=5 after the
     image is shown (which is the peak of the response as shown in Figure 4A), and
     substracting the average responses at t=5 for the other categories."   [sic]

  ops/source-of-truth.md:51, VERIFIED from the model source:
    FmriExtractor(offset=5) means the model learns stimulus(t) -> BOLD(t+5), so the output is
    ALREADY hemodynamically aligned and an event-locked read at onset+5 lands on
    BOLD(onset+10) -- about 18% of peak (canonical HRF h(10)/h(5) = 0.183).

Both cannot hold: the paper says its predicted response PEAKS at t=5; if the output were
already aligned the peak would sit at t=0. Reading the wrong one costs roughly 5.5x in
amplitude and would produce a failed replication for a reason unrelated to the model.

This is NOT resolved by picking one. A single peri-event timecourse yields every lag from
one forward pass, so both reads come free from the same run. The primary follows the paper
(we are replicating it), the alternative is pre-specified, and the MEASURED peak adjudicates.
See DecisionRules.lag_adjudication.

Consequence for the contrast: §5.9 also settles M4. The GLM belongs to the LANGUAGE
experiments ("we fit a General Linear Model ... nilearn FirstLevelModel"); the visual
contrast is the plain peak-minus-mean-of-other-categories subtraction above, with no GLM and
no z-scoring. Figure 4's caption says otherwise and contradicts its own Methods; the Methods
text is the more specific statement and is the one implemented here.
"""

PARCEL_LIST_MISALIGNMENT = """\
Methods §5.9 (arXiv 2605.04326, from the LaTeX source; the HTML renderings truncate before
this section) states verbatim:

    "To extract regions of interest, we used the Glasser Multimodal parcellation. FFA, EBA,
     PPA, VWFA, A5, 45, STS, TPJ, MTG respectively correspond to the following ROI labels:
     FFC, V4t, PH, A5, 45, STSv, PGi, TE1a."

NINE functional names against EIGHT parcel labels. Exactly one name has no parcel.

Where the gap must be, argued from the text alone: `A5` and `45` appear on BOTH sides and
must correspond to themselves. A5 is left-position 5 but right-position 4, so the off-by-one
is already present by position 5 -- the omission therefore lies among the first four, the
visual regions. The tail is then consistent: STS->STSv, TPJ->PGi, MTG->TE1a.

Consequences:
  * FFA -> FFC and EBA -> V4t are SECURE. Moving the omission to position 1 or 2 would force
    EBA->FFC or PPA->V4t, which the remaining assignments cannot absorb.
  * PH is claimed by PPA (if VWFA is the omission) or by VWFA (if PPA is the omission).
    UNRESOLVABLE from the paper. Both readings are carried; neither may gate.

Also: §2.5 orders the regions FFA, PPA, EBA, VWFA while §5.9 orders them FFA, EBA, PPA,
VWFA. Zipping against the results text instead of the methods text silently swaps PPA and
EBA -- which is how our Gate-0 EBA proxy came to pool V4t with PH.
"""

ROI_MAPPING_STATUS = PARCEL_LIST_MISALIGNMENT


def stop_eligible_parcels() -> tuple[Parcel, ...]:
    """The ONLY parcels whose failure may fire "Not recovered -> Stop"."""
    return tuple(p for p in ALL_PARCELS if p.stop_eligible)


def _validate_parcels() -> None:
    names = [p.name for p in ALL_PARCELS]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate parcel names: {names}")
    for p in ALL_PARCELS:
        if p.stop_eligible and p.provenance != "record":
            raise ValueError(
                f"{p.name} is stop-eligible but is labelled '{p.provenance}'. A secondary "
                "analysis must never be able to stop the study."
            )
        if not p.labels:
            raise ValueError(f"{p.name} has no parcel labels")
    if not stop_eligible_parcels():
        raise ValueError("no stop-eligible parcel: the stop rule could never fire")


_validate_parcels()


# ---------------------------------------------------------------------------
# The frozen configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class S2Config:
    """Every number the S2 run depends on. Frozen; change it by making a new one."""

    # --- stimulus ---------------------------------------------------------
    categories: tuple[str, ...] = ("faces", "bodies", "places", "objects", "characters")
    exemplars_per_category: int = 25
    # VERIFIED from the paper: "flashed for 1 second every 8 seconds" (arXiv
    # 2605.04326 §2.5). SOA is the CYCLE, so ISI = SOA - on_duration = 7 s.
    # An earlier plan line read "1 s on / 8 s ISI", which is a 9 s cycle: an 11%
    # cost error and a protocol deviation from the paper being replicated.
    soa_s: float = 8.0
    on_duration_s: float = 1.0
    lead_in_s: float = 25.0        # grey frames before the first onset
    tail_out_s: float = 25.0       # grey frames after the last offset
    fps: int = 8
    frame_size: tuple[int, int] = (224, 224)
    grey_level: int = 128

    # --- randomisation ----------------------------------------------------
    order_seed: int = 20260824     # fixed: the order is part of the frozen design

    # --- analysis ---------------------------------------------------------
    model_id: str = "facebook/tribev2"
    # THE LAG CONFLICT, pre-registered rather than guessed. See LAG_CONFLICT.
    # Methods §5.9 reads the contrast at t=5 and calls that the peak. Our own
    # VERIFIED source-of-truth says TRIBE output is already offset by 5 s, so a
    # read at 5 lands on BOLD(onset+10), ~18% of peak. Both cannot be true.
    # Primary follows the PAPER, because this is a replication. The alternative is
    # pre-specified, not chosen after the fact, and BOTH come from the SAME run:
    # one peri-event timecourse yields every lag, so resolving this costs no extra
    # GPU at all.
    primary_lag_trs: int = 5              # the paper's t=5
    alternative_lag_trs: int = 0          # implied by FmriExtractor(offset=5)
    report_lags: tuple[int, ...] = (-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    peak_lag_policy: str = "measure_and_report_only"
    primary_statistic: str = "event_locked_contrast"
    secondary_statistics: tuple[str, ...] = ("glm_contrast_z", "roi_minus_reference", "spatial_z")
    n_perm: int = 10000
    perm_seed: int = 0
    alpha: float = 0.025           # one-sided, per the IUT convention
    # A category is "recovered" only if it clears alpha AND exceeds its own
    # detection floor (doctrine D-3: no verdict without a floor).
    require_detection_floor: bool = True

    # --- C2: the 100 s window --------------------------------------------
    # Primary is the replication as published. The ISI-baseline read is a
    # PRE-SPECIFIED SECONDARY, so it cannot become the headline after the fact.
    isi_baseline_as_category: bool = True
    isi_baseline_role: Literal["primary", "secondary"] = "secondary"
    log_window_packing: bool = True

    # --- compute ----------------------------------------------------------
    model_window_s: float = 100.0
    compute_s_per_stimulus_s: float = 11.5   # §3.6 cost model

    @property
    def isi_s(self) -> float:
        return self.soa_s - self.on_duration_s

    @property
    def n_events(self) -> int:
        return len(self.categories) * self.exemplars_per_category

    @property
    def stimulus_duration_s(self) -> float:
        """Total rendered video length, lead-in and tail-out included.

        The last event needs its full SOA cycle so its post-onset window is not
        truncated -- reading a response requires the frames after the onset.
        """
        return self.lead_in_s + self.n_events * self.soa_s + self.tail_out_s

    def fingerprint(self) -> str:
        """Stable hash of the whole design. Any change gives a new id."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


S2 = S2Config()


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    event_id: int
    stimulus_id: str
    category: str
    order_index: int
    onset_s: float
    duration_s: float
    offset_s: float
    isi_s: float
    soa_s: float


def build_schedule(cfg: S2Config = S2) -> list[Event]:
    """Deterministic randomised presentation order.

    Randomisation is a *seeded* property of the design, not a run-time accident:
    the same config always yields the same schedule, so the manifest can be
    regenerated and diffed after the fact.
    """
    rng = np.random.default_rng(cfg.order_seed)
    items = [(c, i) for c in cfg.categories for i in range(cfg.exemplars_per_category)]
    order = rng.permutation(len(items))
    events: list[Event] = []
    for k, idx in enumerate(order):
        category, exemplar = items[int(idx)]
        onset = cfg.lead_in_s + k * cfg.soa_s
        events.append(Event(
            event_id=k,
            stimulus_id=f"{category}_{exemplar:03d}",
            category=category,
            order_index=k,
            onset_s=round(onset, 6),
            duration_s=cfg.on_duration_s,
            offset_s=round(onset + cfg.on_duration_s, 6),
            isi_s=cfg.isi_s,
            soa_s=cfg.soa_s,
        ))
    return events


def grey_intervals(cfg: S2Config, events: list[Event]) -> list[tuple[float, float]]:
    """Every interval that must be RENDERED as real grey frames.

    The ISI is not an absence of stimulus; it is grey frames the model sees. If it
    is omitted the timeline shortens and every later onset is wrong.
    """
    out = [(0.0, cfg.lead_in_s)]
    for i, ev in enumerate(events):
        # The final interval runs to the END OF THE TIMELINE, not to
        # offset + tail_out: the last event still owns the remainder of its SOA
        # cycle, and dropping it left 7 s unrendered while every downstream
        # duration still assumed the full schedule.
        end = events[i + 1].onset_s if i + 1 < len(events) else cfg.stimulus_duration_s
        out.append((ev.offset_s, end))
    return [(round(a, 6), round(b, 6)) for a, b in out if b > a]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(cfg: S2Config = S2, *, code_commit: str | None = None) -> dict:
    """The machine-readable record of exactly what was presented.

    Written BEFORE the GPU run and shipped beside the results, so the stimulus can
    be reconstructed without reading the generator.
    """
    events = build_schedule(cfg)
    greys = grey_intervals(cfg, events)
    return {
        "schema_version": 1,
        "design_fingerprint": cfg.fingerprint(),
        "config": asdict(cfg),
        "provenance": {
            "model_id": cfg.model_id,
            "code_commit": code_commit,
            "roi_mapping_status": ROI_MAPPING_STATUS.strip(),
        },
        "timing": {
            "soa_s": cfg.soa_s,
            "on_duration_s": cfg.on_duration_s,
            "isi_s": cfg.isi_s,
            "lead_in_s": cfg.lead_in_s,
            "tail_out_s": cfg.tail_out_s,
            "total_duration_s": cfg.stimulus_duration_s,
            "n_events": cfg.n_events,
        },
        "parcels": [asdict(p) for p in ALL_PARCELS],
        "stop_eligible": [p.name for p in stop_eligible_parcels()],
        "events": [asdict(e) for e in events],
        "grey_intervals_s": greys,
        "compute": gpu_cost_estimate(cfg),
    }


def events_dataframe(cfg: S2Config = S2):
    """Hand-built events table, derived from the frozen schedule.

    Deliberately does NOT call ``get_events_dataframe``: that helper runs WhisperX
    to derive word events, and over a silent video it transcribes nothing while
    still costing ~65 s/clip -- and any timing it derived would come from the audio
    track rather than from this schedule. The events must come from the schedule
    that also generated the pixels, or the two can disagree.
    """
    import pandas as pd

    events = build_schedule(cfg)
    return pd.DataFrame([
        {"onset": e.onset_s, "duration": e.duration_s, "trial_type": e.category,
         "stimulus_id": e.stimulus_id, "event_id": e.event_id}
        for e in events
    ])


# ---------------------------------------------------------------------------
# The three-way consistency gate
# ---------------------------------------------------------------------------

def check_three_way_consistency(cfg: S2Config, manifest: dict, rendered_duration_s: float,
                                rendered_fps: int, events_df) -> list[str]:
    """rendered stimulus <-> event manifest <-> analysis events must agree.

    Returns a list of disagreements; empty means the gate passes. This is a GATE,
    not a diagnostic: a mismatch here means the numbers describe different things.
    """
    problems: list[str] = []
    m_events = manifest["events"]

    if manifest["design_fingerprint"] != cfg.fingerprint():
        problems.append(
            f"manifest fingerprint {manifest['design_fingerprint']} != config "
            f"{cfg.fingerprint()}: the manifest was built from a different design"
        )
    if abs(rendered_duration_s - cfg.stimulus_duration_s) > 1.0 / max(rendered_fps, 1):
        problems.append(
            f"rendered video is {rendered_duration_s:.3f}s but the schedule needs "
            f"{cfg.stimulus_duration_s:.3f}s (tolerance one frame)"
        )
    if rendered_fps != cfg.fps:
        problems.append(f"rendered fps {rendered_fps} != configured {cfg.fps}")
    if len(events_df) != len(m_events):
        problems.append(f"analysis events {len(events_df)} != manifest events {len(m_events)}")
    else:
        for me, (_, ae) in zip(m_events, events_df.iterrows()):
            if abs(float(ae["onset"]) - me["onset_s"]) > 1e-6:
                problems.append(
                    f"event {me['event_id']}: analysis onset {ae['onset']} != manifest "
                    f"{me['onset_s']}")
            if str(ae["trial_type"]) != me["category"]:
                problems.append(
                    f"event {me['event_id']}: category {ae['trial_type']} != {me['category']}")
            if int(ae["event_id"]) != me["event_id"]:
                problems.append(f"event id misalignment at {me['event_id']}")

    # every onset must sit inside the rendered timeline with a full post-onset window
    for me in m_events:
        if me["onset_s"] < cfg.lead_in_s - 1e-9:
            problems.append(f"event {me['event_id']} starts inside the lead-in")
        if me["offset_s"] > rendered_duration_s + 1e-9:
            problems.append(f"event {me['event_id']} ends past the rendered video")

    # the schedule must be strictly increasing and exactly one SOA apart
    onsets = [e["onset_s"] for e in m_events]
    if any(b - a <= 0 for a, b in zip(onsets, onsets[1:])):
        problems.append("event onsets are not strictly increasing")
    gaps = {round(b - a, 6) for a, b in zip(onsets, onsets[1:])}
    if gaps and gaps != {round(cfg.soa_s, 6)}:
        problems.append(f"onset spacing is not a constant SOA: {sorted(gaps)}")

    # grey frames must tile every non-stimulus interval, with no gaps or overlaps
    covered = sorted(manifest["grey_intervals_s"] +
                     [(e["onset_s"], e["offset_s"]) for e in m_events])
    cursor = 0.0
    for a, b in covered:
        if abs(a - cursor) > 1e-6:
            problems.append(f"timeline not tiled: gap or overlap at {cursor:.3f}s (next {a:.3f}s)")
            break
        cursor = b
    else:
        if abs(cursor - cfg.stimulus_duration_s) > 1e-6:
            problems.append(f"timeline tiles to {cursor:.3f}s, expected "
                            f"{cfg.stimulus_duration_s:.3f}s")
    return problems


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def gpu_cost_estimate(cfg: S2Config = S2) -> dict:
    """Cost derived from the ACTUAL schedule, not from a category count.

    Charged on the whole rendered timeline -- lead-in, every ISI and the tail-out
    are real frames the model consumes. Costing only the 1 s presentations would
    understate this by ~8x.
    """
    stim_s = cfg.stimulus_duration_s
    compute_s = stim_s * cfg.compute_s_per_stimulus_s
    n_windows = int(np.ceil(stim_s / cfg.model_window_s))
    return {
        "n_events": cfg.n_events,
        "runs": 1,
        "subjects": 1,          # in-silico: one model, no subject dimension
        "repetitions": 1,
        "rendered_stimulus_s": round(stim_s, 3),
        "billed_stimulus_s": round(stim_s, 3),
        "compute_s_per_stimulus_s": cfg.compute_s_per_stimulus_s,
        "estimated_gpu_s": round(compute_s, 1),
        "estimated_gpu_h": round(compute_s / 3600.0, 2),
        "n_model_windows": n_windows,
        "events_per_window": round(cfg.n_events / n_windows, 2) if n_windows else 0.0,
        "basis": "§3.6 cost model, 11.5 s compute per 1 s of stimulus, measured at N=1 cold. "
                 "Charged on the full rendered timeline including lead-in, ISIs and tail-out.",
    }


def cost_table() -> list[dict]:
    """Cost of the design alternatives, so the choice is made on numbers."""
    out = []
    for n_cat, n_ex in ((5, 50), (5, 25), (3, 50), (3, 25)):
        cfg = replace(S2, categories=S2.categories[:n_cat], exemplars_per_category=n_ex)
        est = gpu_cost_estimate(cfg)
        out.append({"categories": n_cat, "exemplars": n_ex, "events": est["n_events"],
                    "stimulus_s": est["rendered_stimulus_s"], "gpu_h": est["estimated_gpu_h"]})
    return out


# ---------------------------------------------------------------------------
# Decision rules — pre-registered
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionRules:
    """Written down BEFORE the answer is seen. Phase B's lesson, applied.

    Each field names exactly which input controls the branch, so no observed
    result can redefine the rule after the fact.
    """

    blocking_gate: str = (
        "speech -> auditory cortex must be strongly positive. Both TRIBE papers report "
        "auditory/language at near-ceiling and our v3b run got p=0.1448. Controlled by: "
        "p_value(A1, speech contrast) < alpha AND effect > detection_floor(A1). "
        "If this fails the pipeline is broken and NOTHING downstream is interpretable."
    )
    recovered: str = (
        "A stop-eligible parcel is RECOVERED iff p_one_sided < alpha AND observed effect "
        "exceeds its own detection floor. Controlled by: (p, effect, floor) for that parcel "
        "under the PRIMARY statistic at the PRIMARY lag."
    )
    not_recovered_stop: str = (
        "Fires iff EVERY stop-eligible parcel fails `recovered`. Controlled by the "
        "stop_eligible set ONLY -- currently FFA and PPA. EBA, VWFA and every secondary "
        "parcel are structurally incapable of firing it (Parcel.stop_eligible=False, "
        "asserted in _validate_parcels)."
    )
    secondary_reporting: str = (
        "Gate-0 unions, V1 control, the ISI-baseline read and all secondary statistics are "
        "REPORTED ALWAYS and DECIDE NOTHING. They may not be promoted to primary after the "
        "results are seen."
    )
    estimand_note: str = (
        "The primary estimand is the published contrast: category response minus the mean "
        "of the other categories, at the primary lag, in the record parcels. The C2 "
        "ISI-baseline variant answers a DIFFERENT question (response vs rest, not vs other "
        "categories) and is therefore secondary by construction, not by outcome."
    )
    lag_policy: str = (
        "PRIMARY = the paper's t=5 (§5.9). ALTERNATIVE = t=0, implied by FmriExtractor("
        "offset=5). Both are fixed in advance and both are read from the SAME peri-event "
        "timecourse, so no extra GPU is needed and neither can be chosen after the fact. "
        "peak_lag_trs is computed and REPORTED as a diagnostic, never used to select the lag "
        "the test is run at -- selecting a lag on the same data inflates type-I to a "
        "measured 0.2032. Adjudication rule below."
    )
    lag_adjudication: str = (
        "If PRIMARY (t=5) recovers -> the replication succeeds on the paper's own protocol; "
        "report the t=0 read as a secondary. "
        "If PRIMARY fails, ALTERNATIVE (t=0) recovers, AND the measured peak lag is ~0 -> "
        "that is evidence for the double-lag hypothesis (source-of-truth.md:51), and it must "
        "be reported as 'not replicated at the published lag; recovered at the lag implied "
        "by the model card' -- NOT as a plain replication. "
        "If BOTH fail -> 'Not recovered' for that parcel, and the stop rule applies. "
        "The measured peak lag is reported in every case, since it is the quantity that "
        "adjudicates the conflict."
    )


RULES = DecisionRules()


def replication_verdict(results: dict, cfg: S2Config = S2) -> dict:
    """Apply the pre-registered rules. Pure function of results + config.

    Args:
        results: {parcel_name: {"p": float, "effect": float, "floor": float}}.

    Returns:
        verdict dict; ``stop`` is True only when every stop-eligible parcel failed.
    """
    eligible = stop_eligible_parcels()
    per_parcel = {}
    for p in ALL_PARCELS:
        r = results.get(p.name)
        if r is None:
            per_parcel[p.name] = {"status": "not_run", "stop_eligible": p.stop_eligible}
            continue
        floor_ok = (not cfg.require_detection_floor) or (r["effect"] > r["floor"])
        recovered = (r["p"] < cfg.alpha) and floor_ok
        per_parcel[p.name] = {
            "status": "recovered" if recovered else "not_recovered",
            "p": r["p"], "effect": r["effect"], "floor": r["floor"],
            "cleared_alpha": r["p"] < cfg.alpha, "cleared_floor": floor_ok,
            "stop_eligible": p.stop_eligible, "provenance": p.provenance,
        }
    eligible_states = [per_parcel[p.name]["status"] for p in eligible]
    ran = [s for s in eligible_states if s != "not_run"]
    stop = bool(ran) and all(s == "not_recovered" for s in ran)
    return {
        "per_parcel": per_parcel,
        "stop_eligible": [p.name for p in eligible],
        "any_record_recovered": any(s == "recovered" for s in ran),
        "stop": stop,
        "rule": RULES.not_recovered_stop,
    }
