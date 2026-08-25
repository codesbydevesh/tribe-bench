"""S2 CPU dry run — exercise the whole control flow with no GPU.

The point is NOT statistical validity. It is that every function is called, every
expected file appears, event ids line up, parcel labels resolve, the decision
branches behave, randomisation is reproducible, the manifest is written, and the
output schema is stable — before any GPU hour is spent discovering otherwise.

Runs a SMALL config end to end (rendered video included, since the renderer is
part of what must be proven), then validates the FULL config's manifest, cost and
consistency without rendering 1050 s of frames.

    python3 scripts/s2_dry_run.py [--out data/s2_dry_run]

Exit code 0 = every gate passed. Non-zero = at least one gate failed; the run is
NOT GPU-ready.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurocheck.s2_design import (  # noqa: E402
    ALL_PARCELS, S2, build_manifest, build_schedule, check_three_way_consistency,
    cost_table, events_dataframe, gpu_cost_estimate, replication_verdict,
    stop_eligible_parcels,
)
from neurocheck.s2_stimulus import (  # noqa: E402
    frame_plan, render, verify_rendered_frames,
)

RESULTS: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def _code_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/s2_dry_run")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # A small but structurally identical design: same SOA, same lead-in rule,
    # fewer exemplars so the video is seconds rather than minutes.
    tiny = replace(S2, exemplars_per_category=2, lead_in_s=3.0, tail_out_s=3.0)

    print(f"\n=== S2 CPU dry run ===\nfull design  {S2.fingerprint()}"
          f"\ntiny design  {tiny.fingerprint()}\n")

    # ---------------------------------------------------------------- schedule
    print("[1] schedule and randomisation")
    ev = build_schedule(tiny)
    gate("schedule is reproducible",
         [e.stimulus_id for e in build_schedule(tiny)] == [e.stimulus_id for e in ev])
    gate("every exemplar appears exactly once",
         len({e.stimulus_id for e in ev}) == tiny.n_events == len(ev),
         f"{len(ev)} events")
    gaps = {round(b.onset_s - a.onset_s, 6) for a, b in zip(ev, ev[1:])}
    gate("onset spacing equals the SOA", gaps == {tiny.soa_s}, f"gaps={sorted(gaps)}")

    # ---------------------------------------------------------------- manifest
    print("\n[2] manifest")
    manifest = build_manifest(tiny, code_commit=_code_commit())
    mpath = out / "s2_manifest_tiny.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    gate("manifest written", mpath.exists(), str(mpath))
    reread = json.loads(mpath.read_text())
    gate("manifest survives a JSON round trip", reread["events"] == manifest["events"])
    gate("manifest records model id and code commit",
         bool(reread["provenance"]["model_id"]) and reread["provenance"]["code_commit"] is not None,
         f"{reread['provenance']['model_id']} @ {str(reread['provenance']['code_commit'])[:8]}")
    gate("manifest quotes the parcel list verbatim, not a paraphrase",
         "FFC, V4t, PH, A5, 45, STSv, PGi, TE1a" in reread["provenance"]["roi_mapping_status"])

    # ------------------------------------------------------------------ render
    print("\n[3] rendered stimulus (real file, real frames)")
    vpath = out / "s2_stimulus_tiny.mp4"
    try:
        r = render(tiny, vpath)
        gate("video file produced", vpath.exists() and vpath.stat().st_size > 0,
             f"{vpath.stat().st_size/1024:.0f} KB")
        gate("rendered duration matches the schedule",
             abs(r.duration_s - tiny.stimulus_duration_s) <= 1.0 / tiny.fps,
             f"{r.duration_s:.3f}s vs {tiny.stimulus_duration_s:.3f}s")
        expected_stim = int(round(tiny.n_events * tiny.on_duration_s * tiny.fps))
        gate("presentation frame count matches the schedule",
             r.stimulus_frames == expected_stim, f"{r.stimulus_frames} vs {expected_stim}")
        gate("grey ISI frames were physically rendered",
             r.grey_frames > 5 * r.stimulus_frames,
             f"{r.grey_frames} grey vs {r.stimulus_frames} stimulus")
        px = verify_rendered_frames(tiny, vpath, ev)
        gate("sampled pixels agree with the schedule", not px, "; ".join(px[:3]))
    except Exception as exc:                                    # pragma: no cover
        gate("stimulus rendering", False, f"{type(exc).__name__}: {exc}")
        r = None

    # ------------------------------------------------------------------ events
    print("\n[4] events table")
    df = events_dataframe(tiny)
    gate("event ids line up with the manifest",
         list(df["event_id"]) == [e["event_id"] for e in manifest["events"]])
    gate("no whisper / get_events_dataframe in the path",
         "get_events_dataframe" not in Path("neurocheck/s2_design.py").read_text()
         .replace("``get_events_dataframe``", "").replace("get_events_dataframe`` runs", ""))
    dpath = out / "s2_events_tiny.csv"
    df.to_csv(dpath, index=False)
    gate("events table written", dpath.exists(), str(dpath))

    # ------------------------------------------------------- three-way gate
    print("\n[5] rendered <-> manifest <-> analysis events")
    if r is not None:
        problems = check_three_way_consistency(tiny, manifest, r.duration_s, r.fps, df)
        gate("three-way consistency", not problems, "; ".join(problems[:3]))
        bad = df.copy(); bad.loc[0, "onset"] = float(bad.loc[0, "onset"]) + 0.5
        gate("the gate CATCHES an injected drift",
             bool(check_three_way_consistency(tiny, manifest, r.duration_s, r.fps, bad)))

    # ------------------------------------------------------------------ parcels
    print("\n[6] parcel labels resolve")
    try:
        from tribe_tools.atlas import get_vertices
        unresolved = []
        for p in ALL_PARCELS:
            for lab in p.labels:
                try:
                    v = get_vertices(lab, hemi=p.hemi)
                    if v is None or len(v) == 0:
                        unresolved.append(f"{p.name}:{lab} (empty)")
                except Exception as exc:
                    unresolved.append(f"{p.name}:{lab} ({type(exc).__name__})")
        gate("every parcel label resolves to vertices", not unresolved,
             "; ".join(unresolved[:4]) if unresolved else f"{len(ALL_PARCELS)} parcels")
    except Exception as exc:
        gate("atlas available", False,
             f"{type(exc).__name__}: {exc} — needs MNE + HCP atlas (internet on first fetch)")

    # ----------------------------------------------------------- decision path
    print("\n[7] decision rules")
    def _r(p_val, effect, floor, peak=None):
        """Lag-keyed result: the verdict scores BOTH pre-specified lags."""
        one = {"p": p_val, "effect": effect, "floor": floor}
        return {"by_lag": {S2.primary_lag_trs: one, S2.alternative_lag_trs: one},
                "peak_lag_trs": peak, "statistic": S2.primary_statistic}

    win, fail = _r(0.001, 0.5, 0.05), _r(0.9, 0.01, 0.05)
    gate("stop fires when all stop-eligible parcels fail",
         replication_verdict({"FFA": fail, "EBA": fail}, S2)["stop"] is True)
    gate("stop does NOT fire when one record parcel recovers",
         replication_verdict({"FFA": win, "EBA": fail}, S2)["stop"] is False)
    v = replication_verdict({"FFA": win, "EBA": win, "PPA": fail, "VWFA": fail,
                             "PPA_literature": fail, "EBA_gate0_union": fail,
                             "V1_control": fail}, S2)
    gate("secondary parcels cannot fire the stop rule", v["stop"] is False)
    gate("a below-floor 'significant' result is NOT a recovery",
         replication_verdict({"FFA": _r(0.001, 0.01, 0.05)},
                             S2)["per_parcel"]["FFA"]["status"] == "not_recovered")
    # the lag dimension itself
    only_alt = {"by_lag": {S2.primary_lag_trs: {"p": 0.9, "effect": 0.01, "floor": 0.05},
                           S2.alternative_lag_trs: {"p": 0.001, "effect": 0.5, "floor": 0.05}},
                "peak_lag_trs": 0, "statistic": S2.primary_statistic}
    v_alt = replication_verdict({"FFA": only_alt, "EBA": only_alt}, S2)
    gate("recovery only at the alternative lag is a DISTINCT status",
         v_alt["per_parcel"]["FFA"]["status"] == "recovered_at_alternative_lag",
         v_alt["per_parcel"]["FFA"]["status"])
    gate("that outcome does NOT fire the stop rule", v_alt["stop"] is False)
    gate("a missing parcel blocks the stop rule",
         replication_verdict({"FFA": fail}, S2)["stop"] is False)
    nan = float("nan")
    gate("a non-finite number is invalid, not a null",
         replication_verdict({"FFA": _r(nan, 0.5, 0.05), "EBA": _r(nan, 0.5, 0.05)},
                             S2)["stop"] is False)
    gate("results keyed by HCP label instead of functional name are flagged",
         bool(replication_verdict({"FFC": fail}, S2)["warnings"]))
    gate("only the securely mapped parcels are stop-eligible",
         {p.name for p in stop_eligible_parcels()} == {"FFA", "EBA"},
         "PPA/VWFA share the contested PH parcel and cannot gate")

    # ------------------------------------------------------- mock analysis run
    print("\n[8] analysis reaches the reporting stage on mock predictions")
    try:
        from tribe_tools.roi_stats import event_locked_contrast, perm_p
        rng = np.random.default_rng(0)
        per_cat = {c: rng.normal(0.0, 1.0, tiny.exemplars_per_category)
                   for c in tiny.categories}
        per_cat["faces"] = per_cat["faces"] + 3.0          # a plantable effect
        target = per_cat["faces"]
        others = [per_cat[c] for c in tiny.categories if c != "faces"]
        eff = event_locked_contrast(target, others)
        p = perm_p(list(target), list(np.concatenate(others)), n_perm=500, seed=0)
        gate("contrast + p-value computed on mock data",
             np.isfinite(eff) and 0.0 < p <= 1.0, f"effect={eff:.4f}, p={p:.4f}")
        report = {
            "design_fingerprint": tiny.fingerprint(),
            "provenance": manifest["provenance"],
            "results": {"FFA": {
                "by_lag": {S2.primary_lag_trs: {"p": float(p), "effect": float(eff),
                                                "floor": 0.05},
                           S2.alternative_lag_trs: {"p": float(p), "effect": float(eff),
                                                    "floor": 0.05}},
                "peak_lag_trs": None, "statistic": S2.primary_statistic}},
        }
        report["verdict"] = replication_verdict(report["results"], S2)
        rpath = out / "s2_report_tiny.json"
        rpath.write_text(json.dumps(report, indent=2, default=str))
        gate("report written with a stable schema",
             rpath.exists() and set(json.loads(rpath.read_text())) ==
             {"design_fingerprint", "provenance", "results", "verdict"}, str(rpath))
    except Exception as exc:
        gate("mock analysis", False, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ full design
    print("\n[9] FULL design (manifest + cost only, no render)")
    full = build_manifest(S2, code_commit=_code_commit())
    fpath = out / "s2_manifest_full.json"
    fpath.write_text(json.dumps(full, indent=2, default=str))
    gate("full manifest written", fpath.exists(), f"{len(full['events'])} events")
    gate("full manifest is internally consistent",
         not check_three_way_consistency(S2, full, S2.stimulus_duration_s, S2.fps,
                                         events_dataframe(S2)))
    plan = frame_plan(S2, build_schedule(S2))
    gate("full frame plan covers the whole timeline",
         len(plan) == int(round(S2.stimulus_duration_s * S2.fps)),
         f"{len(plan)} frames @ {S2.fps} fps")
    est = gpu_cost_estimate(S2)
    print(f"\n    GPU budget from the ACTUAL schedule:")
    print(f"      events {est['n_events']}, runs {est['runs']}, subjects {est['subjects']}, "
          f"repetitions {est['repetitions']}")
    print(f"      rendered stimulus {est['rendered_stimulus_s']:.0f} s "
          f"({est['rendered_stimulus_s']/60:.1f} min)")
    print(f"      estimated GPU {est['estimated_gpu_h']:.2f} h "
          f"({est['n_model_windows']} windows, {est['events_per_window']:.1f} events/window)")
    print("\n    alternatives:")
    for row in cost_table():
        print(f"      {row['categories']} cat x {row['exemplars']:>3} = {row['events']:>3} events"
              f"  -> {row['stimulus_s']:>6.0f} s  {row['gpu_h']:>5.2f} h")

    # ------------------------------------------------------------------ verdict
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n=== {len(RESULTS) - len(failed)}/{len(RESULTS)} gates passed ===")
    if failed:
        print("FAILED GATES (NOT GPU-ready):")
        for n in failed:
            print(f"  - {n}")
        return 1
    print("All dry-run gates passed. This is necessary, not sufficient: see the "
          "GPU go/no-go checklist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
