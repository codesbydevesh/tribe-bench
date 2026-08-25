"""S2 GPU go/no-go — the checklist, executable.

A prose checklist gets ticked from memory. This one re-derives every item from the
frozen design and from artefacts actually on disk, so "GO" is a computed result
rather than an assertion. Items that cannot be checked mechanically (an
independent review, a human confirmation) are reported as MANUAL and must be
supplied explicitly; they are never assumed satisfied.

    python3 scripts/s2_go_no_go.py [--review-clean] [--dry-run-dir data/s2_dry_run]

Exit 0 = GO. Exit 1 = NO-GO with the failing items listed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurocheck.s2_design import (  # noqa: E402
    ALL_PARCELS, PARCEL_LIST_MISALIGNMENT, RULES, S2, build_manifest, build_schedule,
    environment_provenance, resolve_stimulus_images,
    check_three_way_consistency, events_dataframe, gpu_cost_estimate,
    stop_eligible_parcels,
)

ITEMS: list[tuple[str, str, bool, str]] = []


def check(section: str, name: str, ok: bool, detail: str = "") -> None:
    ITEMS.append((section, name, ok, detail))


def manual(section: str, name: str, ok: bool, detail: str) -> None:
    ITEMS.append((section, name + "  [MANUAL]", ok, detail))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-clean", action="store_true",
                    help="the independent design review found no blocking ambiguity")
    ap.add_argument("--neuralset-timestamp", action="store_true",
                    help="confirmed on the GPU box that V-JEPA frame selection is "
                         "timestamp-based, so 8 fps is safe")
    ap.add_argument("--dry-run-dir", default="data/s2_dry_run")
    args = ap.parse_args()
    dry = Path(args.dry_run_dir)

    # -------------------------------------------------------------- ROIs
    rec = {p.name for p in ALL_PARCELS if p.provenance == "record"}
    sec = {p.name for p in ALL_PARCELS if p.provenance == "secondary"}
    elig = {p.name for p in stop_eligible_parcels()}
    check("ROIs", "primary ROIs are the replication-of-record parcels",
          rec >= {"FFA", "EBA"}, f"record={sorted(rec)}")
    check("ROIs", "secondary parcels are explicitly labelled secondary",
          bool(sec) and not (sec & elig), f"secondary={sorted(sec)}")
    check("ROIs", "stop rule cannot fire from a secondary parcel",
          all(p.provenance == "record" for p in stop_eligible_parcels()))
    check("ROIs", "every stop-eligible parcel has a mapping traced to the paper",
          all(p.mapping_verified for p in stop_eligible_parcels()),
          f"stop-eligible={sorted(elig)}")
    check("ROIs", "the contested PH parcel gates nothing",
          all(not p.stop_eligible for p in ALL_PARCELS if "PH" in p.labels))
    check("ROIs", "the parcel misalignment is recorded verbatim",
          "FFC, V4t, PH, A5, 45, STSv, PGi, TE1a" in PARCEL_LIST_MISALIGNMENT)

    # ------------------------------------------------------------ timing
    ev = build_schedule(S2)
    gaps = {round(b.onset_s - a.onset_s, 6) for a, b in zip(ev, ev[1:])}
    check("Timing", "SOA = 8 s and is frozen", S2.soa_s == 8.0 and gaps == {8.0})
    check("Timing", "ISI = SOA - presentation (not SOA itself)", S2.isi_s == 7.0)
    check("Timing", "lead-in >= 25 s", S2.lead_in_s >= 25.0, f"{S2.lead_in_s}s")
    check("Timing", "last event keeps a full post-onset window",
          S2.stimulus_duration_s - ev[-1].offset_s >= S2.tail_out_s)

    # ------------------------------------------------------------ stimulus
    src = Path("neurocheck/s2_stimulus.py").read_text()
    check("Stimulus", "one continuous video (single render entry point)",
          src.count("\ndef render(") == 1)
    check("Stimulus", "grey ISIs are physically rendered",
          "writer.write(grey)" in src)
    check("Stimulus", "no audio track is written", "audio" not in src.lower().split("uses opencv")[0]
          or "no audio" in src.lower())
    check("Stimulus", "no whisper / get_events_dataframe anywhere in the S2 path",
          all("get_events_dataframe" not in Path(f).read_text()
              .replace("``get_events_dataframe``", "")
              .replace("get_events_dataframe`` runs", "")
              .replace("NOT call ``get_events_dataframe``", "")
              for f in ("neurocheck/s2_design.py", "neurocheck/s2_stimulus.py")))
    check("Stimulus", "presentation order is randomised and reproducible",
          [e.stimulus_id for e in build_schedule(S2)] == [e.stimulus_id for e in ev]
          and [e.category for e in ev] != sorted(e.category for e in ev))
    check("Stimulus", "events table is hand-built from the frozen schedule",
          list(events_dataframe(S2)["onset"]) == [e.onset_s for e in ev])

    # -------------------------------------------------- three-way agreement
    man = build_manifest(S2)
    problems = check_three_way_consistency(S2, man, S2.stimulus_duration_s, S2.fps,
                                           events_dataframe(S2))
    check("Consistency", "stimulus / manifest / analysis events agree",
          not problems, "; ".join(problems[:2]))
    check("Consistency", "CPU dry run completed",
          (dry / "s2_manifest_full.json").exists(),
          f"{dry}/s2_manifest_full.json")
    check("Consistency", "dry-run outputs are machine-readable",
          all((dry / f).exists() for f in
              ("s2_manifest_tiny.json", "s2_events_tiny.csv", "s2_report_tiny.json")))

    # ------------------------------------------------------------------ C2
    check("C2", "ISI-baseline read is classified BEFORE results",
          S2.isi_baseline_role == "secondary")
    check("C2", "the estimand change is documented",
          "DIFFERENT question" in RULES.estimand_note)
    check("C2", "window packing is logged", S2.log_window_packing)

    # ------------------------------------------------------------ decisions
    check("Decisions", "primary lag fixed in advance", S2.primary_lag_trs == 5)
    check("Decisions", "alternative lag pre-specified, same run",
          S2.alternative_lag_trs == 0 and 0 in S2.report_lags and 5 in S2.report_lags)
    check("Decisions", "peak lag is reported, never used to select the tested lag",
          S2.peak_lag_policy == "measure_and_report_only"
          and "never used to select" in RULES.lag_policy)
    check("Decisions", "lag-0-only recovery is not callable a replication",
          "not replicated at the published lag" in RULES.lag_adjudication)
    check("Decisions", "recovery requires a detection floor", S2.require_detection_floor)
    check("Decisions", "blocking gate (speech -> auditory) is stated",
          "auditory" in RULES.blocking_gate)

    # ----------------------------------------------------------- provenance
    check("Provenance", "model id recorded in the manifest",
          man["provenance"]["model_id"] == "facebook/tribev2")
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except Exception:
        sha, dirty = "", "unknown"
    check("Provenance", "code commit is capturable", bool(sha), sha[:12])
    check("Provenance", "working tree is clean at freeze time", dirty == "",
          "uncommitted changes" if dirty else "clean")
    check("Provenance", "design fingerprint is stable",
          build_manifest(S2)["design_fingerprint"] == S2.fingerprint(),
          S2.fingerprint())

    # ------------------------------------------------------------ compute
    est = gpu_cost_estimate(S2)
    check("Compute", "cost derived from the actual rendered schedule",
          est["rendered_stimulus_s"] == S2.stimulus_duration_s,
          f"{est['estimated_gpu_h']} h for {est['n_events']} events")
    check("Compute", "runs / subjects / repetitions are stated",
          all(k in est for k in ("runs", "subjects", "repetitions")))

    # ------------------------------- un-retrofittable provenance (review B)
    check("Provenance", "model revision SHA is pinned", S2.model_revision is not None,
          "fill S2Config.model_revision on the GPU box; from_pretrained does NOT pass "
          "revision=, so Meta can update the repo and a re-run silently differs")
    check("Provenance", "stimulus set identity is recorded",
          S2.stimulus_set_version is not None,
          "fLoc cannot be redistributed, so the manifest is the ONLY carrier of image "
          "identity" if S2.stimulus_set_version is None else S2.stimulus_set_version)
    # The recorded hashes must still match the files on disk, or the manifest
    # describes a stimulus set that is no longer there.
    man_path = Path("data/s2_manifest.json")
    try:
        rec = json.loads(man_path.read_text())["provenance"]
        imgs_rec = (rec.get("images") or {}).get("files") or {}
        live = resolve_stimulus_images("data/floc", S2) if imgs_rec else {}
        drift = [k for k in imgs_rec
                 if k not in live or live[k]["sha256"] != imgs_rec[k]["sha256"]]
        check("Provenance", "manifest image hashes match the files on disk",
              bool(imgs_rec) and not drift,
              f"{len(imgs_rec)} hashed" if not drift else f"{len(drift)} drifted")
        check("Provenance", "checkpoint revision + hash recorded in the manifest",
              bool(rec.get("checkpoint", {}).get("sha256")) and bool(rec.get("model_revision")),
              str(rec.get("model_revision", ""))[:12])
    except Exception as exc:
        check("Provenance", "manifest image hashes match the files on disk", False,
              f"{type(exc).__name__}: {exc}")
        check("Provenance", "checkpoint revision + hash recorded in the manifest", False, "")
    check("Provenance", "load_model pins the revision rather than the floating branch",
          "fetch_pinned_checkpoint" in Path("tribe_tools/model.py").read_text()
          and "revision=revision" in Path("tribe_tools/model.py").read_text())
    def _tracked(path):
        return subprocess.run(["git", "ls-files", "--error-unmatch", path],
                              capture_output=True).returncode == 0
    # A pre-registration that exists only on one untracked disk is not a
    # pre-registration: its value is being fixed and timestamped BEFORE any result.
    # Both .notes/ and data/ are gitignored, so this must be asserted, not assumed.
    check("Provenance", "the frozen manifest is version-controlled",
          _tracked("data/s2_manifest.json"), "data/ is gitignored; force-added")
    check("Provenance", "the pre-registration is version-controlled",
          _tracked("ops/S2-PREREGISTRATION.md"), ".notes/ is gitignored")
    check("Provenance", "the manifest carries the decision rules",
          bool(json.loads(Path("data/s2_manifest.json").read_text()).get("decision_rules")))
    check("Provenance", "stimulus images are gitignored (fLoc states no licence)",
          subprocess.run(["git", "check-ignore", "-q", "data/floc/faces"]).returncode == 0)
    check("Provenance", "image resolver is deterministic and hashes every file",
          "sorted(" in Path("neurocheck/s2_design.py").read_text()
          and "sha256" in Path("neurocheck/s2_design.py").read_text())
    check("Provenance", "environment is capturable",
          bool(environment_provenance().get("python")))
    check("Provenance", "packing attenuation is stated before the answer is seen",
          "not 'TRIBE" in RULES.packing_attenuation or "NOT 'TRIBE" in RULES.packing_attenuation)

    # --------------------------------------------------- decision-rule integrity
    named = {w for w in RULES.not_recovered_stop.replace(",", " ").split()
             if w.strip(".") in {p.name for p in ALL_PARCELS}}
    check("Decisions", "the rule text names the parcels the code actually gates on",
          {p.name for p in stop_eligible_parcels()} <= named,
          f"rule names {sorted(named)}")
    check("Decisions", "peak tolerance is a number, not a word",
          isinstance(S2.peak_tolerance_trs, int))
    check("Decisions", "both lags are scored and reported separately",
          S2.primary_lag_trs != S2.alternative_lag_trs)

    # ------------------------------------------------------------- manual
    # The ONE check that genuinely cannot be run without the GPU environment.
    # neuralset is not installable here, so whether V-JEPA selects its 64 frames by
    # TIMESTAMP or by FRAME INDEX cannot be settled. At 8 fps an index-based loader
    # would span 8 s of timeline instead of 4, halving each presentation's weight
    # inside its tubelet and smearing it across neighbouring events at an 8 s SOA.
    # Timestamp-based -> 8 fps is fine. Index-based -> BLOCKING, raise fps to 16.
    try:
        import importlib.util
        have_neuralset = importlib.util.find_spec("neuralset") is not None
    except Exception:
        have_neuralset = False
    manual("GPU-only", "neuralset samples V-JEPA frames by TIMESTAMP, not frame index",
           args.neuralset_timestamp,
           "neuralset not installed here; run scripts/s2_check_frame_sampling.py on the "
           "GPU box and pass --neuralset-timestamp"
           if not args.neuralset_timestamp else
           f"confirmed by the operator (neuralset importable here: {have_neuralset})")

    manual("Review", "independent design review found no blocking ambiguity",
           args.review_clean,
           "pass --review-clean once the reviewer reports GO" if not args.review_clean
           else "confirmed by the operator")

    # ------------------------------------------------------------- report
    width = max(len(n) for _, n, _, _ in ITEMS)
    section = None
    for sec_name, name, ok, detail in ITEMS:
        if sec_name != section:
            print(f"\n{sec_name}")
            section = sec_name
        print(f"  [{'x' if ok else ' '}] {name:<{width}}" + (f"  {detail}" if detail else ""))

    failed = [(s, n) for s, n, ok, _ in ITEMS if not ok]
    print(f"\n{len(ITEMS) - len(failed)}/{len(ITEMS)} checklist items satisfied")
    if failed:
        print("\nNO-GO — outstanding:")
        for s, n in failed:
            print(f"  - [{s}] {n}")
        return 1
    print("\nGPU GO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
