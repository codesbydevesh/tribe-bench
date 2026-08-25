"""S2 — execute the frozen design. The implementation, not a reinterpretation.

Every parameter comes from :data:`neurocheck.s2_design.S2`. This script contains no
experimental choices of its own: if a number here disagrees with the frozen design,
the design wins and this script is wrong.

**CPU preparation is separate from GPU inference.** ``--prepare`` does everything
that needs no GPU (resolve images, render, verify, hash) and ``--infer`` does only
the forward pass and analysis, so the Kaggle session is one command over
already-verified inputs.

    # on this box, no GPU:
    python3 scripts/s2_run.py --prepare
    python3 scripts/s2_run.py --infer --stub        # end-to-end with a fake model

    # on the GPU box:
    python3 scripts/s2_run.py --infer

Refuses to run on any mismatch. A silently substituted input is the failure this
whole phase exists to prevent.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurocheck.s2_design import (  # noqa: E402
    ALL_PARCELS, LAG_CONFLICT, RULES, S2, S2Config, build_manifest, build_schedule,
    check_three_way_consistency, environment_provenance, events_dataframe,
    replication_verdict, resolve_stimulus_images, stop_eligible_parcels,
)
from neurocheck.s2_stimulus import frame_plan, probe_video, render, verify_rendered_frames  # noqa: E402

# The manifest and the probe ship IN GIT, so they arrive with the repo clone and
# stay repo-relative. The images and the video cannot ship (fLoc has no licence),
# so on a GPU box they live wherever the dataset is mounted -- on Kaggle that is
# /kaggle/input/<dataset-slug>, which is READ-ONLY. Hence one knob, resolved once,
# rather than paths guessed inside the session.
STIMULUS_ROOT = Path(os.environ.get("S2_STIMULUS_ROOT", "data"))
PROBE = Path("data/s2_stimulus_probe.json")
MANIFEST = Path("data/s2_manifest.json")


def video_path(root: Path = None) -> Path:
    return (root or STIMULUS_ROOT) / "s2_stimulus.mp4"


def image_root(root: Path = None) -> Path:
    return (root or STIMULUS_ROOT) / "floc"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def die(msg: str) -> None:
    """Fail loudly. Never substitute, never continue on a mismatch."""
    print(f"\nREFUSING TO RUN: {msg}", file=sys.stderr)
    raise SystemExit(2)


# ---------------------------------------------------------------- preparation

def prepare(cfg: S2Config) -> dict:
    """CPU only: resolve images, render, verify against the spec, hash."""
    print(f"=== S2 prepare === design {cfg.fingerprint()}\n")
    images = resolve_stimulus_images(image_root(), cfg)
    print(f"  images     {len(images)} resolved and hashed from {image_root()}")

    events = build_schedule(cfg)
    r = render(cfg, video_path(), events=events, images=images)
    if r.placeholders:
        die("rendered with placeholder frames; a recorded run needs the real images")

    # verify against the SPEC, not against the renderer's own arithmetic
    plan = frame_plan(cfg, events)
    per_event = np.bincount(plan[plan >= 0], minlength=cfg.n_events)
    problems = []
    if abs(r.duration_s - cfg.stimulus_duration_s) > 1.0 / cfg.fps:
        problems.append(f"duration {r.duration_s} != spec {cfg.stimulus_duration_s}")
    if r.fps != cfg.fps:
        problems.append(f"fps {r.fps} != spec {cfg.fps}")
    if (r.height, r.width) != cfg.frame_size:
        problems.append(f"resolution {r.width}x{r.height} != spec {cfg.frame_size}")
    if r.n_frames != int(round(cfg.stimulus_duration_s * cfg.fps)):
        problems.append(f"frame count {r.n_frames} != spec")
    if len(set(per_event.tolist())) != 1:
        problems.append("events differ in frame count")
    problems += verify_rendered_frames(cfg, video_path(), events, sample=60)
    problems += check_three_way_consistency(cfg, build_manifest(cfg), r.duration_s,
                                            r.fps, events_dataframe(cfg))
    if problems:
        die("rendered stimulus does not match the frozen spec:\n  - " +
            "\n  - ".join(problems[:8]))

    video = {"path": str(video_path()), "sha256": _sha256(video_path()),
             "bytes": video_path().stat().st_size, "n_frames": r.n_frames, "fps": r.fps,
             "duration_s": r.duration_s, "width": r.width, "height": r.height,
             "stimulus_frames": r.stimulus_frames, "grey_frames": r.grey_frames,
             "placeholders": r.placeholders}
    PROBE.write_text(json.dumps(video, indent=2))
    print(f"  video      {r.n_frames} frames @ {r.fps}fps, {r.duration_s:.3f}s, "
          f"{r.width}x{r.height}")
    print(f"             sha256 {video['sha256'][:32]}...")

    idx = json.loads((image_root() / "index.json").read_text())
    man = build_manifest(
        cfg, code_commit=_git("rev-parse", "HEAD"),
        tree_dirty=bool(_git("status", "--porcelain")),
        environment=environment_provenance(), video=video,
        images={"source_repo": idx["source_repo"], "commit": idx["commit"],
                "selection_rule": idx["selection_rule"], "licence": idx["licence"],
                "n": len(images), "files": images})
    from tribe_tools.model import TRIBEV2_CKPT_BYTES, TRIBEV2_CKPT_SHA256, TRIBEV2_REVISION
    man["provenance"]["checkpoint"] = {
        "repo": "facebook/tribev2", "revision": TRIBEV2_REVISION, "file": "best.ckpt",
        "sha256": TRIBEV2_CKPT_SHA256, "bytes": TRIBEV2_CKPT_BYTES}
    MANIFEST.write_text(json.dumps(man, indent=2, default=str))
    print(f"  manifest   {MANIFEST} ({len(json.dumps(man)) / 1024:.0f} KB)")
    print("\nprepare: OK. Inputs verified against the frozen spec.")
    return man


# ------------------------------------------------------------------ inference

def _check_inputs(cfg: S2Config) -> dict:
    """Every input must match what prepare() recorded, or refuse."""
    if not MANIFEST.exists():
        die(f"{MANIFEST} missing — run --prepare first")
    man = json.loads(MANIFEST.read_text())
    if man["design_fingerprint"] != cfg.fingerprint():
        die(f"manifest is for design {man['design_fingerprint']}, code is "
            f"{cfg.fingerprint()} — the design changed since prepare")
    v = (man.get("provenance") or {}).get("video")
    if not v:
        die("manifest records no video — run --prepare first")
    if not video_path().exists():
        die(f"{video_path()} missing — run --prepare first")
    got = _sha256(video_path())
    if got != v["sha256"]:
        die(f"stimulus hash mismatch: manifest {v['sha256'][:16]}..., file "
            f"{got[:16]}... — this is not the video the manifest describes")
    if v.get("placeholders"):
        die("the recorded video was rendered with placeholder frames")
    live = resolve_stimulus_images(image_root(), cfg)
    rec = man["provenance"]["images"]["files"]
    drift = [k for k in rec if k not in live or live[k]["sha256"] != rec[k]["sha256"]]
    if drift:
        die(f"{len(drift)} stimulus images differ from the manifest: {drift[:5]}")
    return man


def _peri_event(preds, row_times, verts, onsets, cfg):
    """Peri-event timecourse over the reported lag grid, one pass, all lags."""
    from tribe_tools.roi_stats import peri_event_timecourse
    pre = -min(cfg.report_lags)
    post = max(cfg.report_lags)
    return peri_event_timecourse(preds, verts, onsets, row_times, pre, post), pre


def analyse(preds, segments, cfg: S2Config, man: dict) -> dict:
    """Score every parcel at BOTH pre-specified lags from the SAME timecourse."""
    from tribe_tools.atlas import get_vertices
    from tribe_tools.roi_stats import (
        detection_floor, event_locked_contrast, peak_lag_trs, perm_p,
        row_times_from_segments,
    )

    row_times = row_times_from_segments(segments)
    events = build_schedule(cfg)
    results: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}

    for parcel in ALL_PARCELS:
        try:
            verts = np.concatenate([get_vertices(l, hemi=parcel.hemi)
                                    for l in parcel.labels])
        except Exception as exc:
            die(f"parcel {parcel.name} ({parcel.labels}) did not resolve: {exc}")
        verts = np.unique(verts)

        # ONE timecourse per category; every lag is a column of it.
        courses = {}
        for cat in cfg.categories:
            onsets = [e.onset_s for e in events if e.category == cat]
            tc, pre = _peri_event(preds, row_times, verts, onsets, cfg)
            courses[cat] = tc
        peak = peak_lag_trs([courses[c] for c in cfg.categories], pre_trs=pre)

        by_lag = {}
        for lag in (cfg.primary_lag_trs, cfg.alternative_lag_trs):
            col = pre + lag
            per_cat = {c: np.asarray(courses[c])[:, col] for c in cfg.categories}
            # the parcel's own category is the one the paper pairs with it
            target_cat = PARCEL_CATEGORY[parcel.name]
            tgt = per_cat[target_cat]
            others = [per_cat[c] for c in cfg.categories if c != target_cat]
            effect = event_locked_contrast(tgt, others)
            p = perm_p(list(tgt), list(np.concatenate(others)),
                       n_perm=cfg.n_perm, seed=cfg.perm_seed)
            # the floor is computed from THIS parcel's own noise and n
            noise = float(np.std(np.concatenate(others), ddof=1))
            floor = detection_floor(n_per_group=len(tgt), noise_sd=noise,
                                    alpha=cfg.alpha, seed=cfg.perm_seed)
            by_lag[lag] = {"p": float(p), "effect": float(effect), "floor": float(floor),
                           "n_target": int(len(tgt)), "noise_sd": noise}
        results[parcel.name] = {"by_lag": by_lag, "peak_lag_trs": int(peak),
                                "statistic": cfg.primary_statistic}
        diagnostics[parcel.name] = {
            "n_vertices": int(verts.size), "target_category": PARCEL_CATEGORY[parcel.name],
            "timecourse_by_category": {c: np.asarray(courses[c]).mean(axis=0).tolist()
                                       for c in cfg.categories},
            "lag_grid": list(cfg.report_lags)}
    return {"results": results, "diagnostics": diagnostics}


# Which category each parcel is the paper's target for. Frozen: the contrast is
# category-minus-others, so this decides what "recovered" means per parcel.
PARCEL_CATEGORY = {
    "FFA": "faces", "EBA": "bodies", "PPA": "places", "VWFA": "characters",
    "PPA_literature": "places", "EBA_gate0_union": "bodies", "V1_control": "faces",
}


def _validate_parcel_categories() -> None:
    """Every parcel needs a target category, or analyse() dies mid-run."""
    missing = [p.name for p in ALL_PARCELS if p.name not in PARCEL_CATEGORY]
    if missing:
        die(f"no target category declared for {missing}; the contrast is "
            "category-minus-others, so this decides what 'recovered' means")
    bad = {k: v for k, v in PARCEL_CATEGORY.items() if v not in S2.categories}
    if bad:
        die(f"target categories not in the design: {bad}")


def infer(cfg: S2Config, stub: bool) -> int:
    _validate_parcel_categories()
    man = _check_inputs(cfg)
    print(f"=== S2 infer === design {cfg.fingerprint()}"
          f"{'  [STUB — no GPU, structure only]' if stub else ''}\n")
    print(f"  model      facebook/tribev2 @ {cfg.model_revision[:12]}")
    print(f"  stimulus   {video_path()} sha256 {man['provenance']['video']['sha256'][:16]}...")
    print(f"  lags       primary {cfg.primary_lag_trs}, alternative "
          f"{cfg.alternative_lag_trs}, reported {list(cfg.report_lags)}")
    print(f"  gating     {[p.name for p in stop_eligible_parcels()]} "
          f"(all others report-only)\n")

    t0 = time.time()
    if stub:
        # Structure only: a fake model with the real row grid, so the whole
        # pipeline executes on CPU. Never a scientific result.
        n_rows = int(round(cfg.stimulus_duration_s))
        rng = np.random.default_rng(0)
        preds = rng.normal(0, 1, (n_rows, 20484)).astype(np.float32)

        class _Seg:
            def __init__(self, t): self.start = float(t)
        segments = [_Seg(t) for t in range(n_rows)]
    else:
        from tribe_tools.inference import predict_single
        from tribe_tools.model import load_model
        model = load_model(device="cuda", revision=cfg.model_revision,
                           config_update={
                               "data.video_feature.infra.keep_in_ram": False,
                               "data.audio_feature.infra.keep_in_ram": False,
                               "data.text_feature.infra.keep_in_ram": False})
        preds, segments = predict_single(model, video_path())
    elapsed = time.time() - t0
    print(f"  inference  {elapsed:.1f}s, preds {np.shape(preds)}, "
          f"{len(segments)} segments")

    out = analyse(preds, segments, cfg, man)
    verdict = replication_verdict(out["results"], cfg)

    report = {
        "schema_version": 1,
        "stub": stub,
        "design_fingerprint": cfg.fingerprint(),
        "provenance": {**man["provenance"],
                       "inference_seconds": round(elapsed, 1),
                       "run_environment": environment_provenance()},
        "lag_conflict": LAG_CONFLICT.strip(),
        "decision_rules": {k: v for k, v in vars(RULES).items()
                           if not k.startswith("_")} or None,
        "results": out["results"],
        "diagnostics": out["diagnostics"],
        "verdict": verdict,
    }
    dest = Path("data/s2_report_stub.json" if stub else "data/s2_report.json")
    dest.write_text(json.dumps(report, indent=2, default=str))

    print("\n  parcel            status                          p(primary)  effect  floor")
    for name, pp in verdict["per_parcel"].items():
        bl = (pp.get("by_lag") or {}).get(str(cfg.primary_lag_trs), {})
        gate = "*" if pp.get("stop_eligible") else " "
        print(f"  {gate}{name:16s} {pp['status']:30s} "
              f"{bl.get('p', float('nan')):9.4f} {bl.get('effect', float('nan')):8.3f} "
              f"{bl.get('floor', float('nan')):7.3f}")
    print("  (* = may fire the stop rule)")
    for w in verdict["warnings"]:
        print(f"\n  WARNING: {w}")
    print(f"\n  stop = {verdict['stop']}   incomplete = {verdict['incomplete']}")
    print(f"  report -> {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="CPU: render and verify")
    ap.add_argument("--infer", action="store_true", help="run the model and analyse")
    ap.add_argument("--stimulus-root", default=None,
                    help="where floc/ and s2_stimulus.mp4 live. Default 'data'; on "
                         "Kaggle pass /kaggle/input/<dataset-slug>. Env: S2_STIMULUS_ROOT")
    ap.add_argument("--stub", action="store_true",
                    help="with --infer: fake model, CPU only, structure check")
    args = ap.parse_args()
    if args.stimulus_root:
        global STIMULUS_ROOT
        STIMULUS_ROOT = Path(args.stimulus_root)
    if args.prepare and not os.access(STIMULUS_ROOT, os.W_OK):
        die(f"--prepare writes the rendered video into {STIMULUS_ROOT}, which is not "
            "writable. On Kaggle the dataset mount is READ-ONLY: prepare on a machine "
            "you control, upload the result, and use --infer only.")
    if not (args.prepare or args.infer):
        ap.error("choose --prepare and/or --infer")
    if args.prepare:
        prepare(S2)
    if args.infer:
        return infer(S2, stub=args.stub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
