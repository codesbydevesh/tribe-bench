"""Validate the submission chain against REAL measured fMRI, locally, for free.

This is the rung below the leaderboard, and it is the one that actually protects us.

The Algonauts challenge withholds the OOD and Friends-season-7 responses, but the Friends
seasons 1-6 and Movie10 responses are PUBLIC, in the challenge's own .h5 files. So we can
correlate TRIBE's predictions against measured human fMRI without spending a submission,
without a server, and without a limit -- and if our surface-to-parcel mapping or our
temporal origin is wrong, this says so immediately instead of returning a mystery score.

WHY A SHORT SLICE IS ENOUGH. A full Friends episode split is ~12 minutes of video, which
at the measured cost (~11.5 s of GPU per second of video) is ~2.3 GPU-hours. We do not
need it. In synthetic testing the ordering check separated correct from LH/RH-swapped at
r = 1.000 vs -0.000, so ~80-120 TR samples -- two to three minutes of video, ~25-35 GPU
minutes -- is ample to catch a wiring error. Predict a slice from the START of a segment
and compare against the first N samples of the measured response.

WHAT A PASS AND A FAIL LOOK LIKE:
  mean_r clearly positive (say > 0.05, and the published in-distribution figure for a
    strong model is ~0.32)      -> parcel ordering and temporal origin are sane. Proceed.
  mean_r ~ 0                    -> STOP. Ordering or alignment is wrong (assumptions A1
    and A2 in prepare_submission.py). No GPU spend and no submission until this is fixed.
  mean_r strongly negative      -> sign or time inversion. Also stop.

Data needed (small; both are git-annex content in the competitors repo):
    datalad get fmri/sub-01/func/                        # ~500 MB, one subject
    datalad get stimuli/movies/friends/s1/friends_s01e01a.mkv

CPU only.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_submission import (  # noqa: E402
    N_PARCELS,
    TR,
    load_parcellation,
    parcellate,
    resample_to_tr,
)

# The model window is 100 s and TRIBE already applies a 5 s haemodynamic offset, so the
# opening samples of any segment are dominated by zero padding. The challenge's own scorer
# drops 5 samples at each end; we drop more at the start for a local sanity check because
# there is no cost to being conservative here.
DROP_HEAD = 10
DROP_TAIL = 5


def list_segments(h5_path: Path) -> list[str]:
    """Dataset keys inside a challenge fMRI .h5, e.g. ses-001_task-s01e01a."""
    with h5py.File(h5_path, "r") as f:
        keys: list[str] = []
        f.visit(lambda name: keys.append(name) if isinstance(f[name], h5py.Dataset) else None)
    return sorted(keys)


def load_segment(h5_path: Path, key: str) -> np.ndarray:
    """(N samples, 1000 parcels) measured response for one segment."""
    with h5py.File(h5_path, "r") as f:
        if key not in f:
            matches = [k for k in list_segments(h5_path) if key in k]
            raise KeyError(
                f"{key!r} not in {h5_path.name}. "
                + (f"Did you mean one of {matches[:5]}?" if matches else "See --list.")
            )
        arr = np.asarray(f[key][:], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != N_PARCELS:
        raise ValueError(f"expected (N, {N_PARCELS}), got {arr.shape}")
    return arr


def validate(
    preds_vertices: np.ndarray,
    measured: np.ndarray,
    parcels: np.ndarray,
) -> dict[str, float | str | int]:
    """Correlate predicted against measured, per parcel, over the overlapping span.

    `preds_vertices` is (T, 20484) at 1 Hz for a slice starting at the segment's start;
    `measured` is the full (N, 1000) measured response. Only the TR samples covered by
    the prediction slice are compared, so a short slice is legitimate rather than being
    silently padded out.
    """
    parcelled = parcellate(preds_vertices, parcels)
    covered_seconds = (parcelled.shape[0] - 1) / 1.0
    n_usable = int(covered_seconds / TR) + 1
    n = min(n_usable, measured.shape[0])
    if n <= DROP_HEAD + DROP_TAIL + 20:
        raise ValueError(
            f"only {n} TR samples are covered by a {parcelled.shape[0]}s prediction; "
            "predict a longer slice (aim for >= 120 s of video)"
        )

    pred = resample_to_tr(parcelled, n)
    pred = pred[DROP_HEAD : n - DROP_TAIL]
    meas = measured[DROP_HEAD : n - DROP_TAIL]

    r = np.zeros(N_PARCELS, dtype=np.float64)
    for p in range(N_PARCELS):
        a, b = pred[:, p], meas[:, p]
        if a.std() > 1e-12 and b.std() > 1e-12:
            r[p] = np.corrcoef(a, b)[0, 1]

    mean_r = float(r.mean())
    if mean_r > 0.05:
        verdict = "PASS - ordering and alignment sane, proceed to the noise submission"
    elif mean_r < -0.05:
        verdict = "FAIL - sign or time inversion. Stop."
    else:
        verdict = "FAIL - r ~ 0. Ordering or alignment wrong (A1/A2). Stop. No GPU spend."

    return {
        "n_tr_compared": int(pred.shape[0]),
        "seconds_of_video": round(covered_seconds, 1),
        "mean_r": round(mean_r, 4),
        "median_r": round(float(np.median(r)), 4),
        "p95_r": round(float(np.percentile(r, 95)), 4),
        "max_r": round(float(r.max()), 4),
        "frac_positive": round(float((r > 0).mean()), 3),
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h5", type=Path, required=True, help="challenge fMRI .h5 for one subject")
    ap.add_argument("--key", help="dataset key, e.g. ses-001_task-s01e01a")
    ap.add_argument("--preds", type=Path, help=".npy of (T, 20484) TRIBE predictions at 1 Hz")
    ap.add_argument("--list", action="store_true", help="list dataset keys and exit")
    args = ap.parse_args()

    if args.h5.stat().st_size < 1000:
        raise SystemExit(
            f"{args.h5} is {args.h5.stat().st_size} bytes -- a git-annex pointer, not the "
            "data. Run: datalad get fmri/sub-01/func/"
        )

    if args.list:
        keys = list_segments(args.h5)
        print(f"{len(keys)} segments in {args.h5.name}:")
        for k in keys[:40]:
            print("  ", k)
        if len(keys) > 40:
            print(f"   ... and {len(keys) - 40} more")
        return

    if not args.key or not args.preds:
        raise SystemExit("--key and --preds are both required (or use --list)")

    parcels = load_parcellation()
    measured = load_segment(args.h5, args.key)
    preds = np.load(args.preds)
    print(f"measured {measured.shape} | predictions {preds.shape}")

    for k, v in validate(preds, measured, parcels).items():
        print(f"  {k:<18} {v}")


if __name__ == "__main__":
    main()
