"""Turn TRIBE v2 fsaverage5 predictions into an Algonauts 2025 Codabench submission.

The post-challenge OOD leaderboard (codabench.org/competitions/9483) scores predicted
fMRI against withheld responses from four CNeuroMod subjects, server-side, with Pearson
r over 1,000 Schaefer parcels. It is the only number in this project that we do not
compute ourselves, which is the whole reason for building this.

Submission format, read off the challenge's own baseline model
(`code/challenge_baseline_model/03_encoding_model_testing/01_test_encoding_ood.py`):

    fmri_test_pred['sub-01']['chaplin1'] = np.ndarray (N_samples, 1000) float32
    np.save(path, fmri_test_pred)          # one .npy holding a nested dict

Subjects are sub-01, sub-02, sub-03, sub-05 (no sub-04). The scorer discards the first
and last 5 samples of every segment before correlating, but each array must still be
exactly N_samples long, where N comes from the challenge's own
`sub-0X_ood_fmri_samples.npy`.

THREE ASSUMPTIONS THAT ARE NOT VERIFIED AND MUST BE, in this order (cheapest first):

  A1. PARCEL ORDERING. The challenge derived its 1,000 parcels from subject-specific
      MNI *volumetric* atlases (97x115x97, voxel values 1..1000). We map TRIBE's
      fsaverage5 *surface* output through the Schaefer fsaverage5 annotation instead.
      Surface-averaged is not volume-averaged, and if the LH/RH ordering convention
      differs the score collapses to ~0. Test it for free and locally with
      `validate_against_friends()` -- the Friends s1-6 fMRI is public.

  A2. TEMPORAL ORIGIN. TRIBE emits one prediction per second; fMRI samples arrive every
      1.49 s. We map TRIBE sample i to t = i seconds and fMRI sample j to t = j * 1.49 s
      and interpolate. Any constant offset error costs accuracy silently.

  A3. HRF. TRIBE's own README states its predictions are already offset by 5 s to
      compensate for haemodynamic lag. We therefore apply NO further shift. Applying the
      challenge tutorial's `hrf_delay=3` on top would double-count it.

  A4. NUMPY PICKLE VERSION -- CONFIRMED AND HANDLED, not an open risk. The scoring image
      `dommybe/codabench_algonauts25:latest` (built 2025-06-02) pins numpy==1.22.4 on
      python3.9, read from the image's own build history. numpy >= 2.0 pickles arrays
      referencing `numpy._core`, absent before 2.0, so a plain `np.save` from this box
      (numpy 2.5.1) raises ModuleNotFoundError server-side -- verified by emulation.
      `save_submission(numpy1_compat=True)` is the default and writes a file that loads
      under both. Leave it on.

CPU only. No model, no GPU.
"""

import argparse
import io
import pickle
import json
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
ATLAS_DIR = REPO / "data" / "atlas"
ANNOT = "{hemi}.Schaefer2018_1000Parcels_7Networks_order.annot"

SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05")
N_PARCELS = 1000
N_VERTICES = 20484  # fsaverage5, both hemispheres, left first
TR = 1.49  # seconds, CNeuroMod
TRIBE_HZ = 1.0  # TRIBE emits one prediction per second

# Six OOD movies, two segments each. `chaplin` is silent Chaplin and ships with NO
# transcript, which is why the challenge baseline trains a separate language-free model
# for it -- predict it without text features.
OOD_MOVIES = ("chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot")
OOD_SEGMENTS = tuple(f"{m}{i}" for m in OOD_MOVIES for i in (1, 2))
SILENT_MOVIES = ("chaplin",)


def load_parcellation(atlas_dir: Path = ATLAS_DIR) -> np.ndarray:
    """Per-vertex Schaefer parcel id for fsaverage5, left hemisphere first.

    Returns (20484,) int32 with values 1..1000; 0 marks unlabelled vertices (medial
    wall), which are excluded from every parcel mean.

    Each hemisphere's annotation numbers its own parcels 1..500, so the right hemisphere
    is offset by 500 to produce the standard combined 1..1000 ordering. This offset is
    assumption A1 -- if the challenge ordered parcels differently, every score is noise.
    """
    labels = []
    for offset, hemi in ((0, "lh"), (500, "rh")):
        path = atlas_dir / ANNOT.format(hemi=hemi)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}. Fetch from ThomasYeoLab/CBIG: stable_projects/"
                "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/"
                "FreeSurfer5.3/fsaverage5/label/"
            )
        lab, _, _ = nib.freesurfer.read_annot(str(path))
        lab = np.asarray(lab, dtype=np.int32)
        lab[lab > 0] += offset  # 0 stays 0: unlabelled
        labels.append(lab)

    out = np.concatenate(labels)
    if out.shape[0] != N_VERTICES:
        raise ValueError(f"expected {N_VERTICES} vertices, got {out.shape[0]}")
    present = np.unique(out[out > 0])
    if present.shape[0] != N_PARCELS:
        raise ValueError(f"expected {N_PARCELS} parcels, found {present.shape[0]}")
    return out


def parcellate(preds: np.ndarray, parcels: np.ndarray) -> np.ndarray:
    """(T, 20484) vertex predictions -> (T, 1000) parcel means.

    Unlabelled vertices are dropped rather than assigned to a neighbour. Vectorised with
    bincount because a Python loop over 1,000 parcels x thousands of timepoints is the
    kind of thing that quietly turns a CPU step into a coffee break.
    """
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim != 2 or preds.shape[1] != N_VERTICES:
        raise ValueError(f"expected (T, {N_VERTICES}), got {preds.shape}")

    keep = parcels > 0
    idx = parcels[keep] - 1  # 0-based parcel index
    counts = np.bincount(idx, minlength=N_PARCELS).astype(np.float64)
    if (counts == 0).any():
        raise ValueError("a parcel has no vertices; parcellation is broken")

    sums = np.stack(
        [np.bincount(idx, weights=row[keep], minlength=N_PARCELS) for row in preds]
    )
    return (sums / counts).astype(np.float32)


def resample_to_tr(
    series: np.ndarray,
    n_target: int,
    tr: float = TR,
    strict: bool = True,
    mode: str = "area",
) -> np.ndarray:
    """(T, P) predictions at TRIBE_HZ -> (n_target, P) on the fMRI TR grid.

    THE CONVENTION, read off the challenge's own feature extractor
    (`01_stimulus_feature_extraction/feature_extraction_ood_utils.py`):

        start_times = [x for x in np.arange(0, clip.duration, args.tr)][:-1]
        clip_chunk = clip.subclip(start, start + args.tr)

    So fMRI sample j corresponds to the movie INTERVAL [j*tr, (j+1)*tr) -- a window
    average, not a point reading at its leading edge. Sampling at t = j*tr instead (which
    an earlier version of this function did) puts every value half a TR -- 0.745 s -- early,
    a systematic misalignment that costs accuracy silently and looks like model error.

    mode="area" (default) integrates the prediction over each TR window, treating
    prediction i as covering [i, i+1) / TRIBE_HZ. This matches the convention above and
    incidentally anti-aliases: going from 1 Hz to 1/1.49 Hz is a DOWNSAMPLE, and point
    sampling would alias anything above the new 0.336 Hz Nyquist. Post-HRF BOLD has little
    power up there, so the effect is small, but averaging costs nothing.

    mode="linear" keeps the old point interpolation, at the window CENTRE (j + 0.5) * tr,
    for comparison. No HRF shift is applied in either mode -- see A3.

    If the predictions run out before the target grid ends, `strict` refuses rather than
    silently emitting a flat tail that correlates with nothing.
    """
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 2:
        raise ValueError(f"expected 2D (T, P), got {series.shape}")
    if n_target < 1:
        raise ValueError(f"n_target must be >= 1, got {n_target}")
    if mode not in ("area", "linear"):
        raise ValueError(f"mode must be 'area' or 'linear', got {mode!r}")

    n_src = series.shape[0]
    src_end = n_src / TRIBE_HZ  # predictions cover [0, n_src/TRIBE_HZ)
    needed = n_target * tr
    shortfall = needed - src_end
    if strict and shortfall > tr:
        raise ValueError(
            f"predictions cover {src_end:.1f}s but the target grid needs {needed:.1f}s "
            f"({n_target} samples x {tr}s) -- short by {shortfall:.1f}s "
            f"({shortfall / tr:.1f} TRs). Predict the full segment, or pass strict=False."
        )

    if mode == "linear":
        src_t = np.arange(n_src, dtype=np.float64) / TRIBE_HZ
        dst_t = (np.arange(n_target, dtype=np.float64) + 0.5) * tr
        out = np.empty((n_target, series.shape[1]), dtype=np.float32)
        for p in range(series.shape[1]):
            out[:, p] = np.interp(dst_t, src_t, series[:, p])
        return out

    # Area average: overlap of each TR window with each 1-second prediction bin.
    src_edges = np.arange(n_src + 1, dtype=np.float64) / TRIBE_HZ
    dst_edges = np.arange(n_target + 1, dtype=np.float64) * tr
    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None], src_edges[None, 1:])
    w = np.clip(hi - lo, 0.0, None)  # (n_target, n_src) overlap in seconds
    total = w.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("a TR window has no overlap with the predictions")
    return ((w @ series) / total).astype(np.float32)


def load_sample_counts(fmri_dir: Path, subject: str) -> dict[str, int]:
    """Read the challenge's withheld-sample counts for one subject.

    File lives at fmri/<subject>/target_sample_number/<subject>_ood_fmri_samples.npy and
    is git-annex content, so a bare `git clone` leaves a pointer file behind. Fetch with:

        datalad install -r https://github.com/courtois-neuromod/algonauts_2025.competitors.git
        cd algonauts_2025.competitors
        datalad get fmri/*/target_sample_number/
    """
    path = fmri_dir / subject / "target_sample_number" / f"{subject}_ood_fmri_samples.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    if path.stat().st_size < 200:
        raise ValueError(
            f"{path} is {path.stat().st_size} bytes -- that is a git-annex pointer, not "
            "the array. Run `datalad get` on it (see this function's docstring)."
        )
    counts = np.load(path, allow_pickle=True)
    counts = counts.item() if isinstance(counts, np.ndarray) else counts
    return {str(k): int(v) for k, v in dict(counts).items()}


def assemble(
    predictions: dict[str, np.ndarray],
    sample_counts: dict[str, dict[str, int]],
    parcels: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Build the nested submission dict.

    `predictions` maps segment name -> (T, 20484) vertex predictions. The released TRIBE
    checkpoint predicts an average subject, so the same prediction goes to all four
    subjects, resampled to each one's own sample count. That structurally caps the score
    below the subject-specific challenge winners and must be stated wherever the number
    is reported.
    """
    missing = set(OOD_SEGMENTS) - set(predictions)
    if missing:
        raise ValueError(f"missing predictions for: {sorted(missing)}")

    parcelled = {seg: parcellate(p, parcels) for seg, p in predictions.items()}
    out: dict[str, dict[str, np.ndarray]] = {}
    for subject in SUBJECTS:
        counts = sample_counts[subject]
        out[subject] = {}
        for seg in OOD_SEGMENTS:
            if seg not in counts:
                raise ValueError(f"{subject} has no sample count for {seg}")
            out[subject][seg] = resample_to_tr(parcelled[seg], counts[seg])
    return out


class _Numpy122Unpickler(pickle.Unpickler):
    """Emulates numpy 1.22.4, in which `numpy._core` does not exist."""

    def find_class(self, module: str, name: str):  # noqa: ANN001, ANN201
        if module.startswith("numpy._core"):
            raise ModuleNotFoundError(
                f"No module named {module!r} -- numpy 1.22 cannot load this pickle"
            )
        return super().find_class(module, name)


def _load_as_numpy122(path: Path):  # noqa: ANN202
    """Load a .npy the way the scoring server's numpy 1.22.4 would."""
    raw = path.read_bytes()
    if raw[:6] != np.lib.format.MAGIC_PREFIX:
        return _Numpy122Unpickler(io.BytesIO(raw)).load()
    fp = io.BytesIO(raw)
    fp.read(6)
    major = fp.read(2)[0]
    header_len = int.from_bytes(fp.read(2 if major == 1 else 4), "little")
    fp.read(header_len)
    return _Numpy122Unpickler(fp).load()


def _write_numpy1_compatible(path: Path, payload: dict) -> None:
    """Write a genuine .npy that the scorer's numpy 1.22.4 can actually unpickle.

    The Codabench scoring image (`dommybe/codabench_algonauts25:latest`, built 2025-06-02)
    pins **numpy==1.22.4** on python3.9 -- read out of the image's own build history, not
    guessed. numpy >= 2.0 pickles ndarrays referencing `numpy._core.multiarray`, which does
    not exist before 2.0, so a plain `np.save` from a numpy 2.x box raises
    ModuleNotFoundError server-side. Verified by emulation: a plain save fails, this
    function's output loads.

    Two details, both load-bearing:

    * `protocol=3` is pinned. At protocol 3 globals are emitted with the newline-terminated
      GLOBAL opcode (`c<module>\\n<name>\\n`), so rewriting `numpy._core` -> `numpy.core`
      is safe even though it shortens the name by a byte. Protocol 4+ uses length-prefixed
      STACK_GLOBAL, where the same edit would corrupt the stream.
    * `np.lib.format.write_array` is used rather than raw `pickle.dumps` so the file keeps
      a real `\\x93NUMPY` header. A bare pickle happens to load via `np.load`'s fallback
      path, but it is not a .npy and we should not ship one.

    Do NOT test this against numpy 1.26 and conclude it is unnecessary: 1.26 ships a
    `numpy._core` shim added for the 2.0 transition, so it reads numpy-2.x pickles happily
    and hides the bug. 1.22 has no such shim.
    """
    boxed = np.empty((), dtype=object)
    boxed[()] = payload

    # numpy 2.x's write_array hardcodes protocol=4, and protocol 4 emits length-prefixed
    # STACK_GLOBAL, where the module rename below would corrupt the stream. So write the
    # header and a protocol-3 pickle by hand -- which is byte-for-byte what numpy 1.22's
    # own write_array produced.
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(buf, np.lib.format.header_data_from_array_1_0(boxed))
    pickle.dump(boxed, buf, protocol=3)
    raw = buf.getvalue()

    if b"\x80\x03" not in raw:
        raise RuntimeError("expected a protocol-3 pickle; refusing to patch blindly")

    path.write_bytes(raw.replace(b"numpy._core", b"numpy.core"))

    # Self-verify, so this can never silently regress into an unreadable submission.
    if path.read_bytes()[:6] != np.lib.format.MAGIC_PREFIX:
        raise RuntimeError(f"{path} lost its .npy header")
    restored = _load_as_numpy122(path)
    restored = restored.item() if isinstance(restored, np.ndarray) else restored
    if set(restored) != set(payload):
        raise RuntimeError("numpy-1.22 round-trip changed the payload keys")
    if not isinstance(np.load(path, allow_pickle=True).item(), dict):
        raise RuntimeError("file is unreadable by the current numpy")


def save_submission(
    payload: dict, out_dir: Path, stem: str = "submission", numpy1_compat: bool = True
) -> Path:
    """Write the .npy and zip it, which is what Codabench ingests.

    `numpy1_compat` writes a pickle the scorer's numpy 1.22.4 can actually read. Leave it
    on unless you have verified the server has been upgraded.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / f"{stem}.npy"
    if numpy1_compat:
        _write_numpy1_compatible(npy, payload)
    else:
        np.save(npy, payload, allow_pickle=True)

    zip_path = out_dir / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(npy, arcname=npy.name)

    shapes = {
        s: {seg: list(a.shape) for seg, a in segs.items()} for s, segs in payload.items()
    }
    (out_dir / f"{stem}_shapes.json").write_text(json.dumps(shapes, indent=1))
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB), numpy {np.__version__}")
    return zip_path


def random_submission(sample_counts: dict[str, dict[str, int]], out_dir: Path) -> Path:
    """A submission of pure noise, to test the pipe before spending a GPU-hour on it.

    ACCEPTANCE TEST: the server must accept this file and return a score near 0.
      - accepted, r ~ 0        -> format and plumbing are correct. Proceed.
      - rejected / error       -> format is wrong (see A4 first). Fix before any GPU.
      - accepted, r far from 0 -> something is very wrong; do not proceed.

    This is M001 applied to the artifact that actually matters: a server score.
    """
    rng = np.random.default_rng(0)
    payload = {
        s: {
            seg: rng.standard_normal((n, N_PARCELS)).astype(np.float32)
            for seg, n in sample_counts[s].items()
        }
        for s in SUBJECTS
    }
    return save_submission(payload, out_dir, stem="submission_random")


def validate_against_friends(
    predictions: np.ndarray,
    measured: np.ndarray,
    parcels: np.ndarray,
) -> dict[str, float]:
    """Check assumption A1 locally, for free, before ever touching the leaderboard.

    Feed TRIBE predictions for any Friends s1-6 or Movie10 segment (the fMRI for those
    IS public, in the challenge's own .h5 files) alongside the measured (N, 1000) array.
    Correct parcel ordering gives a clearly positive mean correlation. A wrong LH/RH
    convention or a bad temporal origin gives ~0, and that is the cheapest possible way
    to catch it.
    """
    pred = resample_to_tr(parcellate(predictions, parcels), measured.shape[0])
    pred, meas = pred[5:-5], np.asarray(measured, dtype=np.float64)[5:-5]
    if pred.shape != meas.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {meas.shape}")

    r = np.zeros(N_PARCELS, dtype=np.float64)
    for p in range(N_PARCELS):
        a, b = pred[:, p], meas[:, p]
        if a.std() > 0 and b.std() > 0:
            r[p] = np.corrcoef(a, b)[0, 1]
    return {
        "mean_r": float(r.mean()),
        "median_r": float(np.median(r)),
        "max_r": float(r.max()),
        "frac_positive": float((r > 0).mean()),
        "verdict": "ORDERING PLAUSIBLE" if r.mean() > 0.05 else "SUSPECT -- check A1/A2",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--random", action="store_true", help="write a noise submission (the pipe test)"
    )
    ap.add_argument(
        "--fmri-dir",
        type=Path,
        required=True,
        help="path to the competitors repo's fmri/ directory",
    )
    ap.add_argument("--out-dir", type=Path, default=REPO / "results" / "algonauts")
    args = ap.parse_args()

    parcels = load_parcellation()
    print(f"parcellation OK: {N_VERTICES} vertices -> {N_PARCELS} parcels, "
          f"{int((parcels == 0).sum())} unlabelled")

    counts = {s: load_sample_counts(args.fmri_dir, s) for s in SUBJECTS}
    for s, c in counts.items():
        print(f"  {s}: {len(c)} segments, {sum(c.values())} samples total")

    if args.random:
        random_submission(counts, args.out_dir)
        print("\nSubmit this to codabench.org/competitions/9483 and report the score.")
        print("Expect ~0. Anything else means stop and read A1-A4 in this file.")
        return

    raise NotImplementedError(
        "Real predictions require a GPU run. Build the events frame with "
        "tribe_tools.inference, keep one timeline per clip, then call assemble(). "
        "Run --random first and get a server score back before spending that GPU time."
    )


if __name__ == "__main__":
    main()
