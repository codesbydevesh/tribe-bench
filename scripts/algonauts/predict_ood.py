"""Predict TRIBE v2 responses for the 12 Algonauts OOD segments. GPU required.

This is the run that produces both deliverables at once: the leaderboard submission and
the founding contents of the prediction bank. The stimuli are CC0, so the predictions are
freely redistributable.

BUDGET, from the measured cost model (378 s/clip cold at N=1, decomposed in
`.notes/plans/corticall/MASTER-PLAN.md` §3.6):

    ~11.5 s of GPU per second of video, plus ~65 s of ASR per segment
    12 segments x ~610 s of video = 7,328 s of stimulus  ->  ~23 GPU-hours

Kaggle sessions cap at 12 h, so this WILL take two or three sessions. Everything here is
built around that fact:

  * ONE `predict()` CALL PER SEGMENT, not one batched call over all twelve. Batching would
    amortise the ~175 s encoder load (saving ~35 min total, i.e. 2.5%), but `predict()`
    accumulates into a list and returns only after the final batch — so a session dying at
    hour 11 of a single 23 h call loses everything. Per-segment calls hand back a finished
    array every ~2 h.
  * PREDICTIONS ARE WRITTEN THE MOMENT A SEGMENT FINISHES, and a rerun skips whatever is
    already on disk. Resume is the default, not a flag.
  * THE FEATURE CACHE IS THE REAL CHECKPOINT. `exca` keys each feature on
    (absolute filepath, offset, duration) and flushes it to disk as it is computed, so even
    a segment killed midway keeps every feature it got through. Point `--cache-dir` at
    persisted storage or this guarantee is worthless.

INTEGRITY. The manifest URLs are git-annex keys of the form `MD5E-s<size>--<md5>.mkv`, so
every download is checked against the size AND md5 the dataset itself published. That is
M001 applied to the delivered artifact rather than to a proxy.

CHAPLIN. `chaplin` is silent Chaplin and the challenge ships no transcript for it — which is
why the challenge's own baseline trains a separate language-free model for that movie. We
cannot retrain, so the choice is between feeding TRIBE whatever Whisper hallucinates over a
music-only track, or giving it no text events at all. Default is the latter (`--mask-text-for
chaplin`), recorded as a deliberate decision rather than an accident. Word counts are logged
for every segment so the hallucination question stays visible.

Run `verify_parcel_mapping.py` and get a non-zero score from `--random` in
`prepare_submission.py` BEFORE spending GPU on this.
"""

import argparse
import hashlib
import json
import logging
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
MANIFEST = REPO / "data" / "algonauts" / "ood_video_manifest.json"
N_VERTICES = 20484
SILENT_MOVIES = ("chaplin",)

logger = logging.getLogger("predict_ood")


def load_manifest(path: Path = MANIFEST) -> dict:
    manifest = json.loads(path.read_text())
    if len(manifest) != 12:
        raise ValueError(f"expected 12 OOD segments in {path}, found {len(manifest)}")
    return manifest


def _expected_md5(url: str) -> tuple[int, str]:
    """Pull the published size and md5 out of a git-annex MD5E key in the URL."""
    m = re.search(r"MD5E-s(\d+)--([0-9a-f]{32})", url)
    if not m:
        raise ValueError(f"no MD5E key in url: {url}")
    return int(m.group(1)), m.group(2)


def _file_md5(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_segment(segment: str, entry: dict, dest_dir: Path) -> Path:
    """Fetch one OOD video over plain HTTPS and verify it against the dataset's own md5.

    No git-annex, no datalad, no sudo: the dataset registers a public RIA store in its
    `git-annex` branch, so annexed content is a normal HTTPS GET.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{segment}.mkv"
    size, md5 = _expected_md5(entry["url"])

    if dest.is_file():
        # Size alone is not enough. A file that is the right length but wrong content --
        # a clobbered partial write, an interrupted resume, bit rot -- would be accepted,
        # and because `exca` keys features on (filepath, offset, duration) rather than on
        # content, the resulting garbage features would be cached against this path
        # permanently. Re-hashing costs ~1 s per segment against ~2 h of GPU, so always.
        if dest.stat().st_size == size and _file_md5(dest) == md5:
            logger.info("%s already downloaded and md5-verified (%.1f MB)",
                        segment, size / 1e6)
            return dest
        logger.warning("%s on disk fails verification; deleting and refetching", segment)
        dest.unlink()

    logger.info("downloading %s (%.1f MB)", segment, size / 1e6)
    with urllib.request.urlopen(entry["url"], timeout=600) as response:
        digest = hashlib.md5()
        with dest.open("wb") as handle:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                handle.write(chunk)
                digest.update(chunk)

    actual_size = dest.stat().st_size
    if actual_size != size:
        dest.unlink()
        raise IOError(f"{segment}: got {actual_size} bytes, dataset says {size}")
    if digest.hexdigest() != md5:
        dest.unlink()
        raise IOError(f"{segment}: md5 {digest.hexdigest()} != published {md5}")
    logger.info("%s verified against the published md5", segment)
    return dest


def predict_segment(model, video_path: Path, mask_text: bool) -> tuple[np.ndarray, int]:
    """Run TRIBE on one segment. Returns (T, 20484) float32 and the ASR word count.

    The word count is returned because it is the only cheap signal about whether Whisper
    hallucinated dialogue over a silent movie.
    """
    from tribe_tools.model import _find_features_to_use

    events = model.get_events_dataframe(video_path=str(video_path))
    words = int((events.get("type") == "Word").sum()) if "type" in events else -1
    logger.info("%s: %d events, %d words", video_path.stem, len(events), words)

    location = _find_features_to_use(model) if mask_text else None
    original = None
    if location is not None:
        parent, attribute = location
        original = list(getattr(parent, attribute))
        keep = [f for f in original if f != "text"]
        if not keep:
            raise ValueError("masking text left no features to use")
        setattr(parent, attribute, keep)
        logger.info("%s: text masked, using %s", video_path.stem, keep)

    try:
        predictions, segments = model.predict(events=events)
    finally:
        if location is not None and original is not None:
            setattr(location[0], location[1], original)

    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim != 2 or predictions.shape[1] != N_VERTICES:
        raise ValueError(f"expected (T, {N_VERTICES}), got {predictions.shape}")
    logger.info("%s: %d timepoints from %d segments", video_path.stem,
                predictions.shape[0], len(segments))
    return predictions, words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working"),
                        help="videos, predictions and logs land here")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="exca feature cache. MUST be persisted storage or a killed "
                             "session throws away every feature it computed.")
    parser.add_argument("--segments", nargs="*", default=None,
                        help="subset to run, e.g. --segments chaplin1 wot2. Default: all 12")
    parser.add_argument("--mask-text-for", nargs="*", default=list(SILENT_MOVIES),
                        help="movies to predict without text features (default: chaplin)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="predict() materialises (batch_size, 20484, n_TRs) float32")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest = load_manifest()
    wanted = args.segments or sorted(manifest)
    unknown = set(wanted) - set(manifest)
    if unknown:
        raise SystemExit(f"unknown segments: {sorted(unknown)}")

    videos = args.work_dir / "ood_videos"
    out_dir = args.work_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [s for s in wanted if not (out_dir / f"{s}.npy").is_file()]
    done = sorted(set(wanted) - set(todo))
    if done:
        logger.info("already done, skipping: %s", done)
    if not todo:
        logger.info("nothing to do — all %d requested segments are on disk", len(wanted))
        return
    logger.info("to predict: %s", todo)

    from tribe_tools.model import load_model

    model = load_model(
        device="cuda",
        cache_folder=args.cache_dir,
        config_update={
            # keep_in_ram defaults True on all three extractors; without this, RSS grows
            # with every feature read and a long run dies on memory rather than time.
            "data.video_feature.infra.keep_in_ram": False,
            "data.audio_feature.infra.keep_in_ram": False,
            "data.text_feature.infra.keep_in_ram": False,
            "data.batch_size": args.batch_size,
        },
    )

    provenance_path = out_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}

    for segment in todo:
        movie = re.sub(r"\d+$", "", segment)
        mask_text = movie in args.mask_text_for
        started = time.time()

        video = download_segment(segment, manifest[segment], videos)
        predictions, words = predict_segment(model, video, mask_text=mask_text)

        # Write to a temp name first so a kill mid-write cannot leave a truncated .npy that
        # the resume logic would then treat as finished.
        target = out_dir / f"{segment}.npy"
        staging = target.with_suffix(".npy.partial")
        np.save(staging, predictions)
        staging.replace(target)

        elapsed = time.time() - started
        provenance[segment] = {
            "shape": list(predictions.shape),
            "asr_words": words,
            "text_masked": mask_text,
            "seconds": round(elapsed, 1),
            "video_md5": _expected_md5(manifest[segment]["url"])[1],
        }
        provenance_path.write_text(json.dumps(provenance, indent=1, sort_keys=True))
        logger.info("%s -> %s %s in %.1f min", segment, target.name,
                    predictions.shape, elapsed / 60)

    remaining = [s for s in sorted(manifest) if not (out_dir / f"{s}.npy").is_file()]
    logger.info("done. %d/12 segments predicted; remaining: %s",
                12 - len(remaining), remaining or "none")
    if remaining:
        logger.info("UPLOAD %s SOMEWHERE PERSISTED NOW, then rerun for the rest.", out_dir)
    else:
        logger.info("all 12 done — build the submission with prepare_submission.py")


if __name__ == "__main__":
    main()
