"""Thin wrapper around TRIBE v2's TribeModel.

TRIBE v2's pipeline already handles sequential encoder loading internally.
Each extractor loads, caches features to disk, and frees GPU before the
next one loads via _free_extractor_model(). No custom loading needed.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_GPU_MSG = "GPU not available. To run inference, use Kaggle/Colab with a T4 GPU."


def _check_tribev2():
    try:
        from tribev2.demo_utils import TribeModel  # noqa: F401
    except ImportError:
        raise ImportError(
            "tribev2 is not installed. Install it with: pip install tribev2\n"
            "Also requires: neuralset, neuraltrain, exca"
        )


# TRIBE v2 checkpoint, PINNED. Resolved from the HuggingFace API 2026-08-24;
# repo lastModified 2026-03-27T09:07:48Z.
TRIBEV2_REPO = "facebook/tribev2"
TRIBEV2_REVISION = "f894e783020944dcd96e5568550afe2aa9743f9f"
TRIBEV2_CKPT_SHA256 = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"
TRIBEV2_CKPT_BYTES = 708_856_138


def fetch_pinned_checkpoint(revision: str = TRIBEV2_REVISION,
                            expected_sha256: Optional[str] = TRIBEV2_CKPT_SHA256,
                            dest: Optional[Path] = None) -> Path:
    """Resolve TRIBE v2 at an EXACT revision and verify the checkpoint hash.

    ``TribeModel.from_pretrained`` has **no** ``revision`` parameter: it calls
    ``hf_hub_download(repo_id, filename)`` against the floating branch, so a re-run
    six months from now silently gets whatever Meta last pushed, with no signal
    anywhere. It DOES accept a local directory, so the pin is done by resolving the
    revision here and handing it a path.

    Returns:
        Local directory containing config.yaml and best.ckpt.

    Raises:
        ValueError: if the downloaded checkpoint does not match ``expected_sha256``.
            A silently different checkpoint is exactly the failure this exists to
            prevent, so a mismatch is fatal rather than a warning.
    """
    import hashlib
    from huggingface_hub import hf_hub_download

    kw = {"repo_id": TRIBEV2_REPO}
    if revision:
        kw["revision"] = revision
    if dest is not None:
        Path(dest).mkdir(parents=True, exist_ok=True)
        kw["local_dir"] = str(dest)

    cfg = Path(hf_hub_download(filename="config.yaml", **kw))
    ckpt = Path(hf_hub_download(filename="best.ckpt", **kw))

    if expected_sha256:
        h = hashlib.sha256()
        with open(ckpt, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != expected_sha256:
            raise ValueError(
                f"checkpoint hash mismatch at revision {revision}: expected "
                f"{expected_sha256}, got {got}. Refusing to run -- the weights are not "
                "the ones this experiment was designed against."
            )
    logger.info("TRIBE v2 pinned at revision %s (checkpoint verified)", revision)
    return ckpt.parent if ckpt.parent == cfg.parent else cfg.parent


def load_model(
    device: str = "cuda",
    cache_folder: Optional[Path] = None,
    config_update: Optional[dict] = None,
    revision: Optional[str] = TRIBEV2_REVISION,
    checkpoint_dir: Optional[Path] = None,
):
    """Load TRIBE v2 via TribeModel.from_pretrained, PINNED to a known revision.

    Args:
        device: "cuda" or "cpu". CPU works but is very slow.
        cache_folder: Where to cache downloaded model weights.
        config_update: Dotted-path config overrides forwarded to from_pretrained.
            Required for anything that must be set at CONSTRUCTION time, because
            each extractor freezes once its cache uid is first computed. The two
            that matter for long runs:
              - `data.<modality>_feature.infra.keep_in_ram: False` — defaults to
                True on all three extractors, so every feature read during
                dataloading is kept forever and RSS grows linearly with the number
                of stimuli. This is the hard ceiling on corpus size.
              - `data.batch_size: <int>` — predict() materialises
                (batch_size, n_vertices, n_TRs) float32, ~524 MB at 64.
            None of these keys participates in an extractor cache uid, so passing
            them does not invalidate already-cached features.

        revision: HuggingFace commit SHA to pin. Defaults to TRIBEV2_REVISION.
            Pass None to accept the floating branch -- never for a recorded run.
        checkpoint_dir: a pre-fetched local directory; skips resolution entirely.

    Returns:
        TribeModel instance ready for prediction.

    Raises:
        RuntimeError: If device="cuda" but no GPU is available.
        ImportError: If tribev2 is not installed.
    """
    _check_tribev2()

    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(_GPU_MSG)
        except ImportError:
            raise RuntimeError(_GPU_MSG)

    from tribev2.demo_utils import TribeModel

    kwargs = {"device": device}
    if cache_folder is not None:
        kwargs["cache_folder"] = str(cache_folder)
    if config_update:
        kwargs["config_update"] = dict(config_update)

    # Resolve the pin OURSELVES and pass a local directory: from_pretrained takes
    # no revision, so handing it the bare repo id would use the floating branch.
    if checkpoint_dir is None and revision:
        checkpoint_dir = fetch_pinned_checkpoint(revision=revision)
    source = str(checkpoint_dir) if checkpoint_dir is not None else TRIBEV2_REPO
    if checkpoint_dir is None:
        logger.warning("Loading TRIBE v2 from the FLOATING branch (revision=None). "
                       "Results from this load are not reproducible.")

    logger.info("Loading TRIBE v2 model from %s ...", source)
    model = TribeModel.from_pretrained(source, **kwargs)
    logger.info("TRIBE v2 model loaded.")
    return model


def _find_features_to_use(model):
    """Discover the features_to_use attribute on the TribeModel.

    TRIBE v2 stores the list of active feature names (e.g., ["video", "audio", "text"])
    somewhere in its config hierarchy. We probe known paths to find it.
    Returns (parent_obj, attr_name) or None if not found.
    """
    paths = [
        ["data", "features_to_use"],
        ["xp", "data", "features_to_use"],
        ["xp", "cfg", "data", "features_to_use"],
        ["cfg", "data", "features_to_use"],
    ]
    for path_parts in paths:
        obj = model
        for part in path_parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, path_parts[-1]):
            val = getattr(obj, path_parts[-1])
            if isinstance(val, (list, tuple)):
                return obj, path_parts[-1]
    return None


def predict_single(
    model,
    video_path: Path,
    features_to_mask: Optional[list[str]] = None,
) -> tuple[np.ndarray, list]:
    """Run TRIBE v2 prediction on a single video.

    Args:
        model: Loaded TribeModel from load_model().
        video_path: Path to video file.
        features_to_mask: Feature names to zero out for modality ablation.
            None or [] = full prediction (all modalities active).
            e.g., ["audio", "text"] = video-only.

    Returns:
        (preds, segments) tuple:
        - preds: np.ndarray shape (n_kept_segments, n_vertices)
        - segments: list of Segment objects with timing info
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    events = model.get_events_dataframe(video_path=str(video_path))

    if features_to_mask:
        # Modality ablation: temporarily restrict which extractors run.
        # The brain model substitutes zeros for missing modalities (model.py:189-192).
        #
        # Strategy: mutate features_to_use on the model's data config so only
        # the desired modalities are extracted. Restore after prediction.
        loc = _find_features_to_use(model)
        if loc is None:
            raise RuntimeError(
                "Cannot find features_to_use on TribeModel. "
                "Modality ablation needs GPU testing to identify the correct API path. "
                "See ops/knowledge-gaps.md G018."
            )
        parent, attr_name = loc
        original = list(getattr(parent, attr_name))
        keep = [f for f in original if f not in features_to_mask]
        if not keep:
            raise ValueError(
                f"Masking {features_to_mask} leaves no features. "
                f"Available features: {original}"
            )
        setattr(parent, attr_name, keep)
        try:
            preds, segments = model.predict(events=events)
        finally:
            setattr(parent, attr_name, original)
    else:
        preds, segments = model.predict(events=events)

    return preds, segments


# Mapping from our modality names to features_to_mask configs
MODALITY_MASKS = {
    "full": [],
    "video_only": ["audio", "text"],
    "audio_only": ["video", "text"],
    "text_only": ["video", "audio"],
}
