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


def load_model(
    device: str = "cuda",
    cache_folder: Optional[Path] = None,
):
    """Load TRIBE v2 via TribeModel.from_pretrained("facebook/tribev2").

    Args:
        device: "cuda" or "cpu". CPU works but is very slow.
        cache_folder: Where to cache downloaded model weights.

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

    logger.info("Loading TRIBE v2 model...")
    model = TribeModel.from_pretrained("facebook/tribev2", **kwargs)
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
