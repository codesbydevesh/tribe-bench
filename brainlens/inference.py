"""4-pass modality ablation for BrainLens.

Runs TRIBE v2 with different features_to_mask configurations:
- full:       no mask (all modalities active)
- video_only: mask audio + text
- audio_only: mask video + text
- text_only:  mask video + audio
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from tribe_tools.cache import get_cache
from tribe_tools.model import MODALITY_MASKS, predict_single

logger = logging.getLogger(__name__)


def run_ablation(
    model,
    video_path: Path,
    cache_dir: Optional[Path] = None,
) -> dict[str, tuple[np.ndarray, list]]:
    """Run 4-pass modality ablation on a single video.

    Args:
        model: Loaded TribeModel.
        video_path: Path to video file.
        cache_dir: If provided, cache results for resume support.

    Returns:
        Dict mapping modality name to (preds, segments) tuple.
        Keys: "full", "video_only", "audio_only", "text_only".
    """
    video_path = Path(video_path)
    cache = get_cache(cache_dir)
    results = {}

    for modality, mask in MODALITY_MASKS.items():
        cache_key = f"brainlens_{video_path.name}_{modality}"

        # Check cache
        if cache and cache.has(cache_key):
            logger.info("Loading cached %s pass", modality)
            preds = cache.load(cache_key)
            results[modality] = (preds, [])
            continue

        logger.info("Running %s pass (masking: %s)", modality, mask or "none")
        preds, segments = predict_single(
            model, video_path, features_to_mask=mask or None
        )
        results[modality] = (preds, segments)

        # Cache (save() persists immediately — no separate flush needed)
        if cache:
            cache.save(cache_key, preds)

    return results
