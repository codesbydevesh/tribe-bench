"""Batch inference with progress tracking and HDF5 checkpointing."""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from tribe_tools.cache import PredictionCache, get_cache
from tribe_tools.model import predict_single

logger = logging.getLogger(__name__)


def batch_predict(
    model,
    video_paths: list[Path],
    features_to_mask: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
    checkpoint_every: int = 10,
) -> dict[Path, tuple[np.ndarray, list]]:
    """Run TRIBE v2 on multiple videos with progress tracking and checkpointing.

    Args:
        model: Loaded TribeModel.
        video_paths: List of video file paths.
        features_to_mask: Features to mask for ALL videos in this batch.
        cache_dir: If provided, save/load results from HDF5 cache.
            Already-cached videos are skipped (resume support).
        checkpoint_every: Save checkpoint after every N videos.

    Returns:
        Dict mapping each video path to (preds, segments) tuple.
    """
    cache = get_cache(cache_dir) if cache_dir else None
    mask_key = _mask_key(features_to_mask)
    results = {}

    skipped = 0
    for i, video_path in enumerate(tqdm(video_paths, desc="Batch inference")):
        video_path = Path(video_path)
        cache_k = _cache_key(video_path, mask_key)

        # Check cache
        if cache and cache.has(cache_k):
            preds = cache.load(cache_k)
            results[video_path] = (preds, [])  # segments not cached
            skipped += 1
            continue

        # Run prediction
        try:
            preds, segments = predict_single(
                model, video_path, features_to_mask=features_to_mask
            )
            results[video_path] = (preds, segments)

            # Save to cache
            if cache:
                cache.save(cache_k, preds)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA OOM on %s. Try shorter clips or smaller batch.", video_path)
                raise  # OOM is unrecoverable — don't silently skip
            logger.error("Failed on %s: %s", video_path, e)
            continue
        except Exception as e:
            logger.error("Failed on %s: %s", video_path, e)
            continue

        # Checkpoint
        if cache and (i + 1) % checkpoint_every == 0:
            cache.flush()
            logger.info("Checkpoint at %d/%d videos", i + 1, len(video_paths))

    if cache:
        cache.flush()

    if skipped:
        logger.info("Skipped %d cached videos", skipped)

    return results


def _mask_key(features_to_mask: Optional[list[str]]) -> str:
    if not features_to_mask:
        return "full"
    return "_".join(sorted(features_to_mask))


def _cache_key(video_path: Path, mask_key: str) -> str:
    """Generate an HDF5-safe cache key from video path + mask config.

    Uses SHA256 of the absolute path to avoid HDF5 treating '/' as groups.
    """
    path_hash = hashlib.sha256(str(video_path.resolve()).encode()).hexdigest()[:16]
    name = video_path.stem
    return f"{name}_{path_hash}_{mask_key}"
