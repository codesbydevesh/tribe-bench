"""HDF5-based prediction cache for persisting results across sessions."""

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class PredictionCache:
    """HDF5 cache for brain predictions.

    Note: Only prediction arrays are cached, not segment timing info.
    When loading from cache, segments will be empty. For workflows that
    need segment data, re-run inference without cache.
    """

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def has(self, key: str) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            with h5py.File(self.cache_path, "r") as f:
                return key in f
        except Exception:
            return False

    def save(self, key: str, data: np.ndarray, metadata: Optional[dict] = None):
        """Save a prediction array to cache.

        Opens the HDF5 file, writes (or overwrites) the dataset, and closes
        it in a single context, so the entry is on disk by the time this
        returns — there is no separate flush step. Not concurrency-safe:
        assumes a single writer.
        """
        mode = "a" if self.cache_path.exists() else "w"
        with h5py.File(self.cache_path, mode) as f:
            if key in f:
                del f[key]
            ds = f.create_dataset(key, data=data, compression="gzip")
            if metadata:
                for k, v in metadata.items():
                    ds.attrs[k] = v

    def load(self, key: str) -> Optional[np.ndarray]:
        """Load a prediction from cache. Returns None if not found."""
        if not self.has(key):
            return None
        with h5py.File(self.cache_path, "r") as f:
            return np.array(f[key])

    def keys(self) -> list[str]:
        """List all cached keys."""
        if not self.cache_path.exists():
            return []
        with h5py.File(self.cache_path, "r") as f:
            return list(f.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        n = len(self)
        return f"PredictionCache({self.cache_path}, {n} entries)"


def get_cache(cache_dir: Optional[Path]) -> Optional[PredictionCache]:
    """Open or create a prediction cache.

    Args:
        cache_dir: Directory for the cache file. If None, returns None.

    Returns:
        PredictionCache instance, or None if cache_dir is None.
    """
    if cache_dir is None:
        return None
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return PredictionCache(cache_dir / "predictions.h5")
