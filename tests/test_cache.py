"""PredictionCache: round-trip, overwrite, and cross-session persistence.

These guard the durability claim in the docstring — each save() must be on
disk by the time it returns, so a killed Kaggle session resumes cleanly.
"""

import numpy as np

from tribe_tools.cache import PredictionCache, get_cache


def test_save_load_roundtrip(tmp_path):
    cache = PredictionCache(tmp_path / "c.h5")
    arr = np.arange(12, dtype="float32").reshape(3, 4)
    cache.save("k1", arr)
    assert cache.has("k1")
    np.testing.assert_array_equal(cache.load("k1"), arr)


def test_missing_key_returns_none(tmp_path):
    cache = PredictionCache(tmp_path / "c.h5")
    assert cache.load("nope") is None
    assert cache.has("nope") is False


def test_overwrite_key(tmp_path):
    cache = PredictionCache(tmp_path / "c.h5")
    cache.save("k", np.zeros(3, dtype="float32"))
    cache.save("k", np.ones(3, dtype="float32"))
    np.testing.assert_array_equal(cache.load("k"), np.ones(3, dtype="float32"))
    assert cache.keys() == ["k"]


def test_persists_across_instances(tmp_path):
    # A fresh instance models a new session — resume support depends on this.
    path = tmp_path / "c.h5"
    PredictionCache(path).save("k", np.array([1.0, 2.0], dtype="float32"))
    assert PredictionCache(path).has("k")


def test_len_counts_entries(tmp_path):
    cache = PredictionCache(tmp_path / "c.h5")
    cache.save("a", np.zeros(1, dtype="float32"))
    cache.save("b", np.zeros(1, dtype="float32"))
    assert len(cache) == 2


def test_get_cache_none_returns_none():
    assert get_cache(None) is None
