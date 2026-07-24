"""Cache-key derivation: deterministic, mask-sensitive, and HDF5-safe.

A key must be stable for the same (video, mask) so cache hits work, must
differ across masks so the 8 coalition passes don't collide, and must not
contain '/' (HDF5 would treat it as a group separator).
"""

from pathlib import Path

from tribe_tools.inference import _cache_key, _mask_key


def test_mask_key_full_when_empty():
    assert _mask_key(None) == "full"
    assert _mask_key([]) == "full"


def test_mask_key_order_independent():
    assert _mask_key(["b", "a"]) == _mask_key(["a", "b"])


def test_cache_key_deterministic():
    p = Path("/data/clip.mp4")
    assert _cache_key(p, "full") == _cache_key(p, "full")


def test_cache_key_varies_by_mask():
    p = Path("/data/clip.mp4")
    assert _cache_key(p, "full") != _cache_key(p, "audio")


def test_cache_key_is_hdf5_safe():
    key = _cache_key(Path("/a/b/c.mp4"), "full")
    assert "/" not in key
