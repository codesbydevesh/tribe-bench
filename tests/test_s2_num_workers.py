"""Regression for the num_workers config key.

Commit 323a65c shipped `config_update={"num_workers": 0}` to stop the forked-CUDA
crash of 2026-08-25. It does not work: num_workers is a field of tribev2's `Data`
sub-model (main.py:112), consumed at main.py:270, not a field of the experiment
root. exca's ConfDict nests strictly on dots (confdict.py:54-58), so the bare key
writes a NEW root key and leaves `data.num_workers` at the checkpoint's 20.

The test that accompanied 323a65c asserted `'"num_workers": 0' in src`. The string
was present and the key was wrong, so it went green over the defect -- and correcting
the key turned it red. It has been deleted. These tests evaluate the MERGED config.
"""
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

# tribev2/grids/defaults.py:20,131 -- N_CPUS on Meta's training cluster, frozen into
# the released checkpoint config. Source of "will create 20 worker processes".
CHECKPOINT_DEFAULTS = {"data": {"num_workers": 20, "batch_size": 8}}


def _payload():
    spec = importlib.util.spec_from_file_location("s2run", REPO / "scripts" / "s2_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.model_config_update()


def _merge_dotted(base, update):
    """exca ConfDict's nesting rule; pinned to the real library below."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, val in update.items():
        node, *rest = key.split(".")
        cur = out
        while rest:
            cur = cur.setdefault(node, {})
            node, *rest = rest
        cur[node] = val
    return out


def test_the_effective_num_workers_reaching_Data_is_zero():
    merged = _merge_dotted(CHECKPOINT_DEFAULTS, _payload())
    assert merged["data"]["num_workers"] == 0, (
        "data.num_workers is still "
        f"{merged['data']['num_workers']}: the DataLoader will fork that many workers "
        "and each re-initialises CUDA inside neuralset/extractors/video.py:265")


def test_no_payload_key_lands_at_the_config_root():
    """A root key is not merely ignored -- TribeExperiment sets extra='forbid'
    (main.py:281) and neuraltrain's BaseExperiment declares only `infra`, so an
    undotted key aborts from_pretrained with a pydantic ValidationError."""
    for key in _payload():
        assert key.startswith("data."), f"{key!r} does not address the tribev2 data tree"


def test_our_dotted_merge_matches_real_exca():
    confdict = pytest.importorskip("exca.confdict", reason="pip install exca==0.5.20")
    real = confdict.ConfDict({"data": dict(CHECKPOINT_DEFAULTS["data"])})
    real.update(_payload())
    assert dict(real.flat())["data.num_workers"] == 0
