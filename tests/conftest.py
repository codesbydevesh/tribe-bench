"""Test-session wiring.

`exca` is a hard dependency of the S2 reliability contract: the tests that prove
"the consume stage cannot recompute" and "the encode counter counts items" are
meaningless without the real library, and the version that matters is the one the
GPU box runs (0.5.20, pinned transitively by tribev2).

It is NOT installable into this dev box's system python (PEP 668). So:

* if `exca` imports, the contract tests run for real;
* if `S2_DEV_SITE_PACKAGES` is set (colon-separated), those paths are prepended
  first -- that is how a developer points at a venv holding exca 0.5.20;
* if `S2_REQUIRE_EXCA=1` and exca still will not import, the session FAILS rather
  than quietly skipping. Kaggle and any CI must set it.

The last rule exists because a skipped test protects nothing, and this project has
already shipped one test that went green over the bug it was written to catch.
"""
import os
import sys

import pytest

for _p in reversed([p for p in os.environ.get("S2_DEV_SITE_PACKAGES", "").split(os.pathsep) if p]):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import exca as _exca  # noqa: F401
    EXCA_AVAILABLE = True
except Exception:
    EXCA_AVAILABLE = False


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_exca: requires the real exca library")
    if os.environ.get("S2_REQUIRE_EXCA") == "1" and not EXCA_AVAILABLE:
        raise pytest.UsageError(
            "S2_REQUIRE_EXCA=1 but exca is not importable. The cache-contract tests "
            "would skip, which protects nothing. Install exca==0.5.20 or point "
            "S2_DEV_SITE_PACKAGES at a venv that has it.")


def pytest_report_header(config):
    return (f"exca: {'AVAILABLE - cache-contract tests are live' if EXCA_AVAILABLE else 'ABSENT - cache-contract tests will SKIP (set S2_DEV_SITE_PACKAGES)'}")


def pytest_collection_modifyitems(session, config, items):
    """Fail the session if any test_*.py file contributes ZERO tests.

    A 509-line test file that collected nothing sat in tests/ on 2026-08-26 -- an
    agent was interrupted mid-write, leaving scaffolding and no assertions. It looked
    like coverage in every listing and provided none. Same class of error as a
    source-string test going green over its own bug, so it fails here.

    Only polices a WHOLE-SUITE run. Selecting one file or one test is a normal thing
    to do, and an earlier version of this hook wrongly aborted those -- a guard that
    misfires on ordinary use gets disabled, which is worse than not having it.
    """
    import pathlib
    root = pathlib.Path(__file__).parent
    targets = [a for a in (config.args or []) if not a.startswith("-")]
    whole_suite = not targets or all(
        pathlib.Path(t).resolve() in (root, root.parent) for t in targets)
    if not whole_suite:
        return
    have = {pathlib.Path(str(i.fspath)).name for i in items}
    empty = sorted({p.name for p in root.glob("test_*.py")} - have)
    if empty:
        raise pytest.UsageError(
            "these test files collected no tests, so they protect nothing: "
            + ", ".join(empty))
