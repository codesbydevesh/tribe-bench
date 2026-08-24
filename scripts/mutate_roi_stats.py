"""Mutation battery for the Phase B mechanism corrections.

Each mutation reintroduces a defect by patching the source text, runs the suite in
a scratch copy, and records whether the suite FAILS (detected) or PASSES (survived).
A survivor means the invariant it belongs to is asserted nowhere.

Three groups:
  ORIGINAL  — the 16 author-designed mutations from the first pass.
  REVIEWER  — the 5 that survived the independent review. Mandatory regressions.
  CLASS     — new mutations probing each newly identified invariant, including
              cases neither the author nor the reviewer demonstrated.

Usage:  python3 scripts/mutate_roi_stats.py [--group NAME] [--jobs N]
Never modifies the repository: every run happens in a fresh copy under /tmp.
"""
from __future__ import annotations

import argparse
import concurrent.futures as _cf
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = "tribe_tools/roi_stats.py"
TESTS = "tests/test_roi_stats.py"

# (id, group, file, find, replace, what it reintroduces)
MUTATIONS = [
    # ---------------------------------------------------------------- ORIGINAL
    ("O01", "ORIGINAL", SRC, 'if (idx < 0).any():', 'if False:',
     "negative indices accepted; they defeat the overlap guard"),
    ("O02", "ORIGINAL", SRC, 'if np.unique(idx).size != idx.size:', 'if False:',
     "duplicate indices double-weight a vertex in every ROI mean"),
    ("O03", "ORIGINAL", SRC, 'and np.all((idx == 0) | (idx == 1))):', 'and False):',
     "0/1 ambiguity resolved by guessing, the original silent wrong ROI"),
    ("O04", "ORIGINAL", SRC, 'if n_vertices is not None and (idx >= n_vertices).any():', 'if False:',
     "out-of-range indices accepted"),
    ("O05", "ORIGINAL", SRC, 'if c.shape[0] == 0:', 'if False:',
     "empty category fabricates a lag via all-NaN argmax"),
    ("O06", "ORIGINAL", SRC, 'if np.allclose(c.std(axis=0).sum(), 0.0)', 'if False and np.allclose(c.std(axis=0).sum(), 0.0)',
     "flat filler category dilutes the pool back to target-only"),
    ("O07", "ORIGINAL", SRC, 'pooled = np.mean(means, axis=0)', 'pooled = means[0]',
     "pooling reduced to the target's own course (C5)"),
    ("O08", "ORIGINAL", SRC, 'av.var(axis=0, ddof=1) / na + bv.var(axis=0, ddof=1) / nb',
     '(av.var(axis=0, ddof=1) * (na - 1) + bv.var(axis=0, ddof=1) * (nb - 1)) / (na + nb - 2) * (1 / na + 1 / nb)',
     "pooled SE instead of Welch; not level-alpha at unequal n"),
    ("O09", "ORIGINAL", SRC, 'ddof=1', 'ddof=0',
     "biased variance in the contrast SE"),
    ("O10", "ORIGINAL", SRC, 'if k >= pv.size:', 'if False:',
     "top_n >= parcel size, a silent no-op selection"),
    ("O11", "ORIGINAL", SRC, 'if pv.size == 0:', 'if False:',
     "empty parcel accepted"),
    ("O12", "ORIGINAL", SRC, 'if len(verts) == 0:', 'if False:',
     "empty ROI returns nan instead of raising"),
    ("O13", "ORIGINAL", SRC, 'return np.sort(keep)', 'return keep',
     "fROI returned unsorted"),
    ("O14", "ORIGINAL", SRC, 'if not np.all(np.equal(np.mod(arr, 1), 0)):', 'if False:',
     "non-integer float indices silently truncated"),
    ("O15", "ORIGINAL", SRC, '(ge + 1) / (n_perm + 1)', 'ge / n_perm',
     "permutation p can be exactly zero; invalid estimator"),
    ("O16", "ORIGINAL", SRC, 'raise ValueError("event_locked_contrast needs at least one other category")',
     'return float(tgt.mean())',
     "contrast with no comparison category returns a bare mean"),

    # ---------------------------------------------------------------- REVIEWER
    # The five that survived the independent review. Each MUST now be detected.
    ("R01", "REVIEWER", SRC, '    if arr.ndim != 1:\n        raise ValueError(\n            f"{what} must be 1-D', '    if False:\n        raise ValueError(\n            f"{what} must be 1-D',
     "F1: 2-D boolean mask of size n read in flat C order -> wrong vertex set"),
    ("R02", "REVIEWER", SRC, 'if arr.ndim != 1:', 'if arr.ndim != 1 and arr.dtype != bool:',
     "F1: 1-D rule exempts booleans, the exact shape of the original bypass"),
    ("R03", "REVIEWER", SRC, 'g = _require_finite(g, "spatial_z map")', 'g = _require_finite(g[verts], "spatial_z map") and g or g',
     "F4: spatial_z guards only the ROI while dividing by whole-map statistics"),
    ("R04", "REVIEWER", SRC, 'if np.isclose(float(mi @ mj) / (ni * nj), 1.0):',
     'if courses[i].shape == courses[j].shape and np.allclose(courses[i], courses[j]):',
     "F5: syntactic duplicate check; scaled/offset/row-duplicated copies pass"),
    ("R05", "REVIEWER", SRC, '    for f in face_list:\n        for s in scene_list:',
     '    for f in face_vals:\n        for s in scene_vals:',
     "F6: u_statistic recomputes from exhausted iterators -> finite wrong U=0.0"),

    # ------------------------------------------------------------------- CLASS
    # I1 — selector canonicalisation
    ("C01", "CLASS", SRC, 'return np.sort(idx)', 'return idx',
     "I1: integer selectors keep caller order; representations diverge"),
    ("C02", "CLASS", SRC, 'if arr.ndim == 0:', 'if False:',
     "I1: bare scalar selector accepted"),
    ("C03", "CLASS", SRC, 'if n_vertices is not None and arr.size != n_vertices:', 'if False:',
     "I1: boolean mask of the wrong length accepted"),
    ("C04", "CLASS", SRC, 'or np.issubdtype(arr.dtype, np.floating)):', 'or True):',
     "I1: object/string dtype reaches np.isfinite and raises TypeError"),
    ("C05", "CLASS", SRC, 'keep = pv[np.argsort(-sel, kind="stable")[:k]]', 'keep = pv[np.argsort(sel)[::-1][:k]]',
     "I1/F7: unstable tie-break; two encodings give different fROIs"),
    # I2 — guard what the computation reads
    ("C06", "CLASS", SRC, 'g = _require_finite(g, "spatial_z map")', 'pass',
     "I2: spatial_z accepts non-finite anywhere and returns nan"),
    ("C07", "CLASS", SRC, '_require_finite(bv, "glm_contrast_z condition B over the ROI")', 'pass',
     "I2: only one of two conditions guarded"),
    ("C08", "CLASS", SRC, '_require_finite(g[ref_verts], "roi_minus_reference reference values")', 'pass',
     "I2: only the ROI guarded, not the reference"),
    # I3 — consume once
    ("C09", "CLASS", SRC, 'u_obs = u_statistic(vals[:n_face], vals[n_face:])',
     'u_obs = u_statistic(face_vals, scene_vals)',
     "I3: exact_perm_p re-reads consumed arguments"),
    ("C10", "CLASS", SRC, '    face_list = list(face_vals)\n    n = len(face_list)', '    n = len(face_vals)',
     "I3: mc_perm_p measures an argument it has already consumed"),
    ("C11", "CLASS", SRC, '    face_list = list(face_vals)\n    n_face = len(face_list)\n    vals = _require_finite(np.array(face_list + list(scene_vals), dtype=float),',
     '    vals = _require_finite(np.array(list(face_vals) + list(scene_vals), dtype=float),',
     "I3: perm_null_deltas calls len() on a consumed argument"),
    # I4 — semantic degeneracy
    ("C12", "CLASS", SRC, 'means = [c.mean(axis=0) for c in courses]\n    # With only two lags',
     'means = [c[:1].mean(axis=0) for c in courses]\n    # With only two lags',
     "I4: degeneracy judged from the first event only, not the mean course"),
    ("C13", "CLASS", SRC, 'if means[0].size >= 3:', 'if False:',
     "I4: degeneracy check disabled entirely"),
    ("C14", "CLASS", SRC, 'mi, mj = means[i] - means[i].mean(), means[j] - means[j].mean()',
     'mi, mj = means[i], means[j]',
     "I4: no mean-centring, so a constant-offset duplicate slips through"),
    # I5 — coverage machinery. Mutating the TESTS: does the machinery catch its own gaps?
    ("C15", "CLASS", TESTS, 'covered = set(_nonfinite_entry_points())',
     'covered = {k.split(":")[0] for k in _nonfinite_entry_points()} | set(_nonfinite_entry_points())',
     "I5/F3: coverage keyed on function name again, so one argument stands for all"),
    ("C16", "CLASS", TESTS, '"u_statistic:scene_vals":       lambda: _R.u_statistic(ok, nanvals),', '',
     "I5: an array argument silently loses its non-finite coverage"),
    ("C17", "CLASS", TESTS, 'for arg in _SELECTOR_ARGS:\n            if arg in params:',
     'for arg in _SELECTOR_ARGS:\n            if arg in params and False:',
     "I5/F2: selector discovery returns nothing, so every selector rule is vacuous"),
    ("C18", "CLASS", TESTS, 'if not n.startswith("_") and callable(getattr(_R, n))}',
     'if not n.startswith("_") and callable(getattr(_R, n))\n            and getattr(getattr(_R, n), "__module__", None) == _R.__name__}',
     "I5/F2: enumeration filters on __module__ again, hiding partials/wrappers"),
]


def _run_one(mut, keep_going: bool):
    mid, group, target, find, repl, what = mut
    with tempfile.TemporaryDirectory(prefix=f"mut_{mid}_") as tmp:
        work = pathlib.Path(tmp) / "repo"
        shutil.copytree(REPO, work, symlinks=True,
                        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc",
                                                      "data", "notebooks", "*.npy",
                                                      "*.nii.gz", "*.png"))
        path = work / target
        text = path.read_text()
        if find not in text:
            return mid, group, "NOT_APPLIED", what
        path.write_text(text.replace(find, repl, 1))
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_roi_stats.py", "-x", "-q",
             "--no-header", "-p", "no:cacheprovider"],
            cwd=work, capture_output=True, text=True, timeout=900)
        return mid, group, ("DETECTED" if proc.returncode != 0 else "SURVIVED"), what


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None, help="ORIGINAL | REVIEWER | CLASS")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    # BASELINE GUARD. If the unmutated suite is red, `pytest -x` fails for every
    # mutation and the whole battery reports a perfect score that means nothing.
    # This happened once; it must never be able to happen silently again.
    base = _run_one(("BASE", "BASELINE", SRC, "import numpy as np", "import numpy as np",
                     "unmutated baseline"), False)
    if base[2] != "SURVIVED":
        print("BASELINE IS NOT GREEN — every mutation would report DETECTED.\n"
              "Fix the suite before trusting any mutation result.")
        return 2
    print("baseline green (unmutated suite passes)\n")

    muts = [m for m in MUTATIONS if args.group is None or m[1] == args.group]
    print(f"running {len(muts)} mutations, {args.jobs} at a time\n")

    results = []
    with _cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for res in pool.map(lambda m: _run_one(m, False), muts):
            results.append(res)
            mid, group, status, what = res
            flag = {"DETECTED": "  ok  ", "SURVIVED": " MISS ", "NOT_APPLIED": " STALE"}[status]
            print(f"{flag} {mid} [{group:8s}] {status:12s} {what}")

    survived = [r for r in results if r[2] == "SURVIVED"]
    stale = [r for r in results if r[2] == "NOT_APPLIED"]
    print(f"\n{len(results) - len(survived) - len(stale)}/{len(results)} detected")
    if stale:
        print(f"STALE (pattern no longer in source — the mutation tests nothing): "
              f"{[r[0] for r in stale]}")
    if survived:
        print(f"SURVIVED (no test asserts this invariant): {[r[0] for r in survived]}")
    return 1 if (survived or stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
