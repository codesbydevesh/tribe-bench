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
    ("O07", "ORIGINAL", SRC, 'pooled = np.mean(means, axis=0)', 'pooled = means[0]',
     "pooling reduced to the target's own course (C5)"),
    ("O08", "ORIGINAL", SRC, 'av.var(axis=0, ddof=1) / na + bv.var(axis=0, ddof=1) / nb',
     '(av.var(axis=0, ddof=1) * (na - 1) + bv.var(axis=0, ddof=1) * (nb - 1)) / (na + nb - 2) * (1 / na + 1 / nb)',
     "pooled SE instead of Welch; not level-alpha at unequal n"),
    ("O09", "ORIGINAL", SRC, 'ddof=1', 'ddof=0',
     "biased variance in ONE arm of the contrast SE (asymmetric estimator)"),
    ("O10", "ORIGINAL", SRC, 'if k >= pv.size:', 'if False:',
     "top_n >= parcel size, a silent no-op selection"),
    ("O11", "ORIGINAL", SRC, 'if pv.size == 0:', 'if False:',
     "empty parcel accepted"),
    ("O12", "ORIGINAL", SRC, 'if len(verts) == 0:', 'if False:',
     "empty ROI returns nan instead of raising (first occurrence: spatial_z)"),
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
    ("R02", "REVIEWER", SRC, 'if arr.ndim != 1:\n        raise ValueError(\n            f"{what} must be 1-D, got shape {arr.shape}. A multi-dimensional',
     'if arr.ndim != 1 and arr.dtype != bool:\n        raise ValueError(\n            f"{what} must be 1-D, got shape {arr.shape}. A multi-dimensional',
     "F1: 1-D rule exempts booleans, the exact shape of the original bypass"),
    ("R03", "REVIEWER", SRC, 'g = _require_finite(g, "spatial_z map")',
     '_require_finite(g[verts], "spatial_z map")',
     "F4: spatial_z guards only the ROI while dividing by whole-map statistics"),
    ("R05", "REVIEWER", SRC, '    for f in face_list:\n        for s in scene_list:',
     '    for f in face_vals:\n        for s in scene_vals:',
     "F6: u_statistic recomputes from exhausted iterators -> finite wrong U=0.0"),

    # ------------------------------------------------------------------- CLASS
    # I1 — selector canonicalisation
    ("C01", "CLASS", SRC, 'return np.sort(idx)', 'return idx',
     "I1: integer selectors keep caller order; representations diverge"),
    ("C02", "CLASS", SRC, 'if arr.ndim == 0:\n        raise ValueError(\n            f"{what} is a scalar', 'if False:\n        raise ValueError(\n            f"{what} is a scalar',
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
    ("C10", "CLASS", SRC, '    n = int(face_arr.size)', '    n = len(face_vals)',
     "I3: mc_perm_p measures an argument it has already consumed"),
    ("C11", "CLASS", SRC, '    n_face = int(face_arr.size)\n    vals = np.concatenate([face_arr, scene_arr])',
     '    n_face = len(face_vals)\n    vals = np.concatenate([face_arr, scene_arr])',
     "I3: perm_null_deltas calls len() on a consumed argument"),
    # I4 — semantic degeneracy, expressed on the pooled course
    ("O06", "CLASS", SRC, 'if _is_zero(nrm, m):', 'if False:',
     "a category whose MEAN course is flat dilutes the pool back to target-only"),
    ("R04", "REVIEWER", SRC, 'if np.isclose(float(pc @ c) / (pn * nrm), 1.0):', 'if False:',
     "F5: pooled-collinearity rule removed; scaled/offset/row-duplicated copies pass"),
    ("C12", "CLASS", SRC, 'means = [c.mean(axis=0) for c in courses]\n    centred =',
     'means = [c[:1].mean(axis=0) for c in courses]\n    centred =',
     "I4: degeneracy judged from the first event only, not the mean course"),
    ("C13", "CLASS", SRC, 'if n_lags < 3:', 'if False:',
     "I4: the <3-lag hole reopened, where every course pair is collinear"),
    ("C14", "CLASS", SRC, 'centred = [m - m.mean() for m in means]', 'centred = [m for m in means]',
     "I4: no mean-centring, so a constant-offset duplicate slips through"),
    ("C19", "CLASS", SRC, 'if _is_zero(pn, pooled):', 'if False:',
     "I4: total cancellation accepted; argmax of a flat pool fabricates lag 0"),
    ("C20", "CLASS", SRC, 'return nrm <= 1e-12 * max(1.0, float(np.abs(m).max()))',
     'return nrm <= 1e-1 * max(1.0, float(np.abs(m).max()))',
     "I4 over-rejection: a weak but real mean response treated as flat"),
    # I1 — rank and representation plumbing found by the second review
    ("C21", "CLASS", SRC, 'if arr.ndim not in (1, 2):', 'if False:',
     "I1: 3-D preds accepted; n_vertices then describes an axis verts never indexes"),
    ("C22", "CLASS", SRC, 'if isinstance(v, np.ma.MaskedArray):', 'if False:',
     "I1: masked selector silently unmasked, two encodings of one set disagree"),
    ("C23", "CLASS", SRC, '    segments = list(segments)', '    pass',
     "I3: row_times_from_segments rejects a single-pass iterable"),
    ("C24", "CLASS", SRC, 'rt = _require_finite(np.array(list(row_times_s), dtype=float), "peri_event_timecourse row_times_s")',
     'rt = _require_finite(np.asarray(row_times_s, dtype=float), "peri_event_timecourse row_times_s")',
     "I3: row_times_s generator becomes a 0-d object array"),
    # --- third review: S6 rank guard on the permutation entry points
    ("C25", "CLASS", SRC, 'face_list = _as_event_vector(_materialise(face_vals), "u_statistic face_vals")',
     'face_list = _require_finite(_materialise(face_vals), "u_statistic face_vals")',
     "S6: u_statistic accepts a (n_events, n_lags) time course"),
    ("C26", "CLASS", SRC, 'other_arr = _as_event_vector(_materialise(other_vals), "mc_perm_p other_vals")',
     'other_arr = np.asarray(_materialise(other_vals), dtype=float)',
     "S6: mc_perm_p broadcasts a 2-D input to a finite wrong p-value (0.567 -> 0.0005)"),
    ("C27", "CLASS", SRC, 'scene_arr = _as_event_vector(_materialise(scene_vals), "perm_null_deltas scene_vals")',
     'scene_arr = np.asarray(_materialise(scene_vals), dtype=float)',
     "S6: the G2 magnitude null means over both axes"),
    # --- the rank-collapse arithmetic itself
    ("C28", "CLASS", SRC, 'return arr.mean(axis=0) if arr.ndim == 2 else arr', 'return arr[0] if arr.ndim == 2 else arr',
     "I1: rank collapse takes row 0 instead of the row mean"),
    ("C29", "CLASS", SRC, 'return arr.mean(axis=0) if arr.ndim == 2 else arr', 'return arr.mean(axis=1) if arr.ndim == 2 else arr',
     "I1: rank collapse averages the wrong axis"),
    # --- roi_minus_reference routed through the shared helper
    ("C30", "CLASS", SRC, '    g = _as_vertex_map(preds, "roi_minus_reference preds")\n    _n = g.shape[-1]',
     '    _n = preds.shape[-1] if np.ndim(preds) else None\n    g = _as_vertex_map(preds, "roi_minus_reference preds")',
     "I1: n_vertices from a raw .shape again, skipping the rank precondition"),
    # --- read-extent completeness derived rather than hand-listed
    ("C31", "CLASS", TESTS, '    return _read_extent_required() - (_WHOLE_MAP_CONSUMERS | _REGION_LOCAL_CONSUMERS)',
     '    return set()',
     "I2/I5: read-extent requirement no longer derived, so a new function is invisible"),
    ("C32", "CLASS", TESTS, 'if params & set(_SELECTOR_ARGS) and params & _ARRAY_DATA_PARAMS:', 'if False:',
     "I2/I5: read-extent derivation returns nothing"),
    # --- scalar guard
    ("C33", "CLASS", SRC, 'if not 0 <= pre_trs < n_lags:', 'if False:',
     "pre_trs outside the lag grid reports a lag that does not exist"),
    # --- the row/time 1:1 pairing, the reason row_times_from_segments exists
    ("C34", "CLASS", SRC, 'if len(rt) != p.shape[0]:', 'if False:',
     "row index treated as TR index; the confusion the module was built to prevent"),
    # I5 — coverage machinery. Mutating the TESTS: does the machinery catch its own gaps?
    ("C15", "CLASS", TESTS, '    return covered, required',
     '    return {k.split(":")[0] for k in covered}, {k.split(":")[0] for k in required}',
     "I5/F3: coverage keyed on function name on BOTH sides, the faithful F3 revert"),
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
                                                      "data", "notebooks", "results",
                                                      "*.npy", "*.nii.gz", "*.png",
                                                      "*.zip", "*.h5", "*.pdf"))
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
