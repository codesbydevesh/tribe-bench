"""Fetch the fLoc images S2 needs, pinned to an exact commit, and hash them.

fLoc states **no licence** (VPNL/fLoc), so the images may be used but must never be
redistributed — not in the repo, not beside the results. That makes the manifest the
only possible carrier of image identity, which is why every file is hashed here.

Downloads ONLY the images the frozen design selects (125 of 1592) at a pinned commit,
so this is ~15 MB rather than the repo's 188 MB — a shared box with 21 GB free does
not need the other 1467 files.

    python3 scripts/s2_fetch_stimuli.py [--dest data/floc] [--dry-run]

Selection is deterministic: round-robin across each category's subcategories, files
sorted by name within each. An unsorted glob would map the SAME manifest to different
pixels on different machines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurocheck.s2_design import S2, S2Config  # noqa: E402

REPO = "VPNL/fLoc"
# Pinned. Repo HEAD as resolved from the GitHub API on 2026-08-24; last pushed
# 2024-02-21T20:15:59Z. fLoc is not versioned by tags, so the commit IS the version.
COMMIT = "de6a26cc269a2c7075461a4c839bfd628f225c95"
RAW = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
API_TREE = "https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "corticall-s2"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def select_paths(cfg: S2Config = S2, *, tree: list | None = None) -> dict:
    """{stimulus_id: repo_path}, by the frozen selection rule."""
    if tree is None:
        tree = json.loads(_get(API_TREE.format(repo=REPO, sha=COMMIT)))["tree"]
    by_dir: dict[str, list[str]] = {}
    for t in tree:
        path = t["path"]
        if not path.lower().endswith((".jpg", ".jpeg", ".png")) or "/" not in path:
            continue
        by_dir.setdefault(path.rsplit("/", 1)[0], []).append(path)
    for k in by_dir:
        by_dir[k].sort()                      # deterministic, not filesystem order

    out: dict[str, str] = {}
    for category, subcats in cfg.stimulus_subcategories:
        pools = [by_dir.get(f"stimuli/{sc}", []) for sc in subcats]
        missing = [sc for sc, pool in zip(subcats, pools) if not pool]
        if missing:
            raise ValueError(f"no images found for subcategories {missing}")
        picked, i = [], 0
        while len(picked) < cfg.exemplars_per_category:
            pool = pools[len(picked) % len(pools)]     # round-robin
            idx = i if len(pools) == 1 else len(picked) // len(pools)
            if idx >= len(pool):
                raise ValueError(f"{category}: ran out of images in {subcats}")
            picked.append(pool[idx])
            i += 1
        for n, repo_path in enumerate(picked):
            out[f"{category}_{n:03d}"] = repo_path
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/floc")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dest = Path(args.dest)

    print(f"fLoc @ {REPO} commit {COMMIT[:12]} (no licence — never redistribute)")
    sel = select_paths(S2)
    print(f"selected {len(sel)} of 1592 images "
          f"({S2.exemplars_per_category} per category, round-robin, sorted)\n")
    for category, subcats in S2.stimulus_subcategories:
        ids = [k for k in sel if k.startswith(category + "_")]
        srcs = sorted({sel[k].split('/')[1] for k in ids})
        print(f"  {category:12s} {len(ids):3d} from {srcs}")
    if args.dry_run:
        print("\n--dry-run: nothing downloaded")
        return 0

    print()
    records, total = {}, 0
    for sid, repo_path in sorted(sel.items()):
        category = sid.rsplit("_", 1)[0]
        out = dest / category / Path(repo_path).name
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            blob = out.read_bytes()
        else:
            blob = _get(RAW.format(repo=REPO, sha=COMMIT, path=repo_path))
            out.write_bytes(blob)
        h = hashlib.sha256(blob).hexdigest()
        records[sid] = {"path": str(out), "repo_path": repo_path,
                        "sha256": h, "bytes": len(blob)}
        total += len(blob)
    print(f"fetched {len(records)} images, {total/1e6:.1f} MB -> {dest}")

    index = {"source_repo": REPO, "commit": COMMIT,
             "selection_rule": S2.stimulus_selection_rule,
             "licence": "none stated by VPNL/fLoc — DO NOT REDISTRIBUTE",
             "images": records}
    idx_path = dest / "index.json"
    idx_path.write_text(json.dumps(index, indent=2))
    print(f"index -> {idx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
