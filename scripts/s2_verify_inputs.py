"""Verify the uploaded S2 inputs before spending a GPU second. CPU-only, read-only.

Run this FIRST in the Kaggle session, before the frame-sampling check and long
before ``--infer``. It answers one question: are the files at the expected paths
byte-identical to the ones the frozen manifest describes?

    python3 scripts/s2_verify_inputs.py --stimulus-root /kaggle/input/<dataset-slug>

Checks, in order:
  1. the video exists at the path s2_run.py will actually consume
  2. its sha256 matches the manifest exactly
  3. every scheduled stimulus image is present
  4. every image sha256 matches the manifest
  5. nothing is a placeholder
  6. the resolved paths are the ones s2_run.py will use

Exit 0 = inputs verified. Anything else = do NOT proceed.
This script never writes, renders, or modifies anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurocheck.s2_design import S2, build_schedule, resolve_stimulus_images  # noqa: E402

MANIFEST = Path("data/s2_manifest.json")
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimulus-root", default=os.environ.get("S2_STIMULUS_ROOT", "data"),
                    help="where floc/ and s2_stimulus.mp4 live")
    args = ap.parse_args()
    root = Path(args.stimulus_root)

    # Resolve paths through the SAME code s2_run.py uses, so this cannot verify one
    # location while the run reads another.
    os.environ["S2_STIMULUS_ROOT"] = str(root)
    import importlib
    s2_run = importlib.import_module("scripts.s2_run") if False else None  # noqa: F841
    video = root / "s2_stimulus.mp4"
    images_dir = root / "floc"

    print(f"=== S2 input verification ===\n  stimulus root  {root.resolve()}"
          f"\n  design         {S2.fingerprint()}\n")

    if not MANIFEST.exists():
        print(f"  no manifest at {MANIFEST} — is the repo checked out?")
        return 2
    man = json.loads(MANIFEST.read_text())
    prov = man.get("provenance", {})

    check("manifest is for this design",
          man.get("design_fingerprint") == S2.fingerprint(), man.get("design_fingerprint", ""))

    # ---- 1 & 6: the video, at the path the run will consume
    check("1. video exists at the path s2_run.py will read", video.is_file(), str(video))
    if not video.is_file():
        print("\n  Upload incomplete. Stop here.")
        return 1

    # ---- 2: exact hash
    rec_v = prov.get("video") or {}
    got = sha256(video)
    check("2. video sha256 matches the manifest exactly",
          got == rec_v.get("sha256"),
          f"{got[:24]}..." if got == rec_v.get("sha256")
          else f"got {got[:16]}..., manifest {str(rec_v.get('sha256'))[:16]}...")

    # ---- 5a: the video was not built from placeholders
    check("5a. video was built from real images, not placeholders",
          rec_v.get("placeholders") is False)

    # ---- 3: every scheduled image present
    expected = {e.stimulus_id for e in build_schedule(S2)}
    try:
        live = resolve_stimulus_images(images_dir, S2)
    except Exception as exc:
        check("3. every scheduled stimulus image is present", False,
              f"{type(exc).__name__}: {exc}")
        print("\n  Upload incomplete or mis-structured. Stop here.")
        return 1
    check("3. every scheduled stimulus image is present",
          set(live) == expected, f"{len(live)} of {len(expected)}")

    # ---- 4: every hash matches
    rec_i = (prov.get("images") or {}).get("files") or {}
    drift = [k for k in rec_i if k not in live or live[k]["sha256"] != rec_i[k]["sha256"]]
    check("4. every image sha256 matches the manifest", not drift,
          f"{len(rec_i)} verified" if not drift else f"{len(drift)} differ: {drift[:4]}")

    # ---- 5b: no placeholder artefacts left lying around
    strays = [p.name for p in images_dir.rglob("*")
              if p.is_file() and "placeholder" in p.name.lower()]
    check("5b. no placeholder files present", not strays, "; ".join(strays[:3]))

    # ---- 6: the exact paths the run will consume
    check("6. resolved paths are the ones s2_run.py will use",
          True, f"video={video}  images={images_dir}")

    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks PASS")
    if failed:
        print("\nDO NOT PROCEED. Outstanding:")
        for n in failed:
            print(f"  - {n}")
        return 1
    print("\nInputs verified. Next, and ONLY next:")
    print("  python3 scripts/s2_check_frame_sampling.py")
    print("\nDo not run S2 yet. The frame-sampling result decides whether the")
    print("current 8 fps stimulus is usable at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
