"""Settle the one S2 question that needs the GPU environment.

Does neuralset's video extractor address V-JEPA's 64 frames by **timestamp** or by
**frame index**? The model is ``vjepa2-vitg-fpc64-256`` at ``frequency: 2``,
``clip_duration: 4`` — 64 frames per 4 s clip. Our stimulus renders at 8 fps.

  timestamp-based -> each clip spans 4 s of timeline whatever the source fps.
                     Frames resolve to duplicates of their neighbours, which is a
                     near-no-op for a STATIC image, and the paper's stimulus was
                     static too. **8 fps is fine.**
  index-based     -> 64 consecutive frame indices at 8 fps span EIGHT seconds, not
                     four: each 1 s presentation's weight inside its tubelet is
                     halved and smeared across neighbouring events at an 8 s SOA.
                     **BLOCKING** — re-render at fps=16 before spending GPU hours.

    python3 scripts/s2_check_frame_sampling.py

Exit 0 = timestamp-based (safe). 1 = index-based (STOP). 2 = undetermined.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import pathlib
import re
import sys

# Verified against neuralset 0.2.3 by reading the wheel from PyPI on 2026-08-24.
# The extractor builds a list of TIMES in seconds and fetches each frame by time:
#     subtimes = [k / self.num_frames * T for k in reversed(range(self.num_frames))]
#     times    = np.linspace(0, video.duration, expect_frames + 1)[1:]
#     ims      = [_VideoImage(video=video, time=max(0, t - t2)) for t2 in subtimes]
# and _VideoImage._read does:
#     img = self.video.get_frame(self.time)
# T is clip_duration (4 s), so a clip spans 4 s of timeline at ANY source fps.
TIMESTAMP_PATTERNS = [
    (r"_VideoImage\([^)]*\btime\s*=", "frames requested by time= keyword"),
    (r"get_frame\(\s*self\.time\s*\)", "video.get_frame(self.time)"),
    (r"np\.linspace\(\s*0\s*,\s*video\.duration", "sample times built from duration in seconds"),
    (r"subtimes\s*=.*num_frames.*T", "sub-frame offsets computed in seconds"),
    (r"CAP_PROP_POS_MSEC", "OpenCV seek by milliseconds"),
]
INDEX_PATTERNS = [
    (r"get_batch\(\s*\[?\s*(?:frame_)?(?:ind|idx|indices)", "decord get_batch(indices)"),
    (r"CAP_PROP_POS_FRAMES", "OpenCV seek by frame index"),
    (r"frames\s*\[\s*\w+\s*:\s*\w+\s*\+\s*(?:self\.)?num_frames", "slice of consecutive frames"),
    (r"np\.arange\([^)]*num_frames[^)]*\)\s*\+\s*\w*(?:start|offset)", "consecutive index range"),
]


def _sources() -> list[tuple[str, str]]:
    """(label, source) for the modules that could do frame selection."""
    out = []
    try:
        import neuralset
    except Exception as exc:
        print(f"neuralset is not importable here ({type(exc).__name__}). Run this on the "
              "GPU box, where tribev2 installs it.")
        return []
    root = pathlib.Path(inspect.getsourcefile(neuralset)).parent
    ver = getattr(neuralset, "__version__", "unknown")
    print(f"neuralset {ver} at {root}\n")
    for rel in ("extractors/video.py", "extractors/image.py"):
        f = root / rel
        if f.is_file():
            out.append((rel, f.read_text(errors="ignore")))
    if not out:                      # layout changed: fall back to a scoped sweep
        for f in sorted(root.rglob("*.py")):
            if "test" in f.name:
                continue
            txt = f.read_text(errors="ignore")
            if re.search(r"num_frames|get_frame|clip_duration", txt):
                out.append((str(f.relative_to(root)), txt))
    return out


def main() -> int:
    srcs = _sources()
    if not srcs:
        return 2

    ts, ix = [], []
    for label, src in srcs:
        for pat, why in TIMESTAMP_PATTERNS:
            for m in re.finditer(pat, src):
                line = src[:m.start()].count("\n") + 1
                ts.append((label, line, why, src.splitlines()[line - 1].strip()[:88]))
        for pat, why in INDEX_PATTERNS:
            for m in re.finditer(pat, src):
                line = src[:m.start()].count("\n") + 1
                ix.append((label, line, why, src.splitlines()[line - 1].strip()[:88]))

    for tag, hits in (("TIMESTAMP", ts), ("INDEX", ix)):
        for label, line, why, text in hits:
            print(f"  [{tag:9s}] {label}:{line}  {why}")
            print(f"              {text}")
    if not (ts or ix):
        print("  no frame-selection pattern matched.")

    print()
    if ts and not ix:
        print("TIMESTAMP-based: each clip spans clip_duration seconds regardless of the\n"
              "source fps, so 8 fps is safe. Re-run the checklist with --neuralset-timestamp.")
        return 0
    if ix and not ts:
        print("INDEX-based. BLOCKING: 64 consecutive frames at 8 fps span 8 s instead of 4.\n"
              "Set S2Config.fps = 16, re-run --prepare, re-upload, THEN run S2.")
        return 1
    if ts and ix:
        print("MIXED evidence. Resolve by hand before running -- read the lines above and\n"
              "determine which path the video extractor actually takes.")
        return 2
    print("UNDETERMINED. Inspect the loader by hand:\n"
          "  python3 -c \"import neuralset,inspect,pathlib;"
          "print(pathlib.Path(inspect.getsourcefile(neuralset)).parent/'extractors'/'video.py')\"")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
