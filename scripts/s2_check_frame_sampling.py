"""Settle the one S2 question that needs the GPU environment.

Does neuralset's video loader select V-JEPA's 64 frames by TIMESTAMP or by FRAME
INDEX? The model is ``vjepa2-vitg-fpc64-256`` at ``frequency: 2``, ``clip_duration:
4`` — 64 frames per 4 s clip, i.e. 16 fps of intended coverage. Our stimulus is
rendered at 8 fps.

  timestamp-based -> each clip still spans 4 s; frames resolve to duplicates of
                     their neighbours. Harmless for a STATIC image, and the paper's
                     stimulus was static too. 8 fps is fine.
  index-based     -> 64 consecutive frame indices at 8 fps span EIGHT seconds, not
                     four. That halves each 1 s presentation's weight inside its
                     tubelet and smears it across neighbouring events at an 8 s SOA.
                     BLOCKING: re-render at fps=16 before spending GPU hours.

Run this on the GPU box:  python3 scripts/s2_check_frame_sampling.py
"""
from __future__ import annotations

import importlib.util
import inspect
import re
import sys


def main() -> int:
    if importlib.util.find_spec("neuralset") is None:
        print("neuralset is NOT importable here. Run this on the GPU box.")
        return 2
    import neuralset  # noqa: F401

    hits = []
    for modname in ("neuralset.extractors", "neuralset.video", "neuralset"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        for pat, verdict in (
            (r"\.set\(\s*cv2\.CAP_PROP_POS_FRAMES", "INDEX"),
            (r"\.set\(\s*cv2\.CAP_PROP_POS_MSEC", "TIMESTAMP"),
            (r"pts|presentation_timestamp|time_base", "TIMESTAMP"),
            (r"frame_indices|np\.arange\([^)]*n_frames", "INDEX"),
            (r"seek\([^)]*sec|start_sec|\btimestamps?\b", "TIMESTAMP"),
        ):
            for m in re.finditer(pat, src):
                line = src[:m.start()].count("\n") + 1
                hits.append((modname, line, verdict, src.splitlines()[line - 1].strip()[:90]))

    if not hits:
        print("No frame-selection pattern matched. Inspect neuralset's video loader by hand:")
        print("  python3 -c \"import neuralset,inspect;print(inspect.getsourcefile(neuralset))\"")
        return 2
    for modname, line, verdict, text in hits:
        print(f"  [{verdict:9s}] {modname}:{line}  {text}")
    verdicts = {v for _, _, v, _ in hits}
    print()
    if verdicts == {"TIMESTAMP"}:
        print("TIMESTAMP-based. 8 fps is safe. Re-run the checklist with "
              "--neuralset-timestamp.")
        return 0
    if "INDEX" in verdicts:
        print("INDEX-based evidence found. BLOCKING: set S2Config.fps = 16 and "
              "re-render before spending GPU hours.")
        return 1
    print("Ambiguous. Resolve by hand before running.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
