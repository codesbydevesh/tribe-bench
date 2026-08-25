"""S2 stimulus renderer — one continuous silent video, grey ISIs physically drawn.

**Why every ISI frame is really drawn.** The ISI is not an absence of stimulus; it
is what the model sees between presentations. Rendering only the 1 s presentations
would produce a video ~8x too short, and every onset after the first would point at
the wrong frame. The renderer therefore emits a frame for every position on the
timeline and the caller can verify the file's true duration afterwards.

**No audio track, and no transcription.** A silent video means the audio and text
extractors drop out, which is what makes this affordable. Nothing here calls
``get_events_dataframe``: over silence WhisperX transcribes nothing while still
costing ~65 s/clip, and any timing it produced would come from the audio rather
than from the frozen schedule.

Uses OpenCV rather than ffmpeg: ffmpeg is not installed on the development box, and
``tribe_tools.video_utils`` hard-requires it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .s2_design import S2, Event, S2Config, build_schedule


@dataclass(frozen=True)
class RenderResult:
    path: Path
    n_frames: int
    fps: int
    duration_s: float
    width: int
    height: int
    stimulus_frames: int
    grey_frames: int
    placeholders: bool = True


def _exemplar_frame(cfg: S2Config, ev: Event) -> np.ndarray:
    """A deterministic placeholder frame for one exemplar.

    Real runs substitute fLoc images here, resolved through
    ``s2_design.resolve_stimulus_images`` so every id maps to a hashed file. The
    placeholder is deterministic in the stimulus id ACROSS PROCESSES (hashlib, not
    the per-process-salted builtin ``hash``), and is visibly not grey so a
    frame-level check can tell presentation from ISI.
    """
    h, w = cfg.frame_size
    # hashlib, not hash(): Python randomises string hashing per process, so the
    # docstring's reproducibility claim was false across runs.
    import hashlib as _h
    seed = int(_h.sha256(ev.stimulus_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), cfg.grey_level, dtype=np.uint8)
    cat_idx = cfg.categories.index(ev.category)
    band = h // max(len(cfg.categories), 1)
    y0 = cat_idx * band
    img[y0:y0 + band, :, :] = rng.integers(0, 256, (band, w, 3), dtype=np.uint8)
    return img


def _grey_frame(cfg: S2Config) -> np.ndarray:
    h, w = cfg.frame_size
    return np.full((h, w, 3), cfg.grey_level, dtype=np.uint8)


def frame_plan(cfg: S2Config, events: list[Event]) -> np.ndarray:
    """Per-frame event id for the whole timeline; -1 means a grey ISI frame.

    Built by sampling the timeline at frame centres, so the plan is a function of
    the schedule rather than of an accumulating counter that can drift.
    """
    n_frames = int(round(cfg.stimulus_duration_s * cfg.fps))
    t = (np.arange(n_frames) + 0.5) / cfg.fps
    plan = np.full(n_frames, -1, dtype=np.int64)
    for ev in events:
        plan[(t >= ev.onset_s) & (t < ev.offset_s)] = ev.event_id
    return plan


def load_stimulus_image(path: str | Path, cfg: S2Config) -> np.ndarray:
    """Load one real stimulus image at the encoder's native size.

    Resized to ``cfg.frame_size`` with INTER_AREA, which is the correct filter for
    downscaling; the default bilinear aliases fine detail, and characters/VWFA is
    the condition that depends on legible small text.
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read stimulus image: {path}")
    h, w = cfg.frame_size
    if img.shape[:2] != (h, w):
        interp = cv2.INTER_AREA if (img.shape[0] > h or img.shape[1] > w) else cv2.INTER_CUBIC
        img = cv2.resize(img, (w, h), interpolation=interp)
    return img


def render(cfg: S2Config = S2, out_path: str | Path = "data/s2_stimulus.mp4",
           *, events: list[Event] | None = None,
           images: dict | None = None) -> RenderResult:
    """Render the continuous silent video. Returns what was ACTUALLY written.

    Args:
        images: ``{stimulus_id: {"path": ...}}`` from
            ``s2_design.resolve_stimulus_images``. REQUIRED for a real run. If
            omitted, deterministic placeholders are drawn instead -- usable for a
            dry run, never for a recorded one, and ``RenderResult.placeholders``
            records which it was so the two cannot be confused.
    """
    import cv2

    events = events if events is not None else build_schedule(cfg)
    if images is not None:
        missing = sorted({e.stimulus_id for e in events} - set(images))
        if missing:
            raise ValueError(
                f"{len(missing)} scheduled stimuli have no image: {missing[:5]}. "
                "Refusing to render a stimulus that does not match the schedule."
            )
    plan = frame_plan(cfg, events)
    by_id = {e.event_id: e for e in events}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = cfg.frame_size
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(cfg.fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open a video writer for {out_path}")
    grey = _grey_frame(cfg)
    cache: dict[int, np.ndarray] = {}
    n_stim = 0
    try:
        for eid in plan:
            if eid < 0:
                writer.write(grey)
            else:
                if eid not in cache:
                    ev_ = by_id[int(eid)]
                    cache[eid] = (load_stimulus_image(images[ev_.stimulus_id]["path"], cfg)
                                  if images is not None else _exemplar_frame(cfg, ev_))
                writer.write(cache[eid])
                n_stim += 1
    finally:
        writer.release()

    probe = probe_video(out_path)
    return RenderResult(
        path=out_path, n_frames=probe["n_frames"], fps=probe["fps"],
        duration_s=probe["duration_s"], width=probe["width"], height=probe["height"],
        stimulus_frames=n_stim, grey_frames=int(len(plan) - n_stim),
        placeholders=images is None,
    )


def probe_video(path: str | Path) -> dict:
    """Read back what is really on disk. Never trust the writer's own arithmetic."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return {"n_frames": n, "fps": int(round(fps)) if fps else 0,
            "duration_s": (n / fps) if fps else 0.0, "width": w, "height": h}


def verify_rendered_frames(cfg: S2Config, path: str | Path, events: list[Event],
                           *, sample: int = 40, seed: int = 0) -> list[str]:
    """Spot-check that presentation frames differ from ISI frames ON DISK.

    The manifest can be perfectly self-consistent while the pixels are wrong. This
    samples real frames and asserts that a frame the schedule calls a presentation
    is not uniformly grey, and that a frame it calls an ISI is.
    """
    import cv2

    problems: list[str] = []
    plan = frame_plan(cfg, events)
    rng = np.random.default_rng(seed)
    stim_idx = np.flatnonzero(plan >= 0)
    grey_idx = np.flatnonzero(plan < 0)
    if stim_idx.size == 0:
        return ["no presentation frames in the plan"]
    picks = np.concatenate([
        rng.choice(stim_idx, size=min(sample // 2, stim_idx.size), replace=False),
        rng.choice(grey_idx, size=min(sample // 2, grey_idx.size), replace=False),
    ])
    cap = cv2.VideoCapture(str(path))
    try:
        for i in sorted(int(x) for x in picks):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                problems.append(f"frame {i} unreadable")
                continue
            # Test FLATNESS, not the absolute level. mp4v encodes through YUV420,
            # which shifts a nominal 128 grey to ~124 with a few levels of spread,
            # so an exact-value check fails on a perfectly good render. What
            # actually separates an ISI frame from a presentation is variance.
            # The mean is still bounded so that a black or dropped frame -- a real
            # rendering failure -- cannot pass as "flat".
            f = frame.astype(float)
            is_flat = bool(f.std() < 4.0)
            near_grey = bool(abs(f.mean() - cfg.grey_level) <= 16)
            if plan[i] >= 0 and is_flat:
                problems.append(
                    f"frame {i} should show event {plan[i]} but is flat (std={f.std():.2f})")
            if plan[i] < 0:
                if not is_flat:
                    problems.append(
                        f"frame {i} should be an ISI grey frame but has std={f.std():.2f}")
                elif not near_grey:
                    problems.append(
                        f"frame {i} is flat but at level {f.mean():.1f}, not ~{cfg.grey_level}")
    finally:
        cap.release()
    return problems
