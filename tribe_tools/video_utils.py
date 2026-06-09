"""Media preprocessing utilities using ffmpeg."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg not found. Install it with: sudo apt install ffmpeg"
        )


def extract_audio(
    video_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Extract audio track from video as WAV.

    Args:
        video_path: Path to video file.
        output_path: Output .wav path. Defaults to same name with .wav extension.

    Returns:
        Path to extracted .wav file.
    """
    _check_ffmpeg()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    output_path = Path(output_path)

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    logger.info("Extracted audio to %s", output_path)
    return output_path


def image_to_video(
    image_path: Path,
    duration: float = 3.0,
    output_path: Optional[Path] = None,
) -> Path:
    """Convert a static image to a video for TRIBE v2 input.

    Args:
        image_path: Path to image file (jpg, png, etc.).
        duration: Video duration in seconds.
        output_path: Output .mp4 path. Defaults to same name with .mp4 extension.

    Returns:
        Path to created .mp4 file.
    """
    _check_ffmpeg()
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_path is None:
        output_path = image_path.with_suffix(".mp4")
    output_path = Path(output_path)

    cmd = [
        "ffmpeg", "-loop", "1", "-i", str(image_path),
        "-c:v", "libx264", "-t", str(duration),
        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-y", str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    logger.info("Created video from image: %s", output_path)
    return output_path


def segment_video(
    video_path: Path,
    segment_duration: float = 30.0,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Split a video into segments.

    Args:
        video_path: Path to video file.
        segment_duration: Duration of each segment in seconds.
        output_dir: Directory for segment files. Defaults to video's directory.

    Returns:
        List of paths to segment files.
    """
    _check_ffmpeg()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_segments"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = output_dir / f"{video_path.stem}_%03d{video_path.suffix}"
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-c", "copy", "-map", "0",
        "-segment_time", str(segment_duration),
        "-f", "segment", "-reset_timestamps", "1",
        "-y", str(pattern),
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    segments = sorted(output_dir.glob(f"{video_path.stem}_*{video_path.suffix}"))
    logger.info("Split into %d segments in %s", len(segments), output_dir)
    return segments


def get_video_info(video_path: Path) -> dict:
    """Get video metadata.

    Returns:
        Dict with keys: duration_seconds, fps, width, height, has_audio.
    """
    _check_ffmpeg()
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe not found (usually installed with ffmpeg)")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    import json
    probe = json.loads(result.stdout)

    info = {
        "duration_seconds": float(probe.get("format", {}).get("duration", 0)),
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "has_audio": False,
    }

    for stream in probe.get("streams", []):
        if stream["codec_type"] == "video":
            info["width"] = int(stream.get("width", 0))
            info["height"] = int(stream.get("height", 0))
            # Parse fps from r_frame_rate like "30/1"
            fps_str = stream.get("r_frame_rate", "0/1")
            num, den = fps_str.split("/")
            info["fps"] = float(num) / float(den) if float(den) > 0 else 0
        elif stream["codec_type"] == "audio":
            info["has_audio"] = True

    return info
