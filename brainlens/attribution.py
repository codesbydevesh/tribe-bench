"""Per-vertex modality attribution from ablation passes.

Computes how much each modality (visual, auditory, language) contributes
to each vertex's activation by correlating single-modality predictions
with the full-model prediction.
"""

import numpy as np


def compute_attribution(
    predictions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-vertex modality attribution from ablation results.

    For each vertex, we compute the correlation between the full-model
    activation timeseries and each single-modality timeseries. Then
    normalize so contributions sum to 1.

    Args:
        predictions: Dict with keys "full", "video_only", "audio_only",
            "text_only". Each value is np.ndarray shape
            (n_kept_segments, n_vertices).

    Returns:
        (visual, auditory, language) — three arrays shape (n_vertices,),
        each in [0, 1], summing to ~1 per vertex.
    """
    full = predictions["full"]
    video = predictions["video_only"]
    audio = predictions["audio_only"]
    text = predictions["text_only"]

    # Ensure all have same number of segments (use minimum)
    min_segs = min(full.shape[0], video.shape[0], audio.shape[0], text.shape[0])
    full = full[:min_segs]
    video = video[:min_segs]
    audio = audio[:min_segs]
    text = text[:min_segs]

    n_vertices = full.shape[1]

    if min_segs < 3:
        # Not enough segments for meaningful correlation.
        # Fall back to variance-explained approach.
        return _attribution_variance(full, video, audio, text)

    # Correlation-based attribution
    visual_attr = _vertex_correlation(full, video)
    auditory_attr = _vertex_correlation(full, audio)
    language_attr = _vertex_correlation(full, text)

    # Clamp negatives to 0 (negative correlation = modality not contributing)
    visual_attr = np.maximum(visual_attr, 0)
    auditory_attr = np.maximum(auditory_attr, 0)
    language_attr = np.maximum(language_attr, 0)

    # Normalize to sum to 1 per vertex
    total = visual_attr + auditory_attr + language_attr
    # Where all correlations are zero/negative, assign equal weight
    dead = total < 1e-10
    visual_attr[dead] = 1.0 / 3
    auditory_attr[dead] = 1.0 / 3
    language_attr[dead] = 1.0 / 3
    total = visual_attr + auditory_attr + language_attr
    visual_attr /= total
    auditory_attr /= total
    language_attr /= total

    return visual_attr, auditory_attr, language_attr


def _vertex_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation between two (n_segments, n_vertices) arrays, per vertex."""
    a_mean = a.mean(axis=0, keepdims=True)
    b_mean = b.mean(axis=0, keepdims=True)
    a_centered = a - a_mean
    b_centered = b - b_mean
    num = (a_centered * b_centered).sum(axis=0)
    den = np.sqrt((a_centered ** 2).sum(axis=0) * (b_centered ** 2).sum(axis=0))
    den = np.maximum(den, 1e-10)
    return num / den


def _attribution_variance(
    full: np.ndarray,
    video: np.ndarray,
    audio: np.ndarray,
    text: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback: attribute by similarity to full model when too few segments.

    Uses mean absolute difference from full prediction as a dissimilarity
    measure, then inverts: modalities closer to the full model get more credit.
    """
    # Smaller difference from full = larger contribution
    video_diff = np.abs(full - video).mean(axis=0)
    audio_diff = np.abs(full - audio).mean(axis=0)
    text_diff = np.abs(full - text).mean(axis=0)

    # Invert: similarity = max_diff - diff (so low diff = high similarity)
    max_diff = np.maximum(np.maximum(video_diff, audio_diff), text_diff) + 1e-10
    video_sim = max_diff - video_diff
    audio_sim = max_diff - audio_diff
    text_sim = max_diff - text_diff

    total = video_sim + audio_sim + text_sim
    total = np.maximum(total, 1e-10)

    return video_sim / total, audio_sim / total, text_sim / total
