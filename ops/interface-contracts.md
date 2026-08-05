# Interface Contracts — The API Bible

Every public function across the project is defined here BEFORE implementation.
If two modules disagree on a data shape, we catch it here, not at runtime on
a borrowed GPU where debugging costs real time.

**Rule: Change this file FIRST, then change the code. Never the reverse.**

---

## Core Data Types

```python
import numpy as np
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass

# The standard brain prediction array
# Shape: (n_kept_segments, n_vertices), dtype: float32
# Axis 0: kept segments (segments with no events are dropped)
# Axis 1: vertices on fsaverage5 surface (left hemi first, then right hemi)
# Vertex count per hemi: FSAVERAGE_SIZES["fsaverage5"] (check at runtime)
# Total vertices: 2 * FSAVERAGE_SIZES["fsaverage5"]
# IMPORTANT: n_kept_segments <= n_total_segments because empty segments are removed
BrainPrediction = np.ndarray  # (n_kept_segments, n_vertices) float32

# Segment info returned alongside predictions — maps each row to its time range
# Each segment has: start, stop, and the events it contains
SegmentInfo = list  # list of neuralset Segment objects

# Modality masking — which features to zero out
# These are the feature names used by TRIBE v2's features_to_mask config
FeatureName = str  # "video", "audio", "text" — confirmed from main.py:98-100 (G017 closed)

# Modality selection for our ablation API
Modality = Literal["full", "video_only", "audio_only", "text_only"]

# Hemisphere selection
Hemisphere = Literal["left", "right", "both"]
```

---

## tribe_tools/model.py

```python
def load_model(
    device: str = "cuda",
    cache_folder: Optional[Path] = None,
    config_update: Optional[dict] = None,
) -> "TribeModel":
    """
    Load TRIBE v2 model via TribeModel.from_pretrained("facebook/tribev2").

    This is a thin wrapper. TRIBE v2's pipeline already handles sequential
    encoder loading internally — each extractor loads, caches features to disk,
    and frees GPU before the next one loads. No custom loading needed.

    Args:
        device: "cuda" or "cpu". CPU works but is slow.
        cache_folder: Where to cache downloaded model weights.
            Defaults to HuggingFace cache (~/.cache/huggingface/).
        config_update: Dotted-path overrides forwarded to from_pretrained's own
            config_update. Needed for settings that MUST be applied at
            construction, because the extractors freeze once their cache uid is
            first computed and cannot be changed afterwards. Established uses:

                data.video_feature.infra.keep_in_ram: False
                data.audio_feature.infra.keep_in_ram: False
                data.text_feature.infra.keep_in_ram: False
                    keep_in_ram defaults to True on all three extractors, so every
                    feature read during dataloading is retained forever and RSS
                    grows linearly with the number of stimuli. This is the hard
                    ceiling on corpus size on a 13-16 GB Kaggle box.

                data.batch_size: <int>
                    predict() materialises (batch_size, n_vertices, n_TRs) float32.
                    At batch_size=64 that is ~524 MB on GPU plus the same again on
                    the CPU copy.

            None of these keys is part of any extractor cache uid, so setting them
            does not invalidate previously cached features.

    Returns:
        The TribeModel object (from tribev2.demo_utils).

    Raises RuntimeError if GPU requested but not available.
    """

def predict_single(
    model: "TribeModel",
    video_path: Path,
    features_to_mask: Optional[list[str]] = None,
) -> tuple[BrainPrediction, SegmentInfo]:
    """
    Run TRIBE v2 prediction on a single video.

    Wraps model.predict(events=df) with optional modality masking.

    Args:
        model: Loaded TribeModel
        video_path: Path to video file (mp4, etc.)
        features_to_mask: List of feature names to zero out.
            None or [] = full prediction (all modalities active).
            e.g., ["audio", "text"] = video-only (mask audio+text).
            Internally mutates features_to_use via _find_features_to_use()
            so masked modalities are not extracted. The brain model substitutes
            zeros for missing modalities (model.py:189-192).

    Returns:
        (preds, segments) tuple:
        - preds: np.ndarray shape (n_kept_segments, n_vertices), dtype float32
        - segments: list of Segment objects with timing info
        IMPORTANT: Segments with no events are dropped by default.

    Raises:
        RuntimeError: If features_to_use attribute cannot be found on model (G018).

    Internally calls:
        events = model.get_events_dataframe(video_path=video_path)
        preds, segments = model.predict(events=events)
    """
```

---

## tribe_tools/inference.py

```python
def batch_predict(
    model: "TribeModel",
    video_paths: list[Path],
    features_to_mask: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
) -> dict[Path, tuple[BrainPrediction, SegmentInfo]]:
    """
    Run TRIBE v2 on multiple videos with progress tracking and resume support.

    Args:
        model: Loaded TribeModel
        video_paths: List of video file paths
        features_to_mask: Features to mask for ALL videos in this batch.
            For multi-modality runs, call batch_predict once per masking config.
        cache_dir: If provided, save/load results from HDF5 cache here.
            Already-cached videos are skipped (resume support).

    Returns:
        Dict mapping each video path to (preds, segments) tuple.

    Progress is shown via tqdm. Each prediction is persisted by cache.save()
    the moment it completes (the save closes the file), so re-calling this
    function with the same cache_dir resumes from the last completed video —
    there is no separate checkpoint/flush step.
    """
```

---

## tribe_tools/atlas.py

Wraps TRIBE v2's own atlas functions from `tribev2/utils.py`. Uses MNE
(not nilearn) for HCP-MMP1 parcellation. TRIBE v2 already has:
- `get_hcp_labels(mesh, combine, hemi)` → dict[str, np.ndarray]
- `get_hcp_roi_indices(rois, hemi, mesh)` → np.ndarray
- `summarize_by_roi(data, hemi, mesh)` → np.ndarray
- `get_topk_rois(data, hemi, mesh, k)` → list[str]

Our wrapper adds convenience and error handling.

```python
def get_vertices(
    region_name: str,
    hemi: Hemisphere = "both",
    mesh: str = "fsaverage5",
) -> np.ndarray:
    """
    Get vertex indices for a named HCP-MMP1 region.

    Wraps tribev2.utils.get_hcp_roi_indices().
    Supports wildcard matching: "V1*" matches all V1 subregions.

    Args:
        region_name: HCP-MMP1 region label (e.g., "V1", "FFC", "A1")
            or wildcard pattern ("V1*", "*_FEF").
        hemi: "left", "right", or "both"
        mesh: Surface mesh. Default "fsaverage5".

    Returns:
        np.ndarray of vertex indices into the prediction array.

    Raises ValueError if region not found (with list of available regions).
    """

def summarize_by_roi(
    data: np.ndarray,
    hemi: Hemisphere = "both",
    mesh: str = "fsaverage5",
) -> dict[str, float]:
    """
    Compute mean activation per HCP-MMP1 region.

    Wraps tribev2.utils.summarize_by_roi() but returns a dict instead
    of a raw array for easier use.

    Args:
        data: 1D array of per-vertex values, shape (n_vertices,).
            If 2D (n_segments, n_vertices), pass the mean across segments.

    Returns:
        Dict mapping region name → mean activation value.
    """

def get_topk_rois(
    data: np.ndarray,
    k: int = 10,
    hemi: Hemisphere = "both",
    mesh: str = "fsaverage5",
) -> list[str]:
    """
    Get the k most activated regions.

    Wraps tribev2.utils.get_topk_rois().

    Args:
        data: 1D array of per-vertex values.
        k: Number of top regions to return.

    Returns:
        List of region name strings, ordered by activation (descending).
    """

def list_regions(
    hemi: Hemisphere = "both",
    mesh: str = "fsaverage5",
) -> list[str]:
    """Return all HCP-MMP1 region names for the given hemisphere."""
```

---

## tribe_tools/viz.py

```python
def plot_activation(
    data: np.ndarray,
    title: str = "",
    cmap: str = "cold_hot",
    hemisphere: Hemisphere = "both",
    output_path: Optional[Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> "matplotlib.figure.Figure":
    """
    Plot activation map on fsaverage5 brain surface.

    Args:
        data: shape (n_vertices,) — one value per vertex.
            If 2D (n_segments, n_vertices), uses mean across segments.
        title: Plot title
        cmap: Colormap name (nilearn colormaps)
        hemisphere: Which hemisphere(s) to show
        output_path: If provided, saves PNG to this path
        vmin/vmax: Colorbar range. If None, uses data range.

    Returns matplotlib figure. Displays inline in notebooks.
    """

def plot_contrast(
    data_a: np.ndarray,
    data_b: np.ndarray,
    title: str = "A vs B",
    output_path: Optional[Path] = None,
) -> "matplotlib.figure.Figure":
    """
    Plot the difference (A - B) on brain surface.
    data_a, data_b: shape (n_vertices,) or (n_segments, n_vertices).
    Uses mean across segments if 2D.
    Positive values (A > B) shown in warm colors, negative in cool colors.
    """

def plot_rgb_attribution(
    visual: np.ndarray,
    auditory: np.ndarray,
    language: np.ndarray,
    title: str = "Modality Attribution",
    output_path: Optional[Path] = None,
) -> "matplotlib.figure.Figure":
    """
    BrainLens signature visualization.
    Three arrays shape (n_vertices,), each representing one modality's contribution.
    Normalized to [0,1], mapped to RGB: red=visual, blue=auditory, green=language.
    Plotted on fsaverage5 surface using nilearn.
    """

def export_html(
    data: np.ndarray,
    title: str = "",
    output_path: Path = Path("brain_map.html"),
    hemisphere: Hemisphere = "both",
) -> Path:
    """
    Export interactive 3D brain map as HTML using nilearn.
    When hemisphere="both", saves two files ({stem}_left.html, {stem}_right.html).
    Returns path to the first saved HTML file.
    """
```

---

## tribe_tools/cache.py

```python
class PredictionCache:
    """HDF5 cache for brain predictions.

    Note: Only prediction arrays are cached, not segment timing info.
    When loading from cache, segments will be empty. For workflows that
    need segment data, re-run inference without cache.
    """
    def __init__(self, cache_path: Path): ...
    def has(self, key: str) -> bool: ...
    def save(self, key: str, data: np.ndarray, metadata: Optional[dict] = None): ...
    def load(self, key: str) -> Optional[np.ndarray]: ...
    def keys(self) -> list[str]: ...


def get_cache(cache_dir: Optional[Path]) -> Optional["PredictionCache"]:
    """
    Open or create an HDF5 prediction cache at cache_dir/predictions.h5.
    Returns None if cache_dir is None.
    The cache persists across sessions (critical for Kaggle).

    Cache keys are "{video_absolute_path}_{mask_key}" strings.
    """
```

---

## tribe_tools/video_utils.py

```python
def extract_audio(video_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Extract audio track from video using ffmpeg.
    Returns path to .wav file.
    Raises FileNotFoundError if ffmpeg not installed.
    """

def image_to_video(
    image_path: Path,
    duration: float = 3.0,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Convert a static image to a video (for TRIBE v2 which expects video input).
    Creates a video of the image displayed for `duration` seconds.
    Returns path to .mp4 file.
    """

def segment_video(
    video_path: Path,
    segment_duration: float = 30.0,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """
    Split a video into segments of segment_duration seconds.
    Returns list of paths to segment files.
    Useful for processing long videos within memory limits.
    """

def get_video_info(video_path: Path) -> dict:
    """
    Get video metadata: duration, fps, resolution, has_audio.
    Returns dict with keys: duration_seconds, fps, width, height, has_audio.
    """
```

---

## brainlens/

```python
# brainlens/inference.py
def run_ablation(
    model: "TribeModel",
    video_path: Path,
    cache_dir: Optional[Path] = None,
) -> dict[Modality, tuple[BrainPrediction, SegmentInfo]]:
    """
    Run the 4-pass modality ablation: full, video-only, audio-only, text-only.

    Runs predict_single() 4 times with different features_to_mask:
    - full:       features_to_mask=[]
    - video_only: features_to_mask=["audio", "text"]
    - audio_only: features_to_mask=["video", "text"]
    - text_only:  features_to_mask=["video", "audio"]

    Returns dict mapping Modality to (preds, segments) tuple.
    Uses cache if available to skip already-computed passes.
    """

# brainlens/attribution.py
def compute_attribution(
    predictions: dict[Modality, BrainPrediction],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-vertex modality attribution from the 4 ablation passes.

    Method: For each vertex, correlation between full-model activation
    and each single-modality activation across segments.

    Returns:
        visual_attr: shape (n_vertices,) — visual contribution per vertex
        auditory_attr: shape (n_vertices,) — auditory contribution
        language_attr: shape (n_vertices,) — language contribution
    All values in [0, 1], summing to 1 per vertex.
    """

# brainlens/visualization.py
def create_brain_map(
    visual_attr: np.ndarray,
    auditory_attr: np.ndarray,
    language_attr: np.ndarray,
    title: str = "",
    output_path: Optional[Path] = None,
) -> "matplotlib.figure.Figure":
    """
    Create the BrainLens RGB brain map.
    Calls tribe_tools.viz.plot_rgb_attribution() under the hood.
    """

# brainlens/cli.py
# Entry point: python -m brainlens.cli
# Arguments:
#   --video PATH        (required) Path to video file
#   --output PATH       (optional) Output PNG path. Default: brainlens_output.png
#   --cache-dir PATH    (optional) HDF5 cache directory
#   --device DEVICE     (optional) cuda or cpu. Default: cuda
```

---

## neurocheck/claims_db/claims.yaml

```yaml
# Schema for each claim:
- id: "NC001"
  claim: "Faces activate the fusiform face area more than houses"
  citation: "Kanwisher et al., 1997, J Neurosci"
  doi: "10.1523/JNEUROSCI.17-11-04302.1997"
  category: "visual_selectivity"
  roi:
    atlas: "glasser"             # REQUIRED: always "glasser" (HCP-MMP1)
    region: "FFC"                # Glasser atlas label for fusiform face complex
    hemisphere: "both"           # "left", "right", or "both"
  contrast:
    stimulus_a:
      description: "Video/image of human faces"
      source: "CelebA dataset or similar"  # REQUIRED: dataset/origin
    stimulus_b:
      description: "Video/image of houses/buildings"
      source: "Places365 dataset, house category"  # REQUIRED: dataset/origin
    direction: "a_greater_than_b"
    expected_effect_size: 0.5    # Cohen's d minimum
  difficulty: "easy"             # easy/medium/hard based on how well-established
  notes: "One of the most replicated findings in cognitive neuroscience"
```

**Required fields per claim:** id, claim, citation, doi, category, roi.atlas,
roi.region, roi.hemisphere, contrast.stimulus_a.description, contrast.stimulus_a.source,
contrast.stimulus_b.description, contrast.stimulus_b.source, contrast.direction,
contrast.expected_effect_size, difficulty.

**Optional fields:** notes.

---

## Update Protocol

When you need to change a function signature:
1. Update this file first
2. Check which other modules call this function
3. Update all callers
4. Update tests

When you discover the contract is wrong (e.g., TRIBE v2 outputs a different shape):
1. Update source-of-truth.md with the verified fact
2. Update this file to match
3. Update all affected code
