"""BrainLens RGB brain map visualization."""

from pathlib import Path
from typing import Optional

import numpy as np

from tribe_tools.viz import plot_rgb_attribution


def create_brain_map(
    visual_attr: np.ndarray,
    auditory_attr: np.ndarray,
    language_attr: np.ndarray,
    title: str = "",
    output_path: Optional[Path] = None,
):
    """Create the BrainLens RGB brain map.

    Red = visual, Green = language, Blue = auditory.

    Args:
        visual_attr: shape (n_vertices,) — visual contribution.
        auditory_attr: shape (n_vertices,) — auditory contribution.
        language_attr: shape (n_vertices,) — language contribution.
        title: Plot title.
        output_path: If provided, saves PNG.

    Returns:
        matplotlib figure.
    """
    return plot_rgb_attribution(
        visual=visual_attr,
        auditory=auditory_attr,
        language=language_attr,
        title=title or "BrainLens: Modality Attribution",
        output_path=output_path,
    )
