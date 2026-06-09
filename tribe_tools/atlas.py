"""HCP-MMP1 atlas integration.

Wraps TRIBE v2's own atlas functions from tribev2/utils.py, which use MNE
for HCP-MMP1 parcellation on fsaverage5. Falls back to direct MNE calls
if tribev2 is not installed.
"""

import logging
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_TRIBEV2_AVAILABLE = None


def _has_tribev2() -> bool:
    global _TRIBEV2_AVAILABLE
    if _TRIBEV2_AVAILABLE is None:
        try:
            from tribev2.utils import get_hcp_labels  # noqa: F401
            _TRIBEV2_AVAILABLE = True
        except ImportError:
            _TRIBEV2_AVAILABLE = False
    return _TRIBEV2_AVAILABLE


def _get_hcp_labels(mesh="fsaverage5", combine=False, hemi="both"):
    """Get HCP label-to-vertex mapping, using tribev2 if available."""
    if _has_tribev2():
        from tribev2.utils import get_hcp_labels
        return get_hcp_labels(mesh=mesh, combine=combine, hemi=hemi)

    # Fallback: direct MNE implementation (same logic as tribev2/utils.py)
    return _get_hcp_labels_mne(mesh=mesh, combine=combine, hemi=hemi)


@lru_cache(maxsize=8)
def _get_hcp_labels_mne(mesh="fsaverage5", combine=False, hemi="both"):
    """Direct MNE implementation matching tribev2/utils.py logic."""
    from pathlib import Path
    import mne

    if hemi in ["right", "left"]:
        subjects_dir = Path(mne.datasets.sample.data_path()) / "subjects"
        mne.datasets.fetch_hcp_mmp_parcellation(
            subjects_dir=subjects_dir, accept=True, verbose=False, combine=combine
        )
        name = "HCPMMP1_combined" if combine else "HCPMMP1"
        labels = mne.read_labels_from_annot(
            "fsaverage", name, hemi="both", subjects_dir=subjects_dir
        )
        label_to_vertices = {}
        for label in labels:
            lname, vertices = label.name, np.array(label.vertices)
            if not combine:
                lname = lname[2:]
            lname = lname.replace("_ROI", "")
            if (hemi == "right" and "-lh" in lname) or (
                hemi == "left" and "-rh" in lname
            ):
                continue
            lname = lname.replace("-rh", "").replace("-lh", "")
            label_to_vertices[lname] = np.array(vertices)

        # Map to target mesh resolution
        fsaverage_size = _get_fsaverage_size(mesh)
        index_offset = fsaverage_size if hemi == "right" else 0
        label_to_vertices = {
            k: v[v < fsaverage_size] + index_offset
            for k, v in label_to_vertices.items()
        }
        # Sanity check: all vertex indices must be within expected range
        total_vertices = fsaverage_size * 2
        for name, verts in label_to_vertices.items():
            if len(verts) > 0:
                assert verts.max() < total_vertices, (
                    f"Region {name}: vertex {verts.max()} >= {total_vertices}"
                )
        return label_to_vertices
    else:
        left = _get_hcp_labels_mne(mesh=mesh, combine=combine, hemi="left")
        right = _get_hcp_labels_mne(mesh=mesh, combine=combine, hemi="right")
        return {
            k: np.concatenate([left[k], right[k]]) for k in left.keys()
        }


def _get_fsaverage_size(mesh: str) -> int:
    """Get vertex count per hemisphere for a given fsaverage mesh."""
    try:
        from neuralset.extractors.neuro import FSAVERAGE_SIZES
        return FSAVERAGE_SIZES[mesh]
    except ImportError:
        # Fallback: known sizes
        sizes = {
            "fsaverage": 163842,
            "fsaverage6": 40962,
            "fsaverage5": 10242,
            "fsaverage4": 2562,
            "fsaverage3": 642,
        }
        if mesh not in sizes:
            raise ValueError(f"Unknown mesh: {mesh}. Known: {list(sizes.keys())}")
        return sizes[mesh]


def get_vertices(
    region_name: str,
    hemi: str = "both",
    mesh: str = "fsaverage5",
) -> np.ndarray:
    """Get vertex indices for a named HCP-MMP1 region.

    Supports wildcard matching: "V1*" matches all V1 subregions.

    Args:
        region_name: HCP-MMP1 label (e.g., "V1", "FFC", "A1") or wildcard.
        hemi: "left", "right", or "both".
        mesh: Surface mesh. Default "fsaverage5".

    Returns:
        np.ndarray of vertex indices.
    """
    if _has_tribev2():
        from tribev2.utils import get_hcp_roi_indices
        return get_hcp_roi_indices(region_name, hemi=hemi, mesh=mesh)

    # Fallback: manual implementation
    labels = _get_hcp_labels(mesh=mesh, hemi=hemi)
    if isinstance(region_name, str):
        region_name = [region_name]

    selected = []
    for roi in region_name:
        if roi.endswith("*"):
            sel = [l for l in labels if l.startswith(roi[:-1])]
        elif roi.startswith("*"):
            sel = [l for l in labels if l.endswith(roi[1:])]
        else:
            sel = [l for l in labels if l == roi]
        if not sel:
            available = sorted(labels.keys())
            raise ValueError(
                f"ROI '{roi}' not found. Available regions: {available[:20]}..."
            )
        selected.extend(sel)

    return np.concatenate([labels[l] for l in selected])


def summarize_by_roi(
    data: np.ndarray,
    hemi: str = "both",
    mesh: str = "fsaverage5",
) -> dict[str, float]:
    """Compute mean activation per HCP-MMP1 region.

    Args:
        data: 1D array shape (n_vertices,).

    Returns:
        Dict mapping region name to mean activation.
    """
    if data.ndim == 2:
        data = data.mean(axis=0)

    labels = _get_hcp_labels(mesh=mesh, hemi=hemi)
    result = {}
    for name, vertices in labels.items():
        if len(vertices) > 0:
            result[name] = float(data[vertices].mean())
    return result


def get_topk_rois(
    data: np.ndarray,
    k: int = 10,
    hemi: str = "both",
    mesh: str = "fsaverage5",
) -> list[str]:
    """Get the k most activated regions.

    Args:
        data: 1D array shape (n_vertices,), or 2D (n_segments, n_vertices).
        k: Number of top regions.

    Returns:
        List of region names ordered by activation (descending).
    """
    roi_means = summarize_by_roi(data, hemi=hemi, mesh=mesh)
    sorted_rois = sorted(roi_means.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_rois[:k]]


def list_regions(
    hemi: str = "both",
    mesh: str = "fsaverage5",
) -> list[str]:
    """Return all HCP-MMP1 region names."""
    labels = _get_hcp_labels(mesh=mesh, hemi=hemi)
    return sorted(labels.keys())
